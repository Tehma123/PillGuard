"""Train the pill embedding network with an ArcFace loss over the seen drugs.

Usage::

    python -m pillguard.embed.train --root data/vaipe --split splits/vaipe_v1.json --epochs 20

Validation metric: nearest-prototype top-1 accuracy on val crops, where prototypes are the
per-class mean embeddings of train crops (exactly how inference works). The checkpoint with
the best val accuracy is kept as ``best.pt``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from pillguard.config import ARTIFACTS_DIR, EMB_DIM, SEED
from pillguard.data.splits import Split
from pillguard.embed.crops import CropDataset, filter_rows, load_crop_index
from pillguard.embed.model import ArcFaceHead, build_model


@torch.no_grad()
def embed_dataset(model, ds: CropDataset, batch: int = 256, workers: int = 2, device: str = "cuda") -> np.ndarray:
    model.eval()
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=device == "cuda")
    out = []
    for x, _, _ in dl:
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            e = model(x.to(device, non_blocking=True))
        out.append(e.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.emb_dim), dtype=np.float32)


def class_means(embs: np.ndarray, idx: np.ndarray, n_classes: int) -> np.ndarray:
    protos = np.zeros((n_classes, embs.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        m = idx == c
        if m.any():
            v = embs[m].mean(0)
            protos[c] = v / (np.linalg.norm(v) + 1e-9)
    return protos


def nearest_prototype_accuracy(embs: np.ndarray, idx: np.ndarray, protos: np.ndarray) -> float:
    if len(embs) == 0:
        return float("nan")
    pred = (embs @ protos.T).argmax(1)
    return float((pred == idx).mean())


def train(root: Path, split_path: Path, out_dir: Path, backbone: str = "mobilenet_v3_small",
          epochs: int = 20, batch: int = 128, lr: float = 1e-3, backbone_lr: float = 3e-4,
          weight_decay: float = 1e-4, margin: float = 0.30, scale: float = 30.0, workers: int = 4,
          device: str = "cuda", seed: int = SEED, emb_dim: int = EMB_DIM, max_train: int | None = None,
          eval_every: int = 1) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device if torch.cuda.is_available() else "cpu"
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = Split.load(split_path)
    unseen = set(split.unseen_classes)

    rows = load_crop_index(root)
    train_rows = filter_rows(rows, "train")
    classes = sorted({r.label for r in train_rows} - unseen)
    train_rows = [r for r in train_rows if r.label in set(classes)]
    if max_train:
        rng = np.random.default_rng(seed)
        train_rows = [train_rows[i] for i in sorted(rng.choice(len(train_rows), size=min(max_train, len(train_rows)), replace=False))]
    val_rows = filter_rows(rows, "val", set(classes))
    print(f"classes={len(classes)} unseen={sorted(unseen)} train_crops={len(train_rows)} val_crops={len(val_rows)}")

    train_ds = CropDataset(train_rows, root, classes, train=True)
    train_eval_ds = CropDataset(train_rows, root, classes, train=False)
    val_ds = CropDataset(val_rows, root, classes, train=False)

    counts = Counter(r.label for r in train_rows)
    weights = torch.tensor([1.0 / math.sqrt(counts[r.label]) for r in train_rows], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_rows), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    dl = DataLoader(train_ds, batch_size=batch, sampler=sampler, num_workers=workers, drop_last=True,
                    pin_memory=device == "cuda", persistent_workers=workers > 0)

    model = build_model(backbone, emb_dim, pretrained=True).to(device)
    head = ArcFaceHead(len(classes), emb_dim, scale=scale, margin=margin).to(device)
    opt = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": backbone_lr},
        {"params": list(model.proj.parameters()) + list(head.parameters()), "lr": lr},
    ], weight_decay=weight_decay)
    steps = epochs * len(dl)
    warm = max(1, int(0.03 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    log_path = out_dir / "log.jsonl"
    best_acc, best_path = -1.0, out_dir / "best.pt"
    t0 = time.time()
    step = 0
    for ep in range(1, epochs + 1):
        model.train(); head.train()
        tot, n, correct = 0.0, 0, 0
        for x, y, _ in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                emb = model(x)
                logits = head(emb.float(), y)
                loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 5.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            tot += loss.item() * len(y); n += len(y)
            correct += (head(emb.detach().float()).argmax(1) == y).sum().item()
        rec = {"epoch": ep, "loss": tot / max(1, n), "train_logit_acc": correct / max(1, n),
               "lr": sched.get_last_lr()[-1], "minutes": (time.time() - t0) / 60}
        if ep % eval_every == 0 or ep == epochs:
            tr_emb = embed_dataset(model, train_eval_ds, workers=min(2, workers), device=device)
            tr_idx = np.array([train_ds.cls_to_idx[r.label] for r in train_rows])
            protos = class_means(tr_emb, tr_idx, len(classes))
            va_emb = embed_dataset(model, val_ds, workers=min(2, workers), device=device)
            va_idx = np.array([val_ds.cls_to_idx[r.label] for r in val_rows])
            rec["val_proto_acc"] = nearest_prototype_accuracy(va_emb, va_idx, protos)
            rec["train_proto_acc"] = nearest_prototype_accuracy(tr_emb, tr_idx, protos)
            if rec["val_proto_acc"] >= best_acc or math.isnan(rec["val_proto_acc"]):
                best_acc = rec["val_proto_acc"]
                torch.save({"model": model.state_dict(), "backbone": backbone, "emb_dim": emb_dim,
                            "classes": classes, "unseen_classes": sorted(unseen), "epoch": ep,
                            "val_proto_acc": best_acc, "head": head.state_dict()}, best_path)
        print(json.dumps(rec), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    torch.save({"model": model.state_dict(), "backbone": backbone, "emb_dim": emb_dim, "classes": classes,
                "unseen_classes": sorted(unseen), "epoch": epochs, "head": head.state_dict()}, out_dir / "last.pt")
    print(f"best val nearest-prototype acc = {best_acc:.4f} -> {best_path}")
    return best_path


def main(argv: list[str] | None = None) -> None:
    from pillguard.config import VAIPE_DIR
    from pillguard.data.splits import default_split_path

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=VAIPE_DIR)
    ap.add_argument("--split", type=Path, default=default_split_path())
    ap.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "embed" / "mnv3s")
    ap.add_argument("--backbone", default="mobilenet_v3_small", choices=["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=3e-4)
    ap.add_argument("--margin", type=float, default=0.30)
    ap.add_argument("--scale", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-train", type=int, default=None)
    a = ap.parse_args(argv)
    train(a.root, a.split, a.out, backbone=a.backbone, epochs=a.epochs, batch=a.batch, lr=a.lr,
          backbone_lr=a.backbone_lr, margin=a.margin, scale=a.scale, workers=a.workers, device=a.device,
          max_train=a.max_train)


if __name__ == "__main__":
    main()
