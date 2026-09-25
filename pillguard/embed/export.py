"""Export the embedding network to ONNX (fp32 + mixed-precision INT8) and check the cost.

Usage::

    python -m pillguard.embed.export --ckpt artifacts/embed/mnv3s/best.pt --root data/vaipe \
        --split splits/vaipe_v1.json --out artifacts/export

Writes ``embed.fp32.onnx`` / ``embed.int8.onnx`` (input ``(N, 3, 128, 128)`` normalised
float, output ``(N, 128)`` unit vectors) and ``embed_export.json`` with the val
nearest-prototype accuracy of both models and their mean cosine agreement.

The INT8 model keeps the first :data:`N_FP32_CONVS` backbone convolutions in fp32 (see
:func:`early_conv_nodes`) and the export fails rather than shipping embeddings that no
longer point where fp32 points (see :func:`check_embedding_quantisation`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pillguard.config import ARTIFACTS_DIR, CROP_SIZE
from pillguard.embed.crops import eval_crop_array, filter_rows, load_crop_index
from pillguard.embed.model import load_embedding_net
from pillguard.embed.train import class_means, nearest_prototype_accuracy
from pillguard.imaging import to_embed_input
from pillguard.onnx_utils import (
    QuantizationError,
    export_torch,
    onnx_size_mb,
    ort_session,
    quantize_static_int8,
    run,
)

N_FP32_CONVS = 16             # backbone convolutions left unquantised (see early_conv_nodes)
MIN_INT8_COSINE = 0.95        # mean cosine between int8 and fp32 embeddings of the same crop
MAX_INT8_ACC_DROP = 0.05      # nearest-prototype accuracy quantisation may cost


def embed_with_onnx(session, crops_u8: np.ndarray, batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(crops_u8), batch):
        e = run(session, to_embed_input(crops_u8[i:i + batch]))
        out.append(e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9))
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def early_conv_nodes(fp32_path: Path, n: int = N_FP32_CONVS) -> list[str]:
    """First ``n`` convolutions of the backbone, to keep in fp32.

    MobileNetV3's early layers see the normalised crop directly and produce activations with
    a far wider range than the rest of the network, so one uint8 scale wrecks them: quantising
    them costs ~67 points of nearest-prototype accuracy while saving almost no bytes (they
    hold a few thousand weights out of 1.5 M).
    """
    import onnx

    convs = [node.name for node in onnx.load(str(fp32_path)).graph.node if node.op_type == "Conv"]
    return convs[:n]


def check_embedding_quantisation(report: dict) -> None:
    """Raise unless the INT8 embeddings still agree with fp32.

    A quantisation that dropped nearest-prototype accuracy from 0.878 to 0.179 was reported
    faithfully in ``embed_export.json`` and shipped anyway, taking the prototypes, the
    calibration and every threshold fitted on top of them with it.
    """
    cos, acc8 = report.get("int8_vs_fp32_cosine_mean"), report.get("int8_val_proto_acc")
    acc32 = report.get("fp32_val_proto_acc")
    problems = []
    if cos is not None and cos < MIN_INT8_COSINE:
        problems.append(f"mean cosine against fp32 {cos:.3f} < {MIN_INT8_COSINE}")
    if acc8 is not None and acc32 is not None and acc32 - acc8 > MAX_INT8_ACC_DROP:
        problems.append(f"val nearest-prototype accuracy {acc32:.3f} -> {acc8:.3f} "
                        f"(drop {acc32 - acc8:.3f} > {MAX_INT8_ACC_DROP})")
    if problems:
        raise QuantizationError("INT8 embedding disagrees with fp32: " + "; ".join(problems))


def export_embedding(ckpt: Path, root: Path, split_path: Path, out_dir: Path, n_calib: int = 256,
                     n_check: int = 4000, method: str = "percentile", seed: int = 0,
                     n_fp32_convs: int = N_FP32_CONVS, strict: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_embedding_net(ckpt)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    classes = list(ck["classes"])
    fp32 = export_torch(model, torch.zeros(1, 3, CROP_SIZE, CROP_SIZE), out_dir / "embed.fp32.onnx",
                        "input", "embedding", dynamic_batch=True)

    rows = load_crop_index(root)
    seen = set(classes)
    rng = np.random.default_rng(seed)
    train_rows = filter_rows(rows, "train", seen)
    val_rows = filter_rows(rows, "val", seen)
    if len(train_rows) > n_check:
        train_rows = [train_rows[i] for i in sorted(rng.choice(len(train_rows), n_check, replace=False))]
    if len(val_rows) > n_check:
        val_rows = [val_rows[i] for i in sorted(rng.choice(len(val_rows), n_check, replace=False))]
    calib_rows = [train_rows[i] for i in sorted(rng.choice(len(train_rows), min(n_calib, len(train_rows)), replace=False))]
    calib = [to_embed_input(eval_crop_array(calib_rows[i:i + 8], root)) for i in range(0, len(calib_rows), 8)]
    int8 = quantize_static_int8(fp32, out_dir / "embed.int8.onnx", calib, "input", method=method,
                                nodes_to_exclude=early_conv_nodes(fp32, n_fp32_convs))

    tr_u8, va_u8 = eval_crop_array(train_rows, root), eval_crop_array(val_rows, root)
    cls_idx = {c: i for i, c in enumerate(classes)}
    tr_idx = np.array([cls_idx[r.label] for r in train_rows]); va_idx = np.array([cls_idx[r.label] for r in val_rows])
    report = {"fp32_mb": onnx_size_mb(fp32), "int8_mb": onnx_size_mb(int8), "n_train_check": len(train_rows),
              "n_val_check": len(val_rows), "checkpoint": str(ckpt), "epoch": ck.get("epoch")}
    embs = {}
    for name, path in (("fp32", fp32), ("int8", int8)):
        s = ort_session(path)
        tr_e, va_e = embed_with_onnx(s, tr_u8), embed_with_onnx(s, va_u8)
        protos = class_means(tr_e, tr_idx, len(classes))
        report[f"{name}_val_proto_acc"] = nearest_prototype_accuracy(va_e, va_idx, protos)
        embs[name] = va_e
    if len(val_rows):
        report["int8_vs_fp32_cosine_mean"] = float((embs["fp32"] * embs["int8"]).sum(1).mean())
    with torch.no_grad():
        t_e = model(torch.from_numpy(to_embed_input(va_u8[:64]))).numpy() if len(va_u8) else np.zeros((0, 1))
    if len(t_e):
        s = ort_session(fp32)
        report["torch_vs_fp32_max_abs"] = float(np.abs(t_e - run(s, to_embed_input(va_u8[:64]))).max())
    (out_dir / "embed_export.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    try:
        check_embedding_quantisation(report)
    except QuantizationError as e:
        if strict:
            raise
        print(f"  WARNING: {e}")
    return report


def main(argv: list[str] | None = None) -> None:
    from pillguard.config import VAIPE_DIR
    from pillguard.data.splits import default_split_path

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=VAIPE_DIR)
    ap.add_argument("--split", type=Path, default=default_split_path())
    ap.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "export")
    ap.add_argument("--n-calib", type=int, default=256)
    ap.add_argument("--method", default="percentile", choices=["minmax", "entropy", "percentile"])
    ap.add_argument("--n-fp32-convs", type=int, default=N_FP32_CONVS)
    ap.add_argument("--allow-degraded", action="store_true", help="warn instead of failing on INT8 drift")
    a = ap.parse_args(argv)
    export_embedding(a.ckpt, a.root, a.split, a.out, n_calib=a.n_calib, method=a.method,
                     n_fp32_convs=a.n_fp32_convs, strict=not a.allow_degraded)


if __name__ == "__main__":
    main()
