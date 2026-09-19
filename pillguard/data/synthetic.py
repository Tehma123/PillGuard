"""Synthetic pill dataset in the exact layout produced by :mod:`pillguard.data.ingest`.

Used by the test-suite and for smoke-running the whole pipeline without VAIPE. Each class
is a (shape, colour, imprint) triple drawn with PIL; images contain 3-7 rotated pills on a
noisy background. A fraction of images receive one pill from a class that is *not* on the
image's prescription and is labelled ``OUT_OF_PRESCRIPTION_LABEL`` (107), mirroring VAIPE.

Example::

    from pillguard.data.synthetic import make_synthetic_dataset
    make_synthetic_dataset(Path("/tmp/synth"), n_images=40, n_classes=8, seed=0)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from pillguard.config import OUT_OF_PRESCRIPTION_LABEL
from pillguard.data.vaipe import write_jsonl

SHAPES = ("round", "oval", "capsule", "square")
PALETTE = [
    (245, 245, 240), (230, 200, 60), (200, 60, 60), (60, 120, 200), (90, 170, 90),
    (240, 150, 60), (150, 90, 170), (120, 120, 120), (250, 200, 200), (200, 240, 250),
    (100, 60, 30), (20, 20, 20),
]
MARKS = ("none", "line", "cross", "dot", "ring")


@dataclass(frozen=True)
class PillClass:
    id: int
    name: str
    shape: str
    color: tuple[int, int, int]
    mark: str
    size: int  # nominal diameter in px


def make_classes(n_classes: int, seed: int = 0) -> list[PillClass]:
    rng = np.random.default_rng(seed)
    combos = [(s, c, m) for s in SHAPES for c in range(len(PALETTE)) for m in MARKS]
    idx = rng.permutation(len(combos))[:n_classes]
    out = []
    for i, k in enumerate(idx):
        s, c, m = combos[k]
        out.append(PillClass(i, f"SYNTH-{s.upper()}-{m}-{c} {int(rng.integers(5, 500))}mg", s,
                             PALETTE[c], m, int(rng.integers(44, 70))))
    return out


def _draw_pill(cls: PillClass, angle: float, rng: np.random.Generator) -> Image.Image:
    d = cls.size
    w, h = (d, d) if cls.shape in ("round", "square") else (int(d * 1.7), d)
    if cls.shape == "capsule":
        w = int(d * 2.1)
    pad = 6
    layer = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    jitter = tuple(int(np.clip(c + rng.integers(-12, 13), 0, 255)) for c in cls.color)
    box = [pad, pad, pad + w, pad + h]
    if cls.shape == "round" or cls.shape == "oval":
        dr.ellipse(box, fill=jitter + (255,), outline=(0, 0, 0, 90))
    elif cls.shape == "capsule":
        dr.rounded_rectangle(box, radius=h // 2, fill=jitter + (255,), outline=(0, 0, 0, 90))
        # second colour on one half, like real capsules
        second = tuple(int(np.clip(255 - c, 0, 255)) for c in cls.color)
        half = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        ImageDraw.Draw(half).rounded_rectangle(box, radius=h // 2, fill=second + (255,))
        mask = Image.new("L", layer.size, 0)
        ImageDraw.Draw(mask).rectangle([pad + w // 2, 0, layer.size[0], layer.size[1]], fill=255)
        layer.paste(half, (0, 0), mask)
    else:
        dr.rounded_rectangle(box, radius=6, fill=jitter + (255,), outline=(0, 0, 0, 90))
    cx, cy = pad + w / 2, pad + h / 2
    ink = (40, 40, 40, 220) if sum(cls.color) > 300 else (230, 230, 230, 220)
    if cls.mark == "line":
        dr.line([cx - w * 0.3, cy, cx + w * 0.3, cy], fill=ink, width=3)
    elif cls.mark == "cross":
        dr.line([cx - w * 0.3, cy, cx + w * 0.3, cy], fill=ink, width=3)
        dr.line([cx, cy - h * 0.3, cx, cy + h * 0.3], fill=ink, width=3)
    elif cls.mark == "dot":
        r = max(2, d // 8)
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ink)
    elif cls.mark == "ring":
        r = d * 0.3
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=ink, width=3)
    # soft shadow + shading
    shade = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    ImageDraw.Draw(shade).ellipse([box[0] + 4, box[1] + 4, box[2] + 4, box[3] + 4], fill=(0, 0, 0, 70))
    shade = shade.filter(ImageFilter.GaussianBlur(3))
    out = Image.alpha_composite(shade, layer)
    return out.rotate(angle, resample=Image.BILINEAR, expand=True)


def _background(w: int, h: int, rng: np.random.Generator) -> Image.Image:
    base = np.array(rng.integers(120, 235, size=3), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = (xx / w * rng.uniform(-40, 40) + yy / h * rng.uniform(-40, 40))[..., None]
    noise = rng.normal(0, 6, size=(h, w, 3))
    arr = np.clip(base + grad + noise, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr, "RGB")
    if rng.random() < 0.5:  # a "hand" or table edge
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, int(h * rng.uniform(0.6, 0.9)), w, h],
                     fill=tuple(int(v) for v in rng.integers(90, 200, size=3)))
    return im


def make_synthetic_dataset(root: Path, n_images: int = 40, n_classes: int = 8, seed: int = 0,
                           image_size: tuple[int, int] = (640, 480), n_prescriptions: int | None = None,
                           pills_per_image: tuple[int, int] = (3, 7), p_foreign: float = 0.35,
                           split: str = "train") -> dict:
    """Write a synthetic dataset to ``root``. Returns the class table."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    classes = make_classes(n_classes, seed)
    (root / "images" / "pill").mkdir(parents=True, exist_ok=True)
    (root / "images" / "prescription").mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)

    n_pres = n_prescriptions or max(2, n_images // 4)
    prescriptions = []
    for p in range(n_pres):
        k = int(rng.integers(2, min(5, n_classes) + 1))
        drugs = sorted(int(c) for c in rng.choice(n_classes, size=k, replace=False))
        prescriptions.append(drugs)

    W, H = image_size
    pill_rows, pres_map = [], {}
    for i in range(n_images):
        p = i % n_pres
        pres_file = f"SYNTH_P_{split.upper()}_{p}.jpg"
        fname = f"SYNTH_P_{p}_{i}.jpg"
        pres_map.setdefault(pres_file, []).append(fname)
        im = _background(W, H, rng)
        n = int(rng.integers(pills_per_image[0], pills_per_image[1] + 1))
        drugs = prescriptions[p]
        picks = [int(rng.choice(drugs)) for _ in range(n)]
        labels = list(picks)
        foreign = [c for c in range(n_classes) if c not in drugs]
        if foreign and rng.random() < p_foreign:
            picks.append(int(rng.choice(foreign)))
            labels.append(OUT_OF_PRESCRIPTION_LABEL)
        boxes, occupied = [], []
        for cls_id in picks:
            sprite = _draw_pill(classes[cls_id], float(rng.uniform(0, 360)), rng)
            sw, sh = sprite.size
            for _ in range(40):
                x, y = int(rng.integers(0, max(1, W - sw))), int(rng.integers(0, max(1, H - sh)))
                cand = (x, y, x + sw, y + sh)
                if all(cand[2] < o[0] + 10 or cand[0] > o[2] - 10 or cand[3] < o[1] + 10 or cand[1] > o[3] - 10
                       for o in occupied):
                    break
            else:
                continue
            im.paste(sprite, (x, y), sprite)
            a = np.array(sprite.split()[-1])
            ys, xs = np.where(a > 40)
            boxes.append([float(x + xs.min()), float(y + ys.min()), float(x + xs.max() + 1), float(y + ys.max() + 1)])
            occupied.append(cand)
        labels = labels[: len(boxes)]
        im = im.filter(ImageFilter.GaussianBlur(float(rng.uniform(0, 0.8))))
        im.save(root / "images" / "pill" / fname, "JPEG", quality=88)
        pill_rows.append({"file": fname, "orig_file": fname, "src_split": split, "width": W, "height": H,
                          "orig_size": [W, H], "boxes": boxes, "labels": labels, "n_dropped_boxes": 0})

    pres_rows = []
    for p, drugs in enumerate(prescriptions):
        pres_file = f"SYNTH_P_{split.upper()}_{p}.jpg"
        pim = Image.new("RGB", (600, 120 + 40 * len(drugs)), (255, 255, 255))
        dr = ImageDraw.Draw(pim)
        dr.text((20, 20), "TOA THUOC (synthetic)", fill=(0, 0, 0))
        texts, labs, boxes_raw, maps, ids = ["TOA THUOC (synthetic)"], ["other"], [[20, 20, 300, 40]], [-1], [0]
        for j, c in enumerate(drugs):
            y = 60 + 40 * j
            t = f"{j + 1}) {classes[c].name}"
            dr.text((20, y), t, fill=(0, 0, 0))
            texts.append(t); labs.append("drugname"); boxes_raw.append([20, y, 400, y + 20]); maps.append(c); ids.append(j + 1)
            texts.append("SL: 10 Vien"); labs.append("quantity"); boxes_raw.append([420, y, 560, y + 20]); maps.append(-1); ids.append(100 + j)
        pim.save(root / "images" / "prescription" / pres_file, "JPEG", quality=90)
        pres_rows.append({"file": pres_file, "orig_file": pres_file, "src_split": split, "width": pim.width,
                          "height": pim.height, "orig_size": [pim.width, pim.height], "scale": 1.0,
                          "ann_ids": ids, "texts": texts, "ann_labels": labs, "boxes_raw": boxes_raw,
                          "mappings": maps})

    write_jsonl(root / "annotations" / f"pill_{split}.jsonl", pill_rows)
    write_jsonl(root / "annotations" / f"prescription_{split}.jsonl", pres_rows)
    (root / f"pill_pres_map_{split}.json").write_text(json.dumps(pres_map, indent=1), encoding="utf-8")
    table = {
        "classes": [{"id": c.id, "name": c.name, "n_boxes": 0, "n_images": 0} for c in classes],
        "out_of_prescription_label": OUT_OF_PRESCRIPTION_LABEL,
        "synthetic": True,
    }
    (root / "classes.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    return table


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--n-classes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    make_synthetic_dataset(a.out, a.n_images, a.n_classes, a.seed)
    print("wrote", a.out)
