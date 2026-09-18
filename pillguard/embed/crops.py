"""Crop cache and PyTorch datasets for the embedding network.

Decoding a 1280 px photo for every pill crop, every epoch, is the bottleneck, so crops are
cut once into a cache::

    data/vaipe/crops/train160/<stem>_<i>.jpg   160 px, generous context (0.30) for augmentation jitter
    data/vaipe/crops/eval128/<stem>_<i>.jpg    128 px, exact inference framing (CROP_CONTEXT)
    data/vaipe/crops/index.jsonl               one row per pill: id, file, box, label, subset

``eval128`` crops are produced by :func:`pillguard.imaging.extract_crop`, i.e. the same
code path as the ONNX pipeline and the browser, so prototypes and validation numbers are
computed on exactly what the deployed model sees.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from pillguard.config import (
    CROP_CONTEXT,
    CROP_SIZE,
    DET_PAD_VALUE,
    EMB_MEAN,
    EMB_STD,
    OUT_OF_PRESCRIPTION_LABEL,
)
from pillguard.data.vaipe import PillImage, read_jsonl
from pillguard.imaging import extract_crop, load_rgb

TRAIN_CROP_SIZE = 160
TRAIN_CROP_CONTEXT = 0.30


@dataclass
class CropRow:
    id: str
    file: str
    box: list[float]
    label: int
    subset: str

    @property
    def train_crop(self) -> str:
        return f"crops/train160/{self.id}.jpg"

    @property
    def eval_crop(self) -> str:
        return f"crops/eval128/{self.id}.jpg"


def _process_image(args) -> list[dict]:
    root, file, boxes, labels, subset = args
    root = Path(root)
    img = load_rgb(root / "images" / "pill" / file)
    stem = Path(file).stem
    rows = []
    for i, (box, lab) in enumerate(zip(boxes, labels)):
        cid = f"{stem}_{i}"
        tp = root / "crops" / "train160" / f"{cid}.jpg"
        ep = root / "crops" / "eval128" / f"{cid}.jpg"
        if not tp.exists():
            Image.fromarray(extract_crop(img, box, TRAIN_CROP_SIZE, TRAIN_CROP_CONTEXT)).save(tp, "JPEG", quality=95)
        if not ep.exists():
            Image.fromarray(extract_crop(img, box, CROP_SIZE, CROP_CONTEXT)).save(ep, "JPEG", quality=95)
        rows.append({"id": cid, "file": file, "box": list(box), "label": int(lab), "subset": subset})
    return rows


def build_crop_cache(images: list[PillImage], subsets: dict[str, str], root: Path, workers: int = 6) -> Path:
    root = Path(root)
    (root / "crops" / "train160").mkdir(parents=True, exist_ok=True)
    (root / "crops" / "eval128").mkdir(parents=True, exist_ok=True)
    jobs = [(str(root), im.file, im.boxes, im.labels, subsets.get(im.file, "train")) for im in images]
    rows: list[dict] = []
    if workers <= 1:
        for j in jobs:
            rows.extend(_process_image(j))
    else:
        with ProcessPoolExecutor(workers) as ex:
            for i, r in enumerate(ex.map(_process_image, jobs, chunksize=8)):
                rows.extend(r)
                if (i + 1) % 500 == 0:
                    print(f"  crops: {i + 1}/{len(jobs)} images, {len(rows)} pills", flush=True)
    index = root / "crops" / "index.jsonl"
    with open(index, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"crop cache: {len(rows)} pills from {len(jobs)} images -> {index}")
    return index


def load_crop_index(root: Path) -> list[CropRow]:
    return [CropRow(**r) for r in read_jsonl(Path(root) / "crops" / "index.jsonl")]


def filter_rows(rows: list[CropRow], subset: str | None = None, classes: set[int] | None = None,
                drop_foreign: bool = True) -> list[CropRow]:
    out = []
    for r in rows:
        if subset and r.subset != subset:
            continue
        if drop_foreign and r.label == OUT_OF_PRESCRIPTION_LABEL:
            continue
        if classes is not None and r.label not in classes:
            continue
        out.append(r)
    return out


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------
def train_transform(size: int = CROP_SIZE):
    from torchvision.transforms import v2

    return v2.Compose([
        v2.ToImage(),
        v2.RandomRotation(180, fill=DET_PAD_VALUE),
        # centre 62 % of the 160 px crop matches the eval framing; jitter around it
        v2.RandomResizedCrop(size, scale=(0.42, 0.80), ratio=(0.85, 1.18), antialias=True),
        v2.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.30, hue=0.04),
        v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.25),
        v2.RandomGrayscale(p=0.03),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(EMB_MEAN, EMB_STD),
        v2.RandomErasing(p=0.10, scale=(0.02, 0.10), value=0.0),
    ])


def eval_transform():
    from torchvision.transforms import v2

    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(EMB_MEAN, EMB_STD),
    ])


class CropDataset(Dataset):
    """Returns ``(tensor, class_index, row_index)``. ``class_index`` is contiguous over ``classes``."""

    def __init__(self, rows: list[CropRow], root: Path, classes: list[int], train: bool) -> None:
        self.rows = rows
        self.root = Path(root)
        self.cls_to_idx = {c: i for i, c in enumerate(classes)}
        self.train = train
        self.tf = train_transform() if train else eval_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        path = self.root / (r.train_crop if self.train else r.eval_crop)
        with Image.open(path) as im:
            im = im.convert("RGB")
            x = self.tf(im)
        return x, self.cls_to_idx.get(r.label, -1), i


def eval_crop_array(rows: list[CropRow], root: Path) -> np.ndarray:
    """Stack eval crops as uint8 (N, S, S, 3), the input format of the ONNX pipeline."""
    out = np.empty((len(rows), CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    for i, r in enumerate(rows):
        with Image.open(Path(root) / r.eval_crop) as im:
            out[i] = np.asarray(im.convert("RGB"))
    return out
