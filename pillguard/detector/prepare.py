"""Write the single-class YOLO dataset (labels + split lists + data yaml) for Ultralytics.

Ultralytics finds labels by replacing ``/images/`` with ``/labels/`` in each image path, so
labels are written next to the ingested images::

    data/vaipe/images/pill/X.jpg  ->  data/vaipe/labels/pill/X.txt   ("0 cx cy w h", normalised)
    data/vaipe/yolo/{train,val,test}.txt                                absolute image paths
    data/vaipe/yolo/pill.yaml

Every box becomes class 0 ("pill"): the detector only localises, recognition is done by
the embedding network.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from pillguard.data.splits import Split
from pillguard.data.vaipe import PillImage


def write_yolo_labels(images: list[PillImage], root: Path) -> int:
    lab_dir = Path(root) / "labels" / "pill"
    lab_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for im in images:
        lines = []
        for x1, y1, x2, y2 in im.boxes:
            cx, cy = (x1 + x2) / 2 / im.width, (y1 + y2) / 2 / im.height
            w, h = (x2 - x1) / im.width, (y2 - y1) / im.height
            if w <= 0 or h <= 0:
                continue
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            n += 1
        (lab_dir / (Path(im.file).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    return n


def write_yolo_dataset(images: list[PillImage], split: Split, root: Path, name: str = "pill") -> Path:
    root = Path(root)
    n = write_yolo_labels(images, root)
    yolo_dir = root / "yolo"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    by = {im.file: im for im in images}
    lists = {}
    for subset in ("train", "val", "test"):
        files = [by[f] for f in split.subset(subset) if f in by]
        p = yolo_dir / f"{subset}.txt"
        p.write_text("\n".join(str(im.path(root).resolve()) for im in files) + "\n", encoding="utf-8")
        lists[subset] = str(p.resolve())
    data = {"path": str(yolo_dir.resolve()), "train": lists["train"], "val": lists["val"],
            "test": lists["test"], "names": {0: "pill"}, "nc": 1}
    yaml_path = yolo_dir / f"{name}.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"YOLO dataset: {n} boxes, lists {[len(split.subset(s)) for s in ('train', 'val', 'test')]} -> {yaml_path}")
    return yaml_path
