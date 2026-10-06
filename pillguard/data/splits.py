"""Fixed, reproducible train / val / test split plus a held-out set of "unseen" drugs.

Why group by prescription
-------------------------
Every VAIPE prescription comes with several photos of the *same* pills taken seconds
apart. A random split by photo would leak near-duplicates into the test set. Photos are
therefore split by prescription: all photos of one prescription land in one subset.

Unseen classes
--------------
``N_UNSEEN_CLASSES`` drugs are hidden from the embedding network and from the prototype
bank. Their pills still exist in val/test photos, where they must be flagged as "not in
the prescription" (they never appear on the prescription list handed to the matcher). The
detector is class-agnostic and is trained on every box, including those.

Splitting by prescription can also leave a rare drug with no train photo at all. The embedding
network never sees it either and it gets no prototype, so ``all_unseen_classes`` adds those
drugs to the held-out ones wherever scenarios are built: listing a drug the matcher cannot
recognise would count its pills as false alarms that no model could avoid.

The split file is small (file names only) and is committed under ``splits/``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pillguard.config import (
    N_UNSEEN_CLASSES,
    OUT_OF_PRESCRIPTION_LABEL,
    SEED,
    SPLITS_DIR,
    TEST_FRACTION,
    VAL_FRACTION,
)
from pillguard.data.vaipe import PillImage


@dataclass
class Split:
    train: list[str]
    val: list[str]
    test: list[str]
    unseen_classes: list[int]
    seed: int = SEED
    meta: dict = field(default_factory=dict)

    def subset(self, name: str) -> list[str]:
        return getattr(self, name)

    def files(self) -> dict[str, str]:
        out = {}
        for name in ("train", "val", "test"):
            for f in getattr(self, name):
                out[f] = name
        return out

    def to_dict(self) -> dict:
        return {"seed": self.seed, "unseen_classes": self.unseen_classes, "meta": self.meta,
                "train": self.train, "val": self.val, "test": self.test}

    @classmethod
    def from_dict(cls, d: dict) -> Split:
        return cls(train=d["train"], val=d["val"], test=d["test"], unseen_classes=d["unseen_classes"],
                   seed=d.get("seed", SEED), meta=d.get("meta", {}))

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=0), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Split:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def untrained_classes(self, images: list[PillImage]) -> list[int]:
        """Drugs that occur in val/test photos but in no train photo, and were not held out on purpose."""
        by = {im.file: im for im in images}
        labels = {name: {l for f in getattr(self, name) if f in by for l in by[f].labels}
                  for name in ("train", "val", "test")}
        rest = (labels["val"] | labels["test"]) - labels["train"] - set(self.unseen_classes)
        return sorted(int(c) for c in rest - {OUT_OF_PRESCRIPTION_LABEL})

    def all_unseen_classes(self, images: list[PillImage]) -> set[int]:
        """Every drug the model has no prototype for: held out on purpose, or absent from train photos."""
        return set(self.unseen_classes) | set(self.untrained_classes(images))

    def summary(self, images: list[PillImage] | None = None) -> dict:
        s = {"n_train": len(self.train), "n_val": len(self.val), "n_test": len(self.test),
             "unseen_classes": self.unseen_classes}
        if images:
            by = {im.file: im for im in images}
            for name in ("train", "val", "test"):
                ims = [by[f] for f in getattr(self, name) if f in by]
                s[f"{name}_pills"] = sum(im.n_pills for im in ims)
                s[f"{name}_prescriptions"] = len({im.prescription for im in ims})
        return s


def _stable_shuffle(keys: list[str], seed: int) -> list[str]:
    """Order-independent shuffle: sort by a hash of (seed, key)."""
    return sorted(keys, key=lambda k: hashlib.sha1(f"{seed}:{k}".encode()).hexdigest())


def choose_unseen_classes(images: list[PillImage], n_unseen: int, seed: int,
                          exclude: set[int] | None = None) -> list[int]:
    """Pick classes with mid-range frequency (25th-75th percentile of images per class)."""
    exclude = set(exclude or ()) | {OUT_OF_PRESCRIPTION_LABEL}
    img_count: Counter = Counter()
    for im in images:
        img_count.update(set(im.labels))
    cands = {c: n for c, n in img_count.items() if c not in exclude}
    if not cands or n_unseen <= 0:
        return []
    counts = np.array(sorted(cands.values()))
    lo, hi = np.percentile(counts, 25), np.percentile(counts, 75)
    pool = sorted(c for c, n in cands.items() if lo <= n <= hi) or sorted(cands)
    rng = np.random.default_rng(seed)
    k = min(n_unseen, len(pool))
    return sorted(int(c) for c in rng.choice(pool, size=k, replace=False))


def make_split(images: list[PillImage], seed: int = SEED, val_frac: float = VAL_FRACTION,
               test_frac: float = TEST_FRACTION, n_unseen: int = N_UNSEEN_CLASSES) -> Split:
    groups: dict[str, list[PillImage]] = defaultdict(list)
    for im in images:
        groups[im.prescription or f"__solo__{im.file}"].append(im)
    order = _stable_shuffle(list(groups), seed)
    n_total = len(images)
    n_test, n_val = round(test_frac * n_total), round(val_frac * n_total)
    test, val, train = [], [], []
    for g in order:
        files = sorted(im.file for im in groups[g])
        if len(test) < n_test:
            test.extend(files)
        elif len(val) < n_val:
            val.extend(files)
        else:
            train.extend(files)
    unseen = choose_unseen_classes(images, n_unseen, seed)
    split = Split(train=sorted(train), val=sorted(val), test=sorted(test), unseen_classes=unseen, seed=seed,
                  meta={"val_frac": val_frac, "test_frac": test_frac, "grouped_by": "prescription",
                        "n_images": n_total, "n_groups": len(groups)})
    return split


def default_split_path(name: str = "vaipe_v1") -> Path:
    return SPLITS_DIR / f"{name}.json"


def load_or_make_split(images: list[PillImage], path: Path | None = None, **kw) -> Split:
    path = Path(path) if path else default_split_path()
    if path.exists():
        return Split.load(path)
    split = make_split(images, **kw)
    split.save(path)
    return split
