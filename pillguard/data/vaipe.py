"""Loaders for the ingested VAIPE layout produced by :mod:`pillguard.data.ingest`.

The same layout is produced by :mod:`pillguard.data.synthetic`, so everything downstream
(splits, scenarios, training, evaluation) works on either.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pillguard.config import OUT_OF_PRESCRIPTION_LABEL, VAIPE_DIR

_DRUG_PREFIX = re.compile(r"^\s*\d+\s*[\)\.\-:]\s*")


@dataclass
class PillImage:
    """One labelled pill photo. ``boxes`` are ``[x1, y1, x2, y2]`` in saved-image pixels."""

    file: str
    width: int
    height: int
    boxes: list[list[float]]
    labels: list[int]
    prescription: str | None = None
    src_split: str = "train"
    extra: dict = field(default_factory=dict)

    def path(self, root: Path = VAIPE_DIR) -> Path:
        return Path(root) / "images" / "pill" / self.file

    @property
    def n_pills(self) -> int:
        return len(self.boxes)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


# --------------------------------------------------------------------------------------
# Raw tables
# --------------------------------------------------------------------------------------
def load_pill_pres_map(root: Path = VAIPE_DIR, split: str = "train") -> dict[str, list[str]]:
    """prescription file -> list of pill photo files."""
    p = Path(root) / f"pill_pres_map_{split}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def pill_to_prescription(root: Path = VAIPE_DIR, split: str = "train") -> dict[str, str]:
    out: dict[str, str] = {}
    for pres, pills in load_pill_pres_map(root, split).items():
        for pill in pills:
            out[pill] = pres
    return out


def load_prescriptions(root: Path = VAIPE_DIR, split: str = "train") -> list[dict]:
    p = Path(root) / "annotations" / f"prescription_{split}.jsonl"
    return read_jsonl(p) if p.exists() else []


def load_pill_images(root: Path = VAIPE_DIR, split: str = "train",
                     only_labelled: bool = True) -> list[PillImage]:
    root = Path(root)
    rows = read_jsonl(root / "annotations" / f"pill_{split}.jsonl")
    p2p = pill_to_prescription(root, split)
    out: list[PillImage] = []
    for r in rows:
        if only_labelled and not r["labels"]:
            continue
        out.append(PillImage(
            file=r["file"], width=r["width"], height=r["height"], boxes=r["boxes"],
            labels=r["labels"], prescription=p2p.get(r["file"]), src_split=r.get("src_split", split),
            extra={k: v for k, v in r.items() if k not in
                   {"file", "width", "height", "boxes", "labels", "src_split"}},
        ))
    return out


# --------------------------------------------------------------------------------------
# Derived tables
# --------------------------------------------------------------------------------------
def clean_drug_name(text: str) -> str:
    """'3) HOẠT HUYẾT DƯỠNG NÃO 150mg+20mg' -> 'HOẠT HUYẾT DƯỠNG NÃO 150mg+20mg'."""
    return _DRUG_PREFIX.sub("", text).strip()


def class_names_from_prescriptions(prescriptions: list[dict]) -> dict[int, str]:
    """Most common cleaned drug-name line for each pill class id."""
    votes: dict[int, Counter] = defaultdict(Counter)
    for r in prescriptions:
        for text, lab, cls in zip(r["texts"], r["ann_labels"], r["mappings"]):
            if lab == "drugname" and cls is not None and cls >= 0:
                votes[cls][clean_drug_name(text)] += 1
    return {c: votes[c].most_common(1)[0][0] for c in sorted(votes)}


def class_aliases_from_prescriptions(prescriptions: list[dict]) -> dict[int, list[str]]:
    """Every cleaned drug-name line ever mapped to a class id, most common first.

    Prescriptions often print a brand (NOVOXIM-500) for a pill whose usual name is another
    brand (FABAMOX 500); the OCR matcher needs all of them.
    """
    votes: dict[int, Counter] = defaultdict(Counter)
    for r in prescriptions:
        for text, lab, cls in zip(r["texts"], r["ann_labels"], r["mappings"]):
            if lab == "drugname" and cls is not None and cls >= 0:
                votes[cls][clean_drug_name(text)] += 1
    return {c: [n for n, _ in votes[c].most_common()] for c in sorted(votes)}


def prescription_drugs(prescriptions: list[dict]) -> dict[str, list[int]]:
    """prescription file -> sorted list of pill class ids written on it."""
    out: dict[str, list[int]] = {}
    for r in prescriptions:
        ids = sorted({c for c, lab in zip(r["mappings"], r["ann_labels"]) if lab == "drugname" and c >= 0})
        out[r["file"]] = ids
    return out


def build_classes(root: Path = VAIPE_DIR, split: str = "train") -> dict:
    """Class table: id -> name, plus counts from the pill labels. Saved as classes.json."""
    pres = load_prescriptions(root, split)
    names = class_names_from_prescriptions(pres)
    aliases = class_aliases_from_prescriptions(pres)
    images = load_pill_images(root, split)
    box_counts: Counter = Counter()
    image_counts: Counter = Counter()
    for im in images:
        box_counts.update(im.labels)
        image_counts.update(set(im.labels))
    ids = sorted(set(names) | {c for c in box_counts if c != OUT_OF_PRESCRIPTION_LABEL})
    classes = []
    for c in ids:
        classes.append({
            "id": c,
            "name": names.get(c, f"class_{c}"),
            "aliases": aliases.get(c, [])[1:],
            "n_boxes": box_counts.get(c, 0),
            "n_images": image_counts.get(c, 0),
        })
    return {
        "classes": classes,
        "out_of_prescription_label": OUT_OF_PRESCRIPTION_LABEL,
        "n_out_of_prescription_boxes": box_counts.get(OUT_OF_PRESCRIPTION_LABEL, 0),
    }


def save_classes(root: Path = VAIPE_DIR, split: str = "train") -> dict:
    table = build_classes(root, split)
    (Path(root) / "classes.json").write_text(json.dumps(table, indent=1, ensure_ascii=False), encoding="utf-8")
    return table


def load_classes(root: Path = VAIPE_DIR) -> dict[int, str]:
    p = Path(root) / "classes.json"
    if not p.exists():
        return save_classes(root)["classes"] and {c["id"]: c["name"] for c in save_classes(root)["classes"]}
    return {c["id"]: c["name"] for c in json.loads(p.read_text(encoding="utf-8"))["classes"]}


def load_aliases(root: Path = VAIPE_DIR) -> dict[int, list[str]]:
    p = Path(root) / "classes.json"
    if not p.exists():
        return {}
    return {c["id"]: c.get("aliases", []) for c in json.loads(p.read_text(encoding="utf-8"))["classes"]}


def prescription_drug_table(root: Path = VAIPE_DIR, split: str = "train") -> dict[str, list[int]]:
    return prescription_drugs(load_prescriptions(root, split))
