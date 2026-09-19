"""Exploratory statistics of the ingested dataset -> ``docs/eda.md`` + ``docs/figures/``.

Run: ``python -m pillguard.data.eda [--root data/vaipe] [--split splits/vaipe_v1.json]``
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from pillguard.config import OUT_OF_PRESCRIPTION_LABEL, ROOT, VAIPE_DIR
from pillguard.data.splits import Split
from pillguard.data.vaipe import (
    load_pill_images,
    load_prescriptions,
    prescription_drugs,
    save_classes,
)


def compute_stats(root: Path, split: Split | None = None) -> dict:
    images = load_pill_images(root)
    pres = load_prescriptions(root)
    classes = save_classes(root)
    names = {c["id"]: c["name"] for c in classes["classes"]}
    labels = [l for im in images for l in im.labels]
    cnt = Counter(labels)
    per_class = {c: cnt.get(c, 0) for c in sorted(names)}
    ppi = [im.n_pills for im in images]
    widths = [(b[2] - b[0]) / im.width for im in images for b in im.boxes]
    pd = prescription_drugs(pres)
    dpp = [len(v) for v in pd.values()]
    img_per_pres = Counter(im.prescription for im in images if im.prescription)
    stats = {
        "n_images": len(images), "n_pills": len(labels), "n_classes": len(names),
        "n_prescriptions": len(pres), "n_foreign_pills": cnt.get(OUT_OF_PRESCRIPTION_LABEL, 0),
        "n_images_with_foreign": sum(1 for im in images if OUT_OF_PRESCRIPTION_LABEL in im.labels),
        "pills_per_image": {"mean": float(np.mean(ppi)), "median": float(np.median(ppi)), "max": int(max(ppi))},
        "pills_per_class": {"min": int(min(per_class.values())), "p25": float(np.percentile(list(per_class.values()), 25)),
                            "median": float(np.median(list(per_class.values()))), "max": int(max(per_class.values()))},
        "classes_under_20_pills": sorted(c for c, n in per_class.items() if n < 20),
        "box_width_rel": {"p5": float(np.percentile(widths, 5)), "median": float(np.median(widths)), "p95": float(np.percentile(widths, 95))},
        "drugs_per_prescription": {"mean": float(np.mean(dpp)), "max": int(max(dpp))},
        "images_per_prescription": {"mean": float(np.mean(list(img_per_pres.values()))), "max": int(max(img_per_pres.values()))},
        "image_size": Counter(f"{im.width}x{im.height}" for im in images).most_common(3),
    }
    if split:
        stats["split"] = split.summary(images)
    return stats, per_class, ppi, widths, dpp, names


def write_eda(root: Path = VAIPE_DIR, split_path: Path | None = None, docs: Path = ROOT / "docs") -> Path:
    from pillguard.eval.plots import plot_hist, plot_sorted_bars

    split = Split.load(split_path) if split_path and Path(split_path).exists() else None
    stats, per_class, ppi, widths, dpp, names = compute_stats(root, split)
    fig_dir = docs / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(per_class)
    unseen = set(split.unseen_classes) if split else set()
    plot_sorted_bars([per_class[c] for c in ids], fig_dir / "pills_per_class.png",
                     "Pills per drug class (log scale; orange = held-out unseen drugs)", "drug classes, sorted", "pills",
                     highlight_idx={i for i, c in enumerate(ids) if c in unseen})
    plot_hist(ppi, fig_dir / "pills_per_image.png", "Pills per photo", "pills", bins=range(1, max(ppi) + 2))
    plot_hist(widths, fig_dir / "box_width.png", "Pill width relative to photo width", "box width / image width", bins=40)
    plot_hist(dpp, fig_dir / "drugs_per_prescription.png", "Drugs per prescription", "drugs", bins=range(1, max(dpp) + 2))
    (docs / "eda_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    s = stats
    lines = [
        "# VAIPE at a glance", "",
        "Numbers computed by `python -m pillguard.data.eda` on the ingested mirror (labelled train split of the AI4VN-2022 release).", "",
        "| quantity | value |", "|---|---|",
        f"| photos | {s['n_images']} |", f"| pills (boxes) | {s['n_pills']} |", f"| drug classes | {s['n_classes']} |",
        f"| prescriptions | {s['n_prescriptions']} |",
        f"| pills labelled 107 (not on the photo's prescription) | {s['n_foreign_pills']} in {s['n_images_with_foreign']} photos |",
        f"| pills per photo | mean {s['pills_per_image']['mean']:.2f}, max {s['pills_per_image']['max']} |",
        f"| pills per class | min {s['pills_per_class']['min']}, median {s['pills_per_class']['median']:.0f}, max {s['pills_per_class']['max']} |",
        f"| classes with < 20 pills | {len(s['classes_under_20_pills'])}: {s['classes_under_20_pills']} |",
        f"| pill width / photo width | p5 {s['box_width_rel']['p5']:.3f}, median {s['box_width_rel']['median']:.3f}, p95 {s['box_width_rel']['p95']:.3f} |",
        f"| drugs per prescription | mean {s['drugs_per_prescription']['mean']:.2f}, max {s['drugs_per_prescription']['max']} |",
        f"| photos per prescription | mean {s['images_per_prescription']['mean']:.1f}, max {s['images_per_prescription']['max']} |",
        f"| saved photo size | {s['image_size']} |", "",
    ]
    if split:
        sp = s["split"]
        lines += ["## Fixed split (grouped by prescription)", "", "| subset | photos | pills | prescriptions |", "|---|---|---|---|"]
        for k in ("train", "val", "test"):
            lines.append(f"| {k} | {sp['n_' + k]} | {sp.get(k + '_pills', '-')} | {sp.get(k + '_prescriptions', '-')} |")
        lines += ["", f"Unseen (held-out) drugs: {sp['unseen_classes']} = " + ", ".join(names.get(c, str(c)) for c in sp["unseen_classes"]), ""]
    lines += ["## Figures", "", "![pills per class](figures/pills_per_class.png)", "", "![pills per photo](figures/pills_per_image.png)", "",
              "![box width](figures/box_width.png)", "", "![drugs per prescription](figures/drugs_per_prescription.png)", ""]
    out = docs / "eda.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "split"}, indent=1, ensure_ascii=False))
    return out


def main(argv: list[str] | None = None) -> None:
    from pillguard.data.splits import default_split_path

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=VAIPE_DIR)
    ap.add_argument("--split", type=Path, default=default_split_path())
    ap.add_argument("--docs", type=Path, default=ROOT / "docs")
    a = ap.parse_args(argv)
    write_eda(a.root, a.split, a.docs)


if __name__ == "__main__":
    main()
