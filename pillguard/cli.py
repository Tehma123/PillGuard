"""``pillguard`` command line: one sub-command per pipeline stage, in the order you run them.

    pillguard ingest                 download + shrink VAIPE (once, ~25 min)
    pillguard classes                classes.json (id -> drug name, counts)
    pillguard split                  fixed train/val/test split + unseen drugs -> splits/vaipe_v1.json
    pillguard scenarios              out-of-prescription scenarios for val/test -> data/vaipe/scenarios_*.jsonl
    pillguard eda                    docs/eda.md + figures
    pillguard det-prepare            YOLO labels + lists + yaml
    pillguard det-train [...]        train the detector (Ultralytics)
    pillguard det-export [...]       ONNX fp32 + int8
    pillguard crops                  crop cache for the embedding network
    pillguard emb-train [...]        train the embedding network
    pillguard emb-export [...]       ONNX fp32 + int8
    pillguard fit [...]              prototypes + calibration + thresholds
    pillguard export-web [...]       copy models + data into web/
    pillguard eval [...]             full-pipeline evaluation on the test scenarios
    pillguard parity [...]           Python vs browser parity set / comparison

Every sub-command forwards its remaining arguments to the module's own ``--help``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pillguard.config import VAIPE_DIR


def cmd_classes(argv):
    from pillguard.data.vaipe import save_classes

    ap = argparse.ArgumentParser(prog="pillguard classes")
    ap.add_argument("--root", type=Path, default=VAIPE_DIR)
    a = ap.parse_args(argv)
    table = save_classes(a.root)
    print(f"{len(table['classes'])} classes, {table['n_out_of_prescription_boxes']} out-of-prescription boxes -> {a.root / 'classes.json'}")


def cmd_split(argv):
    from pillguard.data.splits import default_split_path, make_split
    from pillguard.data.vaipe import load_pill_images

    ap = argparse.ArgumentParser(prog="pillguard split")
    ap.add_argument("--root", type=Path, default=VAIPE_DIR)
    ap.add_argument("--out", type=Path, default=default_split_path())
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    images = load_pill_images(a.root)
    if a.out.exists() and not a.force:
        print(f"{a.out} exists (use --force to overwrite)")
        return
    split = make_split(images)
    split.save(a.out)
    print(json.dumps(split.summary(images), indent=1))


def cmd_scenarios(argv):
    from pillguard.data.scenarios import make_scenarios, save_scenarios, summarize
    from pillguard.data.splits import Split, default_split_path
    from pillguard.data.vaipe import load_pill_images, prescription_drug_table

    ap = argparse.ArgumentParser(prog="pillguard scenarios")
    ap.add_argument("--root", type=Path, default=VAIPE_DIR)
    ap.add_argument("--split", type=Path, default=default_split_path())
    a = ap.parse_args(argv)
    split = Split.load(a.split)
    images = load_pill_images(a.root)
    by = {im.file: im for im in images}
    pd = prescription_drug_table(a.root)
    for subset in ("val", "test"):
        ims = [by[f] for f in split.subset(subset) if f in by]
        sc = make_scenarios(ims, pd, split.unseen_classes, seed=split.seed)
        save_scenarios(a.root / f"scenarios_{subset}.jsonl", sc)
        print(subset, json.dumps(summarize(sc)))


def cmd_det_prepare(argv):
    from pillguard.data.splits import Split, default_split_path
    from pillguard.data.vaipe import load_pill_images
    from pillguard.detector.prepare import write_yolo_dataset

    ap = argparse.ArgumentParser(prog="pillguard det-prepare")
    ap.add_argument("--root", type=Path, default=VAIPE_DIR)
    ap.add_argument("--split", type=Path, default=default_split_path())
    a = ap.parse_args(argv)
    write_yolo_dataset(load_pill_images(a.root), Split.load(a.split), a.root)


def cmd_crops(argv):
    from pillguard.data.splits import Split, default_split_path
    from pillguard.data.vaipe import load_pill_images
    from pillguard.embed.crops import build_crop_cache

    ap = argparse.ArgumentParser(prog="pillguard crops")
    ap.add_argument("--root", type=Path, default=VAIPE_DIR)
    ap.add_argument("--split", type=Path, default=default_split_path())
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    build_crop_cache(load_pill_images(a.root), Split.load(a.split).files(), a.root, workers=a.workers)


COMMANDS = {
    "ingest": lambda argv: __import__("pillguard.data.ingest", fromlist=["main"]).main(argv),
    "classes": cmd_classes,
    "split": cmd_split,
    "scenarios": cmd_scenarios,
    "eda": lambda argv: __import__("pillguard.data.eda", fromlist=["main"]).main(argv),
    "det-prepare": cmd_det_prepare,
    "det-train": lambda argv: __import__("pillguard.detector.train", fromlist=["main"]).main(argv),
    "det-export": lambda argv: __import__("pillguard.detector.export", fromlist=["main"]).main(argv),
    "crops": cmd_crops,
    "emb-train": lambda argv: __import__("pillguard.embed.train", fromlist=["main"]).main(argv),
    "emb-export": lambda argv: __import__("pillguard.embed.export", fromlist=["main"]).main(argv),
    "fit": lambda argv: __import__("pillguard.decision.fit", fromlist=["main"]).main(argv),
    "export-web": lambda argv: __import__("pillguard.web.export_assets", fromlist=["main"]).main(argv),
    "eval": lambda argv: __import__("pillguard.eval.run_eval", fromlist=["main"]).main(argv),
    "parity": lambda argv: __import__("pillguard.eval.parity", fromlist=["main"]).main(argv),
}


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):   # Vietnamese drug names on a cp1252 console
        stream.reconfigure(encoding="utf-8", errors="replace")
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print(__doc__)
        if argv and argv[0] not in ("-h", "--help"):
            print(f"unknown command: {argv[0]}")
            sys.exit(2)
        return
    COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
