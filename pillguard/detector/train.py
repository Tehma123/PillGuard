"""Train and evaluate the single-class pill detector with Ultralytics YOLO.

Usage::

    python -m pillguard.detector.train --data data/vaipe/yolo/pill.yaml --model yolo11n.pt --epochs 60
    python -m pillguard.detector.train --eval artifacts/detector/yolo11n/weights/best.pt --split test

The default model is ``yolo11n`` (2.6 M parameters): small enough for INT8 ONNX in the
browser, and the task is easy (one class, large objects). Rotation augmentation is left
off because axis-aligned boxes inflate under rotation; vertical flips are on because pills
have no canonical orientation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pillguard.config import ARTIFACTS_DIR, DET_INPUT, SEED


def train(data_yaml: Path, model: str = "yolo11n.pt", epochs: int = 60, imgsz: int = DET_INPUT,
          batch: int = 32, name: str | None = None, device: str | int = 0, workers: int = 4,
          project: Path = ARTIFACTS_DIR / "detector", seed: int = SEED, patience: int = 20,
          fraction: float = 1.0, exist_ok: bool = True, plots: bool = True, **overrides) -> Path:
    from ultralytics import YOLO

    name = name or Path(model).stem
    yolo = YOLO(model)
    yolo.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, device=device, workers=workers,
        project=str(project), name=name, seed=seed, deterministic=False, patience=patience,
        fraction=fraction, exist_ok=exist_ok, single_cls=True, cos_lr=True, close_mosaic=max(1, epochs // 10),
        flipud=0.5, fliplr=0.5, degrees=0.0, scale=0.5, hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, mosaic=1.0,
        plots=plots, verbose=True, **overrides,
    )
    best = Path(project) / name / "weights" / "best.pt"
    return best


def evaluate(weights: Path, data_yaml: Path, split: str = "test", imgsz: int = DET_INPUT, batch: int = 32,
             device: str | int = 0, conf: float = 0.001, iou: float = 0.6, out_json: Path | None = None) -> dict:
    from ultralytics import YOLO

    yolo = YOLO(str(weights))
    res = yolo.val(data=str(data_yaml), split=split, imgsz=imgsz, batch=batch, device=device, conf=conf, iou=iou,
                   plots=False, verbose=False, project=str(Path(weights).parents[1]), name=f"val_{split}",
                   exist_ok=True)
    metrics = {
        "split": split,
        "map50": float(res.box.map50),
        "map50_95": float(res.box.map),
        "precision": float(res.box.mp),
        "recall": float(res.box.mr),
        "imgsz": imgsz,
        "weights": str(weights),
    }
    speed = getattr(res, "speed", None)
    if isinstance(speed, dict):
        metrics["speed_ms"] = {k: float(v) for k, v in speed.items()}
    out_json = out_json or Path(weights).parents[1] / f"metrics_{split}.json"
    Path(out_json).write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=None, help="pill.yaml written by pillguard.detector.prepare")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=DET_INPUT)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train set (quick runs)")
    ap.add_argument("--cache", default=None, choices=[None, "ram", "disk"],
                    help="cache decoded images (ram needs ~6 GB for the full train set; disk writes .npy next to the images)")
    ap.add_argument("--eval", type=Path, default=None, help="skip training; evaluate these weights")
    ap.add_argument("--split", default="test")
    a = ap.parse_args(argv)
    device = int(a.device) if a.device.isdigit() else a.device
    if a.eval:
        evaluate(a.eval, a.data, split=a.split, imgsz=a.imgsz, batch=a.batch, device=device)
        return
    extra = {"cache": a.cache} if a.cache else {}
    best = train(a.data, model=a.model, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, name=a.name,
                 device=device, workers=a.workers, fraction=a.fraction, **extra)
    evaluate(best, a.data, split="val", imgsz=a.imgsz, batch=a.batch, device=device)
    evaluate(best, a.data, split="test", imgsz=a.imgsz, batch=a.batch, device=device)


if __name__ == "__main__":
    main()
