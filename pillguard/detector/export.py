"""Export the trained detector to ONNX (fp32) and static INT8, then sanity-check both.

Usage::

    python -m pillguard.detector.export --weights artifacts/detector/yolo11n/weights/best.pt \
        --calib-list data/vaipe/yolo/val.txt --out artifacts/export

Outputs ``detector.fp32.onnx`` and ``detector.int8.onnx``. The ONNX graph takes
``(1, 3, 640, 640)`` float in [0, 1] and returns ``(1, 5, 8400)``: cx, cy, w, h, score in
letterbox pixels (Ultralytics' non-NMS export). Decoding and NMS live in
:mod:`pillguard.pipeline` and ``web/pipeline.js``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from pillguard.config import ARTIFACTS_DIR, DET_CONF, DET_INPUT, DET_IOU, DET_MAX_DETS
from pillguard.imaging import box_iou_matrix, letterbox, load_rgb, nms, to_detector_input
from pillguard.onnx_utils import onnx_size_mb, ort_session, quantize_static_int8, run, simplify_onnx


def export_fp32(weights: Path, out_path: Path, imgsz: int = DET_INPUT, opset: int = 17) -> Path:
    from ultralytics import YOLO

    yolo = YOLO(str(weights))
    produced = yolo.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True, dynamic=False, half=False, nms=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(produced, out_path)
    simplify_onnx(out_path)
    print(f"  fp32 onnx: {onnx_size_mb(out_path):.1f} MB -> {out_path}")
    return out_path


def decode(raw: np.ndarray, conf: float = DET_CONF, iou: float = DET_IOU, max_dets: int = DET_MAX_DETS):
    """(1, 5, N) -> (boxes xyxy in letterbox px, scores) after threshold + NMS."""
    p = raw[0].T  # (N, 5)
    scores = p[:, 4]
    m = scores >= conf
    p, scores = p[m], scores[m]
    boxes = np.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2, p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], 1)
    keep = nms(boxes, scores, iou, max_dets)
    return boxes[keep], scores[keep]


def fp32_tail_nodes(fp32_path: Path) -> list[str]:
    """Decode nodes between the head convolutions and the graph output, to keep in fp32.

    ``output0`` concatenates box coordinates in letterbox pixels (0..640) with sigmoid
    scores (0..1). A single uint8 scale over that tensor (~2.5 per step) rounds every score
    to zero, so the decode arithmetic after the head convolutions stays unquantised.
    """
    import onnx

    graph = onnx.load(str(fp32_path)).graph
    producer = {out: n for n in graph.node for out in n.output}
    tail, frontier = [], [o.name for o in graph.output]
    while frontier:
        node = producer.get(frontier.pop())
        if node is None or node.op_type == "Conv" or node.name in tail:
            continue
        tail.append(node.name)
        frontier.extend(node.input)
    return tail


def calibration_inputs(image_paths: list[Path], n: int = 64, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(image_paths), size=min(n, len(image_paths)), replace=False)
    return [to_detector_input(letterbox(load_rgb(image_paths[i]))[0]) for i in sorted(idx)]


def compare_models(fp32: Path, int8: Path, image_paths: list[Path], n: int = 50) -> dict:
    """Agreement between fp32 and int8 detections on ``n`` images (count and box IoU)."""
    s32, s8 = ort_session(fp32), ort_session(int8)
    same_count, iou_sum, matched, total32, total8 = 0, 0.0, 0, 0, 0
    for p in image_paths[:n]:
        x = to_detector_input(letterbox(load_rgb(p))[0])
        b32, _ = decode(run(s32, x))
        b8, _ = decode(run(s8, x))
        total32 += len(b32); total8 += len(b8)
        same_count += int(len(b32) == len(b8))
        if len(b32) and len(b8):
            ious = box_iou_matrix(b32, b8).max(1)
            iou_sum += float(ious.sum()); matched += int((ious >= 0.5).sum())
    return {"images": min(n, len(image_paths)), "same_count_rate": same_count / max(1, min(n, len(image_paths))),
            "fp32_boxes": total32, "int8_boxes": total8, "matched_iou50": matched,
            "mean_best_iou": iou_sum / max(1, total32)}


def export_detector(weights: Path, out_dir: Path, calib_images: list[Path], imgsz: int = DET_INPUT,
                    n_calib: int = 64, method: str = "minmax") -> dict:
    out_dir = Path(out_dir)
    fp32 = export_fp32(weights, out_dir / "detector.fp32.onnx", imgsz)
    int8 = out_dir / "detector.int8.onnx"
    calib = calibration_inputs(calib_images, n_calib)
    quantize_static_int8(fp32, int8, calib, input_name="images", method=method,
                         nodes_to_exclude=fp32_tail_nodes(fp32))
    report = {"fp32_mb": onnx_size_mb(fp32), "int8_mb": onnx_size_mb(int8),
              "agreement": compare_models(fp32, int8, calib_images)}
    (out_dir / "detector_export.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--calib-list", type=Path, required=True, help="txt with image paths (e.g. yolo/val.txt)")
    ap.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "export")
    ap.add_argument("--n-calib", type=int, default=64)
    ap.add_argument("--method", default="minmax", choices=["minmax", "entropy", "percentile"])
    a = ap.parse_args(argv)
    paths = [Path(l.strip()) for l in a.calib_list.read_text(encoding="utf-8").splitlines() if l.strip()]
    export_detector(a.weights, a.out, paths, n_calib=a.n_calib, method=a.method)


if __name__ == "__main__":
    main()
