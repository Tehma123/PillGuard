"""Pixel-level preprocessing shared by training, the Python ONNX pipeline and (by design) the
browser. Every function here has a line-for-line counterpart in ``web/pipeline.js``; keep
them in sync so that the parity test (``pillguard.eval.parity``) stays at 100 %.

Conventions
-----------
* Images are ``uint8`` arrays of shape ``(H, W, 3)`` in RGB.
* Resizing is plain bilinear with half-pixel centres (``src = (dst + 0.5) * scale - 0.5``),
  no anti-aliasing, computed in float32 and rounded to nearest. That is what a hand-written
  JS loop does, and it avoids depending on OpenCV's fixed-point arithmetic or on the
  browser's canvas scaler.
"""

from __future__ import annotations

import numpy as np

from pillguard.config import CROP_CONTEXT, CROP_SIZE, DET_INPUT, DET_PAD_VALUE, EMB_MEAN, EMB_STD


def load_rgb(path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def bilinear_resize(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize with half-pixel centres, float32 math, round-half-up to uint8."""
    h, w = img.shape[:2]
    if (h, w) == (out_h, out_w):
        return img.copy()
    sy, sx = h / out_h, w / out_w
    ys = np.clip((np.arange(out_h, dtype=np.float32) + 0.5) * sy - 0.5, 0, h - 1)
    xs = np.clip((np.arange(out_w, dtype=np.float32) + 0.5) * sx - 0.5, 0, w - 1)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0).astype(np.float32)[:, None, None]
    wx = (xs - x0).astype(np.float32)[None, :, None]
    src = img.astype(np.float32)
    top = src[y0][:, x0] * (1 - wx) + src[y0][:, x1] * wx
    bot = src[y1][:, x0] * (1 - wx) + src[y1][:, x1] * wx
    out = top * (1 - wy) + bot * wy
    return np.floor(out + 0.5).clip(0, 255).astype(np.uint8)


def letterbox(img: np.ndarray, size: int = DET_INPUT, pad_value: int = DET_PAD_VALUE):
    """Scale the long side to ``size``, pad the rest with ``pad_value`` (top-left anchored).

    Returns ``(canvas, scale, (pad_x, pad_y))``; padding is split evenly like Ultralytics
    (``auto=False``), rounded down on the top/left side.
    """
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    # floor(x + 0.5) on both sides: Python's round() is half-to-even, JS Math.round is half-up
    nh, nw = max(1, int(np.floor(h * scale + 0.5))), max(1, int(np.floor(w * scale + 0.5)))
    resized = bilinear_resize(img, nh, nw)
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    pad_y, pad_x = (size - nh) // 2, (size - nw) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, (pad_x, pad_y)


def to_detector_input(canvas: np.ndarray) -> np.ndarray:
    """uint8 HWC -> float32 NCHW in [0, 1] (Ultralytics convention)."""
    x = canvas.astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def square_crop_box(box, img_w: int, img_h: int, context: float = CROP_CONTEXT):
    """Expand a box by ``context`` on each side and make it square around its centre.

    Returns integer ``(x1, y1, x2, y2)`` that may extend outside the image; callers pad.
    """
    x1, y1, x2, y2 = box
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    side = max(bw, bh) * (1 + 2 * context)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sx1 = int(np.floor(cx - side / 2))
    sy1 = int(np.floor(cy - side / 2))
    s = int(np.ceil(side))
    return sx1, sy1, sx1 + s, sy1 + s


def extract_crop(img: np.ndarray, box, size: int = CROP_SIZE, context: float = CROP_CONTEXT,
                 pad_value: int = DET_PAD_VALUE) -> np.ndarray:
    """Square, context-padded, ``size``x``size`` RGB crop of one pill (uint8)."""
    h, w = img.shape[:2]
    sx1, sy1, sx2, sy2 = square_crop_box(box, w, h, context)
    s = sx2 - sx1
    patch = np.full((s, s, 3), pad_value, dtype=np.uint8)
    ix1, iy1, ix2, iy2 = max(0, sx1), max(0, sy1), min(w, sx2), min(h, sy2)
    if ix2 > ix1 and iy2 > iy1:
        patch[iy1 - sy1:iy2 - sy1, ix1 - sx1:ix2 - sx1] = img[iy1:iy2, ix1:ix2]
    return bilinear_resize(patch, size, size)


def to_embed_input(crops: np.ndarray) -> np.ndarray:
    """uint8 (N,S,S,3) -> normalised float32 (N,3,S,S) with ImageNet statistics."""
    x = crops.astype(np.float32) / 255.0
    x = (x - np.array(EMB_MEAN, dtype=np.float32)) / np.array(EMB_STD, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def scale_boxes_back(boxes: np.ndarray, scale: float, pad: tuple[int, int], img_w: int, img_h: int) -> np.ndarray:
    """Detector-canvas xyxy -> original image xyxy (clipped)."""
    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad[0]) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - pad[1]) / scale
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, img_w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, img_h)
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, max_dets: int) -> np.ndarray:
    """Greedy NMS on xyxy boxes. Returns kept indices in descending score order."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    keep = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        if len(keep) >= max_dets:
            break
        xx1 = np.maximum(x1[idx], x1); yy1 = np.maximum(y1[idx], y1)
        xx2 = np.minimum(x2[idx], x2); yy2 = np.minimum(y2[idx], y2)
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[idx] + areas - inter + 1e-9)
        suppressed |= iou > iou_thr
        suppressed[idx] = True
    return np.array(keep, dtype=np.int64)


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every pair of xyxy boxes in ``a`` (N,4) and ``b`` (M,4)."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0]); yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2]); yy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)
