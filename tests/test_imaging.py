import numpy as np
import pytest

from pillguard.imaging import (
    bilinear_resize,
    box_iou_matrix,
    extract_crop,
    letterbox,
    nms,
    scale_boxes_back,
    square_crop_box,
    to_detector_input,
    to_embed_input,
)


def _img(h=480, w=640, seed=0):
    return (np.random.default_rng(seed).random((h, w, 3)) * 255).astype(np.uint8)


def test_bilinear_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    img = _img()
    for (oh, ow) in [(240, 320), (128, 128), (700, 900)]:
        ours = bilinear_resize(img, oh, ow)
        ref = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
        assert ours.shape == (oh, ow, 3)
        assert np.abs(ours.astype(int) - ref.astype(int)).max() <= 1


def test_bilinear_identity():
    img = _img(50, 70)
    assert np.array_equal(bilinear_resize(img, 50, 70), img)


def test_letterbox_geometry():
    img = _img(480, 640)
    canvas, scale, (px, py) = letterbox(img, 640, 114)
    assert canvas.shape == (640, 640, 3) and scale == 1.0 and (px, py) == (0, 80)
    assert (canvas[:80] == 114).all() and (canvas[560:] == 114).all()
    tall = _img(1000, 300)
    canvas, scale, (px, py) = letterbox(tall, 640, 114)
    assert abs(scale - 0.64) < 1e-9 and py == 0 and px == (640 - 192) // 2
    x = to_detector_input(canvas)
    assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32 and 0 <= x.min() and x.max() <= 1


def test_scale_boxes_back_roundtrip():
    img = _img(480, 640)
    _, scale, pad = letterbox(img)
    boxes = np.array([[10, 100, 200, 300]], np.float32)
    back = scale_boxes_back(boxes, scale, pad, 640, 480)
    assert np.allclose(back, [[10, 20, 200, 220]])


def test_crop_square_and_padding():
    img = _img()
    sx1, sy1, sx2, sy2 = square_crop_box([100, 100, 160, 130], 640, 480, 0.12)
    assert sx2 - sx1 == sy2 - sy1 >= 60 * 1.24
    crop = extract_crop(img, [100, 100, 160, 130])
    assert crop.shape == (128, 128, 3)
    edge = extract_crop(img, [-20, 450, 30, 500])   # partly outside -> padded, no crash
    assert edge.shape == (128, 128, 3) and (edge == 114).any()
    x = to_embed_input(np.stack([crop, edge]))
    assert x.shape == (2, 3, 128, 128) and x.dtype == np.float32


def test_nms_and_iou():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0, 0, 10, 10]], float)
    scores = np.array([0.9, 0.8, 0.7, 0.9])
    keep = nms(boxes, scores, 0.5, 10)
    assert list(keep) == [0, 2]           # stable tie-break keeps index 0, suppresses 1 and 3
    assert list(nms(boxes, scores, 0.5, 1)) == [0]
    assert len(nms(np.zeros((0, 4)), np.zeros(0), 0.5, 5)) == 0
    iou = box_iou_matrix(boxes[:1], boxes)
    assert iou[0, 0] == pytest.approx(1.0) and iou[0, 2] == 0 and 0.6 < iou[0, 1] < 0.7
