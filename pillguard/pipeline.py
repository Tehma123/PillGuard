"""End-to-end inference with ONNX Runtime: photo + listed drugs -> per-pill verdicts.

This is the reference implementation of what ``web/pipeline.js`` does in the browser;
the parity test feeds both the same photos and expects identical verdicts.

    from pillguard.pipeline import PillGuardPipeline
    pipe = PillGuardPipeline.from_dir("web")          # loads models + prototypes + config
    result = pipe.run(load_rgb("photo.jpg"), listed=[47, 10, 64])
    for box, d in zip(result.boxes, result.decisions): print(box, d.verdict, d.p_in)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pillguard.config import (
    CROP_CONTEXT,
    CROP_SIZE,
    DET_CONF,
    DET_INPUT,
    DET_IOU,
    DET_MAX_DETS,
    DecisionParams,
)
from pillguard.decision.matcher import Decision, Matcher
from pillguard.embed.prototypes import load_prototypes_npz, prototypes_from_json
from pillguard.imaging import (
    extract_crop,
    letterbox,
    nms,
    scale_boxes_back,
    to_detector_input,
    to_embed_input,
)
from pillguard.onnx_utils import ort_session, run


@dataclass
class Result:
    boxes: np.ndarray                       # (n, 4) xyxy in image pixels
    scores: np.ndarray                      # (n,)
    embeddings: np.ndarray                  # (n, D)
    decisions: list[Decision]
    timings_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"boxes": np.round(self.boxes, 1).tolist(), "scores": np.round(self.scores, 4).tolist(),
                "decisions": [d.to_dict() for d in self.decisions], "timings_ms": self.timings_ms}


class PillGuardPipeline:
    def __init__(self, detector_onnx: Path, embed_onnx: Path, prototypes: dict[int, np.ndarray],
                 params: DecisionParams | None = None, class_names: dict[int, str] | None = None,
                 threads: int = 1, det_conf: float = DET_CONF, det_iou: float = DET_IOU,
                 det_max: int = DET_MAX_DETS, det_input: int = DET_INPUT, crop_size: int = CROP_SIZE,
                 crop_context: float = CROP_CONTEXT) -> None:
        self.det = ort_session(detector_onnx, threads)
        self.emb = ort_session(embed_onnx, threads)
        self.matcher = Matcher(prototypes, params or DecisionParams())
        self.class_names = class_names or {}
        self.det_conf, self.det_iou, self.det_max, self.det_input = det_conf, det_iou, det_max, det_input
        self.crop_size, self.crop_context = crop_size, crop_context

    # -- construction ---------------------------------------------------------------------
    @classmethod
    def from_dir(cls, web_dir: Path, threads: int = 1) -> PillGuardPipeline:
        """Load from the browser asset layout (web/models + web/data)."""
        web_dir = Path(web_dir)
        cfg = json.loads((web_dir / "data" / "config.json").read_text(encoding="utf-8"))
        protos = prototypes_from_json(json.loads((web_dir / "data" / "prototypes.json").read_text(encoding="utf-8")))
        drugs = json.loads((web_dir / "data" / "drugs.json").read_text(encoding="utf-8"))
        names = {int(d["id"]): d["name"] for d in drugs}
        return cls(web_dir / "models" / cfg["models"]["detector"], web_dir / "models" / cfg["models"]["embedding"],
                   protos, DecisionParams.from_dict(cfg["decision"]), names, threads=threads,
                   det_conf=cfg["det_conf"], det_iou=cfg["det_iou"], det_max=cfg["det_max_dets"],
                   det_input=cfg["det_input"], crop_size=cfg["crop_size"], crop_context=cfg["crop_context"])

    @classmethod
    def from_artifacts(cls, detector_onnx: Path, embed_onnx: Path, prototypes_npz: Path,
                       params: DecisionParams | None = None, threads: int = 1) -> PillGuardPipeline:
        return cls(detector_onnx, embed_onnx, load_prototypes_npz(prototypes_npz), params, threads=threads)

    # -- stages ---------------------------------------------------------------------------
    def detect(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        canvas, scale, pad = letterbox(img, self.det_input)
        raw = run(self.det, to_detector_input(canvas))       # (1, 5, N)
        p = raw[0].T
        scores = p[:, 4]
        m = scores >= self.det_conf
        p, scores = p[m], scores[m]
        boxes = np.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2,
                          p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], 1) if len(p) else np.zeros((0, 4), np.float32)
        keep = nms(boxes, scores, self.det_iou, self.det_max)
        boxes, scores = boxes[keep], scores[keep]
        return scale_boxes_back(boxes, scale, pad, w, h), scores.astype(np.float32)

    def embed(self, img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros((0, self.matcher.vectors.shape[1]), np.float32)
        crops = np.stack([extract_crop(img, b, self.crop_size, self.crop_context) for b in boxes])
        embs = run(self.emb, to_embed_input(crops))
        return embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)

    def decide(self, embeddings: np.ndarray, listed) -> list[Decision]:
        return self.matcher.decide(embeddings, listed)

    # -- end to end -----------------------------------------------------------------------
    def run(self, img: np.ndarray, listed) -> Result:
        t0 = time.perf_counter()
        boxes, scores = self.detect(img)
        t1 = time.perf_counter()
        embs = self.embed(img, boxes)
        t2 = time.perf_counter()
        decisions = self.decide(embs, listed)
        t3 = time.perf_counter()
        return Result(boxes, scores, embs, decisions,
                      {"detect": (t1 - t0) * 1e3, "embed": (t2 - t1) * 1e3, "decide": (t3 - t2) * 1e3,
                       "total": (t3 - t0) * 1e3})

    def run_with_boxes(self, img: np.ndarray, boxes, listed) -> Result:
        """Oracle-box mode: skip the detector (isolates recognition + decision quality)."""
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        t0 = time.perf_counter()
        embs = self.embed(img, boxes)
        decisions = self.decide(embs, listed)
        return Result(boxes, np.ones(len(boxes), np.float32), embs, decisions,
                      {"total": (time.perf_counter() - t0) * 1e3})
