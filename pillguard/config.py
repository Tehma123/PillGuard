"""Project-wide constants and paths.

Everything that must agree between the Python pipeline and the browser pipeline
(input sizes, normalisation, embedding size, letterbox colour) is defined here and
exported to ``web/data/config.json`` by ``pillguard.web.export_assets`` so the two
never drift apart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = Path(os.environ.get("PILLGUARD_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = Path(os.environ.get("PILLGUARD_DATA", ROOT / "data"))
VAIPE_DIR = DATA_DIR / "vaipe"
ARTIFACTS_DIR = Path(os.environ.get("PILLGUARD_ARTIFACTS", ROOT / "artifacts"))
WEB_DIR = ROOT / "web"
SPLITS_DIR = ROOT / "splits"  # fixed split files are small and committed


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
HF_DATASET = "Elfsong/VAIPE_PILL"
INGEST_MAX_SIDE = 1280        # pill photos are 4032x3024; at 1280 a pill is ~80-120 px, close to the 128 px crop
PRESCRIPTION_MAX_SIDE = 1600  # prescriptions are kept larger for OCR experiments
OUT_OF_PRESCRIPTION_LABEL = 107  # VAIPE challenge convention: 107 == "pill not in this prescription"

SEED = 20260918
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15          # of the labelled pool; grouped by prescription to avoid leakage
N_UNSEEN_CLASSES = 12         # held out of embedding training to test novelty detection


# --------------------------------------------------------------------------------------
# Models (must match web/pipeline.js)
# --------------------------------------------------------------------------------------
DET_INPUT = 640               # detector letterbox size
DET_PAD_VALUE = 114           # letterbox fill (Ultralytics convention)
DET_CONF = 0.25
DET_IOU = 0.6
DET_MAX_DETS = 100

CROP_SIZE = 128               # embedding input (square)
CROP_CONTEXT = 0.12           # fractional padding added around each box before cropping
EMB_DIM = 128
EMB_MEAN = (0.485, 0.456, 0.406)
EMB_STD = (0.229, 0.224, 0.225)
PROTOTYPES_PER_CLASS = 3      # k-means sub-prototypes per drug (front/back/lighting)


@dataclass
class DecisionParams:
    """Parameters of the in/out/abstain decision. Fitted on validation, exported to the web."""

    temperature: float = 0.05     # softmax temperature over cosine similarities
    unknown_bias: float = 0.55    # cosine "similarity" assigned to the virtual unknown class
    theta_in: float = 0.80        # p_in >= theta_in  -> in prescription
    theta_out: float = 0.30       # p_in <= theta_out -> out of prescription
    margin: float = 0.0           # optional extra margin requirement s_in - s_out
    platt_a: float = 1.0          # p_in recalibration sigmoid(a * logit(p_in) + b); (1, 0) = identity
    platt_b: float = 0.0

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "unknown_bias": self.unknown_bias,
            "theta_in": self.theta_in,
            "theta_out": self.theta_out,
            "margin": self.margin,
            "platt_a": self.platt_a,
            "platt_b": self.platt_b,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DecisionParams:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class SharedConfig:
    """The subset of constants the browser needs. Serialised to web/data/config.json."""

    det_input: int = DET_INPUT
    det_pad_value: int = DET_PAD_VALUE
    det_conf: float = DET_CONF
    det_iou: float = DET_IOU
    det_max_dets: int = DET_MAX_DETS
    crop_size: int = CROP_SIZE
    crop_context: float = CROP_CONTEXT
    emb_dim: int = EMB_DIM
    emb_mean: tuple = EMB_MEAN
    emb_std: tuple = EMB_STD
    decision: dict = field(default_factory=lambda: DecisionParams().to_dict())

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["emb_mean"] = list(self.emb_mean)
        d["emb_std"] = list(self.emb_std)
        return d
