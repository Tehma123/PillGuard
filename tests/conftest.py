from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pillguard.config import CROP_SIZE, DET_INPUT, DecisionParams
from pillguard.data.synthetic import make_synthetic_dataset
from pillguard.embed.model import build_model
from pillguard.onnx_utils import export_torch


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synth")
    make_synthetic_dataset(root, n_images=30, n_classes=6, seed=1)
    return root


class _DummyDetector(torch.nn.Module):
    """Emits a fixed set of candidate boxes (letterbox pixels) whose scores depend on the input
    so that preprocessing differences would show up in the parity test."""

    def __init__(self, boxes_cxcywh: list[list[float]], n: int = 8400) -> None:
        super().__init__()
        base = torch.zeros(1, 5, n)
        for i, (cx, cy, w, h) in enumerate(boxes_cxcywh):
            base[0, :4, i] = torch.tensor([cx, cy, w, h])
            base[0, 4, i] = 0.6 + 0.05 * i
        self.register_buffer("base", base)

    def forward(self, x):
        m = x.mean() * 0.1  # small input dependence
        out = self.base.clone()
        out[0, 4] = out[0, 4] + m * (out[0, 4] > 0).float()
        return out


def make_dummy_detector_onnx(path: Path, boxes_cxcywh: list[list[float]]) -> Path:
    model = _DummyDetector(boxes_cxcywh).eval()
    torch.onnx.export(model, torch.zeros(1, 3, DET_INPUT, DET_INPUT), str(path), input_names=["images"],
                      output_names=["output0"], opset_version=17, dynamo=False)
    return path


@pytest.fixture(scope="session")
def dummy_models(tmp_path_factory) -> dict:
    """Random-weight embedding net + dummy detector, exported to ONNX, plus random prototypes."""
    d = tmp_path_factory.mktemp("models")
    torch.manual_seed(0)
    emb = build_model(pretrained=False).eval()
    emb_path = export_torch(emb, torch.zeros(1, 3, CROP_SIZE, CROP_SIZE), d / "embed.int8.onnx", "input", "embedding")
    # boxes in 640x640 letterbox coords for a 640x480 synthetic image (pad_y = 80)
    det_path = make_dummy_detector_onnx(d / "detector.int8.onnx", [[120, 200, 90, 90], [400, 300, 100, 80], [520, 420, 70, 70]])
    rng = np.random.default_rng(0)
    protos = {c: rng.normal(size=(2, 128)).astype(np.float32) for c in range(6)}
    for c in protos:
        protos[c] /= np.linalg.norm(protos[c], axis=1, keepdims=True)
    return {"dir": d, "embed": emb_path, "detector": det_path, "prototypes": protos, "torch_embed": emb}


@pytest.fixture(scope="session")
def web_assets(dummy_models, synth_root, tmp_path_factory) -> Path:
    """A web/ layout (models + data) built from the dummy models via the real exporter."""
    from pillguard.data.splits import make_split
    from pillguard.data.vaipe import load_pill_images
    from pillguard.embed.prototypes import save_prototypes_npz
    from pillguard.web.export_assets import export_web_assets

    web = tmp_path_factory.mktemp("web")
    split_path = synth_root / "split.json"
    if not split_path.exists():
        make_split(load_pill_images(synth_root), seed=1, n_unseen=1).save(split_path)
    npz = dummy_models["dir"] / "prototypes.npz"
    save_prototypes_npz(npz, dummy_models["prototypes"])
    params = dummy_models["dir"] / "params.json"
    params.write_text(json.dumps(DecisionParams(temperature=0.1, unknown_bias=0.3, theta_in=0.7, theta_out=0.3).to_dict()))
    export_web_assets(web, dummy_models["detector"], dummy_models["embed"], npz, params, synth_root, split_path)
    return web
