"""Quantisation guard rails and the p_in recalibration shared with the browser.

Every number quoted here is from the export that shipped a detector finding zero boxes and
an embedding whose vectors had cosine 0.23 against fp32: the reports were correct, nothing
acted on them.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import onnx
import pytest

from pillguard.config import DecisionParams
from pillguard.decision.matcher import Matcher, platt
from pillguard.detector.export import check_detector_quantisation, fp32_tail_nodes
from pillguard.embed.export import N_FP32_CONVS, check_embedding_quantisation, early_conv_nodes
from pillguard.onnx_utils import QuantizationError

ROOT = Path(__file__).resolve().parents[1]


# -- p_in recalibration ---------------------------------------------------------------------

def test_platt_identity_is_a_no_op():
    p = np.array([0.0, 0.02, 0.5, 0.98, 1.0])
    assert platt(p, 1.0, 0.0) is p                       # untouched, so 0 and 1 stay exact


def test_platt_shifts_and_stays_monotonic():
    p = np.linspace(0.001, 0.999, 50)
    q = platt(p, 0.849, 2.861)                           # the parameters the fit chose
    assert np.all(np.diff(q) > 0)
    assert np.all(q > p)                                 # b > 0 corrects underconfidence
    assert float(platt(np.array([0.5]), 1.0, 0.0)[0]) == pytest.approx(0.5)
    assert float(platt(np.array([0.5]), 1.0, 2.0)[0]) == pytest.approx(1 / (1 + np.exp(-2.0)))


def test_platt_survives_the_endpoints():
    q = platt(np.array([0.0, 1.0]), 0.849, 2.861)
    assert np.all(np.isfinite(q)) and np.all((q > 0) & (q < 1))


def test_decision_params_round_trip_carries_platt():
    p = DecisionParams(temperature=0.086, unknown_bias=-0.785, platt_a=0.849, platt_b=2.861)
    assert DecisionParams.from_dict(p.to_dict()) == p
    assert DecisionParams.from_dict({"temperature": 0.1}).platt_a == 1.0   # older params.json


def test_matcher_applies_platt():
    rng = np.random.default_rng(0)
    protos = {}
    for c in range(4):
        v = rng.normal(size=(2, 16)).astype(np.float32)
        protos[c] = v / np.linalg.norm(v, axis=1, keepdims=True)
    plain = DecisionParams(temperature=0.05, unknown_bias=0.3)
    shifted = DecisionParams(temperature=0.05, unknown_bias=0.3, platt_a=0.849, platt_b=2.861)
    m = Matcher(protos, plain)
    sims = m.class_similarities(protos[0][0:1])
    mask = m.listed_mask([1, 2])                          # a pill that is *not* listed: p_in is low
    p_plain, _, _ = m.probabilities(sims, mask, plain)
    p_shift, _, _ = m.probabilities(sims, mask, shifted)
    assert p_shift[0] > p_plain[0]
    assert p_shift[0] == pytest.approx(float(platt(p_plain, 0.849, 2.861)[0]))


def test_platt_matches_the_browser():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    cases = [[0.02, 0.849, 2.861], [0.5, 0.849, 2.861], [0.999, 1.2, -0.4],
             [0.0, 1.0, 0.5], [1.0, 1.0, 0.5], [0.3, 1.0, 0.0]]
    script = (f"import {{ platt }} from {json.dumps((ROOT / 'web' / 'pipeline.js').as_uri())};\n"
              f"console.log(JSON.stringify({json.dumps(cases)}.map(([p, a, b]) => platt(p, a, b))));\n")
    out = subprocess.run([node, "--input-type=module", "-e", script],
                         capture_output=True, text=True, check=True)
    for (p, a, b), got in zip(cases, json.loads(out.stdout)):
        assert got == pytest.approx(float(platt(np.array([p]), a, b)[0]), abs=1e-12)


# -- quantisation guard rails ---------------------------------------------------------------

def test_detector_guard_accepts_a_healthy_export():
    check_detector_quantisation({"fp32_boxes": 160, "int8_boxes": 160, "mean_best_iou": 0.966})


def test_detector_guard_catches_the_zero_box_export():
    with pytest.raises(QuantizationError, match="0 int8 boxes"):
        check_detector_quantisation({"fp32_boxes": 160, "int8_boxes": 0, "mean_best_iou": 0.0})


def test_detector_guard_catches_drifted_boxes():
    with pytest.raises(QuantizationError, match="IoU"):
        check_detector_quantisation({"fp32_boxes": 160, "int8_boxes": 158, "mean_best_iou": 0.31})


def test_embedding_guard_accepts_a_healthy_export():
    check_embedding_quantisation({"fp32_val_proto_acc": 0.8784, "int8_val_proto_acc": 0.8741,
                                  "int8_vs_fp32_cosine_mean": 0.9890})


def test_embedding_guard_catches_the_collapse():
    with pytest.raises(QuantizationError) as e:
        check_embedding_quantisation({"fp32_val_proto_acc": 0.8784, "int8_val_proto_acc": 0.1794,
                                      "int8_vs_fp32_cosine_mean": 0.2293})
    assert "cosine" in str(e.value) and "nearest-prototype" in str(e.value)


def test_embedding_guard_ignores_a_report_without_the_checks():
    check_embedding_quantisation({"fp32_mb": 5.19, "int8_mb": 1.67})   # no val crops available


# -- node selection -------------------------------------------------------------------------

def test_early_conv_nodes_are_the_first_convs(dummy_models):
    path = dummy_models["embed"]
    names = early_conv_nodes(path, N_FP32_CONVS)
    convs = [n.name for n in onnx.load(str(path)).graph.node if n.op_type == "Conv"]
    assert len(names) == N_FP32_CONVS < len(convs)
    assert names == convs[:N_FP32_CONVS]


def test_early_conv_nodes_clamp_to_what_exists(dummy_models):
    convs = [n.name for n in onnx.load(str(dummy_models["embed"])).graph.node if n.op_type == "Conv"]
    assert early_conv_nodes(dummy_models["embed"], 10_000) == convs


def test_fp32_tail_always_covers_the_output_node(dummy_models):
    path = dummy_models["detector"]
    graph = onnx.load(str(path)).graph
    tail = fp32_tail_nodes(path)
    producers = {out: n.name for n in graph.node for out in n.output}
    for out in graph.output:
        assert producers[out.name] in tail        # the tensor mixing boxes and scores
    assert all(n.op_type != "Conv" for n in graph.node if n.name in tail)
