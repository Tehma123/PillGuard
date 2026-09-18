import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pillguard.config import DecisionParams
from pillguard.data.scenarios import make_scenarios
from pillguard.data.splits import Split
from pillguard.data.vaipe import load_pill_images, prescription_drug_table
from pillguard.eval.run_eval import evaluate, match_boxes
from pillguard.imaging import load_rgb
from pillguard.pipeline import PillGuardPipeline

WEB_SRC = Path(__file__).resolve().parents[1] / "web"


def test_pipeline_end_to_end(dummy_models, synth_root):
    pipe = PillGuardPipeline(dummy_models["detector"], dummy_models["embed"], dummy_models["prototypes"],
                             DecisionParams(temperature=0.1, unknown_bias=0.3, theta_in=0.7, theta_out=0.3))
    im = load_pill_images(synth_root)[0]
    img = load_rgb(im.path(synth_root))
    boxes, scores = pipe.detect(img)
    assert boxes.shape == (3, 4) and len(scores) == 3
    # dummy boxes were given in 640-letterbox coords for a 640x480 photo (pad_y = 80);
    # detections come back in descending score order, so look the box up rather than index it
    assert any(np.allclose(b, [75, 75, 165, 165], atol=1) for b in boxes)
    assert all(s1 >= s2 for s1, s2 in zip(scores, scores[1:]))
    r = pipe.run(img, listed=[0, 1])
    assert len(r.decisions) == 3 and all(d.verdict in ("in", "out", "uncertain") for d in r.decisions)
    assert r.embeddings.shape == (3, 128) and np.allclose(np.linalg.norm(r.embeddings, axis=1), 1, atol=1e-4)
    assert set(r.timings_ms) >= {"detect", "embed", "decide", "total"}
    r2 = pipe.run_with_boxes(img, im.boxes, listed=[0])
    assert len(r2.decisions) == len(im.boxes)
    d = r.to_dict()
    assert json.dumps(d)


def test_match_boxes():
    gt = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float)
    det = np.array([[1, 1, 11, 11], [100, 100, 110, 110], [49, 49, 61, 61]], float)
    m = match_boxes(gt, det, np.array([0.9, 0.8, 0.7]))
    assert m == {0: 0, 1: 2}


def test_oracle_evaluation_on_synthetic(dummy_models, synth_root, tmp_path):
    pipe = PillGuardPipeline(dummy_models["detector"], dummy_models["embed"], dummy_models["prototypes"],
                             DecisionParams(temperature=0.1, unknown_bias=0.3, theta_in=0.7, theta_out=0.3))
    ims = load_pill_images(synth_root)
    split = Split.load(synth_root / "split.json") if (synth_root / "split.json").exists() else None
    if split is None:
        from pillguard.data.splits import make_split
        split = make_split(ims, seed=1, n_unseen=1)
    by = {im.file: im for im in ims}
    test_ims = [by[f] for f in split.test]
    scen = make_scenarios(test_ims, prescription_drug_table(synth_root), split.unseen_classes, seed=1)
    records, spurious, timings = evaluate(pipe, test_ims, scen, synth_root, seen=set(range(6)), mode="oracle", verbose=False)
    assert len(records) == sum(len(s.pills) for s in scen) and not spurious
    assert all(r["detected"] for r in records)
    records, spurious, timings = evaluate(pipe, test_ims, scen, synth_root, seen=set(range(6)), mode="detector", verbose=False)
    assert len(records) == sum(len(s.pills) for s in scen)
    from pillguard.eval.metrics import summarize_records
    m = summarize_records(records)
    assert 0 <= m["headline"]["detection_recall"] <= 1


def test_web_assets_roundtrip(web_assets, synth_root):
    cfg = json.loads((web_assets / "data" / "config.json").read_text())
    assert cfg["models"]["detector"] == "detector.int8.onnx" and cfg["decision"]["theta_in"] == 0.7
    drugs = json.loads((web_assets / "data" / "drugs.json").read_text())
    assert [d["id"] for d in drugs] == list(range(6))
    pipe = PillGuardPipeline.from_dir(web_assets)
    img = load_rgb(load_pill_images(synth_root)[0].path(synth_root))
    r = pipe.run(img, listed=[0])
    assert len(r.decisions) == 3


@pytest.mark.skipif(shutil.which("node") is None or not (WEB_SRC / "node_modules" / "onnxruntime-web").exists(),
                    reason="node + web/node_modules (npm install) required")
def test_js_python_parity(web_assets, synth_root, tmp_path):
    """Same PNG pixels through pipeline.js (onnxruntime-web, WASM) and pipeline.py must agree."""
    from pillguard.eval.parity import compare, export_cases, run_python

    parity = tmp_path / "parity"
    export_cases(synth_root, synth_root / "split.json", parity, n=4, n_oracle=2)
    run_python(parity, web_assets)
    # the Node harness resolves pipeline.js + node_modules relative to web/, assets from the given dir
    subprocess.run(["node", str(WEB_SRC / "parity_node.mjs"), str(parity), str(web_assets)], cwd=WEB_SRC, check=True,
                   capture_output=True, text=True, timeout=300)
    rep = compare(parity)
    assert rep["cases"] == 6 and rep["same_box_count"] == 6
    assert rep["verdict_agreement"] == 1.0, rep["mismatches"]
    assert rep["max_abs_box_diff_px"] <= 0.5 and rep["max_abs_p_in_diff"] <= 2e-3
