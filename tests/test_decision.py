import numpy as np
import pytest

from pillguard.config import DecisionParams
from pillguard.decision.calibration import (
    CalibrationSet,
    binary_nll,
    calibration_report,
    expected_calibration_error,
    fit_temperature_and_bias,
    p_in_from,
)
from pillguard.decision.matcher import Matcher, verdicts_from_p
from pillguard.decision.thresholds import choose_thresholds, risk_coverage_curve, selective_metrics
from pillguard.eval.metrics import confusion_pairs, image_level, summarize_records


def _protos(seed=0, n=4, d=16):
    rng = np.random.default_rng(seed)
    p = {}
    for c in range(n):
        v = rng.normal(size=(2, d)).astype(np.float32)
        p[c] = v / np.linalg.norm(v, axis=1, keepdims=True)
    return p


def test_matcher_verdicts():
    protos = _protos()
    m = Matcher(protos, DecisionParams(temperature=0.05, unknown_bias=0.3, theta_in=0.8, theta_out=0.3))
    e0 = protos[0][0:1]
    d = m.decide(e0, listed=[0, 1])[0]
    assert d.verdict == "in" and d.best_listed == 0 and d.best_any == 0 and d.p_in > 0.9
    d = m.decide(e0, listed=[1, 2])[0]
    assert d.verdict == "out" and d.p_in < 0.1 and d.best_any == 0
    sims = m.class_similarities(e0)
    assert sims.shape == (1, 4) and sims[0, 0] == pytest.approx(1.0, abs=1e-5)
    # an embedding far from everything -> unknown dominates -> out
    far = np.random.default_rng(5).normal(size=16).astype(np.float32)
    P = np.concatenate(list(protos.values()))            # project out the span of every prototype
    far = far - P.T @ np.linalg.pinv(P.T) @ far
    far = (far / np.linalg.norm(far))[None].astype(np.float32)
    assert np.abs(m.class_similarities(far)).max() < 0.05
    d = m.decide(far, listed=[0, 1, 2, 3])[0]
    assert d.p_unknown > 0.5 and d.verdict == "out"
    # half-way between two classes, one listed one not -> uncertain
    mid = protos[0][0] + protos[1][0]
    mid = (mid / np.linalg.norm(mid))[None]
    d = m.decide(mid, listed=[0])[0]
    assert d.verdict == "uncertain" and 0.3 < d.p_in < 0.8
    v = verdicts_from_p(np.array([d.p_in, 0.95, 0.05]), None, m.params)
    assert list(v) == [2, 0, 1]


def test_matcher_unknown_listed_ignored():
    m = Matcher(_protos())
    e = _protos()[2][0:1]
    d = m.decide(e, listed=[2, 999])[0]
    assert d.verdict == "in" and d.best_listed == 2


def _cal_set(n=600, seed=0):
    rng = np.random.default_rng(seed)
    protos = _protos(seed)
    m = Matcher(protos)
    labels = rng.integers(0, 4, size=n)
    noise = rng.normal(scale=0.4, size=(n, 16)).astype(np.float32)
    embs = np.stack([protos[c][0] for c in labels]) + noise
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    listed = [sorted(rng.choice(4, size=2, replace=False).tolist()) for _ in range(n)]
    y = np.array([float(labels[i] in listed[i]) for i in range(n)], np.float32)
    sims = m.class_similarities(embs)
    masks = np.stack([m.listed_mask(l) for l in listed])
    return CalibrationSet(sims=sims, masks=masks, y_in=y)


def test_calibration_improves_nll_and_reports_ece():
    cal = _cal_set()
    before = DecisionParams(temperature=1 / 30, unknown_bias=0.0)
    fitted = fit_temperature_and_bias(cal, DecisionParams(temperature=0.1, unknown_bias=0.5))
    p_b = p_in_from(cal.sims, cal.masks, before.temperature, None)
    p_a = p_in_from(cal.sims, cal.masks, fitted.temperature, fitted.unknown_bias)
    assert binary_nll(p_a, cal.y_in) <= binary_nll(p_b, cal.y_in) + 1e-6
    rep = calibration_report(cal, before, fitted)
    assert 0 <= rep["after"]["ece"] <= 1 and rep["after"]["nll"] <= rep["before"]["nll"] + 1e-6
    e = expected_calibration_error(np.array([0.9, 0.9, 0.1, 0.1]), np.array([1, 1, 0, 0]))
    assert e["ece"] == pytest.approx(0.1)


def test_threshold_choice_respects_caps():
    cal = _cal_set()
    fitted = fit_temperature_and_bias(cal)
    p = p_in_from(cal.sims, cal.masks, fitted.temperature, fitted.unknown_bias)
    params, info = choose_thresholds(p, cal.y_in, fitted, max_fpr=0.10, max_abstain=0.30)
    m = info["val_metrics"]
    assert params.theta_out <= params.theta_in
    assert m["false_alarm_rate"] <= 0.10 + 1e-9 and m["abstain_rate"] <= info["max_abstain_used"] + 1e-9
    m2 = selective_metrics(p, cal.y_in, params)
    assert m2 == m
    rc = risk_coverage_curve(p, cal.y_in)
    assert rc["coverage"][-1] == pytest.approx(1.0) and rc["risk"][-1] == pytest.approx(rc["full_coverage_risk"])
    assert all(a <= b for a, b in zip(rc["coverage"], rc["coverage"][1:]))
    assert 0 <= rc["aurc"] <= 1


def test_threshold_choice_minimises_risk_not_recall():
    cal = _cal_set()
    fitted = fit_temperature_and_bias(cal)
    p = p_in_from(cal.sims, cal.masks, fitted.temperature, fitted.unknown_bias)
    params, info = choose_thresholds(p, cal.y_in, fitted, max_fpr=0.10, max_abstain=0.30, min_out_recall=0.0)
    chosen, alt = info["val_metrics"], info["max_out_recall_alternative"]["val_metrics"]
    # maximising out recall is what the search used to do, and it costs risk
    assert chosen["risk"] <= alt["risk"] + 1e-12
    assert alt["out_recall"] >= chosen["out_recall"] - 1e-12
    assert info["objective"].startswith("lowest risk")


def test_threshold_choice_honours_the_recall_floor():
    cal = _cal_set()
    fitted = fit_temperature_and_bias(cal)
    p = p_in_from(cal.sims, cal.masks, fitted.temperature, fitted.unknown_bias)
    _, info = choose_thresholds(p, cal.y_in, fitted, max_fpr=0.10, max_abstain=0.30, min_out_recall=0.80)
    assert info["min_out_recall_used"] == 0.80
    assert info["val_metrics"]["out_recall"] >= 0.80 - 1e-9
    # an unreachable floor is dropped rather than collapsing to the no-reject point
    _, hard = choose_thresholds(p, cal.y_in, fitted, max_fpr=0.10, max_abstain=0.30, min_out_recall=1.01)
    assert hard["min_out_recall_used"] == 0.0 and hard["relaxed"]


def test_metrics_summary():
    recs = [
        dict(scenario="a", kind="remove-1", file="a", label=1, gt="in", reason="listed", seen=True, detected=True, verdict="in", p_in=0.9, best_any=1),
        dict(scenario="a", kind="remove-1", file="a", label=2, gt="out", reason="removed", seen=True, detected=True, verdict="out", p_in=0.1, best_any=2),
        dict(scenario="a", kind="remove-1", file="a", label=3, gt="out", reason="removed", seen=True, detected=True, verdict="uncertain", p_in=0.5, best_any=1),
        dict(scenario="b", kind="clean", file="b", label=1, gt="in", reason="listed", seen=True, detected=False, verdict="missed", p_in=None, best_any=None),
        dict(scenario="b", kind="clean", file="b", label=107, gt="out", reason="foreign", seen=False, detected=True, verdict="in", p_in=0.8, best_any=1),
        dict(scenario="c", kind="clean", file="c", label=4, gt="out", reason="unseen", seen=False, detected=True, verdict="out", p_in=0.2, best_any=1),
    ]
    s = summarize_records(recs)
    h = s["headline"]
    assert h["n"] == 6 and h["n_out"] == 4
    assert h["detection_recall"] == pytest.approx(5 / 6)
    assert h["out_recall"] == pytest.approx(2 / 3)            # decided out pills: 2,107,4 -> 2 correct
    assert h["out_recall_strict"] == pytest.approx(2 / 4)
    assert h["abstain_rate"] == pytest.approx(1 / 5)
    assert h["false_alarm_rate"] == 0.0
    assert s["unseen"]["n"] == 2 and s["by_reason"]["foreign"]["out_recall"] == 0.0
    assert "ece" in s and "risk_coverage" in s and s["no_reject_baseline"]["n"] == 5
    il = image_level(recs)
    assert il["alert_recall"] == pytest.approx(2 / 3) and il["tp"] == 2 and il["fn"] == 1
    cp = confusion_pairs(recs, {1: "one", 3: "three"})
    assert cp and cp[0]["true"] == 3 and cp[0]["predicted"] == 1
