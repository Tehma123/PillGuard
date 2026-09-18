"""Pill-level and image-level metrics for scenario evaluation records.

A record is one ground-truth pill inside one scenario::

    {"scenario": id, "kind": clean|remove-1|remove-2, "file": ..., "label": 12, "gt": "in"|"out",
     "reason": listed|removed|foreign|unseen|unlisted, "seen": bool,
     "detected": bool, "verdict": in|out|uncertain|missed, "p_in": 0.93, "best_any": 12, ...}
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from pillguard.decision.calibration import expected_calibration_error
from pillguard.decision.thresholds import risk_coverage_curve


def _rates(recs: list[dict]) -> dict:
    n = len(recs)
    if n == 0:
        return {"n": 0}
    gt_out = np.array([r["gt"] == "out" for r in recs])
    det = np.array([r.get("detected", True) for r in recs])
    v = np.array([r["verdict"] for r in recs])
    decided = (v == "in") | (v == "out")
    out_v = v == "out"
    wrong = decided & ((out_v & ~gt_out) | ((v == "in") & gt_out))
    m = {
        "n": int(n),
        "n_out": int(gt_out.sum()),
        "n_in": int((~gt_out).sum()),
        "detection_recall": float(det.mean()),
        "abstain_rate": float((v == "uncertain").sum() / max(1, det.sum())),
        "coverage": float(decided.sum() / max(1, det.sum())),
        "out_recall": float((out_v & gt_out).sum() / max(1, (decided & gt_out).sum())),
        "out_recall_strict": float((out_v & gt_out).sum() / max(1, gt_out.sum())),   # misses + abstains count
        "false_alarm_rate": float((out_v & ~gt_out).sum() / max(1, (decided & ~gt_out).sum())),
        "false_alarm_rate_strict": float((out_v & ~gt_out).sum() / max(1, (~gt_out).sum())),
        "out_precision": float((out_v & gt_out).sum() / max(1, out_v.sum())),
        "risk": float(wrong.sum() / max(1, decided.sum())),
        "accuracy_decided": float(1 - wrong.sum() / max(1, decided.sum())),
    }
    return m


def summarize_records(records: list[dict], headline_reasons=("listed", "removed", "foreign", "unseen")) -> dict:
    """Headline numbers exclude ``unlisted`` pills (annotation noise); breakdowns include all."""
    head = [r for r in records if r["reason"] in headline_reasons]
    out = {"headline": _rates(head), "all": _rates(records)}
    by_reason = defaultdict(list)
    by_kind = defaultdict(list)
    for r in records:
        by_reason[r["reason"]].append(r)
        by_kind[r["kind"]].append(r)
    out["by_reason"] = {k: _rates(v) for k, v in sorted(by_reason.items())}
    out["by_kind"] = {k: _rates(v) for k, v in sorted(by_kind.items())}
    out["seen"] = _rates([r for r in head if r.get("seen", True)])
    out["unseen"] = _rates([r for r in head if not r.get("seen", True)])
    dec = [r for r in head if r.get("detected", True) and r.get("p_in") is not None]
    if dec:
        p = np.array([r["p_in"] for r in dec], float)
        y = np.array([r["gt"] == "in" for r in dec], float)
        out["ece"] = expected_calibration_error(p, y)["ece"]
        out["reliability"] = expected_calibration_error(p, y)["bins"]
        out["risk_coverage"] = risk_coverage_curve(p, y)
        # baseline without a reject option: same p_in, verdict at 0.5
        base = [dict(r, verdict=("in" if r["p_in"] >= 0.5 else "out")) for r in dec]
        out["no_reject_baseline"] = _rates(base)
    return out


def confusion_pairs(records: list[dict], names: dict[int, str] | None = None, top: int = 15) -> list[dict]:
    """Which drug the matcher thought a pill was when it was wrong (uses ``best_any``)."""
    names = names or {}
    c: Counter = Counter()
    for r in records:
        if not r.get("detected", True) or r.get("best_any") is None:
            continue
        if r["label"] >= 0 and r["label"] != r["best_any"] and r["reason"] in ("listed", "removed"):
            c[(int(r["label"]), int(r["best_any"]))] += 1
    total_by_label: Counter = Counter(int(r["label"]) for r in records if r["reason"] in ("listed", "removed"))
    rows = []
    for (a, b), n in c.most_common(top):
        rows.append({"true": a, "true_name": names.get(a, str(a)), "predicted": b, "predicted_name": names.get(b, str(b)),
                     "count": n, "share_of_true": n / max(1, total_by_label[a])})
    return rows


def image_level(records: list[dict]) -> dict:
    """Per scenario: did the system raise at least one alert when it should (and only then)?"""
    by = defaultdict(list)
    for r in records:
        by[r["scenario"]].append(r)
    tp = fp = fn = tn = 0
    for recs in by.values():
        should = any(r["gt"] == "out" for r in recs)
        did = any(r["verdict"] in ("out", "uncertain") for r in recs)
        tp += should and did; fp += (not should) and did; fn += should and (not did); tn += (not should) and (not did)
    return {"n_scenarios": len(by), "alert_recall": tp / max(1, tp + fn), "alert_false_rate": fp / max(1, fp + tn),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}
