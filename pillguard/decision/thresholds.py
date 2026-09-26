"""Choose the reject thresholds on validation data and compute risk-coverage curves.

Definitions (per pill; "decided" = verdict is in or out, not uncertain)
----------------------------------------------------------------------
* abstain rate     = uncertain / all
* out recall       = (gt out & verdict out) / (gt out & decided)       ["selective"]
* out recall strict= (gt out & verdict out) / (gt out)                 [abstentions count as misses]
* false alarm rate = (gt in & verdict out)  / (gt in & decided)
* risk             = wrong verdicts / decided ; coverage = decided / all

Threshold search: grid over (theta_out <= theta_in); feasible if false-alarm <= ``max_fpr``,
abstain <= ``max_abstain`` and out recall >= ``min_out_recall``; among those pick the lowest
risk, ties broken by higher out recall then lower abstention. Out recall is a floor the spec
sets, not the thing to maximise: maximising it spends the whole abstention budget buying
recall the floor already guarantees and leaves risk at the no-reject level, which defeats the
point of having a reject option. If nothing is feasible the abstain cap is relaxed step by
step, then the recall floor is dropped, and the report says which happened.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pillguard.config import DecisionParams


def selective_metrics(p_in: np.ndarray, y_in: np.ndarray, params: DecisionParams,
                      margin: np.ndarray | None = None) -> dict:
    from pillguard.decision.matcher import verdicts_from_p

    v = verdicts_from_p(p_in, margin, params)
    y_in = np.asarray(y_in).astype(bool)
    decided = v != 2
    gt_out = ~y_in
    n = len(v)
    out_v = v == 1
    d_out = decided & gt_out
    d_in = decided & y_in
    wrong = decided & ((out_v & y_in) | ((v == 0) & gt_out))
    return {
        "n": int(n),
        "abstain_rate": float((~decided).mean()) if n else 0.0,
        "coverage": float(decided.mean()) if n else 0.0,
        "out_recall": float((out_v & gt_out).sum() / max(1, d_out.sum())),
        "out_recall_strict": float((out_v & gt_out).sum() / max(1, gt_out.sum())),
        "false_alarm_rate": float((out_v & y_in).sum() / max(1, d_in.sum())),
        "false_alarm_rate_strict": float((out_v & y_in).sum() / max(1, y_in.sum())),
        "out_precision": float((out_v & gt_out).sum() / max(1, out_v.sum())),
        "risk": float(wrong.sum() / max(1, decided.sum())),
        "accuracy_decided": float(1 - wrong.sum() / max(1, decided.sum())),
        "n_out": int(gt_out.sum()),
        "n_in": int(y_in.sum()),
    }


def feasible_thresholds(p_in: np.ndarray, y_in: np.ndarray, base: DecisionParams, max_fpr: float,
                        max_abstain: float, min_out_recall: float, grid: int,
                        margin: np.ndarray | None) -> list[tuple[DecisionParams, dict]]:
    out = []
    for lo in np.linspace(0.0, 1.0, grid):
        for hi in np.linspace(0.0, 1.0, grid):
            if hi < lo:
                continue
            params = replace(base, theta_in=float(hi), theta_out=float(lo))
            m = selective_metrics(p_in, y_in, params, margin)
            if (m["false_alarm_rate"] <= max_fpr and m["abstain_rate"] <= max_abstain
                    and m["out_recall"] >= min_out_recall):
                out.append((params, m))
    return out


def choose_thresholds(p_in: np.ndarray, y_in: np.ndarray, base: DecisionParams, max_fpr: float = 0.10,
                      max_abstain: float = 0.20, grid: int = 41, margin: np.ndarray | None = None,
                      min_out_recall: float = 0.95) -> tuple[DecisionParams, dict]:
    cap, relax_steps, floor, feasible = max_abstain, 0, min_out_recall, []
    while not feasible and cap <= 1.0 + 1e-9:
        feasible = feasible_thresholds(p_in, y_in, base, max_fpr, cap, floor, grid, margin)
        if not feasible:
            cap += 0.05
            relax_steps += 1
    if not feasible and floor > 0.0:          # the recall floor, not the budget, was the blocker
        cap, floor = max_abstain, 0.0
        feasible = feasible_thresholds(p_in, y_in, base, max_fpr, cap, floor, grid, margin)
    if not feasible:                          # degenerate data: no reject
        params = replace(base, theta_in=0.5, theta_out=0.5)
        feasible = [(params, selective_metrics(p_in, y_in, params, margin))]
    lowest_risk = min(feasible, key=lambda c: (c[1]["risk"], -c[1]["out_recall"], c[1]["abstain_rate"]))
    most_recall = max(feasible, key=lambda c: (c[1]["out_recall"], -c[1]["abstain_rate"], -c[1]["false_alarm_rate"]))
    params, metrics = lowest_risk
    info = {"max_fpr": max_fpr, "max_abstain_requested": max_abstain, "max_abstain_used": cap,
            "min_out_recall_requested": min_out_recall, "min_out_recall_used": floor,
            "relaxed": relax_steps > 0 or floor != min_out_recall, "n_feasible": len(feasible),
            "objective": "lowest risk among feasible thresholds",
            "val_metrics": metrics,
            "max_out_recall_alternative": {"params": most_recall[0].to_dict(), "val_metrics": most_recall[1]}}
    return params, info


def risk_coverage_curve(p_in: np.ndarray, y_in: np.ndarray, n_points: int = 50) -> dict:
    """Confidence = max(p_in, 1 - p_in); sweep a confidence threshold, record risk vs coverage."""
    p_in = np.asarray(p_in, float)
    y_in = np.asarray(y_in).astype(bool)
    pred_in = p_in >= 0.5
    conf = np.maximum(p_in, 1 - p_in)
    wrong = pred_in != y_in
    order = np.argsort(-conf, kind="stable")
    wrong_sorted = wrong[order]
    n = len(p_in)
    if n == 0:
        return {"coverage": [], "risk": [], "aurc": float("nan"), "full_coverage_risk": float("nan")}
    cum_wrong = np.cumsum(wrong_sorted)
    ks = np.unique(np.clip(np.round(np.linspace(1, n, n_points)).astype(int), 1, n))
    cov = ks / n
    risk = cum_wrong[ks - 1] / ks
    aurc = float(np.mean(cum_wrong / np.arange(1, n + 1)))  # area under the full curve
    return {"coverage": cov.tolist(), "risk": risk.tolist(), "aurc": aurc, "full_coverage_risk": float(wrong.mean())}
