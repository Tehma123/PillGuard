"""Pill-level and image-level metrics for scenario evaluation records.

A record is one ground-truth pill inside one scenario::

    {"scenario": id, "kind": clean|remove-1|remove-2, "file": ..., "pill_index": 0, "label": 12, "gt": "in"|"out",
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
        "out_accepted_rate": float(((v == "in") & gt_out).sum() / max(1, (det & gt_out).sum())),  # the costly error
        "false_alarm_rate": float((out_v & ~gt_out).sum() / max(1, (decided & ~gt_out).sum())),
        "false_alarm_rate_strict": float((out_v & ~gt_out).sum() / max(1, (~gt_out).sum())),
        "out_precision": float((out_v & gt_out).sum() / max(1, out_v.sum())),
        "risk": float(wrong.sum() / max(1, decided.sum())),
        "accuracy_decided": float(1 - wrong.sum() / max(1, decided.sum())),
    }
    return m


def summarize_records(records: list[dict], headline_reasons=("listed", "removed", "foreign", "unseen")) -> dict:
    """Headline numbers exclude ``unlisted`` pills (annotation noise); breakdowns include all.

    ``seen`` / ``unseen`` leave ``foreign`` pills out: label 107 says the pill belongs to another
    prescription, not which drug it is, so it cannot be called seen or unseen.
    """
    head = [r for r in records if r["reason"] in headline_reasons]
    out = {"headline": _rates(head), "all": _rates(records)}
    by_reason = defaultdict(list)
    by_kind = defaultdict(list)
    for r in records:
        by_reason[r["reason"]].append(r)
        by_kind[r["kind"]].append(r)
    out["by_reason"] = {k: _rates(v) for k, v in sorted(by_reason.items())}
    out["by_kind"] = {k: _rates(v) for k, v in sorted(by_kind.items())}
    known = [r for r in head if r["reason"] != "foreign"]
    out["seen"] = _rates([r for r in known if r.get("seen", True)])
    out["unseen"] = _rates([r for r in known if not r.get("seen", True)])
    out["naming"] = naming(records)
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


def naming(records: list[dict]) -> dict:
    """How often the nearest prototype is the pill's own drug, and what a wrong name costs.

    Accuracy counts each detected pill of a trained drug once (a photo recurs across its
    scenarios, its embedding does not change). A wrong name only matters when it lands on a
    listed drug, so for ``removed`` pills that were misnamed the verdicts are reported too.
    """
    known = [r for r in records if r["reason"] in ("listed", "removed") and r.get("seen", True)
             and r.get("detected", True) and r.get("best_any") is not None]
    pills = {(r["file"], r.get("pill_index")): r["best_any"] == r["label"] for r in known}
    removed = [r for r in known if r["reason"] == "removed"]
    misnamed = [r for r in removed if r["best_any"] != r["label"]]
    v = Counter(r["verdict"] for r in misnamed)
    return {"n_pills": len(pills), "accuracy": sum(pills.values()) / max(1, len(pills)),
            "removed": len(removed), "removed_misnamed": len(misnamed),
            "removed_misnamed_verdicts": {k: v[k] for k in ("out", "uncertain", "in")}}


PHOTO_STATES = ("ok", "uncertain", "out")


def image_level(records: list[dict], spurious: list[dict] | None = None) -> dict:
    """What the user sees per scenario, i.e. one photo checked against one prescription.

    A scenario reads ``out`` if any pill is judged out, else ``uncertain`` if any pill is, else
    ``ok``. Spurious detections count because the page shows them; missed pills cannot show.
    ``clean`` scenarios hold only listed pills, ``wrong`` ones at least one pill that is not.
    An alert means at least one ``out``; ``uncertain`` asks the user to check, it does not alert.
    """
    verdicts: dict[str, list[str]] = defaultdict(list)
    wrong: dict[str, bool] = {}
    for r in records:
        verdicts[r["scenario"]].append(r["verdict"])
        wrong[r["scenario"]] = wrong.get(r["scenario"], False) or r["gt"] == "out"
    for s in spurious or []:
        if s["scenario"] in wrong:
            verdicts[s["scenario"]].append(s["verdict"])
    table = {"clean": Counter(), "wrong": Counter()}
    for sid, vs in verdicts.items():
        state = "out" if "out" in vs else "uncertain" if "uncertain" in vs else "ok"
        table["wrong" if wrong[sid] else "clean"][state] += 1
    c, w = table["clean"], table["wrong"]
    nc, nw = sum(c.values()), sum(w.values())
    return {"n_scenarios": nc + nw, "n_clean": nc, "n_wrong": nw,
            "clean": {k: c[k] for k in PHOTO_STATES}, "wrong": {k: w[k] for k in PHOTO_STATES},
            "alert_recall": w["out"] / max(1, nw), "false_alert_rate": c["out"] / max(1, nc),
            "check_rate_clean": c["uncertain"] / max(1, nc), "silent_miss_rate": w["ok"] / max(1, nw)}
