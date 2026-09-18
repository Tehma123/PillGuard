"""Fit the temperature and the unknown-class bias on validation pills; expected calibration error.

Input is a *calibration set*: for every validation pill its class-similarity vector (from
the matcher), the listed-drug mask of its scenario and the binary truth (1 = the pill is
a listed drug). Both parameters are fitted by minimising the binary negative log-likelihood
of ``p_in`` (Nelder-Mead over ``log T`` and ``b``).

"Before calibration" is defined as the network's own training softmax: ``T = 1 / s`` with
``s`` the ArcFace scale and no unknown class. That is what a plain classifier would report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pillguard.config import DecisionParams
from pillguard.decision.matcher import Matcher


@dataclass
class CalibrationSet:
    sims: np.ndarray        # (n, C) class similarities
    masks: np.ndarray       # (n, C) bool, listed drugs of the pill's scenario
    y_in: np.ndarray        # (n,) 1 if the pill's drug is listed
    seen: np.ndarray | None = None  # (n,) bool, drug seen in training (else unseen / foreign)

    def __len__(self) -> int:
        return len(self.y_in)


def p_in_from(sims: np.ndarray, masks: np.ndarray, temperature: float, unknown_bias: float | None) -> np.ndarray:
    n = sims.shape[0]
    if unknown_bias is None:
        z = sims / temperature
        z = z - z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        return (p * masks).sum(1)
    z = np.concatenate([sims, np.full((n, 1), unknown_bias, np.float32)], 1) / temperature
    z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    return (p[:, :-1] * masks).sum(1)


def binary_nll(p: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict:
    """ECE of ``p`` as the probability of ``y == 1`` (binary reliability) plus the diagram bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if not m.any():
            bins.append({"lo": lo, "hi": hi, "n": 0, "conf": None, "acc": None})
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(conf - acc)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "conf": conf, "acc": acc})
    return {"ece": float(ece), "bins": bins}


def fit_temperature_and_bias(cal: CalibrationSet, init: DecisionParams | None = None) -> DecisionParams:
    from scipy.optimize import minimize

    init = init or DecisionParams()
    x0 = np.array([np.log(init.temperature), init.unknown_bias])

    def objective(x):
        T, b = float(np.exp(x[0])), float(x[1])
        return binary_nll(p_in_from(cal.sims, cal.masks, T, b), cal.y_in)

    best = None
    for start in (x0, np.array([np.log(0.05), 0.5]), np.array([np.log(0.1), 0.3]), np.array([np.log(0.02), 0.7])):
        r = minimize(objective, start, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
        if best is None or r.fun < best.fun:
            best = r
    T, b = float(np.exp(best.x[0])), float(best.x[1])
    return DecisionParams(temperature=T, unknown_bias=b, theta_in=init.theta_in, theta_out=init.theta_out,
                          margin=init.margin)


def build_calibration_set(matcher: Matcher, embs: np.ndarray, listed_sets: list[list[int]], y_in: np.ndarray,
                          seen: np.ndarray | None = None) -> CalibrationSet:
    sims = matcher.class_similarities(embs)
    masks = np.stack([matcher.listed_mask(l) for l in listed_sets]) if len(listed_sets) else np.zeros((0, matcher.n_classes), bool)
    return CalibrationSet(sims=sims, masks=masks, y_in=np.asarray(y_in, np.float32), seen=seen)


def calibration_report(cal: CalibrationSet, before: DecisionParams, after: DecisionParams) -> dict:
    p_before = p_in_from(cal.sims, cal.masks, before.temperature, None)
    p_after = p_in_from(cal.sims, cal.masks, after.temperature, after.unknown_bias)
    return {
        "n": len(cal),
        "before": {"temperature": before.temperature, "unknown_bias": None,
                   "nll": binary_nll(p_before, cal.y_in), "ece": expected_calibration_error(p_before, cal.y_in)["ece"]},
        "after": {"temperature": after.temperature, "unknown_bias": after.unknown_bias,
                  "nll": binary_nll(p_after, cal.y_in), "ece": expected_calibration_error(p_after, cal.y_in)["ece"]},
        "reliability_after": expected_calibration_error(p_after, cal.y_in)["bins"],
        "reliability_before": expected_calibration_error(p_before, cal.y_in)["bins"],
    }
