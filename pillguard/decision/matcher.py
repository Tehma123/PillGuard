"""Match pill embeddings against the drugs on a prescription and decide in / out / uncertain.

Scores
------
For a pill embedding ``e`` and every seen drug ``c`` with prototype vectors ``P_c``::

    s_c   = max_{p in P_c} cos(e, p)                          (class similarity)
    z     = [s_1 .. s_C, b] / T                               (b = virtual "unknown drug" score)
    p     = softmax(z)
    p_in  = sum_{c on the prescription} p_c                   (probability the pill is a listed drug)

``T`` (temperature) and ``b`` (unknown bias) are fitted on validation data by
:mod:`pillguard.decision.calibration`. The unknown class is what lets the model say
"none of the drugs I know" for a pill of a drug it was never trained on.

Verdict
-------
* ``p_in >= theta_in``  and ``s_in - s_out >= margin``  -> ``in``   (best listed drug is reported)
* ``p_in <= theta_out``                                 -> ``out``  (alert: not on the prescription)
* otherwise                                             -> ``uncertain`` (ask to retake / ask a pharmacist)

The same arithmetic is implemented in ``web/pipeline.js``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from pillguard.config import DecisionParams


@dataclass
class Decision:
    verdict: str            # in | out | uncertain
    p_in: float
    p_unknown: float
    s_in: float             # best similarity to a listed drug
    s_out: float            # best similarity to a seen drug that is not listed
    best_listed: int | None
    best_any: int | None
    margin: float

    def to_dict(self) -> dict:
        return asdict(self)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Matcher:
    def __init__(self, prototypes: dict[int, np.ndarray], params: DecisionParams | None = None) -> None:
        self.class_ids: list[int] = sorted(int(c) for c in prototypes)
        self.index = {c: i for i, c in enumerate(self.class_ids)}
        self.params = params or DecisionParams()
        mats, owner = [], []
        for i, c in enumerate(self.class_ids):
            v = np.asarray(prototypes[c], dtype=np.float32).reshape(-1, prototypes[c].shape[-1])
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
            mats.append(v)
            owner.extend([i] * len(v))
        self.vectors = np.concatenate(mats) if mats else np.zeros((0, 1), np.float32)
        self.owner = np.asarray(owner, dtype=np.int64)
        self.starts = np.flatnonzero(np.r_[True, np.diff(self.owner) != 0]) if len(self.owner) else np.zeros(0, np.int64)

    # -- scores -------------------------------------------------------------------------
    @property
    def n_classes(self) -> int:
        return len(self.class_ids)

    def class_similarities(self, emb: np.ndarray) -> np.ndarray:
        """(n, D) unit embeddings -> (n, C) max cosine per class."""
        emb = np.asarray(emb, dtype=np.float32).reshape(-1, self.vectors.shape[1])
        sims = emb @ self.vectors.T
        if len(self.starts) == 0:
            return np.zeros((len(emb), 0), np.float32)
        return np.maximum.reduceat(sims, self.starts, axis=1)

    def listed_mask(self, listed) -> np.ndarray:
        m = np.zeros(self.n_classes, dtype=bool)
        for c in listed:
            if int(c) in self.index:
                m[self.index[int(c)]] = True
        return m

    def probabilities(self, sims: np.ndarray, mask: np.ndarray, params: DecisionParams | None = None):
        """Return (p_in, p_unknown, p_classes) for class similarities (n, C) and a listed mask (C,) or (n, C)."""
        params = params or self.params
        n = sims.shape[0]
        z = np.concatenate([sims, np.full((n, 1), params.unknown_bias, np.float32)], 1) / params.temperature
        p = _softmax(z)
        mask = np.broadcast_to(mask, (n, self.n_classes))
        p_in = (p[:, :-1] * mask).sum(1)
        return p_in, p[:, -1], p[:, :-1]

    # -- decisions ----------------------------------------------------------------------
    def decide(self, emb: np.ndarray, listed, params: DecisionParams | None = None) -> list[Decision]:
        params = params or self.params
        sims = self.class_similarities(emb)
        mask = self.listed_mask(listed)
        p_in, p_unk, _ = self.probabilities(sims, mask, params)
        return [self._verdict(sims[i], mask, float(p_in[i]), float(p_unk[i]), params) for i in range(len(sims))]

    def _verdict(self, s: np.ndarray, mask: np.ndarray, p_in: float, p_unk: float, params: DecisionParams) -> Decision:
        neg = np.full_like(s, -2.0)
        s_listed = np.where(mask, s, neg)
        s_other = np.where(~mask, s, neg)
        bi = int(s_listed.argmax()) if mask.any() else None
        s_in = float(s_listed[bi]) if bi is not None else -1.0
        oi = int(s_other.argmax()) if (~mask).any() else None
        s_out = float(s_other[oi]) if oi is not None else -1.0
        margin = s_in - s_out
        if p_in >= params.theta_in and margin >= params.margin:
            verdict = "in"
        elif p_in <= params.theta_out:
            verdict = "out"
        else:
            verdict = "uncertain"
        best_any = int(s.argmax()) if len(s) else None
        return Decision(verdict=verdict, p_in=p_in, p_unknown=p_unk, s_in=s_in, s_out=s_out,
                        best_listed=self.class_ids[bi] if bi is not None else None,
                        best_any=self.class_ids[best_any] if best_any is not None else None, margin=margin)


def verdicts_from_p(p_in: np.ndarray, margin: np.ndarray | None, params: DecisionParams) -> np.ndarray:
    """Vectorised verdicts for threshold search: 0 = in, 1 = out, 2 = uncertain."""
    v = np.full(len(p_in), 2, dtype=np.int8)
    ok_margin = np.ones(len(p_in), bool) if margin is None else margin >= params.margin
    v[(p_in >= params.theta_in) & ok_margin] = 0
    v[p_in <= params.theta_out] = 1
    return v
