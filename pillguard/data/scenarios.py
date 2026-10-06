"""Out-of-prescription test scenarios (SPEC §2, "Giả lập thuốc ngoài đơn").

A scenario pairs one photo with a *modified* copy of its prescription:

* ``clean``    – the prescription as written (minus unseen drugs, which are never listed).
                 Measures false alarms on correct pill sets.
* ``remove-k`` – ``k`` of the drugs that are both on the prescription and visible in the photo
                 are deleted from the list. Every pill of a deleted drug must be flagged.

Per pill the ground truth is ``in`` or ``out`` with a ``reason``:

* ``listed``   – drug is on the (modified) list                       -> in
* ``removed``  – drug was deleted from the list by the scenario       -> out
* ``foreign``  – labelled 107 by VAIPE (pill from another prescription) -> out
* ``unseen``   – drug the model was never trained on (novelty test)    -> out
                 Held out on purpose, or absent from every train photo: pass
                 ``Split.all_unseen_classes`` so both kinds stay off the list.
* ``unlisted`` – annotation says a seen drug that the prescription does not mention -> out
                 (rare label noise; reported separately and excluded from headline numbers)

Scenario generation is deterministic per (seed, photo) so it does not depend on ordering.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from pillguard.config import OUT_OF_PRESCRIPTION_LABEL, SEED
from pillguard.data.vaipe import PillImage


@dataclass
class ScenarioPill:
    box: list[float]
    label: int              # VAIPE label (class id, or 107)
    gt: str                 # "in" | "out"
    reason: str             # listed | removed | foreign | unseen | unlisted


@dataclass
class Scenario:
    id: str
    kind: str               # clean | remove-1 | remove-2
    file: str
    prescription: str | None
    listed: list[int]       # drug ids handed to the matcher
    removed: list[int]
    pills: list[ScenarioPill] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Scenario:
        pills = [ScenarioPill(**p) for p in d.pop("pills")]
        return cls(pills=pills, **d)

    @property
    def n_out(self) -> int:
        return sum(p.gt == "out" for p in self.pills)


def _rng_for(seed: int, key: str) -> np.random.Generator:
    h = int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(h)


def _label_pills(im: PillImage, listed: set[int], removed: set[int], unseen: set[int]) -> list[ScenarioPill]:
    pills = []
    for box, lab in zip(im.boxes, im.labels):
        if lab == OUT_OF_PRESCRIPTION_LABEL:
            gt, reason = "out", "foreign"
        elif lab in unseen:
            gt, reason = "out", "unseen"
        elif lab in removed:
            gt, reason = "out", "removed"
        elif lab in listed:
            gt, reason = "in", "listed"
        else:
            gt, reason = "out", "unlisted"
        pills.append(ScenarioPill(box=list(box), label=int(lab), gt=gt, reason=reason))
    return pills


def make_scenarios(images: list[PillImage], pres_drugs: dict[str, list[int]], unseen: set[int] | list[int],
                   seed: int = SEED, ks: tuple[int, ...] = (1, 2), include_clean: bool = True,
                   fallback_to_labels: bool = True) -> list[Scenario]:
    """Build scenarios for every photo that has a usable prescription.

    ``pres_drugs`` maps prescription file -> drug ids written on it. If a photo has no
    prescription annotation and ``fallback_to_labels`` is set, the set of seen labels in the
    photo is used as the prescription (still a valid "remove-k" test).
    """
    unseen = set(unseen)
    out: list[Scenario] = []
    for im in images:
        drugs = pres_drugs.get(im.prescription or "", None)
        if drugs is None:
            if not fallback_to_labels:
                continue
            drugs = sorted({l for l in im.labels if l != OUT_OF_PRESCRIPTION_LABEL})
        base = sorted(set(drugs) - unseen)
        present = sorted({l for l in im.labels if l in base})
        rng = _rng_for(seed, im.file)
        if include_clean:
            out.append(Scenario(id=f"{im.file}#clean", kind="clean", file=im.file, prescription=im.prescription,
                                listed=base, removed=[], pills=_label_pills(im, set(base), set(), unseen)))
        for k in ks:
            if len(present) < k or (len(present) == k and k > 1 and len(base) == k):
                continue
            removed = sorted(int(c) for c in rng.choice(present, size=k, replace=False))
            listed = [c for c in base if c not in removed]
            out.append(Scenario(id=f"{im.file}#remove-{k}", kind=f"remove-{k}", file=im.file,
                                prescription=im.prescription, listed=listed, removed=removed,
                                pills=_label_pills(im, set(listed), set(removed), unseen)))
    return out


def save_scenarios(path: Path, scenarios: list[Scenario]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(s.to_dict(), ensure_ascii=False) + "\n" for s in scenarios)


def load_scenarios(path: Path) -> list[Scenario]:
    with open(path, encoding="utf-8") as f:
        return [Scenario.from_dict(json.loads(line)) for line in f if line.strip()]


def summarize(scenarios: list[Scenario]) -> dict:
    from collections import Counter

    kinds = Counter(s.kind for s in scenarios)
    reasons = Counter(p.reason for s in scenarios for p in s.pills)
    gts = Counter(p.gt for s in scenarios for p in s.pills)
    return {"n_scenarios": len(scenarios), "kinds": dict(kinds), "pills": dict(gts), "reasons": dict(reasons)}
