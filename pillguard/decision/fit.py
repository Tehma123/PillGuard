"""Build prototypes and fit the decision parameters with the *deployed* (ONNX) embedding model.

Steps
-----
1. Embed all train crops of seen drugs with the ONNX model -> prototypes (mean + k-means).
2. Build validation scenarios (clean / remove-1 / remove-2) and embed every val pill.
3. Fit temperature + unknown bias by NLL; choose ``theta_in`` / ``theta_out`` under the
   false-alarm and abstention caps; compute the val risk-coverage curve and ECE before/after.

Outputs (``artifacts/decision/``): ``prototypes.npz``, ``params.json``, ``fit_report.json``.

Usage::

    python -m pillguard.decision.fit --embed-onnx artifacts/export/embed.int8.onnx \
        --ckpt artifacts/embed/mnv3s/best.pt --root data/vaipe --split splits/vaipe_v1.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from pillguard.config import (
    ARTIFACTS_DIR,
    PROTOTYPES_PER_CLASS,
    DecisionParams,
)
from pillguard.data.scenarios import make_scenarios, summarize
from pillguard.data.splits import Split
from pillguard.data.vaipe import load_pill_images, prescription_drug_table
from pillguard.decision.calibration import (
    CalibrationSet,
    calibration_report,
    fit_platt,
    fit_temperature_and_bias,
    p_in_from,
)
from pillguard.decision.matcher import Matcher
from pillguard.decision.thresholds import choose_thresholds, risk_coverage_curve, selective_metrics
from pillguard.embed.crops import eval_crop_array, filter_rows, load_crop_index
from pillguard.embed.export import embed_with_onnx
from pillguard.embed.prototypes import build_prototypes, save_prototypes_npz
from pillguard.onnx_utils import ort_session


def fit_decision(root: Path, split_path: Path, embed_onnx: Path, ckpt: Path, out_dir: Path,
                 max_fpr: float = 0.10, max_abstain: float = 0.20, k: int = PROTOTYPES_PER_CLASS,
                 arcface_scale: float = 30.0, seed: int = 0, batch: int = 64) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = Split.load(split_path)
    unseen = set(split.unseen_classes)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    seen = set(int(c) for c in ck["classes"])
    session = ort_session(embed_onnx)
    rows = load_crop_index(root)
    by_id = {r.id: r for r in rows}

    # 1. prototypes from train crops
    train_rows = filter_rows(rows, "train", seen)
    print(f"embedding {len(train_rows)} train crops for prototypes", flush=True)
    tr_e = embed_with_onnx(session, eval_crop_array(train_rows, root), batch)
    tr_lab = np.array([r.label for r in train_rows])
    protos = build_prototypes(tr_e, tr_lab, k=k, seed=seed)
    save_prototypes_npz(out_dir / "prototypes.npz", protos)
    matcher = Matcher(protos)

    # 2. validation scenarios -> calibration set
    images = load_pill_images(root)
    by_file = {im.file: im for im in images}
    val_ims = [by_file[f] for f in split.val if f in by_file]
    scen = make_scenarios(val_ims, prescription_drug_table(root), unseen, seed=split.seed)
    print("val scenarios:", summarize(scen), flush=True)
    val_ids = sorted({f"{Path(s.file).stem}_{i}" for s in scen for i in range(len(s.pills))})
    val_rows = [by_id[i] for i in val_ids if i in by_id]
    print(f"embedding {len(val_rows)} val crops", flush=True)
    va_e = embed_with_onnx(session, eval_crop_array(val_rows, root), batch)
    emb_by_id = {r.id: va_e[i] for i, r in enumerate(val_rows)}
    embs, listed, y, seen_flag, reasons = [], [], [], [], []
    for s in scen:
        stem = Path(s.file).stem
        for i, p in enumerate(s.pills):
            cid = f"{stem}_{i}"
            if cid not in emb_by_id or p.reason == "unlisted":
                continue
            embs.append(emb_by_id[cid]); listed.append(s.listed); y.append(1.0 if p.gt == "in" else 0.0)
            seen_flag.append(p.label in seen); reasons.append(p.reason)
    embs = np.stack(embs)
    sims = matcher.class_similarities(embs)
    masks = np.stack([matcher.listed_mask(l) for l in listed])
    cal = CalibrationSet(sims=sims, masks=masks, y_in=np.array(y, np.float32), seen=np.array(seen_flag))

    # 3. calibration + thresholds
    before = DecisionParams(temperature=1.0 / arcface_scale, unknown_bias=0.0)
    fitted = fit_temperature_and_bias(cal)
    raw = p_in_from(cal.sims, cal.masks, fitted.temperature, fitted.unknown_bias)
    platt_a, platt_b = fit_platt(raw, cal.y_in)
    fitted = replace(fitted, platt_a=platt_a, platt_b=platt_b)
    cal_rep = calibration_report(cal, before, fitted)
    p_in = p_in_from(cal.sims, cal.masks, fitted.temperature, fitted.unknown_bias,
                     fitted.platt_a, fitted.platt_b)
    params, th_info = choose_thresholds(p_in, cal.y_in, fitted, max_fpr=max_fpr, max_abstain=max_abstain)
    no_reject = selective_metrics(p_in, cal.y_in, DecisionParams(fitted.temperature, fitted.unknown_bias, 0.5, 0.5))
    report = {
        "n_val_pills": len(y), "n_classes": matcher.n_classes, "prototypes_per_class": k,
        "unseen_classes": sorted(unseen), "calibration": cal_rep, "thresholds": th_info,
        "val_no_reject": no_reject, "val_risk_coverage": risk_coverage_curve(p_in, cal.y_in),
        "val_by_reason": {r: selective_metrics(p_in[np.array(reasons) == r], cal.y_in[np.array(reasons) == r], params)
                          for r in sorted(set(reasons))},
        "params": params.to_dict(),
    }
    (out_dir / "params.json").write_text(json.dumps(params.to_dict(), indent=1), encoding="utf-8")
    (out_dir / "fit_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("val_risk_coverage", "calibration")}, indent=1))
    print("ECE before/after:", cal_rep["before"]["ece"], cal_rep["after"]["ece"])
    return report


def main(argv: list[str] | None = None) -> None:
    from pillguard.config import VAIPE_DIR
    from pillguard.data.splits import default_split_path

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embed-onnx", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=VAIPE_DIR)
    ap.add_argument("--split", type=Path, default=default_split_path())
    ap.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "decision")
    ap.add_argument("--max-fpr", type=float, default=0.10)
    ap.add_argument("--max-abstain", type=float, default=0.20)
    a = ap.parse_args(argv)
    fit_decision(a.root, a.split, a.embed_onnx, a.ckpt, a.out, max_fpr=a.max_fpr, max_abstain=a.max_abstain)


if __name__ == "__main__":
    main()
