"""Python-vs-browser parity check (SPEC §4: same photo, same verdicts).

1. ``--export``  writes ``artifacts/parity/``: lossless PNG copies of N test photos and
   ``cases.json`` (listed drugs per case, some cases with oracle boxes), then runs the
   Python ONNX pipeline and stores ``results_py.json``.
2. ``cd web && npm run parity`` runs the identical JavaScript pipeline under Node with
   onnxruntime-web (WASM) and writes ``results_js.json``.
3. ``--compare`` reports box-count agreement, verdict agreement and the largest
   probability / box deviations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pillguard.config import ARTIFACTS_DIR, VAIPE_DIR, WEB_DIR
from pillguard.data.scenarios import make_scenarios
from pillguard.data.splits import Split, default_split_path
from pillguard.data.vaipe import load_pill_images, prescription_drug_table
from pillguard.imaging import load_rgb
from pillguard.pipeline import PillGuardPipeline


def export_cases(root: Path, split_path: Path, out_dir: Path, n: int = 30, n_oracle: int = 10, seed: int = 0) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = Split.load(split_path)
    images = load_pill_images(root)
    by = {im.file: im for im in images}
    test = [by[f] for f in split.test if f in by]
    rng = np.random.default_rng(seed)
    pick = [test[i] for i in sorted(rng.choice(len(test), size=min(n, len(test)), replace=False))]
    scen = {s.file: s for s in make_scenarios(pick, prescription_drug_table(root), split.unseen_classes, seed=split.seed)
            if s.kind == "remove-1"}
    cases = []
    for k, im in enumerate(pick):
        png = f"{Path(im.file).stem}.png"
        Image.fromarray(load_rgb(im.path(root))).save(out_dir / png, "PNG")
        s = scen.get(im.file)
        listed = s.listed if s else sorted({l for l in im.labels if l != 107})
        cases.append({"id": f"{im.file}#det", "png": png, "listed": listed, "boxes": None})
        if k < n_oracle:
            cases.append({"id": f"{im.file}#oracle", "png": png, "listed": listed, "boxes": [list(b) for b in im.boxes]})
    (out_dir / "cases.json").write_text(json.dumps(cases, indent=1), encoding="utf-8")
    print(f"parity set: {len(cases)} cases -> {out_dir}")
    return cases


def run_python(out_dir: Path, web_dir: Path) -> list[dict]:
    out_dir = Path(out_dir)
    pipe = PillGuardPipeline.from_dir(web_dir)
    cases = json.loads((out_dir / "cases.json").read_text())
    results = []
    for c in cases:
        img = np.asarray(Image.open(out_dir / c["png"]).convert("RGB"))
        r = pipe.run_with_boxes(img, c["boxes"], c["listed"]) if c["boxes"] else pipe.run(img, c["listed"])
        d = r.to_dict()
        results.append({"id": c["id"], "boxes": d["boxes"], "scores": d["scores"], "decisions": d["decisions"],
                        "timings_ms": d["timings_ms"]})
    (out_dir / "results_py.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"python results: {len(results)} cases")
    return results


def compare(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    py = {r["id"]: r for r in json.loads((out_dir / "results_py.json").read_text())}
    js = {r["id"]: r for r in json.loads((out_dir / "results_js.json").read_text())}
    ids = sorted(set(py) & set(js))
    same_count = same_verdicts = 0
    max_box, max_p, max_s, n_pills = 0.0, 0.0, 0.0, 0
    mismatches = []
    for i in ids:
        a, b = py[i], js[i]
        counts_equal = len(a["boxes"]) == len(b["boxes"])
        same_count += counts_equal
        if counts_equal:
            va = [d["verdict"] for d in a["decisions"]]
            vb = [d["verdict"] for d in b["decisions"]]
            ok = va == vb
            same_verdicts += ok
            if not ok:
                mismatches.append({"id": i, "py": va, "js": vb})
            for sa, sb in zip(a["scores"], b["scores"]):
                max_s = max(max_s, abs(sa - sb))
            for da, db, ba, bb in zip(a["decisions"], b["decisions"], a["boxes"], b["boxes"]):
                max_p = max(max_p, abs(da["p_in"] - db["p_in"]))
                max_box = max(max_box, float(np.abs(np.array(ba) - np.array(bb)).max()))
                n_pills += 1
        else:
            mismatches.append({"id": i, "py_boxes": len(a["boxes"]), "js_boxes": len(b["boxes"])})
    rep = {"cases": len(ids), "same_box_count": same_count, "same_verdicts": same_verdicts,
           "verdict_agreement": same_verdicts / max(1, len(ids)), "pills_compared": n_pills,
           "max_abs_p_in_diff": max_p, "max_abs_box_diff_px": max_box, "max_abs_score_diff": max_s, "mismatches": mismatches}
    (out_dir / "parity_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "mismatches"}, indent=1))
    if mismatches:
        print("mismatches:", json.dumps(mismatches[:5], indent=1))
    return rep


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=VAIPE_DIR)
    ap.add_argument("--split", type=Path, default=default_split_path())
    ap.add_argument("--web", type=Path, default=WEB_DIR)
    ap.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "parity")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--export", action="store_true", help="write the PNG cases and run the Python side")
    ap.add_argument("--compare", action="store_true", help="compare results_py.json with results_js.json")
    a = ap.parse_args(argv)
    if a.export:
        export_cases(a.root, a.split, a.out, n=a.n)
        run_python(a.out, a.web)
    if a.compare:
        compare(a.out)
    if not (a.export or a.compare):
        ap.print_help()


if __name__ == "__main__":
    main()
