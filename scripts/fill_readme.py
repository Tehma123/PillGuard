"""Fill the README results block from the evaluation artifacts.

    python scripts/fill_readme.py

Reads artifacts/eval/test_{detector,oracle}/metrics.json, the detector mAP, model sizes,
calibration and parity reports, and rewrites everything between the RESULTS markers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def load(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(x, digits=1):
    return "–" if x is None else f"{100 * x:.{digits}f} %"


def main() -> None:
    det = load(ART / "eval" / "test_detector" / "metrics.json")
    orc = load(ART / "eval" / "test_oracle" / "metrics.json")
    dmap = load(ART / "detector" / "yolo11n" / "metrics_test.json")
    dexp = load(ART / "export" / "detector_export.json")
    eexp = load(ART / "export" / "embed_export.json")
    fit = load(ART / "decision" / "fit_report.json")
    par = load(ART / "parity" / "parity_report.json")
    if det is None:
        raise SystemExit("run `pillguard eval --mode detector` first")

    h, nr = det["headline"], det.get("no_reject_baseline", {})
    lines = ["| metric (test split, pill level) | PillGuard (with reject) | same model, no reject | SPEC target |",
             "|---|---|---|---|",
             f"| out-of-prescription recall | **{pct(h['out_recall'])}** | {pct(nr.get('out_recall'))} | ≥ 95 % |",
             f"| false alarm rate (listed pill flagged) | **{pct(h['false_alarm_rate'])}** | {pct(nr.get('false_alarm_rate'))} | ≤ 10 % |",
             f"| abstain rate | **{pct(h['abstain_rate'])}** | 0 % | ≤ 20 % |",
             f"| error rate on decided pills (risk) | **{pct(h['risk'])}** | {pct(nr.get('risk'))} | lower with reject |",
             f"| out recall, strict (misses and abstains count as missed) | {pct(h['out_recall_strict'])} | {pct(nr.get('out_recall_strict'))} | – |",
             f"| detection recall (pill found at IoU ≥ 0.5) | {pct(h['detection_recall'])} | same | – |",
             f"| ECE of p(in prescription) | {det.get('ece', float('nan')):.3f} | – | lower than uncalibrated |"]
    if fit:
        c = fit["calibration"]
        lines.append(f"| ECE on validation, before → after calibration | {c['before']['ece']:.3f} → {c['after']['ece']:.3f} | – | – |")
    if dmap:
        lines.append(f"| detector mAP@0.5 / mAP@0.5:0.95 (test) | {dmap['map50']:.3f} / {dmap['map50_95']:.3f} | – | set after baseline |")
    if dexp and eexp:
        total = dexp["int8_mb"] + eexp["int8_mb"]
        lines.append(f"| model download (INT8 ONNX) | {total:.1f} MB (detector {dexp['int8_mb']:.1f} + embedding {eexp['int8_mb']:.1f}) | – | ≤ 30 MB |")
    lat = det.get("latency_ms", {})
    if lat.get("median"):
        lines.append(f"| Python ONNX latency, 1 thread (detect + embed) | median {lat['median']:.0f} ms, p90 {lat['p90']:.0f} ms | – | ≤ 1 s in browser |")
    if par:
        lines.append(f"| Python vs browser (Node/WASM) verdict agreement | {pct(par['verdict_agreement'], 0)} on {par['cases']} cases | – | 100 % |")
    lines.append("")
    seen, unseen = det["seen"], det["unseen"]
    lines += ["| subset | pills | out recall | false alarm | abstain |", "|---|---|---|---|---|",
              f"| seen drugs | {seen['n']} | {pct(seen['out_recall'])} | {pct(seen['false_alarm_rate'])} | {pct(seen['abstain_rate'])} |",
              f"| unseen drugs (12 held out) | {unseen['n']} | {pct(unseen['out_recall'])} | {pct(unseen['false_alarm_rate'])} | {pct(unseen['abstain_rate'])} |"]
    for k, label in (("removed", "deleted from the prescription"), ("foreign", "labelled foreign by VAIPE")):
        m = det["by_reason"].get(k)
        if m:
            lines.append(f"| out pills: {label} | {m['n']} | {pct(m['out_recall'])} | – | {pct(m['abstain_rate'])} |")
    if orc:
        o = orc["headline"]
        lines += ["", f"With ground-truth boxes (recognition + decision only): out recall {pct(o['out_recall'])}, "
                      f"false alarms {pct(o['false_alarm_rate'])}, abstain {pct(o['abstain_rate'])}, ECE {orc.get('ece', float('nan')):.3f}."]
    lines += ["", "Full tables, confusion pairs with example crops, risk-coverage and reliability plots: "
                  "`artifacts/eval/test_detector/report.md` (regenerate with `pillguard eval`)."]
    block = "\n".join(lines)
    readme = ROOT / "README.md"
    s = readme.read_text(encoding="utf-8")
    s = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->",
               "<!-- RESULTS:START -->\n" + block + "\n<!-- RESULTS:END -->", s, flags=re.S)
    readme.write_text(s, encoding="utf-8")
    print(block)


if __name__ == "__main__":
    main()
