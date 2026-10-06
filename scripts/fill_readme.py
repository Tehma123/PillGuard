"""Fill the README results block and docs/benchmark.md from the evaluation artifacts.

    python scripts/fill_readme.py

Reads artifacts/eval/test_{detector,oracle}/metrics.json, the detector mAP, model sizes,
calibration and parity reports, rewrites everything between the README's RESULTS markers and
writes the long form to docs/benchmark.md.

``artifacts/`` is not committed because it holds VAIPE pixels, so a clone cannot read the
generated report. Everything written to docs/ is numbers and drug names, and the two figures
copied into docs/figures are aggregate curves; the confusion-pair crops stay behind.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
DOCS = ROOT / "docs"
BENCHMARK_FIGURES = ("risk_coverage.png", "reliability.png")


def load(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(x, digits=1):
    return "–" if x is None else f"{100 * x:.{digits}f} %"


def subset_table(source: dict, keys: list[tuple[str, str]], first: str = "subset") -> list[str]:
    """Rates that have no denominator in a subset (out recall of listed pills, say) print as '–'."""
    rows = [f"| {first} | pill checks | out recall | false alarm | abstain | out pill accepted |", "|---|---|---|---|---|---|"]
    for key, label in keys:
        d = source.get(key)
        if d and d.get("n"):
            has_out, has_in = d["n_out"] > 0, d["n_in"] > 0
            rows.append(f"| {label} | {d['n']} | {pct(d['out_recall'] if has_out else None)} "
                        f"| {pct(d['false_alarm_rate'] if has_in else None)} | {pct(d['abstain_rate'])} "
                        f"| {pct(d['out_accepted_rate'] if has_out else None)} |")
    return rows


def unseen_label(det: dict) -> str:
    g = det.get("unseen_groups", {})
    held, untrained = g.get("held_out", {}), g.get("untrained", {})
    if not held:
        return "drugs never trained on"
    return (f"drugs never trained on ({len(held['present'])} of the {len(held['classes'])} held out, "
            f"{len(untrained.get('present', []))} with no train photo)")


def photo_table(img: dict) -> list[str]:
    rows = ["| photo holds | scenarios | OK | not sure, no OUT | at least one OUT |", "|---|---|---|---|---|"]
    for key, label in (("clean", "only prescribed pills"), ("wrong", "at least one pill not on the list")):
        n, t = img[f"n_{key}"], img[key]
        rows.append(f"| {label} | {n} | " + " | ".join(f"{t[s]} ({pct(t[s] / max(1, n))})" for s in ("ok", "uncertain", "out")) + " |")
    return rows


def benchmark_doc(det: dict, orc: dict | None, dmap: dict | None, dexp: dict | None,
                  eexp: dict | None, fit: dict | None, par: dict | None) -> str:
    h, nr = det["headline"], det.get("no_reject_baseline", {})
    o = (orc or {}).get("headline", {})
    img = det.get("image_level", {})
    sv = det.get("spurious_verdicts", {})
    nan = float("nan")
    br = det.get("by_reason", {})
    nm = det.get("naming", {})
    L = ["# PillGuard benchmark",
         "",
         f"Test split: {det.get('n_pills_unique', '?')} pills in {det.get('n_photos', '?')} photos, each photo checked "
         f"against its prescription as written and with one or two drugs deleted, {img.get('n_scenarios', '?')} "
         f"scenarios in all. Pill-level rates count {h['n']} pill checks ({h['n_in']} on the list, {h['n_out']} "
         "not), one per pill per scenario. Regenerate with `pillguard eval --mode detector`, "
         "`pillguard eval --mode oracle` and `python scripts/fill_readme.py`.",
         "",
         "## End to end",
         "",
         "*Detector boxes* is the whole system. *Ground-truth boxes* isolates recognition and the decision",
         "layer from detection. *No reject* is the same model forced to answer for every pill.",
         "",
         "| metric | detector boxes | ground-truth boxes | no reject |",
         "|---|---|---|---|",
         f"| out-of-prescription recall (decided) | **{pct(h['out_recall'])}** | {pct(o.get('out_recall'))} "
         f"| {pct(nr.get('out_recall'))} |",
         f"| out recall, strict (abstain counts as a miss) | {pct(h['out_recall_strict'])} "
         f"| {pct(o.get('out_recall_strict'))} | {pct(nr.get('out_recall_strict'))} |",
         f"| false alarm rate (decided) | **{pct(h['false_alarm_rate'])}** | {pct(o.get('false_alarm_rate'))} "
         f"| {pct(nr.get('false_alarm_rate'))} |",
         f"| abstain rate | **{pct(h['abstain_rate'])}** | {pct(o.get('abstain_rate'))} | 0 % |",
         f"| risk (error rate on decided pills) | {pct(h['risk'])} | {pct(o.get('risk'))} | {pct(nr.get('risk'))} |",
         f"| detection recall (IoU >= 0.5) | {pct(h['detection_recall'])} | {pct(o.get('detection_recall'))} | – |",
         f"| ECE of p(in prescription) | {det.get('ece', nan):.4f} | {(orc or {}).get('ece', nan):.4f} | – |",
         f"| risk-coverage AURC | {det.get('risk_coverage', {}).get('aurc', nan):.4f} | – | – |"]
    L += ["",
          f"Spurious detections, boxes matching no labelled pill: {det.get('n_spurious_detections')} "
          f"({sv.get('out', 0)} judged out, {sv.get('uncertain', 0)} uncertain, {sv.get('in', 0)} in)."]
    if img:
        c = img["clean"]
        L += ["", "## Per photo", "",
              "What the user sees for one photo checked against one prescription. The photo reads **OUT** if any",
              "pill is flagged out, **not sure** if some pill is uncertain and none is out, and **OK** otherwise.",
              "Spurious boxes count, because the page shows them; missed pills cannot show.", ""]
        L += photo_table(img)
        L += ["",
              f"An alert means at least one OUT. On a photo of only prescribed pills that is a false alert, and it "
              f"happens {pct(img['false_alert_rate'])} of the time. The price of the reject option shows here "
              f"instead: {pct(img['check_rate_clean'])} of such photos ask the user to check at least one pill, "
              f"because a photo holds {h['n'] / max(1, img['n_scenarios']):.1f} pills on average and each can "
              f"abstain. A photo with a wrong pill goes through with nothing flagged {pct(img['silent_miss_rate'])} "
              "of the time.",
              "",
              f"An earlier version of this page counted *not sure* as an alert too and called the sum (now "
              f"{pct((c['out'] + c['uncertain']) / max(1, img['n_clean']))}) the false alert rate."]
    f, r, u = br.get("foreign", {}), br.get("removed", {}), br.get("unseen", {})
    if f and r:
        L += ["", "## Known weaknesses", "",
              f"* **A pill the model has never seen is often left *not sure*.** Foreign pills (VAIPE "
              f"label 107: a pill from another prescription, drug unknown) are left uncertain "
              f"{pct(f['abstain_rate'])} of the time and drugs never trained on {pct(u.get('abstain_rate'))}, "
              f"against {pct(r['abstain_rate'])} for pills deleted from the prescription. That is the safe "
              "direction, since the user is told to check, but it is not a confident alert.",
              f"* **Some wrong pills are accepted.** {pct(f['out_accepted_rate'])} of detected foreign pills and "
              f"{pct(r['out_accepted_rate'])} of deleted ones are judged to be on the prescription. This is the "
              "costly error: nothing tells the user to look.",
              f"* **Correct photos often get a *not sure*:** {pct(img.get('check_rate_clean'))} of them, see above.",
              "* **Naming a drug is weaker than checking a prescription,** see below."]
    if nm.get("n_pills"):
        mv = nm["removed_misnamed_verdicts"]
        L += ["", "## Naming a drug versus checking a prescription", "",
              f"The nearest prototype is the pill's own drug for {pct(nm['accuracy'])} of {nm['n_pills']} detected "
              f"test pills of trained drugs. On validation crops it is {(eexp or {}).get('int8_val_proto_acc', nan):.3f} "
              f"for the INT8 model and {(eexp or {}).get('fp32_val_proto_acc', nan):.3f} in fp32 (INT8 table below). "
              "Out-of-prescription recall is higher because the check never needs the name: it only asks whether "
              "the pill looks like one of the listed drugs.",
              "",
              f"Of {nm['removed']} checks of a deleted pill, {nm['removed_misnamed']} gave it the wrong name. "
              f"{mv['out']} of those were still flagged OUT, {mv['uncertain']} were *not sure* and {mv['in']} were "
              "accepted. A wrong name lets a pill through only when it lands on a drug that is on the list."]
    L += ["", "## Seen vs unseen drugs (detector boxes)", ""]
    L += subset_table(det, [("seen", "drugs the model was trained on"), ("unseen", unseen_label(det))])
    g = det.get("unseen_groups", {})
    held, untrained = g.get("held_out", {}), g.get("untrained", {})
    L += ["",
          "Unseen drugs are never on a prescription: the scenarios take them off the list, as the demo's drug",
          "picker only offers drugs that have prototypes. Every unseen pill is therefore out and the false alarm",
          "column does not apply."]
    if held:
        L += ["",
              f"They are the {len(held['classes'])} drugs held out of training on purpose, of which "
              f"{len(held['present'])} occur in test photos ({held['pills']} checks), and "
              f"{len(untrained.get('classes', []))} rare drugs that the split by prescription left out of every "
              f"train photo, of which ids {', '.join(map(str, untrained.get('present', [])))} occur in test "
              f"({untrained.get('pills', 0)} checks). Foreign pills are in neither row, since label 107 says a pill "
              "belongs to another prescription, not which drug it is."]
    L += ["", "## By reason", ""]
    L += subset_table(br, [(k, k) for k in sorted(br)], first="reason")
    L += ["",
          "`listed` pills are on the prescription, so only their false alarm rate applies. *Out pill accepted* is",
          "the share of detected out pills judged to be on the prescription.",
          "", "## By scenario kind", ""]
    L += subset_table(det.get("by_kind", {}), [(k, k) for k in sorted(det.get("by_kind", {}))], first="kind")
    if fit:
        c, prm = fit["calibration"], fit.get("params", {})
        L += ["", "## Calibration", "",
              "| parameter | value |", "|---|---|",
              f"| softmax temperature | {prm.get('temperature', nan):.4f} |",
              f"| unknown-class bias | {prm.get('unknown_bias', nan):.4f} |",
              f"| Platt a / b on p_in | {prm.get('platt_a', 1.0):.4f} / {prm.get('platt_b', 0.0):.4f} |",
              f"| theta_in / theta_out | {prm.get('theta_in', nan):.3f} / {prm.get('theta_out', nan):.3f} |",
              f"| validation NLL, before to after | {c['before']['nll']:.4f} -> {c['after']['nll']:.4f} |",
              f"| validation ECE, before to after | {c['before']['ece']:.4f} -> {c['after']['ece']:.4f} |",
              "",
              "Before is the network's own training softmax: T = 1/scale with no unknown class. A temperature",
              "alone leaves p_in underconfident across the middle of its range, because it can sharpen the",
              "distribution but not shift the reliability curve, so the fit adds an affine map in logit space.",
              "A positive Platt b is that correction."]
    if dexp and eexp:
        a = dexp.get("agreement", {})
        L += ["", "## INT8 quantisation", "",
              "| model | fp32 | INT8 | agreement with fp32 |", "|---|---|---|---|",
              f"| detector | {dexp['fp32_mb']:.1f} MB | {dexp['int8_mb']:.1f} MB | {a.get('int8_boxes')} / "
              f"{a.get('fp32_boxes')} boxes, mean IoU {a.get('mean_best_iou', nan):.3f} |",
              f"| embedding | {eexp['fp32_mb']:.1f} MB | {eexp['int8_mb']:.1f} MB | nearest-prototype "
              f"{eexp.get('fp32_val_proto_acc', nan):.3f} -> {eexp.get('int8_val_proto_acc', nan):.3f}, cosine "
              f"{eexp.get('int8_vs_fp32_cosine_mean', nan):.3f} |",
              "",
              "Both are mixed precision. One uint8 scale per tensor destroys the detector's output, which",
              "concatenates box pixels with sigmoid scores, and MobileNetV3's early layers, whose activations",
              "span a far wider range than the rest of the network, so those nodes stay in fp32. The export",
              "raises rather than shipping a model that drifts past these margins."]
    if dmap:
        L += ["", f"Detector mAP@0.5 {dmap['map50']:.3f}, mAP@0.5:0.95 {dmap['map50_95']:.3f} on the test split."]
    lat = det.get("latency_ms", {})
    if lat.get("median"):
        L += ["", f"Python ONNX latency per image, 1 thread, detect + embed: median {lat['median']:.0f} ms, "
                  f"p90 {lat['p90']:.0f} ms."]
    if par:
        L += ["", "## Python versus browser parity", "",
              f"The same {par['cases']} images through the Python pipeline and through `web/pipeline.js` on "
              f"Node/WASM: verdicts agree on {pct(par['verdict_agreement'], 0)} of cases "
              f"({par['pills_compared']} pills), boxes within {par['max_abs_box_diff_px']:.3f} px, detector "
              f"scores within {par['max_abs_score_diff']:.1e}.",
              "",
              f"The largest p_in gap is {par['max_abs_p_in_diff']:.3f}. That is not the runtimes: given an "
              "identical input tensor the two produce identical embeddings to float32 precision. It is crop "
              "resampling, amplified by the sharp softmax. Verdicts are unaffected because the pills "
              "concerned sit far from the thresholds."]
    L += ["", "## Figures", "",
          "![risk-coverage](figures/benchmark_risk_coverage.png)", "",
          "![reliability](figures/benchmark_reliability.png)", "",
          "Confusion pairs with example crops are deliberately absent: they are VAIPE photographs.",
          "`pillguard eval` writes them to `artifacts/eval/test_detector/confusion_examples.png`."]
    return "\n".join(L) + "\n"


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
    img = det.get("image_level", {})
    lines = ["| metric (test split) | PillGuard (with reject) | same model, no reject | SPEC target |",
             "|---|---|---|---|",
             f"| out-of-prescription recall | **{pct(h['out_recall'])}** | {pct(nr.get('out_recall'))} | ≥ 95 % |",
             f"| false alarm rate (listed pill flagged) | **{pct(h['false_alarm_rate'])}** | {pct(nr.get('false_alarm_rate'))} | ≤ 10 % |",
             f"| abstain rate | **{pct(h['abstain_rate'])}** | 0 % | ≤ 20 % |",
             f"| error rate on decided pills (risk) | **{pct(h['risk'])}** | {pct(nr.get('risk'))} | lower with reject |",
             f"| out recall, strict (misses and abstains count as missed) | {pct(h['out_recall_strict'])} | {pct(nr.get('out_recall_strict'))} | – |",
             f"| detection recall (pill found at IoU ≥ 0.5) | {pct(h['detection_recall'])} | same | – |",
             f"| ECE of p(in prescription) | {det.get('ece', float('nan')):.3f} | – | lower than uncalibrated |"]
    if img:
        lines += [f"| **per photo:** only prescribed pills, yet a pill flagged OUT (false alert) | **{pct(img['false_alert_rate'])}** | – | not in SPEC |",
                  f"| per photo: only prescribed pills, a pill *not sure* and none OUT | {pct(img['check_rate_clean'])} | – | not in SPEC |",
                  f"| per photo: a wrong pill present, at least one pill flagged OUT | {pct(img['alert_recall'])} | – | not in SPEC |",
                  f"| per photo: a wrong pill present, nothing flagged at all | {pct(img['silent_miss_rate'])} | – | not in SPEC |"]
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
    lines += subset_table(det["by_reason"], [("listed", "on the prescription"), ("removed", "deleted from the prescription"),
                                             ("unseen", unseen_label(det)),
                                             ("foreign", "labelled foreign by VAIPE (drug unknown)")], first="pills")
    if orc:
        o = orc["headline"]
        lines += ["", f"With ground-truth boxes (recognition + decision only): out recall {pct(o['out_recall'])}, "
                      f"false alarms {pct(o['false_alarm_rate'])}, abstain {pct(o['abstain_rate'])}, ECE {orc.get('ece', float('nan')):.3f}."]
    lines += ["", "Full tables, calibration and quantisation detail, risk-coverage and reliability plots: "
                  "[docs/benchmark.md](docs/benchmark.md). Confusion pairs with example crops are VAIPE "
                  "photographs and stay out of the repository; `pillguard eval` writes them under "
                  "`artifacts/eval/`."]
    block = "\n".join(lines)
    readme = ROOT / "README.md"
    s = readme.read_text(encoding="utf-8")
    s = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->",
               "<!-- RESULTS:START -->\n" + block + "\n<!-- RESULTS:END -->", s, flags=re.S)
    readme.write_text(s, encoding="utf-8")

    (DOCS / "figures").mkdir(parents=True, exist_ok=True)
    for name in BENCHMARK_FIGURES:
        src = ART / "eval" / "test_detector" / name
        if src.exists():
            shutil.copy(src, DOCS / "figures" / f"benchmark_{name}")
    (DOCS / "benchmark.md").write_text(benchmark_doc(det, orc, dmap, dexp, eexp, fit, par), encoding="utf-8")

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # the block contains '>=' as U+2265
    print(block)
    print(f"\n-> {DOCS / 'benchmark.md'}")


if __name__ == "__main__":
    main()
