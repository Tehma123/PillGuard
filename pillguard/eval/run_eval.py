"""Evaluate the deployed pipeline on the test scenarios (SPEC §4 protocol).

Two modes:

* ``detector`` - the real thing: detector -> crops -> embeddings -> verdicts. Ground-truth
  pills are matched to detections at IoU >= 0.5; an undetected pill is a miss.
* ``oracle``   - ground-truth boxes are used, isolating recognition + decision quality.

Writes to ``artifacts/eval/<name>/``: ``records.jsonl`` (one row per GT pill per scenario),
``spurious.jsonl`` (detections with no GT pill), ``metrics.json``, ``confusion.json``,
figures, and ``report.md``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from pillguard.config import ARTIFACTS_DIR
from pillguard.data.scenarios import Scenario, make_scenarios, summarize
from pillguard.data.splits import Split
from pillguard.data.vaipe import PillImage, load_classes, load_pill_images, prescription_drug_table
from pillguard.eval.metrics import PHOTO_STATES, confusion_pairs, image_level, summarize_records
from pillguard.imaging import box_iou_matrix, load_rgb
from pillguard.pipeline import PillGuardPipeline


def match_boxes(gt: np.ndarray, det: np.ndarray, scores: np.ndarray, thr: float = 0.5) -> dict[int, int]:
    """Greedy one-to-one matching by detection score. Returns {gt_index: det_index}."""
    if len(gt) == 0 or len(det) == 0:
        return {}
    iou = box_iou_matrix(gt, det)
    taken_gt, out = set(), {}
    for j in np.argsort(-scores, kind="stable"):
        cand = [(iou[i, j], i) for i in range(len(gt)) if i not in taken_gt and iou[i, j] >= thr]
        if cand:
            _, i = max(cand)
            taken_gt.add(i); out[i] = int(j)
    return out


def evaluate(pipe: PillGuardPipeline, images: list[PillImage], scenarios: list[Scenario], root: Path,
             seen: set[int], mode: str = "detector", limit: int | None = None, verbose: bool = True):
    by_file: dict[str, list[Scenario]] = defaultdict(list)
    for s in scenarios:
        by_file[s.file].append(s)
    files = list(by_file)
    if limit:
        files = files[:limit]
    im_by = {im.file: im for im in images}
    records, spurious, timings = [], [], []
    t_start = time.time()
    for n, f in enumerate(files):
        im = im_by[f]
        img = load_rgb(im.path(root))
        gt = np.asarray(im.boxes, np.float32).reshape(-1, 4)
        t0 = time.perf_counter()
        if mode == "detector":
            det, scores = pipe.detect(img)
            embs = pipe.embed(img, det)
            matches = match_boxes(gt, det, scores)
        else:
            det, scores = gt, np.ones(len(gt), np.float32)
            embs = pipe.embed(img, det)
            matches = {i: i for i in range(len(gt))}
        timings.append((time.perf_counter() - t0) * 1e3)
        for s in by_file[f]:
            decisions = pipe.decide(embs, s.listed) if len(embs) else []
            used = set()
            for i, p in enumerate(s.pills):
                j = matches.get(i)
                rec = {"scenario": s.id, "kind": s.kind, "file": f, "pill_index": i, "label": p.label,
                       "gt": p.gt, "reason": p.reason, "seen": p.label in seen, "detected": j is not None,
                       "listed": s.listed}
                if j is None:
                    rec.update(verdict="missed", p_in=None, best_any=None, best_listed=None, s_in=None, s_out=None)
                else:
                    d = decisions[j]; used.add(j)
                    rec.update(verdict=d.verdict, p_in=round(d.p_in, 5), p_unknown=round(d.p_unknown, 5),
                               best_any=d.best_any, best_listed=d.best_listed, s_in=round(d.s_in, 4),
                               s_out=round(d.s_out, 4), det_box=np.round(det[j], 1).tolist(), det_score=float(scores[j]))
                records.append(rec)
            for j, d in enumerate(decisions):
                if j not in used:
                    spurious.append({"scenario": s.id, "file": f, "verdict": d.verdict, "p_in": round(d.p_in, 5),
                                     "det_box": np.round(det[j], 1).tolist(), "det_score": float(scores[j])})
        if verbose and (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(files)} images, {len(records)} pill records, {(time.time() - t_start) / 60:.1f} min", flush=True)
    return records, spurious, timings


def unseen_groups(records: list[dict], held_out, untrained) -> dict:
    """Which drugs the ``unseen`` pills belong to: held out on purpose, or absent from train photos."""
    out = {}
    for name, ids in (("held_out", set(held_out)), ("untrained", set(untrained))):
        recs = [r for r in records if r["reason"] == "unseen" and r["label"] in ids]
        out[name] = {"classes": sorted(ids), "present": sorted({r["label"] for r in recs}), "pills": len(recs)}
    return out


def write_report(out_dir: Path, metrics: dict, conf: list[dict], img_lvl: dict, timings: list[float], mode: str,
                 names: dict[int, str], n_spurious: int) -> Path:
    h = metrics["headline"]
    nr = metrics.get("no_reject_baseline", {})
    lines = [f"# PillGuard evaluation ({mode} boxes)", "",
             f"Pill checks: {h['n']} (in: {h['n_in']}, out: {h['n_out']}), i.e. {metrics.get('n_pills_unique', '?')} pills "
             f"in {metrics.get('n_photos', '?')} photos, each counted once per scenario; spurious detections: {n_spurious}", "",
             "| metric | with reject | no-reject baseline |", "|---|---|---|"]
    for key, label in (("out_recall", "out-of-prescription recall (decided)"), ("out_recall_strict", "out recall, strict (misses + abstains count)"),
                       ("false_alarm_rate", "false alarm rate (decided)"), ("abstain_rate", "abstain rate"),
                       ("risk", "risk = error rate on decided"), ("detection_recall", "detection recall")):
        a = h.get(key); b = nr.get(key)
        fa = f"{a:.3f}" if isinstance(a, (int, float)) else "-"
        fb = f"{b:.3f}" if isinstance(b, (int, float)) else "-"
        lines.append(f"| {label} | {fa} | {fb} |")
    lines += ["", f"ECE (p_in vs truth): {metrics.get('ece', float('nan')):.4f}",
              f"Risk-coverage AURC: {metrics.get('risk_coverage', {}).get('aurc', float('nan')):.4f}", "",
              "## Seen vs unseen drugs (foreign pills are neither)", "",
              "| subset | n | out recall | false alarm | abstain |", "|---|---|---|---|---|"]
    for k in ("seen", "unseen"):
        m = metrics[k]
        if m.get("n"):
            lines.append(f"| {k} | {m['n']} | {m['out_recall']:.3f} | {m['false_alarm_rate']:.3f} | {m['abstain_rate']:.3f} |")
    lines += ["", "## By reason", "",
              "| reason | n | out recall | false alarm | abstain | out pill accepted | detection recall |",
              "|---|---|---|---|---|---|---|"]
    for k, m in metrics["by_reason"].items():
        lines.append(f"| {k} | {m['n']} | {m['out_recall']:.3f} | {m['false_alarm_rate']:.3f} | {m['abstain_rate']:.3f} "
                     f"| {m['out_accepted_rate']:.3f} | {m['detection_recall']:.3f} |")
    lines += ["", "## By scenario kind", "", "| kind | n | out recall | false alarm | abstain |", "|---|---|---|---|---|"]
    for k, m in metrics["by_kind"].items():
        lines.append(f"| {k} | {m['n']} | {m['out_recall']:.3f} | {m['false_alarm_rate']:.3f} | {m['abstain_rate']:.3f} |")
    lines += ["", f"## Per photo over {img_lvl['n_scenarios']} scenarios (worst verdict per photo, spurious boxes included)", "",
              "| photo holds | n | all ok | uncertain, no out | at least one out |", "|---|---|---|---|---|"]
    for k, label in (("clean", "only listed pills"), ("wrong", "a pill not on the list")):
        n, t = img_lvl[f"n_{k}"], img_lvl[k]
        lines.append(f"| {label} | {n} | " + " | ".join(f"{t[s]} ({t[s] / max(1, n):.3f})" for s in PHOTO_STATES) + " |")
    nm = metrics.get("naming", {})
    if nm.get("n_pills"):
        lines += ["", f"Nearest prototype is the pill's own drug for {nm['accuracy']:.3f} of {nm['n_pills']} detected pills "
                      f"of trained drugs. Removed pills given a wrong name: {nm['removed_misnamed']} of {nm['removed']}, "
                      f"verdicts {nm['removed_misnamed_verdicts']}."]
    lines += ["", "## Most confused drug pairs (true -> predicted)", "", "| true | predicted | count | share of true |", "|---|---|---|---|"]
    for c in conf:
        lines.append(f"| {c['true']} {c['true_name']} | {c['predicted']} {c['predicted_name']} | {c['count']} | {c['share_of_true']:.2f} |")
    if timings:
        lines += ["", f"Python ONNX latency per image (1 thread, detect+embed): median {np.median(timings):.0f} ms, p90 {np.percentile(timings, 90):.0f} ms"]
    lines += ["", "![risk-coverage](risk_coverage.png)", "", "![reliability](reliability.png)", ""]
    p = out_dir / "report.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def plot_figures(out_dir: Path, metrics: dict) -> None:
    from pillguard.eval.plots import plot_reliability, plot_risk_coverage

    rc = metrics.get("risk_coverage")
    if rc and rc.get("coverage"):
        plot_risk_coverage(rc, out_dir / "risk_coverage.png")
    rel = metrics.get("reliability")
    if rel:
        plot_reliability(rel, out_dir / "reliability.png", ece=metrics.get("ece"))


def save_confusion_examples(out_dir: Path, records: list[dict], conf: list[dict], root: Path, n_pairs: int = 6,
                            per_pair: int = 4) -> None:
    """Grid of example crops for the most confused pairs."""
    from PIL import Image, ImageDraw

    from pillguard.imaging import extract_crop

    ims = {}
    tiles = []
    for c in conf[:n_pairs]:
        ex = [r for r in records if r["label"] == c["true"] and r.get("best_any") == c["predicted"]][:per_pair]
        row = []
        for r in ex:
            if r["file"] not in ims:
                ims[r["file"]] = load_rgb(Path(root) / "images" / "pill" / r["file"])
            box = r.get("det_box")
            if box is None:
                continue
            row.append(Image.fromarray(extract_crop(ims[r["file"]], box, 96, 0.12)))
        tiles.append((c, row))
    if not tiles:
        return
    W = 96 * per_pair + 260
    grid = Image.new("RGB", (W, 100 * len(tiles)), (250, 250, 248))
    d = ImageDraw.Draw(grid)
    for k, (c, row) in enumerate(tiles):
        d.text((4, 100 * k + 4), f"true {c['true']}: {c['true_name'][:28]}", fill=(20, 20, 20))
        d.text((4, 100 * k + 20), f"pred {c['predicted']}: {c['predicted_name'][:28]}", fill=(120, 20, 20))
        d.text((4, 100 * k + 36), f"{c['count']} pills", fill=(60, 60, 60))
        for j, tile in enumerate(row):
            grid.paste(tile, (260 + 96 * j, 100 * k + 2))
    grid.save(out_dir / "confusion_examples.png")


def run(root: Path, split_path: Path, web_dir: Path | None, out_dir: Path, mode: str = "detector",
        limit: int | None = None, pipe: PillGuardPipeline | None = None, subset: str = "test") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = Split.load(split_path)
    images = load_pill_images(root)
    by = {im.file: im for im in images}
    subset_ims = [by[f] for f in split.subset(subset) if f in by]
    untrained = split.untrained_classes(images)
    scenarios = make_scenarios(subset_ims, prescription_drug_table(root), split.all_unseen_classes(images), seed=split.seed)
    pipe = pipe or PillGuardPipeline.from_dir(web_dir)
    seen = set(pipe.matcher.class_ids)
    names = load_classes(root)
    print(f"[{mode}] {len(subset_ims)} {subset} images, scenarios: {summarize(scenarios)}", flush=True)
    records, spurious, timings = evaluate(pipe, subset_ims, scenarios, root, seen, mode=mode, limit=limit)
    metrics = summarize_records(records)
    conf = confusion_pairs(records, names)
    img_lvl = image_level(records, spurious)
    metrics["image_level"] = img_lvl
    metrics["n_photos"] = len({r["file"] for r in records})
    metrics["n_pills_unique"] = len({(r["file"], r["pill_index"]) for r in records})   # records repeat per scenario
    metrics["unseen_groups"] = unseen_groups(records, split.unseen_classes, untrained)
    # a listed drug without a prototype can never be matched: every one of its pills becomes a false alarm
    blind = sorted({r["label"] for r in records if r["gt"] == "in" and not r["seen"]})
    metrics["listed_without_prototype"] = blind
    if blind:
        print(f"WARNING: listed drugs with no prototype {blind}; their pills cannot be judged in", flush=True)
    metrics["n_spurious_detections"] = len(spurious)
    metrics["spurious_verdicts"] = {v: sum(1 for s in spurious if s["verdict"] == v) for v in ("in", "out", "uncertain")}
    metrics["latency_ms"] = {"median": float(np.median(timings)) if timings else None,
                             "p90": float(np.percentile(timings, 90)) if timings else None}
    metrics["mode"] = mode
    metrics["subset"] = subset
    with open(out_dir / "records.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    with open(out_dir / "spurious.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in spurious)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    (out_dir / "confusion.json").write_text(json.dumps(conf, indent=1, ensure_ascii=False), encoding="utf-8")
    try:
        plot_figures(out_dir, metrics)
        save_confusion_examples(out_dir, records, conf, root)
    except Exception as e:  # figures are a convenience
        print(f"(figures skipped: {e})")
    write_report(out_dir, metrics, conf, img_lvl, timings, mode, names, len(spurious))
    h = metrics["headline"]
    print(json.dumps({k: h[k] for k in ("n", "detection_recall", "out_recall", "out_recall_strict", "false_alarm_rate", "abstain_rate", "risk")}, indent=1))
    print("ECE", metrics.get("ece"), "| report ->", out_dir / "report.md")
    return metrics


def main(argv: list[str] | None = None) -> None:
    from pillguard.config import VAIPE_DIR, WEB_DIR
    from pillguard.data.splits import default_split_path

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=VAIPE_DIR)
    ap.add_argument("--split", type=Path, default=default_split_path())
    ap.add_argument("--web", type=Path, default=WEB_DIR, help="asset dir with models + data (what the browser loads)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--mode", default="detector", choices=["detector", "oracle"])
    ap.add_argument("--subset", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    out = a.out or ARTIFACTS_DIR / "eval" / f"{a.subset}_{a.mode}"
    run(a.root, a.split, a.web, out, mode=a.mode, limit=a.limit, subset=a.subset)


if __name__ == "__main__":
    main()
