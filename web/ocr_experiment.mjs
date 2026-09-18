/**
 * Week-0 experiment: can in-browser OCR (Tesseract.js, Vietnamese model) recover the drugs
 * on real VAIPE prescription scans? Runs under Node on N ingested prescription images and
 * scores recognised class ids against the annotation.
 *
 *   cd web && npm install && node ocr_experiment.mjs ../data/vaipe 20            # run OCR
 *   node ocr_experiment.mjs ../data/vaipe 20 --rescore                            # reuse saved OCR text
 */
import fs from "node:fs";
import path from "node:path";
import { matchOcrLines } from "./ocr_match.js";

const root = path.resolve(process.argv[2] || "../data/vaipe");
const N = Number(process.argv[3] || 20);
const rescore = process.argv.includes("--rescore");
const outPath = path.resolve("../artifacts/ocr_experiment.json");

const rows = fs.readFileSync(path.join(root, "annotations", "prescription_train.jsonl"), "utf-8")
  .split("\n").filter(Boolean).map((l) => JSON.parse(l));

// drug table with aliases: every cleaned drugname line ever mapped to a class id
const names = new Map();
for (const r of rows) r.texts.forEach((t, i) => {
  if (r.ann_labels[i] === "drugname" && r.mappings[i] >= 0) {
    const name = t.replace(/^\s*\d+\s*[).\-:]\s*/, "").trim();
    const m = names.get(r.mappings[i]) || new Map(); m.set(name, (m.get(name) || 0) + 1); names.set(r.mappings[i], m);
  }
});
const drugs = [...names.entries()].map(([id, m]) => {
  const sorted = [...m.entries()].sort((a, b) => b[1] - a[1]).map((x) => x[0]);
  return { id, name: sorted[0], aliases: sorted.slice(1) };
}).sort((a, b) => a.id - b.id);

const sample = rows.filter((_, i) => i % Math.floor(rows.length / N) === 0).slice(0, N);
let saved = null;
if (rescore) saved = new Map(JSON.parse(fs.readFileSync(outPath, "utf-8")).details.map((d) => [d.file, d]));
let worker = null;
if (!rescore) {
  const { createWorker } = await import("tesseract.js");
  worker = await createWorker("vie", 1, { logger: () => {} });
}
let tp = 0, fp = 0, fn = 0, tsum = 0, ntp = 0, nfp = 0, nfn = 0;
const namesOf = new Map(drugs.map((d) => [d.id, new Set([d.name, ...d.aliases])]));
const details = [];
for (const r of sample) {
  let lines, dt;
  if (rescore) { ({ lines, ms: dt } = saved.get(r.file)); }
  else {
    const t0 = Date.now();
    const { data } = await worker.recognize(path.join(root, "images", "prescription", r.file));
    dt = Date.now() - t0;
    lines = data.text.split(/\n+/).filter((l) => l.trim());
  }
  tsum += dt;
  const found = new Set(matchOcrLines(lines, drugs).map((m) => m.drug.id));
  const gt = new Set(r.mappings.filter((m, i) => r.ann_labels[i] === "drugname" && m >= 0));
  const hit = [...gt].filter((g) => found.has(g)).length;
  tp += hit; fn += gt.size - hit; fp += [...found].filter((f) => !gt.has(f)).length;
  // name level: a found id counts as correct if it shares a printed name with a ground-truth id
  const gtNames = new Set([...gt].flatMap((g) => [...namesOf.get(g)]));
  const shares = (f) => [...namesOf.get(f)].some((n) => gtNames.has(n));
  const foundNames = new Set([...found].flatMap((f) => [...namesOf.get(f)]));
  ntp += [...gt].filter((g) => [...namesOf.get(g)].some((n) => foundNames.has(n))).length;
  nfn += [...gt].filter((g) => ![...namesOf.get(g)].some((n) => foundNames.has(n))).length;
  nfp += [...found].filter((f) => !shares(f)).length;
  details.push({ file: r.file, gt: [...gt], found: [...found], ms: dt, lines });
  console.log(`${r.file}: gt=${[...gt]} found=${[...found]} (${dt} ms)`);
}
if (worker) await worker.terminate();
const summary = { n: sample.length, drug_recall: tp / Math.max(1, tp + fn), drug_precision: tp / Math.max(1, tp + fp),
  mean_ms: tsum / sample.length, tp, fp, fn,
  name_level_recall: ntp / Math.max(1, ntp + nfn), name_level_precision: ntp / Math.max(1, ntp + nfp), n_drugs: drugs.length, n_aliases: drugs.reduce((s, d) => s + d.aliases.length, 0) };
console.log(JSON.stringify(summary, null, 1));
fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, JSON.stringify({ summary, details }, null, 1));
