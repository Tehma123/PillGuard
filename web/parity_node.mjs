/**
 * Run the browser pipeline under Node (onnxruntime-web, WASM backend) on the parity set
 * written by `python -m pillguard.eval.parity --export`, and write results for Python to
 * compare. PNG inputs are lossless, so both sides start from identical pixels.
 *
 *   cd web && npm install && node parity_node.mjs ../artifacts/parity
 */
import fs from "node:fs";
import path from "node:path";
import { PNG } from "pngjs";
import * as ort from "onnxruntime-web";
import { PillGuard, rgbaToRgb } from "./pipeline.js";

const parityDir = path.resolve(process.argv[2] || "../artifacts/parity");
const webDir = path.resolve(process.argv[3] || ".");
ort.env.wasm.numThreads = 1;

const fetchJson = async (u) => JSON.parse(fs.readFileSync(u, "utf-8"));
const baseUrl = webDir + path.sep;
const config = await fetchJson(path.join(webDir, "data", "config.json"));
const prototypes = await fetchJson(path.join(webDir, "data", "prototypes.json"));
const det = await ort.InferenceSession.create(fs.readFileSync(path.join(webDir, "models", config.models.detector)));
const emb = await ort.InferenceSession.create(fs.readFileSync(path.join(webDir, "models", config.models.embedding)));
const pg = new PillGuard(ort, det, emb, config, prototypes);

const cases = JSON.parse(fs.readFileSync(path.join(parityDir, "cases.json"), "utf-8"));
const results = [];
for (const c of cases) {
  const png = PNG.sync.read(fs.readFileSync(path.join(parityDir, c.png)));
  const img = rgbaToRgb(png.data, png.width, png.height);
  const r = c.boxes ? await pg.runWithBoxes(img, c.boxes, c.listed) : await pg.run(img, c.listed);
  results.push({
    id: c.id,
    boxes: r.boxes.map((b) => b.map((v) => Math.round(v * 10) / 10)),
    scores: r.scores.map((s) => Math.round(s * 1e4) / 1e4),
    decisions: r.decisions.map((d) => ({ ...d, p_in: Math.round(d.p_in * 1e5) / 1e5 })),
    timings_ms: r.timings_ms || null,
  });
  process.stdout.write(`${c.id}: ${r.boxes.length} pills, ${r.decisions.map((d) => d.verdict).join(",")}\n`);
}
fs.writeFileSync(path.join(parityDir, "results_js.json"), JSON.stringify(results, null, 1));
console.log(`wrote ${results.length} results -> ${path.join(parityDir, "results_js.json")}`);
