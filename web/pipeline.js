/**
 * PillGuard browser pipeline. Pure functions + one class; no DOM access, so the same file
 * runs in the browser and in Node (see parity_node.mjs).
 *
 * Every function mirrors pillguard/imaging.py, pillguard/decision/matcher.py and
 * pillguard/pipeline.py. Keep the arithmetic identical: half-pixel bilinear resize,
 * floor(x + 0.5) rounding, greedy NMS with stable descending sort, square context crops,
 * ImageNet normalisation, max-over-prototypes cosine, softmax with an unknown class.
 *
 * Image objects are { data: Uint8Array RGB (h*w*3), width, height }.
 */

// ---------------------------------------------------------------------------------------
// Pixel helpers
// ---------------------------------------------------------------------------------------
export function rgbaToRgb(rgba, width, height) {
  const n = width * height;
  const out = new Uint8Array(n * 3);
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    out[i * 3] = rgba[j]; out[i * 3 + 1] = rgba[j + 1]; out[i * 3 + 2] = rgba[j + 2];
  }
  return { data: out, width, height };
}

/** Bilinear resize with half-pixel centres, no anti-aliasing, floor(v + 0.5) rounding. */
export function bilinearResize(img, outH, outW) {
  const { data: src, width: w, height: h } = img;
  if (h === outH && w === outW) return { data: new Uint8Array(src), width: w, height: h };
  const out = new Uint8Array(outH * outW * 3);
  const sy = h / outH, sx = w / outW;
  const x0s = new Int32Array(outW), x1s = new Int32Array(outW), wxs = new Float32Array(outW);
  for (let x = 0; x < outW; x++) {
    let xs = (x + 0.5) * sx - 0.5; if (xs < 0) xs = 0; if (xs > w - 1) xs = w - 1;
    const x0 = Math.floor(xs); x0s[x] = x0; x1s[x] = Math.min(x0 + 1, w - 1); wxs[x] = xs - x0;
  }
  for (let y = 0; y < outH; y++) {
    let ys = (y + 0.5) * sy - 0.5; if (ys < 0) ys = 0; if (ys > h - 1) ys = h - 1;
    const y0 = Math.floor(ys), y1 = Math.min(y0 + 1, h - 1), wy = ys - y0;
    const r0 = y0 * w * 3, r1 = y1 * w * 3, ro = y * outW * 3;
    for (let x = 0; x < outW; x++) {
      const x0 = x0s[x] * 3, x1 = x1s[x] * 3, wx = wxs[x];
      for (let c = 0; c < 3; c++) {
        const top = src[r0 + x0 + c] * (1 - wx) + src[r0 + x1 + c] * wx;
        const bot = src[r1 + x0 + c] * (1 - wx) + src[r1 + x1 + c] * wx;
        let v = Math.floor(top * (1 - wy) + bot * wy + 0.5);
        out[ro + x * 3 + c] = v < 0 ? 0 : v > 255 ? 255 : v;
      }
    }
  }
  return { data: out, width: outW, height: outH };
}

export function letterbox(img, size, padValue) {
  const { width: w, height: h } = img;
  const scale = Math.min(size / h, size / w);
  const nh = Math.max(1, Math.floor(h * scale + 0.5)), nw = Math.max(1, Math.floor(w * scale + 0.5));
  const resized = bilinearResize(img, nh, nw);
  const canvas = new Uint8Array(size * size * 3).fill(padValue);
  const padY = Math.floor((size - nh) / 2), padX = Math.floor((size - nw) / 2);
  for (let y = 0; y < nh; y++) {
    canvas.set(resized.data.subarray(y * nw * 3, (y + 1) * nw * 3), ((padY + y) * size + padX) * 3);
  }
  return { canvas: { data: canvas, width: size, height: size }, scale, padX, padY };
}

/** uint8 HWC -> float32 CHW in [0, 1]. */
export function toDetectorInput(canvas) {
  const { data, width: w, height: h } = canvas;
  const n = w * h, out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    out[i] = data[i * 3] / 255; out[n + i] = data[i * 3 + 1] / 255; out[2 * n + i] = data[i * 3 + 2] / 255;
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Detection post-processing
// ---------------------------------------------------------------------------------------
export function nms(boxes, scores, iouThr, maxDets) {
  const n = scores.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => (scores[b] - scores[a]) || (a - b));
  const areas = new Float64Array(n);
  for (let i = 0; i < n; i++) areas[i] = Math.max(0, boxes[i][2] - boxes[i][0]) * Math.max(0, boxes[i][3] - boxes[i][1]);
  const suppressed = new Uint8Array(n);
  const keep = [];
  for (const idx of order) {
    if (suppressed[idx]) continue;
    keep.push(idx);
    if (keep.length >= maxDets) break;
    const [ax1, ay1, ax2, ay2] = boxes[idx];
    for (let j = 0; j < n; j++) {
      if (suppressed[j]) continue;
      const xx1 = Math.max(ax1, boxes[j][0]), yy1 = Math.max(ay1, boxes[j][1]);
      const xx2 = Math.min(ax2, boxes[j][2]), yy2 = Math.min(ay2, boxes[j][3]);
      const inter = Math.max(0, xx2 - xx1) * Math.max(0, yy2 - yy1);
      const iou = inter / (areas[idx] + areas[j] - inter + 1e-9);
      if (iou > iouThr) suppressed[j] = 1;
    }
    suppressed[idx] = 1;
  }
  return keep;
}

/** raw (1, 5, N) float32 -> { boxes: [[x1,y1,x2,y2] in letterbox px], scores } */
export function decodeDetections(raw, dims, conf, iouThr, maxDets) {
  const N = dims[2];
  const boxes = [], scores = [];
  for (let i = 0; i < N; i++) {
    const s = raw[4 * N + i];
    if (s >= conf) {
      const cx = raw[i], cy = raw[N + i], w = raw[2 * N + i], h = raw[3 * N + i];
      boxes.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]); scores.push(s);
    }
  }
  const keep = nms(boxes, scores, iouThr, maxDets);
  return { boxes: keep.map((k) => boxes[k]), scores: keep.map((k) => scores[k]) };
}

export function scaleBoxesBack(boxes, scale, padX, padY, imgW, imgH) {
  const clip = (v, hi) => (v < 0 ? 0 : v > hi ? hi : v);
  return boxes.map(([x1, y1, x2, y2]) => [
    clip((x1 - padX) / scale, imgW), clip((y1 - padY) / scale, imgH),
    clip((x2 - padX) / scale, imgW), clip((y2 - padY) / scale, imgH),
  ]);
}

// ---------------------------------------------------------------------------------------
// Crops + embedding input
// ---------------------------------------------------------------------------------------
export function squareCropBox(box, context) {
  const [x1, y1, x2, y2] = box;
  const bw = Math.max(1, x2 - x1), bh = Math.max(1, y2 - y1);
  const side = Math.max(bw, bh) * (1 + 2 * context);
  const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
  const sx1 = Math.floor(cx - side / 2), sy1 = Math.floor(cy - side / 2), s = Math.ceil(side);
  return [sx1, sy1, sx1 + s, sy1 + s];
}

export function extractCrop(img, box, size, context, padValue) {
  const { data, width: w, height: h } = img;
  const [sx1, sy1, sx2, sy2] = squareCropBox(box, context);
  const s = sx2 - sx1;
  const patch = new Uint8Array(s * s * 3).fill(padValue);
  const ix1 = Math.max(0, sx1), iy1 = Math.max(0, sy1), ix2 = Math.min(w, sx2), iy2 = Math.min(h, sy2);
  if (ix2 > ix1 && iy2 > iy1) {
    for (let y = iy1; y < iy2; y++) {
      patch.set(data.subarray((y * w + ix1) * 3, (y * w + ix2) * 3), ((y - sy1) * s + (ix1 - sx1)) * 3);
    }
  }
  return bilinearResize({ data: patch, width: s, height: s }, size, size);
}

/** crops: array of {data,width,height} (uint8 RGB) -> float32 NCHW, ImageNet-normalised. */
export function toEmbedInput(crops, size, mean, std) {
  const n = size * size, out = new Float32Array(crops.length * 3 * n);
  crops.forEach((c, k) => {
    const base = k * 3 * n;
    for (let i = 0; i < n; i++) {
      for (let ch = 0; ch < 3; ch++) out[base + ch * n + i] = (c.data[i * 3 + ch] / 255 - mean[ch]) / std[ch];
    }
  });
  return out;
}

export function l2normalize(v) {
  let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  const inv = 1 / (Math.sqrt(s) + 1e-9);
  const out = new Float32Array(v.length); for (let i = 0; i < v.length; i++) out[i] = v[i] * inv;
  return out;
}

// ---------------------------------------------------------------------------------------
// Matcher (prescription-aware decision)
// ---------------------------------------------------------------------------------------
export class Matcher {
  /** prototypes: {dim, classes:[{id,name,vectors:[[...]]}]}; params: {temperature, unknown_bias, theta_in, theta_out, margin} */
  constructor(prototypes, params) {
    this.params = params;
    this.dim = prototypes.dim;
    this.classIds = prototypes.classes.map((c) => c.id).slice().sort((a, b) => a - b);
    const byId = new Map(prototypes.classes.map((c) => [c.id, c]));
    this.names = new Map(prototypes.classes.map((c) => [c.id, c.name]));
    this.vectors = this.classIds.map((id) => byId.get(id).vectors.map((v) => l2normalize(Float32Array.from(v))));
  }
  get nClasses() { return this.classIds.length; }

  classSimilarities(emb) {
    const out = new Float32Array(this.nClasses);
    for (let c = 0; c < this.nClasses; c++) {
      let best = -Infinity;
      for (const v of this.vectors[c]) {
        let d = 0; for (let i = 0; i < this.dim; i++) d += emb[i] * v[i];
        if (d > best) best = d;
      }
      out[c] = best;
    }
    return out;
  }

  listedMask(listed) {
    const set = new Set(listed.map(Number));
    return this.classIds.map((id) => set.has(id));
  }

  decide(emb, listed, params) {
    const p = params || this.params;
    const s = this.classSimilarities(emb);
    const mask = this.listedMask(listed);
    const C = this.nClasses;
    // softmax over [s / T, b / T]
    const z = new Float64Array(C + 1);
    let zmax = -Infinity;
    for (let c = 0; c < C; c++) { z[c] = s[c] / p.temperature; if (z[c] > zmax) zmax = z[c]; }
    z[C] = p.unknown_bias / p.temperature; if (z[C] > zmax) zmax = z[C];
    let sum = 0; const e = new Float64Array(C + 1);
    for (let c = 0; c <= C; c++) { e[c] = Math.exp(z[c] - zmax); sum += e[c]; }
    let pIn = 0; for (let c = 0; c < C; c++) if (mask[c]) pIn += e[c] / sum;
    const pUnknown = e[C] / sum;
    let sIn = -1, bi = null, sOut = -1, oi = null, anyListed = false, anyOther = false;
    for (let c = 0; c < C; c++) {
      if (mask[c]) { if (!anyListed || s[c] > sIn) { sIn = s[c]; bi = c; } anyListed = true; }
      else { if (!anyOther || s[c] > sOut) { sOut = s[c]; oi = c; } anyOther = true; }
    }
    if (!anyListed) sIn = -1; if (!anyOther) sOut = -1;
    const margin = sIn - sOut;
    let verdict;
    if (pIn >= p.theta_in && margin >= p.margin) verdict = "in";
    else if (pIn <= p.theta_out) verdict = "out";
    else verdict = "uncertain";
    let bestAny = null, ba = -Infinity;
    for (let c = 0; c < C; c++) if (s[c] > ba) { ba = s[c]; bestAny = c; }
    return {
      verdict, p_in: pIn, p_unknown: pUnknown, s_in: sIn, s_out: sOut, margin,
      best_listed: bi === null ? null : this.classIds[bi],
      best_any: bestAny === null ? null : this.classIds[bestAny],
    };
  }
}

// ---------------------------------------------------------------------------------------
// End-to-end
// ---------------------------------------------------------------------------------------
export class PillGuard {
  constructor(ort, detSession, embSession, config, prototypes) {
    this.ort = ort; this.det = detSession; this.emb = embSession; this.config = config;
    this.matcher = new Matcher(prototypes, config.decision);
    this.detInputName = detSession.inputNames[0];
    this.embInputName = embSession.inputNames[0];
  }

  /** baseUrl must end with '/'; fetchJson(url) -> object; sessionOpts passed to InferenceSession.create */
  static async load(ort, baseUrl, { fetchJson, sessionOpts } = {}) {
    const fj = fetchJson || (async (u) => (await fetch(u)).json());
    const config = await fj(baseUrl + "data/config.json");
    const prototypes = await fj(baseUrl + "data/prototypes.json");
    const det = await ort.InferenceSession.create(baseUrl + "models/" + config.models.detector, sessionOpts);
    const emb = await ort.InferenceSession.create(baseUrl + "models/" + config.models.embedding, sessionOpts);
    return new PillGuard(ort, det, emb, config, prototypes);
  }

  async detect(img) {
    const cfg = this.config;
    const { canvas, scale, padX, padY } = letterbox(img, cfg.det_input, cfg.det_pad_value);
    const x = new this.ort.Tensor("float32", toDetectorInput(canvas), [1, 3, cfg.det_input, cfg.det_input]);
    const out = await this.det.run({ [this.detInputName]: x });
    const t = out[this.det.outputNames[0]];
    const { boxes, scores } = decodeDetections(t.data, t.dims, cfg.det_conf, cfg.det_iou, cfg.det_max_dets);
    return { boxes: scaleBoxesBack(boxes, scale, padX, padY, img.width, img.height), scores };
  }

  async embed(img, boxes) {
    if (boxes.length === 0) return [];
    const cfg = this.config;
    const crops = boxes.map((b) => extractCrop(img, b, cfg.crop_size, cfg.crop_context, cfg.det_pad_value));
    const x = new this.ort.Tensor("float32", toEmbedInput(crops, cfg.crop_size, cfg.emb_mean, cfg.emb_std),
      [crops.length, 3, cfg.crop_size, cfg.crop_size]);
    const out = await this.emb.run({ [this.embInputName]: x });
    const t = out[this.emb.outputNames[0]];
    const d = t.dims[1], embs = [];
    for (let i = 0; i < crops.length; i++) embs.push(l2normalize(t.data.subarray(i * d, (i + 1) * d)));
    return embs;
  }

  decide(embs, listed) { return embs.map((e) => this.matcher.decide(e, listed)); }

  async run(img, listed) {
    const t0 = performance.now();
    const { boxes, scores } = await this.detect(img);
    const t1 = performance.now();
    const embs = await this.embed(img, boxes);
    const t2 = performance.now();
    const decisions = this.decide(embs, listed);
    const t3 = performance.now();
    return { boxes, scores, embeddings: embs, decisions,
      timings_ms: { detect: t1 - t0, embed: t2 - t1, decide: t3 - t2, total: t3 - t0 } };
  }

  async runWithBoxes(img, boxes, listed) {
    const embs = await this.embed(img, boxes);
    return { boxes, scores: boxes.map(() => 1), embeddings: embs, decisions: this.decide(embs, listed) };
  }
}
