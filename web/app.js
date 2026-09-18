/**
 * PillGuard demo page. Everything runs locally: ONNX Runtime Web (WASM) + pipeline.js.
 * Prescription input: pick drugs from the list (C), load a sample prescription (B), or the
 * experimental in-browser OCR of a prescription photo with Tesseract.js (A).
 */
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.min.mjs";
import { PillGuard, rgbaToRgb, bilinearResize } from "./pipeline.js";
import { normalize, matchOcrLines } from "./ocr_match.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const MAX_SIDE = 1280;      // user photos are shrunk like the training data
const OCR_CDN = "https://cdn.jsdelivr.net/npm/tesseract.js@6/dist/tesseract.esm.min.js";

const I18N = {
  vi: {
    tagline: "Kiểm tra viên thuốc theo đơn, chạy hoàn toàn trong trình duyệt",
    disclaimer: "Đây là demo nghiên cứu, không phải thiết bị y tế và không tư vấn liều dùng. Ảnh được xử lý ngay trên máy của bạn và không gửi đi đâu.",
    step1: "Đơn thuốc", step2: "Ảnh nắm thuốc", step3: "Kết quả",
    tabPick: "Chọn từ danh sách", tabSample: "Đơn mẫu", tabOcr: "Quét ảnh đơn (thử nghiệm)",
    searchPlaceholder: "Gõ tên thuốc…", add: "Thêm",
    pickHint: "Chỉ hỗ trợ các thuốc mà mô hình đã học (danh sách gợi ý). Thuốc khác sẽ bị coi là ngoài đơn.",
    sampleHint: "Danh sách thuốc lấy từ các đơn trong tập kiểm tra của bộ dữ liệu VAIPE (chỉ tên thuốc, không kèm ảnh).",
    ocrRun: "Đọc đơn", ocrHint: "OCR tiếng Việt chạy trong trình duyệt (Tesseract.js, tải thêm ~15 MB lần đầu). Kết quả cần được kiểm tra lại.",
    photoHint: "Chụp gần, đủ sáng, viên thuốc không chồng lên nhau. Lật mặt có chữ lên trên nếu có.",
    run: "Kiểm tra", loading: "Đang tải mô hình…", ready: "Sẵn sàng", running: "Đang xử lý…",
    needDrugs: "Hãy thêm ít nhất một thuốc vào đơn.", needPhoto: "Hãy chọn ảnh viên thuốc.",
    noPills: "Không tìm thấy viên thuốc nào. Hãy chụp gần hơn.",
    sumOk: (n) => `Tất cả ${n} viên đều khớp với đơn.`,
    sumAlert: (k, n) => `Cảnh báo: ${k}/${n} viên không có trong đơn.`,
    sumUnsure: (k) => `${k} viên chưa chắc chắn: chụp lại gần hơn, lật mặt có chữ, hoặc hỏi dược sĩ.`,
    vIn: "Đúng đơn", vOut: "Ngoài đơn", vUncertain: "Không chắc",
    whyIn: (name) => `Giống ${name}`, whyOut: (name) => name ? `Không giống thuốc nào trong đơn (gần nhất: ${name})` : "Không giống thuốc nào trong đơn",
    whyUnsure: (a, b) => `Có thể là ${a}${b ? " hoặc " + b : ""}`,
    pill: "Viên", timings: (t) => `Thời gian: phát hiện ${t.detect.toFixed(0)} ms · nhận diện ${t.embed.toFixed(0)} ms · tổng ${t.total.toFixed(0)} ms`,
    ocrLoading: "Đang tải OCR…", ocrWorking: (p) => `Đang đọc… ${p}%`, ocrNone: "Không nhận ra tên thuốc nào. Hãy chọn từ danh sách.",
    ocrFound: (n) => `Tìm thấy ${n} thuốc có thể có trong đơn. Chọn để thêm:`, ocrAdd: "Thêm thuốc đã chọn",
    footer: "PillGuard · demo nghiên cứu, mã nguồn mở (AGPL-3.0)",
    modelInfo: (mb, n) => `mô hình ${mb.toFixed(1)} MB · ${n} thuốc`,
    unknownName: "thuốc chưa biết",
  },
  en: {
    tagline: "Prescription-aware pill check, running entirely in your browser",
    disclaimer: "Research demo. Not a medical device and no dosage advice. Photos are processed on your device and never uploaded.",
    step1: "Prescription", step2: "Pill photo", step3: "Result",
    tabPick: "Pick from list", tabSample: "Sample prescription", tabOcr: "Scan prescription (experimental)",
    searchPlaceholder: "Type a drug name…", add: "Add",
    pickHint: "Only drugs the model was trained on are listed. Anything else counts as 'not on the prescription'.",
    sampleHint: "Drug lists taken from test-split prescriptions of the VAIPE dataset (names only, no images).",
    ocrRun: "Read prescription", ocrHint: "Vietnamese OCR in the browser (Tesseract.js, ~15 MB extra download on first use). Check the result.",
    photoHint: "Shoot close, well lit, pills not overlapping. Imprinted side up if there is one.",
    run: "Check", loading: "Loading models…", ready: "Ready", running: "Working…",
    needDrugs: "Add at least one drug to the prescription.", needPhoto: "Choose a pill photo.",
    noPills: "No pills found. Try a closer photo.",
    sumOk: (n) => `All ${n} pills match the prescription.`,
    sumAlert: (k, n) => `Warning: ${k} of ${n} pills are not on the prescription.`,
    sumUnsure: (k) => `${k} uncertain: retake closer, flip the imprinted side up, or ask a pharmacist.`,
    vIn: "On prescription", vOut: "Not on prescription", vUncertain: "Uncertain",
    whyIn: (name) => `Looks like ${name}`, whyOut: (name) => name ? `Unlike any listed drug (closest: ${name})` : "Unlike any listed drug",
    whyUnsure: (a, b) => `Could be ${a}${b ? " or " + b : ""}`,
    pill: "Pill", timings: (t) => `Timing: detect ${t.detect.toFixed(0)} ms · recognise ${t.embed.toFixed(0)} ms · total ${t.total.toFixed(0)} ms`,
    ocrLoading: "Loading OCR…", ocrWorking: (p) => `Reading… ${p}%`, ocrNone: "No drug names recognised. Pick from the list instead.",
    ocrFound: (n) => `Found ${n} possible drugs. Tick to add:`, ocrAdd: "Add selected",
    footer: "PillGuard · research demo, open source (AGPL-3.0)",
    modelInfo: (mb, n) => `models ${mb.toFixed(1)} MB · ${n} drugs`,
    unknownName: "unknown drug",
  },
};

const $ = (id) => document.getElementById(id);
const state = { lang: "vi", pg: null, drugs: [], names: new Map(), samples: [], listed: new Map(), img: null, result: null };

// ---------------------------------------------------------------------------------------
// i18n
// ---------------------------------------------------------------------------------------
function t(key, ...args) { const v = I18N[state.lang][key]; return typeof v === "function" ? v(...args) : v; }
function applyLang(lang) {
  state.lang = lang;
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => { el.placeholder = t(el.dataset.i18nPlaceholder); });
  document.querySelectorAll(".lang button").forEach((b) => b.classList.toggle("active", b.dataset.lang === lang));
  try { localStorage.setItem("pillguard.lang", lang); } catch (_) { /* private mode */ }
  if (state.result) renderResult(state.result);
  if (state.pg) $("status").textContent = t("ready");
}

// ---------------------------------------------------------------------------------------
// Prescription input
// ---------------------------------------------------------------------------------------
function renderChips() {
  const box = $("chips");
  box.innerHTML = "";
  for (const [id, name] of state.listed) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = `<span>${escapeHtml(name)}</span><span class="id">#${id}</span><button type="button" aria-label="remove">×</button>`;
    chip.querySelector("button").onclick = () => { state.listed.delete(id); renderChips(); };
    box.appendChild(chip);
  }
  updateRunButton();
}
function addDrug(id) {
  if (!state.names.has(id)) return false;
  state.listed.set(id, state.names.get(id));
  renderChips();
  return true;
}
function drugFromInput(text) {
  const q = normalize(text);
  if (!q) return null;
  let best = null;
  for (const d of state.drugs) {
    const n = normalize(d.name);
    if (n === q) return d.id;
    if (n.startsWith(q) && best === null) best = d.id;
  }
  return best;
}

// ---------------------------------------------------------------------------------------
// Experimental OCR (Tesseract.js); matching lives in ocr_match.js
// ---------------------------------------------------------------------------------------
async function runOcr(file) {
  const status = $("ocr-status"), cands = $("ocr-candidates");
  cands.innerHTML = "";
  status.textContent = t("ocrLoading");
  const Tesseract = (await import(OCR_CDN)).default;
  const { data } = await Tesseract.recognize(file, "vie", {
    logger: (m) => { if (m.status === "recognizing text") status.textContent = t("ocrWorking", Math.round(m.progress * 100)); },
  });
  const lines = data.text.split(/\n+/).filter((l) => l.trim());
  const matches = matchOcrLines(lines, state.drugs);
  if (!matches.length) { status.textContent = t("ocrNone"); return; }
  status.textContent = t("ocrFound", matches.length);
  for (const m of matches) {
    const label = document.createElement("label");
    const shown = m.name && m.name !== m.drug.name ? `${m.drug.name} (${m.name})` : m.drug.name;
    label.innerHTML = `<input type="checkbox" checked value="${m.drug.id}"> <span>${escapeHtml(shown)}</span> <span class="line">← “${escapeHtml(m.line)}”</span>`;
    cands.appendChild(label);
  }
  const btn = document.createElement("button");
  btn.type = "button"; btn.className = "secondary"; btn.textContent = t("ocrAdd");
  btn.onclick = () => { cands.querySelectorAll("input:checked").forEach((c) => addDrug(Number(c.value))); };
  cands.appendChild(btn);
}

// ---------------------------------------------------------------------------------------
// Photo handling
// ---------------------------------------------------------------------------------------
async function loadPhoto(file) {
  const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
  const c = document.createElement("canvas");
  c.width = bmp.width; c.height = bmp.height;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  let img = rgbaToRgb(ctx.getImageData(0, 0, c.width, c.height).data, c.width, c.height);
  const scale = Math.min(1, MAX_SIDE / Math.max(img.width, img.height));
  if (scale < 1) img = bilinearResize(img, Math.floor(img.height * scale + 0.5), Math.floor(img.width * scale + 0.5));
  state.img = img; state.result = null;
  drawImage(img);
  $("summary").classList.add("hidden"); $("results").innerHTML = ""; $("timings").textContent = "";
  updateRunButton();
}
function drawImage(img) {
  const canvas = $("canvas");
  canvas.width = img.width; canvas.height = img.height;
  const ctx = canvas.getContext("2d");
  const rgba = new Uint8ClampedArray(img.width * img.height * 4);
  for (let i = 0, j = 0; i < img.width * img.height; i++, j += 4) {
    rgba[j] = img.data[i * 3]; rgba[j + 1] = img.data[i * 3 + 1]; rgba[j + 2] = img.data[i * 3 + 2]; rgba[j + 3] = 255;
  }
  ctx.putImageData(new ImageData(rgba, img.width, img.height), 0, 0);
}
function colorFor(verdict) {
  const css = getComputedStyle(document.documentElement);
  return css.getPropertyValue(verdict === "in" ? "--in" : verdict === "out" ? "--out" : "--unsure").trim();
}

// ---------------------------------------------------------------------------------------
// Run + render
// ---------------------------------------------------------------------------------------
function updateRunButton() { $("run").disabled = !(state.pg && state.img && state.listed.size > 0); }
async function runCheck() {
  if (!state.listed.size) { $("status").textContent = t("needDrugs"); return; }
  if (!state.img) { $("status").textContent = t("needPhoto"); return; }
  $("run").disabled = true; $("status").textContent = t("running");
  await new Promise((r) => setTimeout(r, 20));
  try {
    const result = await state.pg.run(state.img, [...state.listed.keys()]);
    state.result = result;
    renderResult(result);
    $("status").textContent = t("ready");
  } catch (e) {
    console.error(e); $("status").textContent = "Error: " + e.message;
  }
  updateRunButton();
}
function renderResult(result) {
  drawImage(state.img);
  const ctx = $("canvas").getContext("2d");
  const lw = Math.max(2, Math.round(state.img.width / 300));
  ctx.lineWidth = lw; ctx.font = `${Math.max(14, lw * 6)}px system-ui, sans-serif`;
  result.boxes.forEach((b, i) => {
    const col = colorFor(result.decisions[i].verdict);
    ctx.strokeStyle = col; ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
    ctx.fillStyle = col; ctx.fillRect(b[0], Math.max(0, b[1] - lw * 9), lw * 12, lw * 9);
    ctx.fillStyle = "#fff"; ctx.fillText(String(i + 1), b[0] + lw * 2, Math.max(lw * 7, b[1] - lw * 2));
  });
  const n = result.decisions.length;
  const nOut = result.decisions.filter((d) => d.verdict === "out").length;
  const nUns = result.decisions.filter((d) => d.verdict === "uncertain").length;
  const sum = $("summary");
  sum.classList.remove("hidden", "ok", "alert", "unsure");
  if (n === 0) { sum.classList.add("unsure"); sum.textContent = t("noPills"); }
  else if (nOut > 0) { sum.classList.add("alert"); sum.textContent = t("sumAlert", nOut, n) + (nUns ? " " + t("sumUnsure", nUns) : ""); }
  else if (nUns > 0) { sum.classList.add("unsure"); sum.textContent = t("sumUnsure", nUns); }
  else { sum.classList.add("ok"); sum.textContent = t("sumOk", n); }
  const list = $("results");
  list.innerHTML = "";
  result.decisions.forEach((d, i) => {
    const nameL = d.best_listed !== null ? state.names.get(d.best_listed) : null;
    const nameA = d.best_any !== null ? state.names.get(d.best_any) : t("unknownName");
    let why;
    if (d.verdict === "in") why = t("whyIn", nameL);
    else if (d.verdict === "out") why = t("whyOut", d.best_any !== null && !state.listed.has(d.best_any) ? nameA : "");
    else why = t("whyUnsure", nameL || nameA, nameL && nameA !== nameL ? nameA : "");
    const li = document.createElement("li");
    li.className = "v-" + d.verdict;
    li.innerHTML = `<span class="dot" aria-hidden="true"></span><div><div class="name">${t("pill")} ${i + 1}: ${t(d.verdict === "in" ? "vIn" : d.verdict === "out" ? "vOut" : "vUncertain")}</div><div class="why">${escapeHtml(why)}</div></div><span class="p">p(in)=${d.p_in.toFixed(2)}</span>`;
    list.appendChild(li);
  });
  if (result.timings_ms) $("timings").textContent = t("timings", result.timings_ms);
}
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

// ---------------------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------------------
async function boot() {
  let lang = "vi";
  try { lang = localStorage.getItem("pillguard.lang") || (navigator.language.startsWith("vi") ? "vi" : "en"); } catch (_) { /* ignore */ }
  applyLang(lang);
  document.querySelectorAll(".lang button").forEach((b) => (b.onclick = () => applyLang(b.dataset.lang)));
  document.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => {
    document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("active", x === b));
    document.querySelectorAll(".tab").forEach((p) => p.classList.toggle("hidden", p.id !== "tab-" + b.dataset.tab));
  }));
  $("status").textContent = t("loading");
  const [drugs, samples] = await Promise.all([fetch("data/drugs.json").then((r) => r.json()), fetch("data/samples.json").then((r) => r.json())]);
  state.drugs = drugs; state.samples = samples;
  drugs.forEach((d) => state.names.set(d.id, d.name));
  const dl = $("drug-list");
  drugs.forEach((d) => { const o = document.createElement("option"); o.value = d.name; dl.appendChild(o); });
  const sel = $("sample-select");
  sel.innerHTML = `<option value="">—</option>` + samples.map((s, i) => `<option value="${i}">${escapeHtml(s.names.join(" · "))}</option>`).join("");
  sel.onchange = () => { const s = samples[Number(sel.value)]; if (!s) return; state.listed.clear(); s.drugs.forEach(addDrug); };
  $("add-drug").onclick = () => { const id = drugFromInput($("drug-search").value); if (id !== null && addDrug(id)) $("drug-search").value = ""; };
  $("drug-search").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); $("add-drug").click(); } });
  $("drug-search").addEventListener("change", () => { const id = drugFromInput($("drug-search").value); if (id !== null && normalize(state.names.get(id)) === normalize($("drug-search").value) && addDrug(id)) $("drug-search").value = ""; });
  $("pill-file").onchange = (e) => e.target.files[0] && loadPhoto(e.target.files[0]).catch((err) => ($("status").textContent = "Error: " + err.message));
  $("ocr-run").onclick = () => { const f = $("rx-file").files[0]; if (f) runOcr(f).catch((err) => ($("ocr-status").textContent = "OCR error: " + err.message)); };
  $("run").onclick = runCheck;
  try {
    state.pg = await PillGuard.load(ort, "./", { sessionOpts: { executionProviders: ["wasm"] } });
    const cfg = state.pg.config;
    $("model-info").textContent = t("modelInfo", (cfg.models.detector_bytes + cfg.models.embedding_bytes) / 1e6, drugs.length);
    $("status").textContent = t("ready");
  } catch (e) {
    console.error(e); $("status").textContent = "Model load failed: " + e.message;
  }
  updateRunButton();
}
if (typeof document !== "undefined") boot();
