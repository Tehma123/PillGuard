/**
 * Fuzzy matching of OCR text lines against the drug list (shared by app.js and the Node
 * OCR experiment).
 *
 * A drug is recognised when the whole alphabetic token sequence of one of its names
 * (dose, parentheses and tokens under 3 letters removed) appears in a line, each token within
 * a small edit distance. Names must carry at least 6 letters in total, which keeps a common
 * word such as "huyết" from matching "Huyết áp" on every prescription. Drugs may carry
 * ``aliases``: VAIPE prescriptions often print a brand (NOVOXIM-500) for a pill class whose
 * usual name is another brand (FABAMOX 500), so every name seen for a class counts.
 */
export function normalize(s) {
  return s.normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/đ/g, "d").replace(/Đ/g, "D")
    .toLowerCase().replace(/([a-z])(\d)/g, "$1 $2").replace(/(\d)([a-z])/g, "$1 $2")
    .replace(/[^a-z0-9]+/g, " ").trim();
}

export function levenshtein(a, b) {
  const m = a.length, n = b.length;
  if (!m) return n; if (!n) return m;
  let prev = Array.from({ length: n + 1 }, (_, i) => i);
  for (let i = 1; i <= m; i++) {
    const cur = [i];
    for (let j = 1; j <= n; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return prev[n];
}

/** Alphabetic tokens (>= 3 letters) of a drug name, dose and parenthetical parts removed. */
export function nameTokens(name) {
  return normalize(name.replace(/\([^)]*\)/g, " ")).split(" ").filter((t) => t.length >= 3 && /^[a-z]+$/.test(t));
}

/** Alphabetic tokens (>= 3 letters) of an OCR line, same filtering as nameTokens. */
export function lineTokens(line) {
  return normalize(line).split(" ").filter((t) => t.length >= 3 && /^[a-z]+$/.test(t));
}

const tolerance = (tok) => (tok.length >= 8 ? 2 : tok.length >= 5 ? 1 : 0);

function sequenceIn(lineToks, nameToks) {
  outer: for (let i = 0; i + nameToks.length <= lineToks.length; i++) {
    for (let k = 0; k < nameToks.length; k++) {
      const a = lineToks[i + k], b = nameToks[k];
      if (a !== b && levenshtein(a, b) > tolerance(b)) continue outer;
    }
    return true;
  }
  return false;
}

/** lines: OCR text lines; drugs: [{id, name, aliases?}] -> [{drug, line, name}] one per recognised drug. */
export function matchOcrLines(lines, drugs) {
  const found = new Map();
  const keys = [];
  for (const d of drugs) {
    for (const name of [d.name, ...(d.aliases || [])]) {
      const toks = nameTokens(name);
      if (toks.length && toks.join("").length >= 6) keys.push({ d, name, toks });
    }
  }
  for (const raw of lines) {
    const lineToks = lineTokens(raw);
    if (!lineToks.length) continue;
    for (const { d, name, toks } of keys) {
      if (!found.has(d.id) && sequenceIn(lineToks, toks)) found.set(d.id, { drug: d, line: raw.trim(), name });
    }
  }
  return [...found.values()];
}
