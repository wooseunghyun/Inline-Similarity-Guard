import { idbGet, idbPut } from "./idb.js";
import { bpeEncodeText } from "./tokenizer_bpe.js";
import { dot, normalize } from "./vec.js";

// ===== 설정 =====
const MODEL_BASE = "https://2dtfinaltrackbsa.blob.core.windows.net/word-embedding-out/"; // out/를 올려둔 경로(끝에 /)
const MANIFEST_URL = MODEL_BASE + "manifest.json";

// ===== 런타임 캐시(메모리) =====
let model = null; // { tok, vocab, idf, embBuf, embView, anchors, dim, thresholds }

// ----- sha256 (WebCrypto) -----
async function sha256Hex(buffer) {
  const hash = await crypto.subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(hash)].map(b => b.toString(16).padStart(2, "0")).join("");
}

async function fetchJson(url) {
  const res = await fetch(url, { cache: "no-store" });
  console.log("[Guard] fetchJson", url, res.status);
  if (!res.ok) throw new Error("fetch failed: " + url + " status=" + res.status);
  return await res.json();
}

async function fetchText(url) {
  const res = await fetch(url, { cache: "no-store" });
  console.log("[Guard] fetchText", url, res.status);
  if (!res.ok) throw new Error("fetch failed: " + url + " status=" + res.status);
  return await res.text();
}

async function fetchBytes(url) {
  const res = await fetch(url, { cache: "no-store" });
  console.log("[Guard] fetchBytes", url, res.status);
  if (!res.ok) throw new Error("fetch failed: " + url + " status=" + res.status);
  return await res.arrayBuffer();
}

// ===== 정규화/룰/스트리핑 =====
function normalizeText(s) {
  return (s || "")
    .normalize("NFKC")
    .replace(/[\u200B-\u200D\uFEFF]/g, "") // zero-width
    .replace(/\s+/g, " ")
    .trim();
}

function stripSecurityLabels(s) {
  const t = normalizeText(s);

  const reWords =
    /(극\s*비|기\s*밀|대\s*외\s*비|top\s*secret|confidential|internal\s*use\s*only)/gi;

  const reBracketed =
    /[\[({【「『<＜]\s*(극\s*비|기\s*밀|대\s*외\s*비|top\s*secret|confidential|internal\s*use\s*only)\s*[\])}】」』>＞]/gi;

  const reHeading =
    /(^|\n)\s*(극\s*비|기\s*밀|대\s*외\s*비|top\s*secret|confidential|internal\s*use\s*only)\s*[:\-—–]/gim;

  let out = t.replace(reBracketed, " ");
  out = out.replace(reHeading, "\n");
  out = out.replace(reWords, " ");
  out = out.replace(/\s+/g, " ").trim();
  return out;
}

function applyLabelRules(rawText) {
  const t = normalizeText(rawText);

  const hit = {
    hasLabel: false,
    label: null,
    boostScore: 0,    // 룰이 제공하는 score 바이어스(작게)
    minLevel: null,   // 최소 보장 레벨
    reasons: []
  };

  // NOTE:
  // - "극비" 단독은 HIGH로 올리지 않습니다(과잉경고 방지)
  // - 최소 MED + 작은 score bias만 제공합니다.
  const rules = [
    { label: "극비", re: /(^|[^가-힣A-Za-z0-9])극\s*비([^가-힣A-Za-z0-9]|$)/i, boostScore: 0.15, minLevel: "MED" },
    { label: "기밀", re: /(^|[^가-힣A-Za-z0-9])기\s*밀([^가-힣A-Za-z0-9]|$)/i, boostScore: 0.10, minLevel: "MED" },
    { label: "대외비", re: /(^|[^가-힣A-Za-z0-9])대\s*외\s*비([^가-힣A-Za-z0-9]|$)/i, boostScore: 0.08, minLevel: "MED" },

    { label: "극비", re: /\btop\s*secret\b/i, boostScore: 0.15, minLevel: "MED" },
    { label: "기밀", re: /\bconfidential\b/i, boostScore: 0.10, minLevel: "MED" },
    { label: "대외비", re: /\binternal\s*use\s*only\b/i, boostScore: 0.08, minLevel: "MED" }
  ];

  for (const r of rules) {
    if (r.re.test(t)) {
      hit.hasLabel = true;
      hit.label = hit.label || r.label;
      hit.boostScore = Math.max(hit.boostScore, r.boostScore);
      hit.minLevel = hit.minLevel || r.minLevel;
      hit.reasons.push(`label:${r.label}`);
    }
  }

  return hit;
}

function combineScores(ruleBoost, embScore) {
  const r = Math.min(Math.max(ruleBoost, 0), 1);
  const e = Math.min(Math.max(embScore, 0), 1);
  // 1 - (1-r)(1-e): 과도하게 1.0에 붙지 않게 하면서 결합
  return 1 - (1 - r) * (1 - e);
}

// ===== 모델 로드(IDB) =====
async function loadModelFromIDB() {
  const meta = await idbGet("meta", "current");
  if (!meta) return null;

  const tok = await idbGet("files", "tokenizer.json");
  const vocab = await idbGet("files", "vocab.json");
  const idf = await idbGet("files", "idf.json");
  const anchorsObj = await idbGet("files", "anchors.json");
  const embBuf = await idbGet("files", "emb.bin");

  if (!tok || !vocab || !idf || !anchorsObj || !embBuf) return null;

  const dim = meta.dim;
  const thresholds = meta.thresholds;

  const anchors = anchorsObj.anchors.map(a => ({
    ...a,
    vec: a.vec // server에서 정규화되어 있다고 가정(아니면 여기서 normalize 가능)
  }));

  return {
    tok, vocab, idf, anchors,
    embBuf,
    embView: new DataView(embBuf),
    dim,
    vocabSize: meta.vocab_size,
    thresholds
  };
}

function getEmbVec(idx, dim, embView) {
  const off = idx * dim * 4;
  const v = new Float32Array(dim);
  for (let i = 0; i < dim; i++) {
    v[i] = embView.getFloat32(off + i * 4, true);
  }
  return v;
}

function computeDocVec(tokens, vocab, idf, dim, embView) {
  // TF-IDF weighted mean
  const tf = new Map();
  for (const t of tokens) tf.set(t, (tf.get(t) || 0) + 1);

  const vec = new Float32Array(dim);
  let denom = 0;

  for (const [tok, f] of tf.entries()) {
    const idx = vocab[tok];
    if (idx === undefined) continue;

    const w = f * (idf[tok] || 0);
    if (w <= 0) continue;

    const e = getEmbVec(idx, dim, embView);
    for (let i = 0; i < dim; i++) vec[i] += w * e[i];
    denom += w;
  }

  if (denom <= 1e-9) return normalize(vec);
  for (let i = 0; i < dim; i++) vec[i] /= denom;
  return normalize(vec);
}

function topK(arr, k, keyFn) {
  const out = [];
  for (const x of arr) out.push(x);
  out.sort((a, b) => keyFn(b) - keyFn(a));
  return out.slice(0, k);
}

function riskLevel(score, thresholds) {
  if (score >= thresholds.HIGH) return "HIGH";
  if (score >= thresholds.MED) return "MED";
  return "LOW";
}

async function ensureModelLoaded() {
  if (model) return model;
  const m = await loadModelFromIDB();
  model = m;
  return model;
}

// ===== 모델 업데이트 =====
async function updateModelIfNeeded() {
  console.log("[Guard] fetching manifest:", MANIFEST_URL);

  const remote = await fetchJson(MANIFEST_URL);
  const localMeta = await idbGet("meta", "current");

  if (localMeta && localMeta.version === remote.version) {
    return { updated: false, version: remote.version };
  }

  const files = remote.files;

  // ✅ 올바른 방식: 원본 bytes로 해시(텍스트는 text로 받아 그대로 인코딩)
  const tokText = await fetchText(MODEL_BASE + files.tokenizer.path);
  const vocabText = await fetchText(MODEL_BASE + files.vocab.path);
  const idfText = await fetchText(MODEL_BASE + files.idf.path);
  const anchorsText = await fetchText(MODEL_BASE + files.anchors.path);
  const embBuf = await fetchBytes(MODEL_BASE + files.emb.path);

  const tokBuf = new TextEncoder().encode(tokText);
  const vocabBuf = new TextEncoder().encode(vocabText);
  const idfBuf = new TextEncoder().encode(idfText);
  const anchorsBuf = new TextEncoder().encode(anchorsText);

  // verify
  const tokHash = await sha256Hex(tokBuf.buffer);
  const vocabHash = await sha256Hex(vocabBuf.buffer);
  const idfHash = await sha256Hex(idfBuf.buffer);
  const anchorsHash = await sha256Hex(anchorsBuf.buffer);
  const embHash = await sha256Hex(embBuf);

  if (tokHash !== files.tokenizer.sha256) throw new Error("tokenizer hash mismatch");
  if (vocabHash !== files.vocab.sha256) throw new Error("vocab hash mismatch");
  if (idfHash !== files.idf.sha256) throw new Error("idf hash mismatch");
  if (anchorsHash !== files.anchors.sha256) throw new Error("anchors hash mismatch");
  if (embHash !== files.emb.sha256) throw new Error("emb hash mismatch");

  // save JSON objects + emb buffer (중복 fetch 제거)
  await idbPut("files", "tokenizer.json", JSON.parse(tokText));
  await idbPut("files", "vocab.json", JSON.parse(vocabText));
  await idbPut("files", "idf.json", JSON.parse(idfText));
  await idbPut("files", "anchors.json", JSON.parse(anchorsText));
  await idbPut("files", "emb.bin", embBuf);

  await idbPut("meta", "current", {
    version: remote.version,
    vocab_size: remote.vocab_size,
    dim: remote.dim,
    thresholds: remote.thresholds
  });

  model = null; // reload
  return { updated: true, version: remote.version };
}

// alarms: 하루 1회 업데이트 체크
chrome.runtime.onInstalled.addListener(async () => {
  chrome.alarms.create("model_update", { periodInMinutes: 60 * 24 });
  try {
    await updateModelIfNeeded();
  } catch (e) {
    console.error("[Guard] updateModelIfNeeded failed:", e);
  }
});

chrome.alarms.onAlarm.addListener(async (alarm) => {
  if (alarm.name !== "model_update") return;
  try {
    await updateModelIfNeeded();
  } catch (e) {
    console.error("[Guard] updateModelIfNeeded failed:", e);
  }
});

// 메시지 처리: content.js -> sw
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    // ---- 모델 강제 갱신 ----
    if (msg.type === "UPDATE_MODEL_NOW") {
      try {
        const r = await updateModelIfNeeded();
        sendResponse({ ok: true, ...r });
      } catch (e) {
        console.error("[Guard] UPDATE_MODEL_NOW failed:", e);
        sendResponse({ ok: false, error: String(e?.message || e) });
      }
      return;
    }

    // ---- 텍스트 검사 ----
    if (msg.type === "CHECK_TEXT") {
      const m = await ensureModelLoaded();
      if (!m) {
        sendResponse({ ok: false, error: "MODEL_NOT_READY" });
        return;
      }

      const rawText = msg.text || "";

      // 1) 룰: 등급 단어는 여기서만 반영
      const ruleHit = applyLabelRules(rawText);

      // 2) 임베딩용 텍스트: 등급 표식 제거
      const contentText = stripSecurityLabels(rawText);

      // 3) 토큰화/임베딩은 “내용”으로만
      const tokens = bpeEncodeText(contentText, m.tok, m.vocab);

      // 디버깅 필요하면 아래 주석 해제
      // console.log("TEXT(raw):", rawText);
      // console.log("TEXT(content):", contentText);
      // console.log("TOKENS sample:", tokens.slice(0, 50));
      // console.log("Known token count:", tokens.filter(t => m.vocab[t] !== undefined).length);

      const docVec = computeDocVec(tokens, m.vocab, m.idf, m.dim, m.embView);

      const scored = m.anchors.map(a => ({
        class: a.class,
        anchor_id: a.anchor_id,
        title: a.title,
        sim: dot(docVec, a.vec)
      }));

      const topMatches = topK(scored, 3, x => x.sim);
      const embScore = topMatches.length ? topMatches[0].sim : 0;

      // 4) 점수 결합(룰 + 임베딩)
      const finalScore = combineScores(ruleHit.boostScore, embScore);

      // 5) 레벨 결정 + 룰 최소 레벨 보장
      let level = riskLevel(finalScore, m.thresholds);
      if (ruleHit.minLevel === "MED" && level === "LOW") level = "MED";

      sendResponse({
        ok: true,
        risk_score: finalScore,
        risk_level: level,
        emb_score: embScore,
        rule_boost: ruleHit.boostScore,
        reasons: ruleHit.reasons,
        top_matches: topMatches,
        // 디버깅용(원하면 UI에서 숨기거나 제거)
        content_text: contentText
      });
      return;
    }

    sendResponse({ ok: false, error: "UNKNOWN_MSG" });
  })();

  return true; // async response
});
