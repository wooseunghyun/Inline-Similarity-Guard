import { idbGet, idbPut } from "./idb.js";
import { bpeEncodeText } from "./tokenizer_bpe.js";
import { dot, normalize } from "./vec.js";

// ===== 설정 =====
const MODEL_BASE = "https://2dtfinaltrackbsa.blob.core.windows.net/word-embedding-out/"; // 끝에 /
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

async function fetchArrayBuffer(url) {
  const res = await fetch(url, { cache: "no-store" });
  console.log("[Guard] fetchArrayBuffer", url, res.status);
  if (!res.ok) throw new Error("fetch failed: " + url + " status=" + res.status);
  return await res.arrayBuffer();
}

async function fetchText(url) {
  const res = await fetch(url, { cache: "no-store" });
  console.log("[Guard] fetchText", url, res.status);
  if (!res.ok) throw new Error("fetch failed: " + url + " status=" + res.status);
  return await res.text();
}

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
  const thresholds = meta.thresholds || {};
  const anchorsRaw = anchorsObj.anchors || [];

  // ✅ anchors schema v2: {anchor_id,class,group,title,text,vec}
  const anchors = anchorsRaw.map(a => ({
    anchor_id: a.anchor_id,
    class: a.class,
    group: a.group || (a.class === "일반" ? "NEG" : "POS"),
    title: a.title || "",
    text: a.text || "",
    vec: a.vec
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

// ===== 룰 분리: 등급 키워드는 별도 체크 =====
// (등급 단어가 들어갔다고 해서 "내용 유출"은 아닐 수 있으니,
//  이건 rule_hit로만 표시하고, 임베딩 입력에서 제거(옵션))
const CLASS_WORDS = ["극비", "기밀", "대외비"];
function detectClassWords(text) {
  const hits = [];
  for (const w of CLASS_WORDS) {
    if (text.includes(w)) hits.push(w);
  }
  return hits;
}

function stripClassWords(text) {
  // 너무 과격하게 지우면 문장이 망가질 수 있으니, 단순 치환만
  let t = text;
  for (const w of CLASS_WORDS) t = t.split(w).join(" ");
  return t;
}

// ===== 위험도 계산 (POS vs NEG, margin 기반) =====
// 기본값: (나중에 검증셋으로 튜닝 권장)
function riskLevelFromScores(posMax, margin, thresholds) {
  // thresholds에 margin 기반이 있으면 우선 사용
  const t = thresholds || {};
  const HIGH_POS = t.HIGH_POS ?? t.HIGH ?? 0.72;
  const MED_POS  = t.MED_POS  ?? t.MED  ?? 0.60;
  const HIGH_MARGIN = t.HIGH_MARGIN ?? 0.12;
  const MED_MARGIN  = t.MED_MARGIN  ?? 0.06;

  // "민감으로 강하게 붙고" + "일반보다 확실히 더 민감" 이면 HIGH
  if (posMax >= HIGH_POS && margin >= HIGH_MARGIN) return "HIGH";
  if (posMax >= MED_POS && margin >= MED_MARGIN) return "MED";
  return "LOW";
}

async function ensureModelLoaded() {
  if (model) return model;
  const m = await loadModelFromIDB();
  model = m;
  return model;
}

// ----- 모델 업데이트 -----
async function updateModelIfNeeded() {
  console.log("[Guard] fetching manifest:", MANIFEST_URL);
  const remote = await fetchJson(MANIFEST_URL);
  const localMeta = await idbGet("meta", "current");

  if (localMeta && localMeta.version === remote.version) {
    return { updated: false, version: remote.version };
  }

  const files = remote.files;

  // ✅ 원본 텍스트 bytes 기준 해시
  const tokText = await fetchText(MODEL_BASE + files.tokenizer.path);
  const vocabText = await fetchText(MODEL_BASE + files.vocab.path);
  const idfText = await fetchText(MODEL_BASE + files.idf.path);
  const anchorsText = await fetchText(MODEL_BASE + files.anchors.path);
  const embBuf = await fetchArrayBuffer(MODEL_BASE + files.emb.path);

  const tokBuf = new TextEncoder().encode(tokText);
  const vocabBuf = new TextEncoder().encode(vocabText);
  const idfBuf = new TextEncoder().encode(idfText);
  const anchorsBuf = new TextEncoder().encode(anchorsText);

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

  // save JSON objects + emb buffer
  await idbPut("files", "tokenizer.json", JSON.parse(tokText));
  await idbPut("files", "vocab.json", JSON.parse(vocabText));
  await idbPut("files", "idf.json", JSON.parse(idfText));
  await idbPut("files", "anchors.json", JSON.parse(anchorsText));
  await idbPut("files", "emb.bin", embBuf);

  await idbPut("meta", "current", {
    version: remote.version,
    vocab_size: remote.vocab_size,
    dim: remote.dim,
    thresholds: remote.thresholds || {}
  });

  model = null; // reload
  return { updated: true, version: remote.version };
}

// alarms: 하루 1회 업데이트 체크
chrome.runtime.onInstalled.addListener(async () => {
  chrome.alarms.create("model_update", { periodInMinutes: 60 * 24 });
  try { await updateModelIfNeeded(); } catch (e) { console.error("[Guard] updateModelIfNeeded failed:", e); }
});

chrome.alarms.onAlarm.addListener(async (alarm) => {
  if (alarm.name !== "model_update") return;
  try { await updateModelIfNeeded(); } catch (e) { console.error("[Guard] updateModelIfNeeded failed:", e); }
});

// ===== 핵심: POS/NEG 둘 다 비교해서 위험도 계산 =====
function scoreAgainstAnchors(docVec, anchors) {
  const scored = anchors.map(a => ({
    group: a.group || (a.class === "일반" ? "NEG" : "POS"),
    class: a.class,
    anchor_id: a.anchor_id,
    title: a.title,
    text: a.text || "",
    sim: dot(docVec, a.vec)
  }));

  const pos = scored.filter(x => x.group === "POS");
  const neg = scored.filter(x => x.group === "NEG");

  const topPos = topK(pos, 3, x => x.sim);
  const topNeg = topK(neg, 3, x => x.sim);

  const posMax = topPos.length ? topPos[0].sim : 0;
  const negMax = topNeg.length ? topNeg[0].sim : 0;
  const margin = posMax - negMax;

  return { topPos, topNeg, posMax, negMax, margin };
}

// 메시지 처리: content.js -> sw
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    try {
      if (msg.type === "CHECK_TEXT") {
        const m = await ensureModelLoaded();
        if (!m) {
          sendResponse({ ok: false, error: "MODEL_NOT_READY" });
          return;
        }

        const rawText = (msg.text || "");
        const ruleHits = detectClassWords(rawText);

        // ✅ 등급 단어는 룰로 분리, 임베딩 입력에서는 제거(원하던 의도)
        const embedText = stripClassWords(rawText);

        const tokens = bpeEncodeText(embedText, m.tok, m.vocab);
        const docVec = computeDocVec(tokens, m.vocab, m.idf, m.dim, m.embView);

        const { topPos, topNeg, posMax, negMax, margin } = scoreAgainstAnchors(docVec, m.anchors);
        const level = riskLevelFromScores(posMax, margin, m.thresholds);

        // UI에 보여줄 top matches는 POS 위주로(민감 후보)
        const topMatches = topPos.map(x => ({
          class: x.class,
          anchor_id: x.anchor_id,
          title: x.title,
          sim: x.sim,
          // ✅ “어떤 내용이랑 비슷한지”를 보여주려면 text가 필요
          text_preview: (x.text || "").slice(0, 160)
        }));

        sendResponse({
          ok: true,
          // 점수는 posMax를 기본 risk_score로 쓰되, margin도 같이 제공
          risk_score: posMax,
          risk_level: level,

          pos_max: posMax,
          neg_max: negMax,
          margin: margin,

          rule_hits: ruleHits,     // 등급 키워드 발견 여부(룰 기반)
          top_matches: topMatches,

          // 디버깅용: NEG도 보고 싶으면 UI에서 활용 가능
          top_neg_matches: topNeg.map(x => ({
            class: x.class,
            anchor_id: x.anchor_id,
            title: x.title,
            sim: x.sim,
            text_preview: (x.text || "").slice(0, 120)
          }))
        });
        return;
      }

      if (msg.type === "UPDATE_MODEL_NOW") {
        const r = await updateModelIfNeeded();
        sendResponse({ ok: true, ...r });
        return;
      }

      sendResponse({ ok: false, error: "UNKNOWN_MSG" });
    } catch (e) {
      console.error("[Guard] handler error:", e);
      sendResponse({ ok: false, error: String(e?.message || e) });
    }
  })();

  return true; // async response
});
