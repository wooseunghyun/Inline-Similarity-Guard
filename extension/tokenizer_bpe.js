export function normalizeText(s) {
  return s.replace(/\u00a0/g, " ").replace(/[ \t]+/g, " ").trim();
}

export function buildMergeRank(merges) {
  // merges: [[a,b], ...] in order
  const rank = new Map();
  merges.forEach((pair, i) => rank.set(pair[0] + "\u0001" + pair[1], i));
  return rank;
}

function bpeEncodeWord(word, tok, vocab, mergeRank) {
  // char-level + </w>
  let tokens = Array.from(word);
  tokens.push(tok.end_of_word);

  // 반복적으로 "가장 낮은 rank" pair를 병합 (간단 구현)
  // MVP에서는 merges 전체를 순서대로 적용해도 되지만, 비용이 큼.
  // 여기서는 "순서대로 적용"으로 단순화(서버와 동일 방식)
  for (const [a, b] of tok.merges) {
    const merged = a + b;
    const out = [];
    for (let i = 0; i < tokens.length; ) {
      if (i < tokens.length - 1 && tokens[i] === a && tokens[i + 1] === b) {
        out.push(merged);
        i += 2;
      } else {
        out.push(tokens[i]);
        i += 1;
      }
    }
    tokens = out;
  }

  // remove </w>, map OOV -> [UNK]
  const res = [];
  for (const t of tokens) {
    if (t === tok.end_of_word) continue;
    res.push(vocab.hasOwnProperty(t) ? t : tok.unk_token);
  }
  return res;
}

export function bpeEncodeText(text, tok, vocab) {
  const t = normalizeText(text);
  if (!t) return [];
  const mergeRank = null; // MVP에서는 순차 적용이라 rank 불필요
  const out = [];
  for (const w of t.split(" ")) {
    if (!w) continue;
    out.push(...bpeEncodeWord(w, tok, vocab, mergeRank));
  }
  return out;
}
