import os, re, json, math, struct, hashlib, random
from collections import Counter, defaultdict
from pathlib import Path

# 필요 라이브러리: gensim
from gensim.models import Word2Vec

# -----------------------------
# Config (MVP 기본값)
# -----------------------------
CORPUS_DIR = Path("corpus")
OUT_DIR = Path("out")

VOCAB_SIZE = 20000          # subword vocab
DIM = 64                    # embedding dim (브라우저-friendly)
WINDOW = 5
MIN_COUNT = 2
NEGATIVE = 8
EPOCHS = 10

ANCHORS_PER_CLASS = 120     # 클래스당 앵커 수(청크)
CHUNK_CHARS = 350           # 한 앵커 텍스트 길이(문자 수 기준)
SEED = 42

# POS(민감), NEG(일반) 구분을 위해 명시
SENSITIVE_CLASSES = ["극비", "기밀", "대외비"]
NEGATIVE_CLASS = "일반"
CLASSES = SENSITIVE_CLASSES + [NEGATIVE_CLASS]

# -----------------------------
# Utilities
# -----------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def normalize_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def split_into_chunks(text: str, chunk_chars: int):
    text = text.strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_chars]
        chunks.append(chunk)
        i += chunk_chars
    return chunks

# -----------------------------
# 1) Minimal BPE Trainer (toy but workable for MVP)
# -----------------------------
def bpe_train(texts, target_vocab_size=20000, num_merges=18000):
    vocab = Counter()
    for t in texts:
        for w in t.split():
            chars = list(w) + ["</w>"]
            vocab[tuple(chars)] += 1

    def get_pair_stats(vocab_counter):
        pairs = Counter()
        for word, freq in vocab_counter.items():
            for i in range(len(word)-1):
                pairs[(word[i], word[i+1])] += freq
        return pairs

    merges = []
    for _ in range(num_merges):
        pairs = get_pair_stats(vocab)
        if not pairs:
            break
        (a, b), _ = pairs.most_common(1)[0]
        merges.append([a, b])

        new_vocab = Counter()
        ab = a + b
        for word, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and word[i] == a and word[i+1] == b:
                    new_word.append(ab)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] += freq
        vocab = new_vocab

        symbols = set()
        for w in vocab:
            symbols.update(w)
        if len(symbols) >= target_vocab_size:
            break

    symbols = Counter()
    for word, freq in vocab.items():
        for s in word:
            symbols[s] += freq

    special = ["[UNK]"]
    most_common = [s for s, _ in symbols.most_common(target_vocab_size - len(special))]
    final_vocab = special + most_common
    vocab_map = {tok: i for i, tok in enumerate(final_vocab)}

    tok = {
        "version": "bpe-1",
        "unk_token": "[UNK]",
        "end_of_word": "</w>",
        "merges": merges,
        "normalization": {"collapse_spaces": True}
    }
    return tok, vocab_map

def bpe_encode_word(word, tokenizer, vocab_map):
    tokens = list(word) + [tokenizer["end_of_word"]]
    merges = tokenizer["merges"]
    for a, b in merges:
        merged = a + b
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens)-1 and tokens[i] == a and tokens[i+1] == b:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    tokens = [t for t in tokens if t != tokenizer["end_of_word"]]
    return [t if t in vocab_map else tokenizer["unk_token"] for t in tokens]

def bpe_encode_text(text, tokenizer, vocab_map):
    out = []
    for w in text.split():
        out.extend(bpe_encode_word(w, tokenizer, vocab_map))
    return out

# -----------------------------
# 2) Labeling (MVP)
# -----------------------------
def infer_class_from_filename_or_text(fname: str, text: str) -> str:
    # 1) 파일명에 클래스가 들어가면 그걸 최우선
    for c in CLASSES:
        if c in fname:
            return c

    # 2) heuristic keywords (데모용)
    rules = {
        "일반": ["일반", "FAQ", "SLA", "운영", "고객지원", "응답", "안내", "공지", "문의", "템플릿"],
        "극비": ["극비", "비닉", "최상위", "핵심기술", "무기체계", "추진", "엔진", "방호설계"],
        "기밀": ["기밀", "계약", "도면", "설계", "가이드라인", "사양", "시험", "성능"],
        "대외비": ["대외비", "입찰", "제안", "인프라", "협력사", "견적", "공문", "회의"]
    }
    score = {c: 0 for c in CLASSES}
    for c, kws in rules.items():
        for kw in kws:
            if kw in fname or kw in text:
                score[c] += 1

    best = max(score.items(), key=lambda x: x[1])[0]
    return best

# -----------------------------
# 3) Build IDF (doc-level)
# -----------------------------
def build_idf(tokenized_docs):
    N = len(tokenized_docs)
    df = Counter()
    for toks in tokenized_docs:
        df.update(set(toks))
    idf = {}
    for tok, d in df.items():
        idf[tok] = math.log((N + 1) / (d + 1)) + 1.0
    return idf

# -----------------------------
# 4) Compute doc vector via TF-IDF weighted mean
# -----------------------------
def compute_doc_vec(tokens, w2v, idf, dim):
    tf = Counter(tokens)
    num = [0.0] * dim
    denom = 0.0
    for tok, f in tf.items():
        if tok not in w2v.wv:
            continue
        w = f * idf.get(tok, 0.0)
        v = w2v.wv[tok]
        for i in range(dim):
            num[i] += w * float(v[i])
        denom += w

    if denom <= 1e-12:
        return [0.0] * dim

    vec = [x / denom for x in num]
    norm = math.sqrt(sum(x*x for x in vec)) + 1e-12
    return [x / norm for x in vec]

def write_emb_bin(vocab_map, w2v, dim, out_path: Path):
    V = len(vocab_map)
    arr = bytearray()
    inv = [None] * V
    for tok, idx in vocab_map.items():
        inv[idx] = tok
    for idx in range(V):
        tok = inv[idx]
        if tok in w2v.wv:
            vec = [float(x) for x in w2v.wv[tok]]
        else:
            vec = [0.0] * dim
        arr.extend(struct.pack("<" + "f"*dim, *vec))
    out_path.write_bytes(arr)
    return bytes(arr)

def main():
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(CORPUS_DIR.glob("*.txt"))
    if not files:
        raise SystemExit("corpus/*.txt 가 없습니다.")

    raw_docs = []
    meta = []
    for fp in files:
        txt = normalize_text(fp.read_text(encoding="utf-8", errors="ignore"))
        if not txt:
            continue
        cls = infer_class_from_filename_or_text(fp.name, txt)
        raw_docs.append(txt)
        meta.append({"file": fp.name, "class": cls})

    # chunk for training + anchors 후보
    chunks = []
    chunk_meta = []
    for doc, m in zip(raw_docs, meta):
        for ch in split_into_chunks(doc, CHUNK_CHARS):
            chunks.append(ch)
            chunk_meta.append({
                "class": m["class"],
                "src": m["file"]
            })

    # train tokenizer + vocab
    tokenizer, vocab_map = bpe_train(chunks, target_vocab_size=VOCAB_SIZE)
    (OUT_DIR / "tokenizer.json").write_text(json.dumps(tokenizer, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "vocab.json").write_text(json.dumps(vocab_map, ensure_ascii=False), encoding="utf-8")

    # tokenize chunks for training
    token_seqs = [bpe_encode_text(ch, tokenizer, vocab_map) for ch in chunks]

    # train word2vec
    w2v = Word2Vec(
        sentences=token_seqs,
        vector_size=DIM,
        window=WINDOW,
        min_count=MIN_COUNT,
        negative=NEGATIVE,
        sg=1,
        workers=4,
        epochs=EPOCHS
    )

    # idf
    idf = build_idf(token_seqs)
    (OUT_DIR / "idf.json").write_text(json.dumps(idf, ensure_ascii=False), encoding="utf-8")

    # anchors (vectorize chunk)
    per_class = defaultdict(list)
    for ch, m, toks in zip(chunks, chunk_meta, token_seqs):
        vec = compute_doc_vec(toks, w2v, idf, DIM)
        per_class[m["class"]].append({
            "src": m["src"],
            "text": ch,
            "vec": vec
        })

    # 샘플 편향 줄이기: 클래스별 셔플 후 상위 N개
    anchors = []
    for cls in CLASSES:
        items = per_class.get(cls, [])
        if not items:
            continue
        random.shuffle(items)

        group = "NEG" if cls == NEGATIVE_CLASS else "POS"
        for i, it in enumerate(items[:ANCHORS_PER_CLASS]):
            anchors.append({
                "anchor_id": f"{cls}-{i:03d}",
                "class": cls,
                "group": group,           # POS(민감) / NEG(일반)
                "title": it["src"],       # 파일명
                "text": it["text"],       # ✅ 비교 대상 텍스트(청크 원문)
                "vec": it["vec"]          # 임베딩 벡터
            })

    anchors_obj = {
        "anchors": anchors,
        "schema_version": 2,
        "note": "Anchors include both sensitive(POS) and general(NEG) chunks."
    }
    (OUT_DIR / "anchors.json").write_text(
        json.dumps(anchors_obj, ensure_ascii=False),
        encoding="utf-8"
    )

    # emb.bin
    emb_bytes = write_emb_bin(vocab_map, w2v, DIM, OUT_DIR / "emb.bin")

    # manifest.json
    def file_info(name):
        p = OUT_DIR / name
        b = p.read_bytes()
        return {"path": name, "sha256": sha256_bytes(b), "bytes": len(b)}

    manifest = {
        "model_id": "sec-embed-ko",
        "version": "local-build-1",
        "created_at": "LOCAL",
        "vocab_size": len(vocab_map),
        "dim": DIM,
        "tokenizer_version": tokenizer["version"],
        "thresholds": {
            "HIGH_POS": 0.72,
            "MED_POS": 0.60,
            "HIGH_MARGIN": 0.12,
            "MED_MARGIN": 0.06
        },
        "files": {
            "tokenizer": file_info("tokenizer.json"),
            "vocab": file_info("vocab.json"),
            "idf": file_info("idf.json"),
            "emb": {"path": "emb.bin", "sha256": sha256_bytes(emb_bytes), "bytes": len(emb_bytes)},
            "anchors": file_info("anchors.json")
        }
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("DONE. out/ 에 산출물 생성 완료")
    print(f"- anchors.json anchors: {len(anchors)}")
    print(f"- classes: {CLASSES}")

if __name__ == "__main__":
    main()
