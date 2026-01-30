# Inline-Similarity-Guard

브라우저 단에서 텍스트 유출 위험을 **즉시(inline)** 탐지하기 위한  
**임베딩 기반 유사도 분석 + 룰 분리형 보안 가드 PoC**입니다.

본 프로젝트는 “단어 하나”가 아닌 **문맥 기반 판단**을 통해  
기밀·대외비·극비 문서의 **실질적인 유출 위험**을 탐지하는 구조를 검증하는 것을 목표로 합니다.

---

## 1. 프로젝트 개요

기존 DLP(Data Loss Prevention) 솔루션은 다음과 같은 한계를 가집니다.

- 키워드 기반 탐지로 인한 과도한 오탐
- 문서 맥락을 고려하지 못하는 정적 규칙
- 사용자 행위 직전(pre-action) 제어의 어려움

**Inline-Similarity-Guard**는 이를 개선하기 위해,

- ✔️ **등급 단어(극비/기밀/대외비)는 룰로 분리**
- ✔️ **본문 내용은 임베딩 기반 유사도 분석**
- ✔️ **브라우저 입력 단계에서 즉시 판단**

하는 구조를 실험합니다.

---

## 2. 핵심 설계 개념

### 🔹 Rule-based + Embedding-based 분리

| 구분 | 역할 |
|---|---|
| 룰 기반 | 등급 표기, 형식적 단어(극비/기밀/대외비 등) |
| 임베딩 기반 | 문서 실제 내용, 업무 맥락, 의미 유사성 |

> 단순히 “대외비”라는 단어가 있다는 이유만으로 위험으로 판단하지 않습니다.  
> **내용이 실제 기밀 문서와 의미적으로 유사한지**를 판단합니다.

---

### 🔹 Anchor 기반 유사도 판단

- 서버에서 기밀 문서를 **요약 + 청크화**
- 각 청크를 **Anchor Vector**로 생성
- 클라이언트(Chrome Extension)는
  - 입력 텍스트를 벡터화
  - Anchor들과 cosine similarity 계산
  - 가장 유사한 Anchor 기준으로 위험도 산정

---

## 3. 전체 아키텍처 개요

```

[기밀 문서 Corpus]
│
▼
[서버 전처리]

* 정규화
* 문서 청크화
* 요약 Anchor 생성
* 임베딩 벡터화
  │
  ▼
  [Blob Storage / CDN]
* tokenizer.json
* vocab.json
* idf.json
* emb.bin
* anchors.json
* manifest.json
  │
  ▼
  [Chrome Extension]
* 입력 텍스트 inline 수집
* Rule 기반 1차 필터
* Embedding 유사도 분석
* Risk Score / Level 표시

```

---

## 4. 리포지토리 구성

```

Inline-Similarity-Guard/
│
├─ server/
│  ├─ corpus/              # 원본 기밀/일반 문서
│  ├─ out/                 # 모델 산출물 (CDN 업로드 대상)
│  └─ sw.py                # 서버측 모델 빌드 스크립트
│
├─ extension/
│  ├─ sw.js                # Service Worker (모델 로딩/판단)
│  ├─ content.js           # 페이지 텍스트 수집
│  ├─ idb.js               # IndexedDB 캐시
│  ├─ tokenizer_bpe.js
│  ├─ vec.js
│  └─ manifest.json
│
├─ .gitignore
├─ README.md

```

---

## 5. 위험도 판단 방식

1. **Rule Layer**
   - 등급 단어, 형식 규칙 감지
   - 경고 트리거 또는 가중치 조정

2. **Embedding Layer**
   - TF-IDF 가중 평균으로 문서 벡터 생성
   - Anchor Vector들과 cosine similarity 계산

3. **Risk Level 결정**
   - HIGH / MED / LOW (threshold 기반)
   - 가장 유사한 Anchor 정보 함께 표시

---

## 6. 왜 클라이언트 단인가?

- 외부 서버로 텍스트를 전송하지 않음
- 입력 직전(pre-action) 제어 가능
- 내부자 위협 시나리오에 적합

> 본 프로젝트는 **“서버로 보내기 전에 막는 것”**에 초점을 둡니다.

---

## 7. 향후 확장 방향

- Sentinel / SIEM 연계
- 사용자 행동 맥락 기반 Risk 보정
- RAG 기반 “의도된 유출 vs 비의도 유출” 분석
- 정책 문서 전용 Anchor 클래스 분리

---

## 8. 주의 사항

- 본 프로젝트는 **개념 검증(PoC)** 목적입니다.
- 실제 운영 환경에서는:
  - 모델 검증
  - 법/보안 정책 검토
  - 오탐/미탐 관리 체계
가 반드시 필요합니다.

---

## 9. 키워드

`DLP` · `Insider Threat` · `Embedding Similarity` ·  
`Chrome Extension Security` · `Azure` · `Sentinel` · `PoC`
```
