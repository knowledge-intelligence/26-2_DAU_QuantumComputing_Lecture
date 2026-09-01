# M04_2_vector_rag.ipynb — 환경 설치·구축·실행 가이드

> 9~11주차 모듈 3(지식) **2/4편**: **문서 전처리**(구조화·청킹·메타데이터) →
> **벡터 DB 3종**(ChromaDB · FAISS · Qdrant) → **LangChain 조립**(PromptTemplate · LCEL · Agent).
> LLM 은 `.env` 의 `LLM_PROVIDER` 를 그대로 사용합니다.

## 0. 전제

- Windows 11 + **CMD(`cmd.exe`)** + Python **3.11**(`uv` 관리)
- 공통 1회 준비는 [README.md](README.md) 를 먼저 따라 하세요(uv 설치 → `uv venv` → `uv sync` → 커널 등록).
- [1편 M04_1](M04_1_embeddings.md) 의 임베딩 개념·`get_embedder()` 를 먼저 보고 오면 좋습니다.
- **서버·Docker 가 필요 없습니다** — 벡터 DB 세 종류 모두 로컬/인메모리로 동작합니다.

## 1. 이 노트북이 필요로 하는 것

| 구분 | 내용 |
|---|---|
| Python 패키지 | `sentence-transformers`, `numpy`, `langchain`, `chromadb`, `faiss-cpu`, `qdrant-client` (첫 셀이 `utils.uv_install()` 로 보강) |
| 임베딩 | `paraphrase-multilingual-MiniLM-L12-v2` (오프라인, 최초 1회 약 470MB) |
| LLM | `.env` 의 `LLM_PROVIDER`. **4장 RAG Agent 는 도구 호출 지원 모델 필요** |
| 실습 문서 | 노트북이 `notebooks/workspace/personal_docs/` 에 마크다운 6개를 자동 생성 |

구현은 아래 세 모듈에 분리되어 있습니다.

| 모듈 | 역할 |
|---|---|
| [`agentic_lib/doc_prep.py`](../agentic_lib/doc_prep.py) | 샘플 문서 생성, front matter·섹션 파싱, 청킹 4전략, 메타데이터 설계, 골드셋 |
| [`agentic_lib/vector_stores.py`](../agentic_lib/vector_stores.py) | Chroma/FAISS/Qdrant 통일 어댑터, 검색 비교, 청킹 전략 평가 |
| [`agentic_lib/lc_rag.py`](../agentic_lib/lc_rag.py) | PromptTemplate, LCEL 체인, 검색 도구, RAG Agent |

## 2. 추가 패키지 설치 (CMD)

`uv sync` 후에도 노트북 첫 셀이 `utils.uv_install([...])` 로 자동 보강합니다.
수동으로 미리 설치하려면 (프로젝트 루트에서):

```bat
REM 문서 전처리 + 임베딩 + LangChain
uv pip install sentence-transformers numpy langchain langchain-text-splitters

REM 벡터 DB 3종
uv pip install chromadb faiss-cpu qdrant-client
```

> `faiss-cpu` 는 Windows/Python 3.11 프리빌트 휠이 있어 컴파일러가 필요 없습니다(import 는 `faiss`).
> `qdrant-client` 는 `QdrantClient(":memory:")` 로 **서버 없이** 동작합니다.

## 3. LLM 설정

```ini
# notebooks/.env
LLM_PROVIDER=ollama          # 또는 google / nvidia / openrouter
```

**4장 RAG Agent 주의**: 검색을 도구로 넘기므로 **도구 호출(tool calling)** 을 지원해야 합니다.

| 공급자 | 도구 호출 | 비고 |
|---|---|---|
| `ollama` + `qwen3:8b` | ✅ | 네이티브 도구 호출, 로컬·무료 |
| `google` (Gemini) | ✅ | 안정적 |
| `nvidia` | ✅ | 단일 도구 호출 |
| `openrouter` (`openrouter/free`) | ⚠️ | 라우팅되는 모델에 따라 다름 — 실패 시 노트북이 안내 후 건너뜀 |

## 4. 실행 순서

1. Jupyter 에서 커널을 **`Agentic AI Tutorial (uv)`** 로 선택
2. 위에서부터 **순서대로** 실행
   - 0장 setup (`uv_install` + 라이브러리 import)
   - 1장 **문서 전처리** — 파일 생성 → 구조화 → 청킹 4전략 비교 →
     **검색 정확도로 전략 선택** → 메타데이터 스키마
   - 2장 **벡터 DB 3종** — 적재 → 같은 질의 비교 → 메타데이터 필터 → 선택 가이드
   - 3장 **PromptTemplate + LCEL** — `invoke` / `batch` / `stream` / 근거 표시 체인
   - 4장 **RAG Agent** — 검색 도구, 반복 검색, 체인과 비교, 에이전트의 함정
3. 첫 임베딩 모델 로딩은 다운로드 시간이 걸릴 수 있습니다(이후 캐시).
4. 실습 문서는 `notebooks/workspace/` 아래에 생성되며 git 에 올라가지 않습니다(`.gitignore` 등록됨).

## 5. 자주 겪는 문제

| 증상 | 원인/해결 |
|---|---|
| `NameError` (변수 없음) | **첫 setup 셀을 실행하지 않은 것**. 위에서부터 순서대로 실행하세요 |
| `ModuleNotFoundError: faiss` | 패키지명이 다릅니다 → `uv pip install faiss-cpu` |
| `ModuleNotFoundError: qdrant_client` | `uv pip install qdrant-client` |
| Chroma `Collection already exists` | 어댑터가 생성 전에 삭제합니다. 그래도 나면 커널 재시작 후 재실행 |
| FAISS 필터 결과가 적음 | **정상 동작**. FAISS 는 후처리 필터라 넉넉히 뽑아야 합니다(어댑터가 k×10 으로 처리) |
| 세 백엔드 Top-1 이 다름 | 임베딩이 다르거나 거리 척도(cosine/L2)가 다릅니다. 어댑터는 셋 다 코사인으로 맞춰 둡니다 |
| 에이전트가 "검색 0회"로 답함 | 모델이 도구를 무시한 것. 답변이 문서 밖 일반론이면 신뢰하지 마세요(노트북 4-2절) |
| 에이전트 생성/실행 실패 | 도구 호출 미지원 공급자 → `.env` 의 `LLM_PROVIDER` 변경 |
| 답변이 실행할 때마다 달라짐 | `openrouter/free` 는 요청마다 다른 모델로 라우팅됩니다 → 고정 모델 공급자 사용 |

## 6. 명령 요약 (복붙용)

```bat
REM 패키지 (한 줄)
uv pip install sentence-transformers numpy langchain langchain-text-splitters chromadb faiss-cpu qdrant-client

REM .env 준비 (notebooks/ 에서 1회)
cd notebooks && copy .env.example .env
REM .env 에서 LLM_PROVIDER 확인 (4장은 도구 호출 지원 모델 권장)

uv run jupyter notebook
```
