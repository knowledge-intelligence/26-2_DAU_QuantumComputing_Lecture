# M04_4_agent_memory.ipynb — 환경 설치·구축·실행 가이드

> 9~11주차 모듈 3(지식) **4/4편**: **에이전트 메모리**(단기/에피소딕/시맨틱) →
> **Deep-Knowledge Agent**(Hybrid RAG + 메모리 + Multi-hop 추론). 기본 LLM 은
> **로컬 Ollama + `qwen3:8b`**, 비교용은 클라우드 **Google(Gemini)**.

## 0. 전제

- Windows 11 + **CMD(`cmd.exe`)** + Python **3.11**(`uv` 관리)
- 공통 1회 준비는 [README.md](README.md) 를 먼저 따라 하세요(uv 설치 → `uv venv` → `uv sync` → 커널 등록).
- 기본 LLM(로컬 Ollama + `qwen3:8b`) 준비는 [M01_1_intro.md](M01_1_intro.md) 와 동일합니다.
- 이 노트북은 **자기완결**입니다 — 에이전트가 쓰는 [2편](../M04_2_vector_rag.ipynb)·[3편](../M04_3_graph_rag.ipynb)의 Vector RAG · 지식 그래프 · Hybrid RAG 를 앞부분에서 재구성합니다.

## 1. 이 노트북이 필요로 하는 것

| 구분 | 내용 |
|---|---|
| Python 패키지 | `chromadb`, `sentence-transformers`, `numpy` (Vector/Hybrid RAG 재구성 + 에피소딕 메모리용) |
| 로컬 LLM | **Ollama** 서버 + **`qwen3:8b`** 모델 |
| 임베딩(오프라인) | `sentence-transformers` 의 `paraphrase-multilingual-MiniLM-L12-v2` (최초 1회 다운로드 후 **오프라인** 동작) |
| (선택) 클라우드 비교 | `GOOGLE_API_KEY` (Gemini 무료 티어) |

> 메모리 프리미티브는 [`agentic_lib/memory.py`](../agentic_lib/memory.py), RAG/그래프는
> [`agentic_lib/rag.py`](../agentic_lib/rag.py) 로 분리되어 있습니다. LLM 응답은
> `bootstrap.to_text()` 로 정규화됩니다. 이 노트북의 4계층 메모리(`ShortTermMemory`·`EpisodicMemory`·
> `SemanticMemory`·`AgentMemorySystem`)와 `DeepKnowledgeAgent` 는
> [`agentic_lib/agent_memory.py`](../agentic_lib/agent_memory.py) 에 있고, 노트북은 `am` 으로 import 해
> **조립하고 관찰** 합니다(설명은 노트북 6·7절의 표 참고).

## 2. 추가 패키지 설치 (CMD)

`uv sync` 후에도 노트북 첫 셀이 `utils.uv_install([...])` 로 아래를 자동 보강합니다.
수동으로 미리 설치하려면 (프로젝트 루트에서):

```bat
REM Vector/Hybrid RAG 재구성 + 에피소딕 메모리(벡터 저장) 필수 패키지
uv pip install chromadb sentence-transformers numpy
```

> `sentence-transformers` 최초 import 시 임베딩 모델(약 470MB)을 한 번 내려받습니다.
> 이후에는 캐시되어 **네트워크 없이** 동작합니다(`%USERPROFILE%\.cache\huggingface`).
> 이 노트북의 Graph RAG 재구성은 로컬 `KnowledgeGraph` 로 충분해 **Neo4j 는 필요 없습니다**.

## 3. 로컬 LLM 준비 (기본값)

```bat
REM Ollama 설치 (1회)
winget install --id Ollama.Ollama -e

REM 서버 실행 (자동 실행 안 되면 수동으로)
ollama serve

REM 모델 다운로드 (약 5GB, 1회)
ollama pull qwen3:8b

REM 동작 확인
curl.exe http://localhost:11434/api/tags
```

`notebooks/.env` 설정:

```ini
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen3:8b
# 클라우드(Gemini)와 비교하려면 위를 google 로 바꾸고 아래 키를 채우세요.
GOOGLE_API_KEY=
```

> **로컬 ↔ 클라우드 전환**: `.env` 의 `LLM_PROVIDER` 만 `ollama` ↔ `google` 로 바꾸면 됩니다.

## 4. 실행 순서

1. Jupyter 에서 커널을 **`Agentic AI Tutorial (uv)`** 로 선택
2. 위에서부터 순서대로 셀 실행
   - 자기완결 setup(`utils.reload_env()` + `agentic_lib` import + `uv_install`)
   - **선행 구축**: ChromaDB → 샘플 문서 → `rag.VectorRAG` → `rag.KnowledgeGraph` → `rag.HybridRAG`
   - **에이전트 메모리**: `am.AgentMemorySystem`(단기/일화/의미/작업) 조립 + 계층별 관찰(`stats()`)
   - **최종 통합**: `am.DeepKnowledgeAgent` — 사고 → 검색 → 추론 → 생성 파이프라인
3. 첫 LLM 호출 시 Ollama 가 모델을 메모리에 올리느라 수십 초 걸릴 수 있습니다(이후 빠름).

## 5. 자주 겪는 문제

| 증상 | 원인/해결 |
|---|---|
| `Connection refused (11434)` | Ollama 서버 미실행 → `ollama serve` |
| `model 'qwen3:8b' not found` | 모델 미설치 → `ollama pull qwen3:8b` |
| `ModuleNotFoundError: chromadb` | `uv pip install chromadb sentence-transformers` |
| 임베딩 모델 다운로드 느림/실패 | 최초 1회 네트워크 필요. 사내망이면 프록시 설정 또는 미리 캐시 후 오프라인 사용 |
| 출력에 `<think>...</think>` 가 보임 | qwen3 는 사고 모델. 노트북은 `bootstrap.to_text()` 로 자동 제거 |

## 6. 명령 요약 (복붙용)

```bat
winget install --id Ollama.Ollama -e
ollama serve
ollama pull qwen3:8b
uv pip install chromadb sentence-transformers numpy
cd notebooks && copy .env.example .env
REM .env 에서 LLM_PROVIDER=ollama 확인 후 Jupyter 실행
uv run jupyter notebook
```
