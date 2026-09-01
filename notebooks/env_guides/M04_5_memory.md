# M04_5_memory.ipynb — 환경 설치·구축·실행 가이드

> 9~11주차 실습: AI 에이전트의 **메모리 시스템**(단기·장기·크로스 스레드·시맨틱)을
> 단계적으로 구현하고 실행합니다. 기본 LLM 은 **로컬 Ollama + `qwen3:8b`** 입니다.

## 0. 전제

- Windows 11 + **CMD(`cmd.exe`)** + Python **3.11**(`uv` 관리)
- 공통 1회 준비는 [README.md](README.md) 를 먼저 따라 하세요(uv 설치 → `uv venv` → `uv sync` → 커널 등록).
- 기본 LLM(로컬 Ollama + `qwen3:8b`) 준비는 [M01_1_intro.md](M01_1_intro.md) 와 동일합니다.

## 1. 이 노트북이 필요로 하는 것

| 구분 | 내용 |
|---|---|
| Python 패키지 | `langgraph`, `langgraph-checkpoint-sqlite`, `langchain`, `langchain-core` (대부분 `uv sync` 에 포함; 노트북이 `utils.uv_install()` 로 보강 설치) |
| 로컬 LLM | **Ollama** 서버 + **`qwen3:8b`** 모델 (네이티브 도구 호출 지원 → Store 도구 실습에 필요) |
| (선택) 클라우드 비교 | `GOOGLE_API_KEY` (Gemini 무료 티어) |
| 저장 공간 | `notebooks/workspace/agent_memory.db` (SqliteSaver 영구 저장 파일, 자동 생성) |

> 이 노트북의 **시맨틱 메모리**는 임베딩/벡터 DB 없이 **LLM 자체**로 관련 기억을
> 선별합니다. 따라서 `chromadb`·`sentence-transformers` 는 **필수가 아닙니다**(본문
> 표에서 대안으로만 언급). 실제 대규모 벡터 검색을 직접 실습하려면 아래 선택 설치를 참고하세요.

## 2. 추가 패키지 설치 (CMD)

`uv sync` 후에도 노트북 첫 셀이 `utils.uv_install([...])` 로 아래를 자동 보강합니다.
수동으로 미리 설치하려면 (프로젝트 루트에서):

```bat
REM 메모리 실습 필수 패키지
uv pip install "langgraph>=0.2.32" "langchain>=0.3.0" "langchain-core>=0.3.0" langgraph-checkpoint-sqlite

REM (선택) 임베딩 기반 벡터 검색을 직접 실습하고 싶을 때만
uv pip install chromadb sentence-transformers
```

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
> 노트북 코드는 한 줄도 고치지 않습니다(`utils.get_llm()` + `bootstrap.to_text()` 가 공급자 차이를 흡수).

## 4. 실행 순서

1. Jupyter 에서 커널을 **`Agentic AI (uv)`** 로 선택
2. 위에서부터 순서대로 셀 실행
   - 섹션 0: 환경 셋업(`utils.reload_env()` + `agentic_lib` import + `uv_install`)
   - 섹션 1: 단기 기억(버퍼 / 슬라이딩 윈도우 / 요약 / LangGraph MessagesState)
   - 섹션 2: 장기 기억(MemorySaver / SqliteSaver)
   - 섹션 3: 크로스 스레드(LangGraph Store + 저장/조회 도구 에이전트)
   - 섹션 4: 시맨틱 메모리(LLM 기반 관련 기억 검색)
   - 섹션 5: 통합 메모리 에이전트(FullMemoryAgent)
3. 메모리 클래스 구현은 [`agentic_lib/memory_advanced.py`](../agentic_lib/memory_advanced.py) 에 분리되어 있고 노트북은 import 만 합니다.

## 5. 자주 겪는 문제

| 증상 | 원인/해결 |
|---|---|
| `Connection refused (11434)` | Ollama 서버 미실행 → `ollama serve` |
| `model 'qwen3:8b' not found` | 모델 미설치 → `ollama pull qwen3:8b` |
| Store 도구가 호출되지 않음 | 도구 호출 미지원 모델 사용 중. `qwen3:8b`(네이티브 도구 호출) 또는 `google` 권장 |
| `ImportError: SqliteSaver` | `uv pip install langgraph-checkpoint-sqlite` |
| 출력에 `<think>...</think>` 가 보임 | qwen3 는 사고 모델. 노트북은 `bootstrap.to_text()` 로 자동 제거 |
| `agent_memory.db` 가 계속 커짐 | SqliteSaver 가 모든 체크포인트를 누적 저장. 초기화하려면 파일 삭제 |

## 6. 명령 요약 (복붙용)

```bat
winget install --id Ollama.Ollama -e
ollama serve
ollama pull qwen3:8b
uv pip install "langgraph>=0.2.32" "langchain>=0.3.0" "langchain-core>=0.3.0" langgraph-checkpoint-sqlite
cd notebooks && copy .env.example .env
REM .env 에서 LLM_PROVIDER=ollama 확인 후 Jupyter 실행
uv run jupyter notebook
```
