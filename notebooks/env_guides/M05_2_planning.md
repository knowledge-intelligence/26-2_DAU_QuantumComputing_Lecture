# M05_2_planning.ipynb — 환경 설치·구축·실행 가이드

> 12~14주차 실습: **고급 계획 수립·실행(Planning)**. 구조화 플래닝 → DAG 실행 →
> Plan-and-Execute → 동적 재계획 → 계층적 플래닝 → 멀티 에이전트까지 다룹니다.
> 핵심 구현은 [`agentic_lib/dag_planning.py`](../agentic_lib/dag_planning.py) 로 분리되어 있습니다.

## 0. 전제

- Windows 11 + **CMD(`cmd.exe`)** + Python **3.11**(`uv` 관리)
- 공통 1회 준비는 [README.md](README.md) 를 먼저 따라 하세요(uv 설치 → `uv venv` → `uv sync` → 커널 등록).
- 기본 LLM 은 **로컬 Ollama + `qwen3:8b`**(구조화 출력·도구 호출 지원), 클라우드 비교는 **google(Gemini)**.

## 1. 이 노트북이 필요로 하는 것

| 구분 | 내용 |
|---|---|
| Python 패키지 | `langgraph`, `langchain`, `langchain-core`, `pydantic` (대부분 `uv sync` 에 포함) |
| 로컬 LLM | **Ollama** 서버 + **`qwen3:8b`** 모델 |
| (선택) 클라우드 비교 | `GOOGLE_API_KEY` (Gemini 무료 티어) |

> LangGraph 상태 기계(`StateGraph`)와 Pydantic 구조화 출력(`with_structured_output`)을
> 사용하므로, 작은 로컬 모델보다 **`qwen3:8b` 이상**을 권장합니다(스키마 준수 안정성).

## 2. 추가 패키지 설치 (CMD)

기본 `uv sync` 에 langgraph/langchain/pydantic 이 포함되어 있지만, 버전을 명시적으로
맞추려면 아래처럼 설치합니다(노트북 첫 셀의 `utils.uv_install(...)` 도 같은 일을 합니다).

```bat
REM 프로젝트 루트에서
uv pip install "langgraph>=0.2.32" "langchain>=0.3.0" "langchain-core>=0.3.0" "pydantic>=2.0.0"
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

## 4. `.env` 설정

`notebooks/.env.example` 를 복사해 `notebooks/.env` 를 만들고 아래처럼 둡니다.

```bat
cd notebooks
copy .env.example .env
```

```ini
# notebooks/.env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen3:8b
# 클라우드(Gemini)와 비교하려면 위를 google 로 바꾸고 아래 키를 채우세요.
GOOGLE_API_KEY=
```

> **로컬 ↔ 클라우드 전환**: `.env` 의 `LLM_PROVIDER` 만 `ollama` ↔ `google` 로 바꾸면 됩니다.
> 노트북 코드는 한 줄도 고치지 않습니다(`utils.get_llm()` 이 공급자 차이를 흡수).

## 5. 실행 순서

1. Jupyter 에서 커널을 **`Agentic AI (uv)`** 로 선택
2. 위에서부터 순서대로 셀 실행
   - 섹션 0: 셋업(패키지 설치 + `agentic_lib` import + LLM 연결)
   - 섹션 1: 구조화 플래닝(Pydantic 스키마로 계획 JSON 생성)
   - 섹션 2: DAG 실행 엔진(`DAGExecutor`) + 의존성 그래프 시각화
   - 섹션 3: Plan-and-Execute(LangGraph 상태 기계)
   - 섹션 4: 동적 재계획(replan 노드)
   - 섹션 5: 계층적 플래닝(목표 → 서브목표 → 태스크)
   - 섹션 6: 멀티 에이전트(플래너·실행자·검증자·재작성자·최종화)
3. 각 섹션은 앞 섹션의 결과(`plan`, `hplan` 등)를 사용하므로 **순서대로** 실행하세요.
4. `write_document` 도구는 `notebooks/workspace/` 폴더에 마크다운 파일을 생성합니다.

## 6. 자주 겪는 문제

| 증상 | 원인/해결 |
|---|---|
| `Connection refused (11434)` | Ollama 서버 미실행 → `ollama serve` |
| `model 'qwen3:8b' not found` | 모델 미설치 → `ollama pull qwen3:8b` |
| `with_structured_output` 결과가 비정상 | 모델이 너무 작음 → `qwen3:8b` 이상 사용 권장 |
| `ModuleNotFoundError: langgraph` | 첫 셀의 `uv_install(...)` 실행 또는 §2 명령으로 설치 |
| 출력에 `<think>...</think>` 가 보임 | qwen3 사고 모델. 노트북·라이브러리가 `to_text()` 로 자동 제거 |
| 그래프가 무한 반복 | 재계획/재작성 횟수는 `max_replans`·`max_revisions` 로 제한됨(기본 2회) |

## 7. 명령 요약 (복붙용)

```bat
uv pip install "langgraph>=0.2.32" "langchain>=0.3.0" "langchain-core>=0.3.0" "pydantic>=2.0.0"
winget install --id Ollama.Ollama -e
ollama serve
ollama pull qwen3:8b
cd notebooks && copy .env.example .env
REM .env 에서 LLM_PROVIDER=ollama 확인 후 Jupyter 실행
uv run jupyter notebook
```
