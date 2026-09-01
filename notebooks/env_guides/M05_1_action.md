# M05_1_action.ipynb — 환경 설치·구축·실행 가이드

> 12~14주차 모듈4: **실행(Action & Evaluation)**. LangGraph 상태 그래프 →
> 멀티 에이전트 → OpenClaw 업무 자동화 → RAGAS 스타일 평가 → LLM-as-a-Judge →
> 최종 통합 시스템까지 다룹니다. 핵심 구현은 공통 라이브러리로 분리되어 있습니다:
> [`agentic_lib/graph.py`](../agentic_lib/graph.py),
> [`agentic_lib/evaluation.py`](../agentic_lib/evaluation.py),
> [`agentic_lib/automation.py`](../agentic_lib/automation.py).

## 0. 전제

- Windows 11 + **CMD(`cmd.exe`)** + Python **3.11**(`uv` 관리)
- 공통 1회 준비는 [README.md](README.md) 를 먼저 따라 하세요(uv 설치 → `uv venv` → `uv sync` → 커널 등록).
- 기본 LLM 은 **로컬 Ollama + `qwen3:8b`**(네이티브 도구 호출 지원), 클라우드 비교는 **google(Gemini)**.

## 1. 이 노트북이 필요로 하는 것

| 구분 | 내용 |
|---|---|
| Python 패키지 | `langgraph`, `langchain`, `langchain-openai`, `langchain-google-genai`, `pandas`, `matplotlib` (대부분 `uv sync` 에 포함) |
| 로컬 LLM | **Ollama** 서버 + **`qwen3:8b`** 모델 |
| (선택) 클라우드 비교 | `GOOGLE_API_KEY` (Gemini 무료 티어) |

> ReAct 에이전트(`create_react_agent`)는 **도구 호출**을 사용하므로, 작은 모델보다
> **`qwen3:8b` 이상**을 권장합니다(도구 호출 안정성).

## 2. 추가 패키지 설치 (CMD)

기본 `uv sync` 에 langgraph/langchain 이 포함되어 있지만, 명시적으로 맞추려면 아래처럼
설치합니다(노트북 첫 셀의 `utils.uv_install(...)` 도 같은 일을 합니다).

```bat
REM 프로젝트 루트에서
uv pip install "langgraph>=0.2.32" "langchain>=0.3.0" "langchain-openai>=0.2.0" ^
               "langchain-google-genai>=2.0.0" pandas matplotlib
```

### RAGAS 는 설치하지 않습니다 (자체 구현)

원래 RAGAS 패키지(`ragas`, `datasets`)는 의존성이 무겁고 일부 환경에서 설치가
까다롭습니다. 본 노트북은 **RAGAS 핵심 지표를 외부 의존성 없이 직접 구현**한
[`agentic_lib/evaluation.py`](../agentic_lib/evaluation.py) 의 `RAGEvaluator` 를 사용합니다
(faithfulness / answer_relevancy / context_recall / context_precision). LLM 키도 필요 없고
오프라인에서 결정론적으로 동작합니다. 진짜 RAGAS 를 써 보고 싶다면 별도로
`uv pip install ragas datasets` 후 LLM 키를 설정하세요(선택, 본 강의 범위 밖).

LLM 기반 평가는 같은 모듈의 `LLMJudge`(LLM-as-a-Judge)로 다룹니다 — 기본 LLM(`llm`)을
그대로 채점기로 주입하며, 모델이 없거나 JSON 파싱에 실패하면 규칙 기반으로 폴백합니다.

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
   - 섹션 1: 셋업(패키지 설치 + `agentic_lib` import + LLM 연결)
   - 섹션 2: LangGraph 기초(`graph.build_plan_execute_app` — plan→execute→evaluate 루프)
   - 섹션 3: 실제 LLM ReAct 에이전트(`graph.build_react_agent`)
   - 섹션 4: 멀티 에이전트 연구 시스템(`graph.build_research_app`)
   - 섹션 5: OpenClaw 업무 자동화(`automation.WorkflowAutomationAgent`)
   - 섹션 6~7: RAGAS 스타일 평가(`evaluation.RAGEvaluator`) + LLM-as-a-Judge(`evaluation.LLMJudge`)
   - 섹션 8~9: 성능 대시보드 + 최종 통합 시스템(`automation.FinalAgentSystem`)
3. 첫 LLM 호출 시 Ollama 가 모델을 메모리에 올리느라 수십 초 걸릴 수 있습니다.

> **파일 자동화 안전장치**: `WorkflowAutomationAgent` 는 모든 파일 입출력을
> `notebooks/workspace/` **하위로만** 허용합니다(폴더 밖·`..` 탈출 경로는 차단).
> 섹션 5 를 실행하면 `notebooks/workspace/` 에 `draft.txt`, `tech_stats.csv`,
> `report_*.md` 가 생성됩니다.

## 6. 자주 겪는 문제

| 증상 | 원인/해결 |
|---|---|
| `Connection refused (11434)` | Ollama 서버 미실행 → `ollama serve` |
| `model 'qwen3:8b' not found` | 모델 미설치 → `ollama pull qwen3:8b` |
| `ModuleNotFoundError: langgraph` | 첫 셀의 `uv_install(...)` 실행 또는 §2 명령으로 설치 |
| ReAct 에이전트가 도구를 안 부름 | 모델이 너무 작음 → `qwen3:8b` 이상 권장 |
| 출력에 `<think>...</think>` 가 보임 | qwen3 사고 모델. 노트북·라이브러리가 `to_text()` 로 자동 제거 |
| `작업 폴더 밖의 경로에는 접근할 수 없습니다` | 자동화 경로 가드 작동(정상). `notebooks/workspace/` 하위 경로만 사용 |
| 대시보드 한글 깨짐 | `matplotlib` 폰트 `Malgun Gothic` 필요(Windows 기본 포함) |

## 7. 명령 요약 (복붙용)

```bat
uv pip install "langgraph>=0.2.32" "langchain>=0.3.0" "langchain-openai>=0.2.0" ^
               "langchain-google-genai>=2.0.0" pandas matplotlib
winget install --id Ollama.Ollama -e
ollama serve
ollama pull qwen3:8b
cd notebooks && copy .env.example .env
REM .env 에서 LLM_PROVIDER=ollama 확인 후 Jupyter 실행
uv run jupyter notebook
```
