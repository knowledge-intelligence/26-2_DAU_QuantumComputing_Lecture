# M03_2_tracing.ipynb — 환경 설치·구축·실행 가이드

> 5-7주차(모듈 2: 통제) · (2) LangSmith 트레이싱. Chain of Thought 추적(AgentTracer) +
> Safe & Traceable Agent + LangSmith 추적 분석/시각화. 기본 LLM 은 **로컬 Ollama + `qwen3:8b`**.

## 0. 전제

- Windows 11 + **CMD(`cmd.exe`)** + Python **3.11**(`uv` 관리)
- 공통 1회 준비는 [README.md](README.md) 를 먼저 따라 하세요(uv 설치 → `uv venv` → `uv sync` → 커널 등록).

## 1. 이 노트북이 필요로 하는 것

| 구분 | 내용 |
|---|---|
| Python 패키지 | `langsmith`, `langchain`, `langchain-openai`, `langchain-google-genai`, `langchain-anthropic`, `matplotlib` (노트북이 `uv_install` 로 자동 설치) |
| 로컬 LLM | **Ollama** 서버 + **`qwen3:8b`** 모델 |
| (선택) 클라우드 비교 | `GOOGLE_API_KEY` (Gemini 무료 티어) |
| (선택) 추적 | `LANGSMITH_API_KEY` (없으면 로컬 로깅만) |

## 2. 로컬 LLM 준비 (기본값)

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

## 3. `.env` 설정

`notebooks/.env.example` 를 복사해 `notebooks/.env` 를 만듭니다.

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
# (선택) LangSmith 추적 — 없으면 노트북은 로컬 로깅만 사용
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=agentic-ai-tutorial
```

> **LangSmith 는 선택 사항입니다.** `LANGSMITH_API_KEY` 를 비워 두면 노트북은 **로컬 로깅**으로 정상 동작하고,
> 키를 채우면 [smith.langchain.com](https://smith.langchain.com) 에 실행 추적이 전송됩니다(무료 티어 발급 가능).

## 4. 실행 순서

1. Jupyter 에서 커널을 **`Agentic AI (uv)`** 로 선택
2. 위에서부터 순서대로 셀 실행 (setup → LangSmith 환경 설정 → AgentTracer → SafeTraceableAgent → 시각화)
3. 첫 LLM 호출 시 Ollama 가 모델을 메모리에 올리느라 수십 초 걸릴 수 있습니다(이후 빠름).

## 5. 자주 겪는 문제

| 증상 | 원인/해결 |
|---|---|
| `Connection refused (11434)` | Ollama 서버 미실행 → `ollama serve` |
| `model 'qwen3:8b' not found` | 모델 미설치 → `ollama pull qwen3:8b` |
| LangSmith 추적이 안 보임 | `LANGSMITH_API_KEY` 미설정 — 노트북은 로컬 로깅으로 동작(정상) |
| 대시보드 한글이 깨짐 | matplotlib 폰트를 `Malgun Gothic` 으로 지정(노트북에 포함). Windows 기본 폰트 |
| 출력에 `<think>...</think>` 가 보임 | qwen3 는 사고 모델. 노트북은 `bootstrap.to_text()` 로 자동 제거 |

## 6. 명령 요약 (복붙용)

```bat
winget install --id Ollama.Ollama -e
ollama serve
ollama pull qwen3:8b
cd notebooks && copy .env.example .env
REM (선택) .env 에 LANGSMITH_API_KEY 채우기
uv run jupyter notebook
```
