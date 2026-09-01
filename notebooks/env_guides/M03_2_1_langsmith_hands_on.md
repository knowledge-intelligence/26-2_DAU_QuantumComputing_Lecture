# M03_2_1_langsmith_hands_on.ipynb — 환경 설치·구축·실행 가이드

> 5-7주차(모듈 2: 통제) · (2-1) **LangSmith 실전 핸즈온**. 실제 [smith.langchain.com](https://smith.langchain.com)
> 에 직접 붙어 트레이싱·데이터셋·평가(`evaluate()`)·피드백을 다룬다. 기본 LLM 은 **로컬 Ollama + `qwen3:8b`**.
> 자매 노트북 [`M03_2_tracing`](../M03_2_tracing.ipynb) 의 **시뮬레이션**과 달리, 이 노트북은 **실제 LangSmith 서비스**를 사용한다.

## 0. 전제

- Windows 11 + **CMD(`cmd.exe`)** + Python **3.11**(`uv` 관리)
- 공통 1회 준비는 [README.md](README.md) 를 먼저 따라 하세요(uv 설치 → `uv venv` → `uv sync` → 커널 등록).

## 1. 이 노트북이 필요로 하는 것

| 구분 | 내용 |
|---|---|
| Python 패키지 | `langsmith`, `langchain`, `langchain-core`, `langchain-openai` (노트북이 `uv_install` 로 자동 설치) |
| 로컬 LLM | **Ollama** 서버 + **`qwen3:8b`** 모델 (또는 `.env` 의 다른 `LLM_PROVIDER`) |
| **추적(핵심)** | **`LANGSMITH_API_KEY`** — 이 노트북의 목적이 *실제 LangSmith 사용*이므로 발급 권장 |

> **키가 없어도** 모든 셀은 오류 없이 진행됩니다(트레이스·데이터셋·평가 전송만 건너뜀).
> 하지만 대시보드에서 결과를 보려면 키가 필요합니다.

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

## 3. LangSmith API 키 발급

1. [smith.langchain.com](https://smith.langchain.com) 에 가입(무료 티어 제공).
2. **Settings → API Keys → Create API Key** 로 키 발급 후 복사.

## 4. `.env` 설정

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
# LangSmith 실전 추적 (이 노트북의 핵심)
LANGSMITH_API_KEY=lsv2_...             # 위에서 발급한 키
LANGSMITH_PROJECT=agentic-ai-tutorial  # 트레이스를 묶을 프로젝트 이름(자유)
# (선택) 자체 호스팅/리전 엔드포인트를 쓸 때만 변경
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

> 노트북은 최신 이름(`LANGSMITH_*`)과 구버전 별칭(`LANGCHAIN_*`)을 함께 설정해 호환성을 확보합니다.

## 5. 실행 순서

1. Jupyter 에서 커널을 **`Agentic AI (uv)`** 로 선택
2. 위에서부터 순서대로 셀 실행
   (셋업 → 추적 켜기/연결 → 자동 트레이스 → `@traceable` → 데이터셋 → `evaluate()` → 메타데이터·피드백 → UI)
3. 각 셀 실행 후 [smith.langchain.com](https://smith.langchain.com) 의 프로젝트에서 트레이스가 쌓이는 것을 확인
4. 첫 LLM 호출 시 Ollama 가 모델을 메모리에 올리느라 수십 초 걸릴 수 있습니다(이후 빠름).

## 6. 자주 겪는 문제

| 증상 | 원인/해결 |
|---|---|
| `Connection refused (11434)` | Ollama 서버 미실행 → `ollama serve` |
| `model 'qwen3:8b' not found` | 모델 미설치 → `ollama pull qwen3:8b` |
| 트레이스가 대시보드에 안 보임 | `LANGSMITH_API_KEY` 미설정/오타 — 노트북 2번 셀 출력의 ⚠️ 메시지 확인 |
| 데이터셋/평가 셀이 "(건너뜀)" 출력 | `LANGSMITH_API_KEY` 없음 — 키를 넣고 재실행 |
| `401 Unauthorized` | 키가 잘못됨/만료 → 새 키 발급 후 `.env` 갱신, 커널 재시작 |
| 출력에 `<think>...</think>` 가 보임 | qwen3 는 사고 모델. 노트북은 `bootstrap.to_text()`/`strip_think()` 로 자동 제거 |

## 7. 명령 요약 (복붙용)

```bat
winget install --id Ollama.Ollama -e
ollama serve
ollama pull qwen3:8b
cd notebooks && copy .env.example .env
REM .env 에 LANGSMITH_API_KEY 채우기 (smith.langchain.com 에서 발급)
uv run jupyter notebook
```
