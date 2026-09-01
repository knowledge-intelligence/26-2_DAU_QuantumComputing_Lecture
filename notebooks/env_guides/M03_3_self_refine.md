# M03_3_self_refine.ipynb — 환경 설치·구축·실행 가이드

> 5-7주차(모듈 2: 통제) · (3) Self-Refine. 자기 비평 → 반복 개선(SelfCorrectingAgent).
> **모든 셀이 실제 LLM 을 호출합니다**(생성·비평·개선). 기본 LLM 은 **로컬 Ollama + `qwen3:8b`**.

## 0. 전제

- Windows 11 + **CMD(`cmd.exe`)** + Python **3.10**(`uv` 관리)
- 공통 1회 준비는 [README.md](README.md) 를 먼저 따라 하세요(uv 설치 → `uv venv` → `uv sync` → 커널 등록).

## 1. 이 노트북이 필요로 하는 것

| 구분 | 내용 |
|---|---|
| Python 패키지 | `langchain`, `python-dotenv` (노트북이 `uv_install` 로 자동 설치) |
| 로컬 LLM | **Ollama** 서버 + **`qwen3:8b`** 모델 (setup 의 `get_llm()` 용) |
| (선택) 클라우드 비교 | `GOOGLE_API_KEY` (Gemini 무료 티어) |

> ⏱ **실행 비용/시간**: 자기 수정은 반복 1회마다 LLM 을 2회(비평 + 개선) 호출합니다.
> 7.1 은 최대 3회, 7.2 는 최대 5회 반복하므로 노트북 전체가 최대 20여 회의 LLM 호출을 냅니다.
> 로컬 8B 모델 기준 수 분이 걸릴 수 있습니다. 빨리 보고 싶으면 `max_iterations` 를 줄이세요.

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
GOOGLE_API_KEY=        # google 로 비교 전환 시에만 필요
```

## 4. 실행 순서

1. Jupyter 에서 커널을 **`Agentic AI (uv)`** 로 선택
2. 위에서부터 순서대로 셀 실행 (setup → 자기 수정 메커니즘)

## 5. 자주 겪는 문제

| 증상 | 원인/해결 |
|---|---|
| `Connection refused (11434)` | Ollama 서버 미실행 → `ollama serve` (setup 의 `get_llm()` 용) |
| 출력에 `<think>...</think>` 가 보임 | qwen3 는 사고 모델. 노트북은 `bootstrap.to_text()` 로 자동 제거 |
| 문제점에 `비평 JSON 파싱 실패` 가 뜸 | 모델이 JSON 대신 산문을 반환. `SelfCorrectingAgent._parse_critique()` 가 0.5 로 폴백해 루프는 계속된다. 자주 뜨면 지시를 더 잘 따르는 모델로 바꾸거나 `critic_llm` 을 교체 |
| 7.1 점수가 오르내리며 수렴하지 않음 | **정상 동작이자 학습 포인트.** 자기채점은 불안정하다 — 그래서 7.2 에서 검증기로 근거를 잡는다 |

## 6. 참고

- Self-Refine 논문: https://arxiv.org/abs/2303.17651
