# M03_1_nemo_guardrails.ipynb — 환경 설치·구축·실행 가이드

> 5-7주차(모듈 2: 통제) · (1) NeMo Guardrails. 입출력 가드레일(코드 구현) + NeMo Guardrails(Colang/Docker).
> 이 노트북은 **로컬 Ollama + `qwen3:8b`** 를 기본 LLM 으로 쓰고, **NeMo Guardrails 는
> Linux Docker 컨테이너**에서 실행합니다(호스트 직접 설치 불가).

## 0. 전제

- Windows 11 + **CMD(`cmd.exe`)** + Python **3.11**(`uv` 관리)
- 공통 1회 준비는 [README.md](README.md) 를 먼저 따라 하세요(uv 설치 → `uv venv` → `uv sync` → 커널 등록).
- **NeMo Guardrails 실습에는 Docker Desktop for Windows 가 필요합니다.**

## 1. 이 노트북이 필요로 하는 것

| 구분 | 내용 |
|---|---|
| Python 패키지 | `langchain`, `langchain-openai`, `langchain-google-genai`, `python-dotenv` (노트북이 `uv_install` 로 자동 설치) |
| 로컬 LLM | **Ollama** 서버 + **`qwen3:8b`** 모델 |
| (선택) 클라우드 비교 | `GOOGLE_API_KEY` (Gemini 무료 티어) |
| NeMo Guardrails | **Docker Desktop for Windows** (이미지 `nemo-guardrails:local` 빌드) |

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
```

> **로컬 ↔ 클라우드 전환**: `.env` 의 `LLM_PROVIDER` 만 `ollama` ↔ `google` 로 바꾸면 됩니다.
> NeMo Guardrails 컨테이너도 같은 `.env` 를 `--env-file` 로 받으므로 동일 공급자를 사용합니다.

## 4. NeMo Guardrails — Docker 실행 (필수: Docker Desktop)

`nemoguardrails` 의 의존성 `annoy` 는 **Windows 사전 빌드 휠이 없고**(소스 컴파일에 Visual
Studio C++ 빌드 도구 필요) 호스트 설치가 어렵습니다. 따라서 **Linux 컨테이너**에서 구동합니다.

- `notebooks/nemo_docker/Dockerfile` : `python:3.11` 기반, `nemoguardrails` + LangChain 공급자 설치(annoy 는 Linux 에서 정상 빌드)
- `notebooks/nemo_docker/runner.py` : 컨테이너 안에서 가드레일을 실행하는 러너(`.env` 의 LLM 공급자 사용)
- `notebooks/nemo_config/` : `config.yml`(모델 설정 형식)·`main.co`(Colang 대화 레일) — 노트북 셀이 생성

### 4-1. Docker Desktop 준비

```bat
REM Docker Desktop 이 실행 중인지 확인 (버전 출력되면 정상)
docker version
```

### 4-2. 이미지 빌드 (1회)

노트북의 빌드 셀이 자동 실행하지만, CMD 에서 직접 빌드하려면(노트북 폴더 기준):

```bat
cd notebooks
docker build -t nemo-guardrails:local nemo_docker
docker images nemo-guardrails:local --format "{{.Repository}}:{{.Tag}}  {{.Size}}"
```

최초 빌드는 수 분 소요되며 이후에는 캐시됩니다.

### 4-3. 컨테이너 실행 (가드레일 평가)

노트북은 `utils.run_cmd(...)` 로 아래 형태의 명령을 실행합니다(메시지는 `messages.json` 으로 전달).

```bat
docker run --rm --env-file ".env" ^
  -e HF_HUB_DISABLE_PROGRESS_BARS=1 ^
  -v nemo-cache:/root/.cache ^
  -v "%CD%\nemo_config:/work/config" -v "%CD%\nemo_docker:/work/app" ^
  nemo-guardrails:local python /work/app/runner.py
```

## 5. 실행 순서

1. Jupyter 에서 커널을 **`Agentic AI (uv)`** 로 선택
2. 위에서부터 순서대로 셀 실행 (setup → 입출력 가드레일 → NeMo 설정 생성 → 이미지 빌드 → 컨테이너 실행)
3. 첫 LLM 호출 시 Ollama 가 모델을 메모리에 올리느라 수십 초 걸릴 수 있습니다(이후 빠름).

## 6. 자주 겪는 문제

| 증상 | 원인/해결 |
|---|---|
| `Connection refused (11434)` | Ollama 서버 미실행 → `ollama serve` |
| `model 'qwen3:8b' not found` | 모델 미설치 → `ollama pull qwen3:8b` |
| `docker: command not found` / 빌드 실패 | Docker Desktop 미실행 → 실행 후 `docker version` 확인 |
| 컨테이너 첫 실행이 느림 | 임베딩 모델 최초 다운로드 — `nemo-cache` 볼륨에 캐시되어 다음부터 빠름 |
| 출력에 `<think>...</think>` 가 보임 | qwen3 는 사고 모델. 노트북은 `bootstrap.to_text()` 로 자동 제거 |
| 한글이 깨짐(`cp949`) | 노트북(Jupyter)은 UTF-8 이라 정상. 일반 CMD 출력만 영향 |

## 7. 명령 요약 (복붙용)

```bat
winget install --id Ollama.Ollama -e
ollama serve
ollama pull qwen3:8b
cd notebooks && copy .env.example .env
docker version
docker build -t nemo-guardrails:local nemo_docker
uv run jupyter notebook
```
