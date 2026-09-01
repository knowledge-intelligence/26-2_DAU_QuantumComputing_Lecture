# M04_1_embeddings.ipynb — 환경 설치·구축·실행 가이드

> 9~11주차 모듈 3(지식) **1/4편**: 벡터 임베딩과 시맨틱 유사도.
> **오프라인** 임베딩(`sentence-transformers`, 모델 카탈로그·2종 비교·query/passage 접두사)
> → **온라인 무료** 임베딩 API 3종(Google / NVIDIA / OpenRouter) → **성능 비교 · 의미 공간 시각화**.
> (Neo4j 벡터 인덱스 실습은 [3편 M04_3_graph_rag](M04_3_graph_rag.md) 로 옮겼습니다.)
> LLM 은 `.env` 의 `LLM_PROVIDER` 를 그대로 사용합니다(이 노트북에서는 거의 쓰지 않음).

## 0. 전제

- Windows 11 + **CMD(`cmd.exe`)** + Python **3.11**(`uv` 관리)
- 공통 1회 준비는 [README.md](README.md) 를 먼저 따라 하세요(uv 설치 → `uv venv` → `uv sync` → 커널 등록).
- 기본 LLM(로컬 Ollama + `qwen3:8b`) 준비는 [M01_1_intro.md](M01_1_intro.md) 와 동일합니다.

## 1. 이 노트북이 필요로 하는 것

| 구분 | 내용 |
|---|---|
| Python 패키지 | `sentence-transformers`, `numpy`, `requests`, `matplotlib`, `scikit-learn` (첫 셀이 `utils.uv_install()` 로 보강) |
| 시각화 | `matplotlib`(막대·의미 공간 그림) + `scikit-learn`(PCA 차원 축소) — 없으면 시각화만 건너뛰고 표는 그대로 출력 |
| 임베딩(오프라인) | `paraphrase-multilingual-MiniLM-L12-v2`(기본) + `jhgan/ko-sroberta-multitask`(2-2) + `intfloat/multilingual-e5-small`(2-3 접두사 실습). 최초 1회 다운로드 후 **오프라인** 동작 |
| 임베딩(온라인, 선택) | `GOOGLE_API_KEY` / `NVIDIA_API_KEY` / `OPENROUTER_API_KEY` 중 **있는 것만** 사용 |
| LLM | `.env` 의 `LLM_PROVIDER` (연결 확인용으로만 1회 호출) |

> 임베딩 공급자 추상화는 [`agentic_lib/embeddings.py`](../agentic_lib/embeddings.py) 입니다.

## 2. 추가 패키지 설치 (CMD)

`uv sync` 후에도 노트북 첫 셀이 `utils.uv_install([...])` 로 아래를 자동 보강합니다.
수동으로 미리 설치하려면 (프로젝트 루트에서):

```bat
REM 임베딩 실습 필수 패키지
uv pip install sentence-transformers numpy requests

REM 비교 그래프 · 의미 공간 시각화(PCA)
uv pip install matplotlib scikit-learn
```

> `sentence-transformers` 최초 사용 시 임베딩 모델(약 470MB)을 한 번 내려받습니다.
> 이후에는 캐시되어 **네트워크 없이** 동작합니다(`%USERPROFILE%\.cache\huggingface`).

## 3. 온라인 임베딩 API 키 발급 (선택 — 있는 것만 쓰면 됨)

세 곳 **모두 무료** 로 임베딩을 쓸 수 있습니다. LLM 용으로 이미 발급해 둔 키를 그대로 씁니다.

| 공급자 | 발급처 | 무료 조건 | `.env` 변수 |
|---|---|---|---|
| **Google AI Studio** | https://aistudio.google.com/apikey | 무료 티어(분당 토큰 제한) | `GOOGLE_API_KEY` |
| **NVIDIA build** | https://build.nvidia.com | 가입 시 무료 크레딧 | `NVIDIA_API_KEY` |
| **OpenRouter** | https://openrouter.ai/keys | `:free` 접미사 모델은 $0 | `OPENROUTER_API_KEY` |

`notebooks/.env`:

```ini
# LLM (이 노트북에서는 연결 확인용)
LLM_PROVIDER=ollama

# 임베딩 — 기본 공급자(코드에서 provider 를 명시하지 않을 때 사용)
EMBED_PROVIDER=local

# 온라인 임베딩용 키 (있는 것만 채우면 됨 — 없는 공급자는 자동으로 건너뜀)
GOOGLE_API_KEY=
NVIDIA_API_KEY=
OPENROUTER_API_KEY=
```

### 공급자별 기본 모델과 바꾸는 법

| `EMBED_PROVIDER` | 기본 모델 | 차원 | 모델 변경 환경변수 |
|---|---|---|---|
| `local` | `paraphrase-multilingual-MiniLM-L12-v2` | 384 | `LOCAL_EMBED_MODEL` |
| `google` | `gemini-embedding-001` | 768 (`EMBED_DIM` 으로 조절, 최대 3072) | `GOOGLE_EMBED_MODEL` |
| `nvidia` | `nvidia/nemotron-3-embed-1b` | 2048 | `NVIDIA_EMBED_MODEL` |
| `openrouter` | `nvidia/llama-nemotron-embed-vl-1b-v2:free` | 2048 | `OPENROUTER_EMBED_MODEL` |
| `ollama` | `bge-m3` (`ollama pull bge-m3` 필요) | 1024 | `OLLAMA_EMBED_MODEL` |

> ⚠️ **한국어에서는 다국어 모델을 골라야 합니다.** 예를 들어 NVIDIA 의
> `nvidia/nv-embedqa-e5-v5` 는 영어 특화라 한국어 검색 품질(MRR)이 크게 떨어집니다.
> 노트북 4장의 벤치마크로 직접 확인해 보세요.

## 4. 실행 순서

1. Jupyter 에서 커널을 **`Agentic AI Tutorial (uv)`** 로 선택
2. 위에서부터 **순서대로** 셀 실행 — 첫 셀(setup)을 건너뛰면 `NameError` 가 납니다
   - 1장 setup(`utils.reload_env()` + `agentic_lib` import + `uv_install`)
   - 2장 오프라인 임베딩 + 유사도 매트릭스 → **모델 카탈로그**(2-1) → **오프라인 2종 비교**(2-2)
     → **query/passage 접두사**(2-3)
   - 3장 온라인 임베딩 API(원시 REST 호출 → 통일 인터페이스 → **공급자별 유사도 매트릭스 비교**)
   - 4장 오프라인 ↔ 온라인 벤치마크(속도 · 차원 · Top-1/Top-3/MRR) + **의미 공간 종합 시각화**(4-2)
3. 첫 임베딩 모델 로딩은 다운로드 시간이 걸릴 수 있습니다(이후 캐시).
   2-2 의 `jhgan/ko-sroberta-multitask`(약 440MB), 2-3 의 `intfloat/multilingual-e5-small`(약 470MB)도
   **최초 1회** 내려받습니다.
4. 그림은 **파일로 저장하지 않고 노트북에 인라인으로** 표시됩니다.

## 5. 자주 겪는 문제

| 증상 | 원인/해결 |
|---|---|
| `NameError: name 'cosine_similarity' is not defined` | **첫 setup 셀을 실행하지 않은 것**. 노트북은 위에서부터 순서대로 실행해야 합니다(커널 재시작 후에도 동일) |
| `ModuleNotFoundError: sentence_transformers` | `uv pip install sentence-transformers numpy` |
| 임베딩 모델 다운로드 느림/실패 | 최초 1회 네트워크 필요. 사내망이면 프록시 설정 또는 미리 캐시 후 오프라인 사용 |
| 온라인 임베딩 `HTTP 401` | 키 오타/만료 → `.env` 의 해당 `*_API_KEY` 확인 후 `utils.reload_env()` 재실행 |
| 온라인 임베딩 `HTTP 429` | 무료 티어 분당 쿼터 초과 → 잠시 후 재시도하거나 배치 크기를 줄임 |
| NVIDIA `HTTP 400 input_type` | `nv-embedqa` 계열은 `input_type`(query/passage) 필수 — 라이브러리가 자동 처리 |
| 한국어 검색 품질이 유난히 낮음 | 영어 전용 임베딩 모델일 가능성 → 다국어 모델로 교체(3절 표 참고) |
| Gemini 벡터의 L2 norm 이 1 이 아님 | 정상. 3072 미만으로 축소(MRL)하면 정규화가 풀림 → 라이브러리가 재정규화 |
| e5 모델 검색 품질이 낮음 | `query: `/`passage: ` 접두사 누락. `get_embedder(..., auto_prefix=True)`(기본) 확인 |
| 그래프 라벨이 □□□ 로 깨짐 | 한글 폰트 없음(`Malgun Gothic`). 표 수치로 확인하면 됩니다 |

## 6. 명령 요약 (복붙용)

```bat
REM 패키지
uv pip install sentence-transformers numpy requests matplotlib scikit-learn

REM .env 준비 (notebooks/ 에서 1회)
cd notebooks && copy .env.example .env
REM .env 에서 LLM_PROVIDER 와 (선택) GOOGLE/NVIDIA/OPENROUTER 키 확인

uv run jupyter notebook
```
