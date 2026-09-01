# M03_4_nemo_agent_toolkit.ipynb — 환경 설치·구축·실행 가이드

> 5-7주차(모듈 2: 통제) · (4) NVIDIA NeMo-Agent-Toolkit(NAT) 소개. 개념·아키텍처·설치 안내 위주.
> ℹ️ NAT 는 **Python 3.11–3.13** 을 요구합니다. 강의 본체가 3.11 이므로 **버전 조건은 이제 충족**되지만,
> NAT 는 의존성(LangChain 등)을 자체 범위로 고정하므로 강의 환경을 오염시키지 않도록 **별도 전용 venv** 를 유지합니다.

## 0. 전제

- Windows 11 + **CMD(`cmd.exe`)**
- 노트북 자체(개념/점검)는 강의 기본 커널 **`Agentic AI (uv)`**(Python 3.11)에서 열립니다.
- **NAT 설치·실행**은 아래 3절의 **NAT 전용 가상환경(Python 3.11)** 에서 진행합니다.

## 1. NAT 요약 (출처 명시)

- **패키지명**: `nvidia-nat` (구 이름: Agent Intelligence Toolkit / AIQ toolkit)
- **목적**: 프레임워크를 대체하지 않고 감싸서 **관측·프로파일링·평가·최적화** 를 더하는 오픈소스 라이브러리
- **지원 프레임워크**: LangChain, LlamaIndex, CrewAI, Microsoft Semantic Kernel, Google ADK, 커스텀
- **Python 요구**: **3.11 / 3.12 / 3.13** (강의 고정 3.11 과 호환 → 버전 제약은 해소, 의존성 격리 목적으로만 별도 venv 사용)
- **CLI**: `nat`
- 출처: GitHub https://github.com/NVIDIA/NeMo-Agent-Toolkit · 문서 https://docs.nvidia.com/nemo/agent-toolkit/

## 2. 이 노트북이 필요로 하는 것

| 구분 | 내용 |
|---|---|
| 노트북 열람 | 강의 기본 커널 `Agentic AI (uv)`(3.11) — 개념 설명 + `nat` 설치 여부 점검 셀 |
| NAT 설치(선택) | **별도 venv(Python 3.11)** + `nvidia-nat` |
| NAT 예시 실행(선택) | `NVIDIA_API_KEY` (build.nvidia.com 발급) + 저장소 클론 |

## 3. NAT 설치 (별도 전용 venv, CMD)

NAT 전용 venv 는 NAT 실습 시리즈 폴더인 **`notebooks/NAT_Tutorial/.venv-nat`** 에 만듭니다.
아래 명령은 프로젝트 루트에서 실행합니다.

```bat
REM NAT 전용 가상환경 (강의 venv 와 분리 — 의존성 격리 목적)
uv venv notebooks\NAT_Tutorial\.venv-nat --python 3.11

REM 코어 + 프레임워크 플러그인(extras) 설치
uv pip install --python notebooks\NAT_Tutorial\.venv-nat "nvidia-nat[langchain]"

REM 설치 확인
notebooks\NAT_Tutorial\.venv-nat\Scripts\nat.exe --version
```

> pip 사용자: `pip install nvidia-nat` / `pip install "nvidia-nat[langchain]"` 로 동일하게 설치됩니다.
>
> ⚠️ venv 는 폴더를 옮기면 깨집니다(실행 파일에 절대 경로가 박힘 →
> `uv trampoline failed to canonicalize script path`). 이동했다면 `.venv-nat` 을 지우고 위 명령으로 재생성하세요.
>
> NAT 실습 시리즈(`NAT_01`~`NAT_07`)의 상세 설치는 [`../NAT_Tutorial/INSTALL_NAT.md`](../NAT_Tutorial/INSTALL_NAT.md) 참고.

## 4. API 키 & 예시 실행 (선택)

```bat
REM API 키 발급: https://build.nvidia.com 에서 계정 생성 후 키 발급
set NVIDIA_API_KEY=nvapi-여기에_키

REM 예시 실행에는 저장소 클론이 필요 (예시 파일 포함)
git clone https://github.com/NVIDIA/NeMo-Agent-Toolkit.git
cd NeMo-Agent-Toolkit
nat run --config_file workflow.yml --input "List five subspecies of Aardvarks"
```

## 5. 주의사항 (정직한 안내)

- NAT 는 **Python 3.11+** 요구 → 강의 `agentic-ai-venv`(3.11)와 버전은 호환되지만, NAT 가 LangChain 등 의존성을
  자체 범위로 고정하므로 강의 환경 오염을 피하려면 **별도 venv 사용을 권장**합니다.
- 예시 실행은 **NVIDIA_API_KEY** 와 **저장소 클론**이 필요하므로 설치가 무겁습니다. 본 노트북은
  (NeMo Guardrails 를 Docker 로 분리한 것과 같은 취지로) **개념·설치 가이드 위주**로 다룹니다.
- 공식 문서에 Windows 특정 제약은 명시되어 있지 않으나, 세부 스키마·명령은 버전에 따라 달라질 수 있으니
  항상 [공식 문서](https://docs.nvidia.com/nemo/agent-toolkit/) 를 확인하세요.

## 6. 참고 자료

- GitHub: https://github.com/NVIDIA/NeMo-Agent-Toolkit
- 공식 문서: https://docs.nvidia.com/nemo/agent-toolkit/
- API 키 발급: https://build.nvidia.com
