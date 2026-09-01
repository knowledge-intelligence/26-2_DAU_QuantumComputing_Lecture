"""컨테이너 안에서 NeMo Guardrails 를 실행하는 러너.

- 설정(config.yml, main.co)은 /work/config 에 마운트된다.
- 입력 메시지는 /work/app/messages.json (문자열 리스트) 에서 읽는다.
- 결과는 /work/app/results.json 에 저장하고, 요약을 stdout 으로 출력한다.
- LLM 공급자는 호스트 .env(--env-file)로 전달된 환경변수를 그대로 사용한다.
"""
import json
import os
import sys
import warnings

# 컨테이너 실행 로그를 깔끔히 유지하기 위한 경고 억제:
#  - HF_HUB_DISABLE_PROGRESS_BARS=1 로 진행바를 끄면 huggingface_hub 가 UserWarning 을 낸다.
#  - LangChain/NeMo 의 일부 DeprecationWarning 도 교육용 출력에서는 잡음이다.
# (onnxruntime 의 네이티브 GPU 탐색 경고는 import 시점 stderr 라 여기서 제어되지 않는다 —
#  노트북은 컨테이너 원시 출력 대신 results.json 을 정리해 출력해 이를 숨긴다.)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

APP_DIR = "/work/app"
CONFIG_DIR = "/work/config"


def _flatten_content(msg):
    """AIMessage.content 가 list(Gemini 등)면 일반 문자열로 평탄화한다 (in-place)."""
    if isinstance(msg.content, list):
        msg.content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in msg.content
        )


def _container_url(url: str) -> str:
    """호스트 .env 의 localhost URL 을 '컨테이너에서 호스트로' 접근 가능한 주소로 치환한다.

    컨테이너 안에서 localhost/127.0.0.1 은 컨테이너 자신을 가리키므로, 호스트에서 뜬
    로컬 LLM 서버(Ollama, llama.cpp 등)에 닿지 못한다. Docker Desktop 이 자동 제공하는
    host.docker.internal 로 바꿔 호스트 포트에 접근한다.
    """
    if not url:
        return url
    return url.replace("localhost", "host.docker.internal").replace(
        "127.0.0.1", "host.docker.internal"
    )


def make_llm():
    """utils.get_llm 과 동일한 규칙으로 LangChain LLM 을 생성한다 (컨테이너용 독립 구현).

    지원 공급자는 노트북의 utils.get_llm 과 동일하게 맞춘다:
    google / anthropic / anthropic_oauth / openai / vllm / ollama / llamacpp / nvidia / openrouter.
    로컬 서버(vllm/ollama/llamacpp)의 base_url 은 _container_url 로 호스트 접근 주소로 바꾼다.
    (config.yml 의 models 항목은 형식상 유지될 뿐, 실제 호출에는 여기서 만든 llm 이 쓰인다.)
    """
    provider = os.getenv("LLM_PROVIDER", "google")
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Gemini 는 content 를 [{'type':'text','text':...}] 리스트로 반환한다.
        # NeMo Guardrails 는 문자열을 기대하므로, 생성 결과의 content 를 문자열로 평탄화한다.
        class StringChatGoogle(ChatGoogleGenerativeAI):
            def _generate(self, *args, **kwargs):
                result = super()._generate(*args, **kwargs)
                for gen in result.generations:
                    _flatten_content(gen.message)
                return result

            async def _agenerate(self, *args, **kwargs):
                result = await super()._agenerate(*args, **kwargs)
                for gen in result.generations:
                    _flatten_content(gen.message)
                return result

        return StringChatGoogle(
            model="gemini-3.1-flash-lite",
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            temperature=0,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            temperature=0, max_tokens=1024,
        )
    if provider == "anthropic_oauth":
        from langchain_anthropic import ChatAnthropic
        token = os.getenv("ANTHROPIC_OAUTH_TOKEN", "")
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001", api_key=token,
            default_headers={"Authorization": f"Bearer {token}"},
            temperature=0, max_tokens=1024,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY", ""),
            temperature=0, max_tokens=1024,
        )
    if provider == "vllm":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("VLLM_MODEL", ""),
            base_url=_container_url(os.getenv("VLLM_BASE_URL", "")),
            api_key="EMPTY", temperature=0, max_tokens=1024,
        )
    if provider == "ollama":
        # Ollama 의 OpenAI 호환 엔드포인트(/v1). 호스트에서 뜬 서버이므로 host.docker.internal 로 접근.
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            base_url=_container_url(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")),
            api_key="ollama", temperature=0, max_tokens=1024,
        )
    if provider == "llamacpp":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("LLAMACPP_MODEL", "llamacpp"),
            base_url=_container_url(os.getenv("LLAMACPP_BASE_URL", "http://localhost:8000/v1")),
            api_key="sk-no-key-required", temperature=0, max_tokens=1024,
        )
    if provider == "openrouter":
        # OpenRouter — OpenAI 호환 엔드포인트(외부 URL 이라 치환 불필요). 기본은 무료 라우터.
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "openrouter/free"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            temperature=0, max_tokens=1024,
        )
    if provider == "nvidia":
        # NVIDIA build — ChatNVIDIA 커넥터(이미지에 langchain-nvidia-ai-endpoints 설치 필요).
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        kwargs = dict(
            model=os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct"),
            api_key=os.getenv("NVIDIA_API_KEY", ""),
            temperature=0, max_tokens=1024,
        )
        base = os.getenv("NVIDIA_BASE_URL", "")
        if base:
            kwargs["base_url"] = base
        return ChatNVIDIA(**kwargs)
    raise SystemExit(
        f"지원하지 않는 LLM 공급자: {provider}. "
        "사용 가능: google, anthropic, anthropic_oauth, openai, vllm, ollama, llamacpp, openrouter, nvidia"
    )


def main():
    from nemoguardrails import LLMRails, RailsConfig
    from nemoguardrails.integrations.langchain.llm_adapter import LangChainLLMAdapter

    config = RailsConfig.from_path(CONFIG_DIR)
    # LangChain LLM 은 어댑터로 감싸 전달한다(raw LLM 직접 전달은 deprecated).
    rails = LLMRails(config, llm=LangChainLLMAdapter(make_llm()))

    # 입력 메시지 로드 (없으면 기본 테스트 케이스)
    msg_path = os.path.join(APP_DIR, "messages.json")
    if os.path.exists(msg_path):
        with open(msg_path, encoding="utf-8") as f:
            messages = json.load(f)
    else:
        messages = ["안녕하세요!"]

    results = []
    for msg in messages:
        try:
            resp = rails.generate(messages=[{"role": "user", "content": msg}])
            content = resp.get("content") if isinstance(resp, dict) else str(resp)
            results.append({"message": msg, "response": content, "error": False})
        except Exception as e:
            results.append({"message": msg, "response": str(e), "error": True})

    # 결과 저장 + 요약 출력
    with open(os.path.join(APP_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("=== NeMo Guardrails (Docker) 결과 ===")
    for r in results:
        print(f"\n[사용자] {r['message']}")
        print(f"[에이전트] {r['response']}")


if __name__ == "__main__":
    sys.exit(main())
