# NeMo Guardrails 실행용 이미지
# Windows 호스트에서는 annoy(=nemoguardrails 의존성)가 사전 빌드 휠이 없어 설치되지 않으므로,
# Linux 컨테이너(컴파일러 포함)에서 nemoguardrails 를 구동한다.
# python:3.11 (slim 아님) 이미지는 gcc/g++ 와 Python 헤더를 포함해 C 확장 빌드가 가능하다.
# 강의 본체(호스트)와 동일한 3.11 로 맞춘다.
FROM python:3.11

# NeMo Guardrails + 과정에서 쓰는 LangChain 공급자들
# (openai 커넥터 하나로 ollama/llamacpp/vllm/openrouter 등 OpenAI 호환 공급자를 모두 처리)
RUN pip install --no-cache-dir \
    nemoguardrails \
    langchain-google-genai langchain-openai langchain-anthropic \
    langchain-nvidia-ai-endpoints

WORKDIR /work
# config(nemo_config) 와 app(nemo_docker)은 런타임에 -v 로 마운트한다.
# (스크립트/설정 수정 시 이미지를 다시 빌드하지 않아도 됨)
