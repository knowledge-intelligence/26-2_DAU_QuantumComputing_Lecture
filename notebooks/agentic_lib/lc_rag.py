"""
lc_rag — LangChain 으로 조립하는 RAG (PromptTemplate · LCEL · Agent)
=====================================================================

앞 단계(`doc_prep` 로 청킹, `vector_stores` 로 색인)에서 만든 검색기를
**LangChain 표준 부품** 으로 감싸 RAG 파이프라인을 완성합니다.

세 가지 조립 방식을 나란히 보여줍니다.

    1) PromptTemplate  — 프롬프트를 문자열이 아닌 '틀'로 다룬다(변수·재사용·검증)
    2) LCEL 체인       — retriever | prompt | llm | parser 를 파이프로 연결한다
                          (스트리밍·배치·비동기가 공짜로 따라온다)
    3) RAG Agent       — 검색을 '도구'로 주고 LLM 이 **언제 몇 번 검색할지 스스로** 정한다

LCEL 과 Agent 의 차이가 이 모듈의 핵심입니다.
LCEL 은 "무조건 1회 검색 후 답변"이라 빠르고 예측 가능하지만, 질문이 복잡하면 부족합니다.
Agent 는 필요하면 질의를 바꿔가며 여러 번 검색하지만 느리고 도구 호출이 가능한 모델이 필요합니다.

의존성 규약:
    - LLM 은 주입받는다(`utils.get_llm()` 이 만든 공급자 무관 모델).
    - 응답은 `bootstrap.to_text()` 로 정규화한다.
"""

from typing import Callable, Dict, List, Optional, Sequence

from .bootstrap import to_text

# LangChain 은 이 프로젝트의 기본 의존성이지만, 없더라도 import 가 깨지지 않게 방어한다.
try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    _LC_AVAILABLE = True
except Exception:  # ModuleNotFoundError 등
    _LC_AVAILABLE = False


# =============================================================================
# 1. 프롬프트 템플릿
# =============================================================================
#: RAG 답변 생성용 시스템 프롬프트.
#: "모르면 모른다고 하라"를 명시하는 것이 환각을 줄이는 가장 값싼 방법이다.
RAG_SYSTEM_PROMPT = """당신은 사용자의 개인 문서를 검색해 답하는 비서입니다.

규칙:
1. 반드시 아래 '문서 발췌'에 있는 내용만 근거로 답하세요.
2. 발췌에 답이 없으면 "제공된 문서에서 찾을 수 없습니다"라고 솔직히 말하세요.
3. 답변 끝에 근거로 삼은 문서를 [출처: 문서제목 › 섹션] 형식으로 표시하세요.
4. 한국어로 간결하게 답하세요."""

RAG_USER_PROMPT = """문서 발췌:
{context}

질문: {question}"""


def build_rag_prompt(system: str = RAG_SYSTEM_PROMPT,
                     user: str = RAG_USER_PROMPT) -> "ChatPromptTemplate":
    """RAG 용 ChatPromptTemplate 을 만든다.

    프롬프트를 문자열 f-string 이 아니라 템플릿 객체로 다루면
    변수 목록(`input_variables`)이 드러나고, 체인 안에서 재사용·교체가 쉬워진다.

    Args:
        system: 시스템 메시지 템플릿.
        user: 사용자 메시지 템플릿({context}, {question} 변수 포함).

    Returns:
        ChatPromptTemplate 인스턴스.
    """
    if not _LC_AVAILABLE:
        raise ImportError("langchain-core 가 필요합니다 (uv pip install langchain)")
    return ChatPromptTemplate.from_messages([("system", system), ("human", user)])


def format_docs(hits: Sequence[Dict], max_chars: int = 500) -> str:
    """검색 결과(hit 목록)를 프롬프트에 넣을 컨텍스트 문자열로 만든다.

    출처를 함께 적어 두면 LLM 이 인용 표기를 훨씬 잘 지킨다.

    Args:
        hits: vector_stores 의 search() 결과.
        max_chars: 청크 하나당 최대 글자 수.

    Returns:
        "[1] (제목 › 섹션) 본문" 형식으로 이어 붙인 문자열.
    """
    blocks = []
    for i, hit in enumerate(hits, 1):
        meta = hit.get("metadata", {})
        origin = f"{meta.get('title', '제목없음')} › {meta.get('section', '') or '본문'}"
        body = hit["text"].strip().replace("\n", " ")[:max_chars]
        blocks.append(f"[{i}] ({origin}) {body}")
    return "\n\n".join(blocks) if blocks else "(검색 결과 없음)"


# =============================================================================
# 2. LCEL 체인
# =============================================================================
def make_retriever(store, k: int = 3,
                   where: Optional[Dict[str, object]] = None) -> Callable[[str], List[Dict]]:
    """벡터 저장소를 '질문 → 검색결과' 함수로 감싼다(LCEL 에 끼워 넣기 위함)."""
    def retrieve(question: str) -> List[Dict]:
        return store.search(question, k=k, where=where)
    return retrieve


def build_lcel_chain(store, llm, k: int = 3,
                     where: Optional[Dict[str, object]] = None,
                     prompt: Optional["ChatPromptTemplate"] = None):
    """검색 → 프롬프트 → LLM → 문자열 파싱을 하나의 LCEL 체인으로 잇는다.

    파이프(|)로 연결된 각 단계는 모두 Runnable 이라 같은 인터페이스를 갖는다.
    그래서 `chain.invoke(q)` / `chain.batch([...])` / `chain.stream(q)` 가 모두 공짜다.

    구조:
        {"context": 질문→검색→문자열, "question": 그대로} | prompt | llm | StrOutputParser

    Args:
        store: vector_stores 의 저장소.
        llm: LangChain BaseChatModel(`utils.get_llm()`).
        k: 검색할 청크 수.
        where: 메타데이터 필터(부서·분류별로 좁힐 때).
        prompt: 사용할 프롬프트(기본 build_rag_prompt()).

    Returns:
        문자열 질문을 받아 문자열 답변을 돌려주는 Runnable.
    """
    if not _LC_AVAILABLE:
        raise ImportError("langchain-core 가 필요합니다 (uv pip install langchain)")
    retrieve = make_retriever(store, k=k, where=where)

    # RunnableLambda: 평범한 파이썬 함수를 체인 부품으로 끌어들인다
    context_step = RunnableLambda(retrieve) | RunnableLambda(format_docs)

    return (
        {"context": context_step, "question": RunnablePassthrough()}
        | (prompt or build_rag_prompt())
        | llm
        | StrOutputParser()
    )


def build_traced_chain(store, llm, k: int = 3):
    """검색 결과까지 함께 돌려주는 체인 — 근거를 눈으로 확인할 때 쓴다.

    Returns:
        질문 → {"question", "hits", "context", "answer"} 를 돌려주는 Runnable.
    """
    if not _LC_AVAILABLE:
        raise ImportError("langchain-core 가 필요합니다 (uv pip install langchain)")
    retrieve = make_retriever(store, k=k)
    prompt = build_rag_prompt()

    def build_inputs(question: str) -> Dict:
        hits = retrieve(question)
        return {"question": question, "hits": hits, "context": format_docs(hits)}

    def answer(payload: Dict) -> Dict:
        message = prompt.invoke({"context": payload["context"],
                                 "question": payload["question"]})
        return {**payload, "answer": to_text(llm.invoke(message).content)}

    return RunnableLambda(build_inputs) | RunnableLambda(answer)


# -----------------------------------------------------------------------------
# 2-1. 체인 중간을 들여다보기
#      LCEL 은 파이프로 이어 놓으면 편하지만 그만큼 **속이 안 보인다**.
#      "검색은 뭘 찾았고, 프롬프트에는 뭐가 들어갔고, 어디서 시간을 썼나"를 보는 두 가지 방법.
# -----------------------------------------------------------------------------
def _brief(value, limit: int = 62) -> str:
    """이벤트 payload 를 한 줄로 요약한다 — 타입마다 보여 줄 것이 다르다."""
    if value is None or value == "":
        return ""
    # 검색 결과(hit 목록): 몇 건을 어디서 찾았는지가 궁금하다
    if isinstance(value, list) and value and isinstance(value[0], dict) and "metadata" in value[0]:
        heads = ", ".join(f"{h['metadata'].get('title', '')} › {h['metadata'].get('section', '')}"
                          for h in value[:2])
        return f"{len(value)}건 — {heads[:limit]}{'…' if len(heads) > limit else ''}"
    # 프롬프트 결과(ChatPromptValue): 최종적으로 LLM 에 무엇이 갔는지
    messages = getattr(value, "messages", None)
    if messages is not None:
        parts = ", ".join(f"{type(m).__name__.replace('Message', '').lower()} "
                          f"{len(to_text(m.content))}자" for m in messages)
        return f"메시지 {len(messages)}개 ({parts})"
    # LLM 응답(AIMessage) 등 content 를 가진 객체
    if hasattr(value, "content"):
        value = to_text(value.content)
    if isinstance(value, dict):
        return "{" + ", ".join(value.keys()) + "}"
    text = str(value).replace("\n", " ")
    return f"{len(text)}자  '{text[:limit]}{'…' if len(text) > limit else ''}'"


#: 추적에서 제외할 이름 — 체인을 감싸는 껍데기라 단계로 보여 줄 가치가 없다
_TRACE_SKIP = {"RunnableSequence", "RunnablePassthrough"}


async def trace_lcel_events(chain, question: str) -> Dict:
    """**이미 만든 체인을 그대로 두고** 실행 과정을 단계별로 출력한다.

    `astream_events()` 는 LCEL 의 각 부품이 언제 시작하고 끝났는지, 무엇을 받아
    무엇을 내놓았는지를 이벤트로 흘려 준다. 체인을 고치지 않아도 되는 것이 장점이다.

    각 단계의 **시작·종료 시각** 을 함께 찍는다. 하나로 합친 '소요' 를 쓰지 않는 이유가 있다 —
    LCEL 단계는 줄 서서 하나씩 도는 것이 아니라 **겹쳐서** 돌기 때문이다.

        - `StrOutputParser` 는 첫 토큰이 도착하자마자 열려서 `ChatOpenAI` 와 나란히 돈다(스트리밍).
        - `RunnableParallel` 은 `retrieve`·`format_docs` 를 감싸는 부모라 구간이 자식을 포함한다.

    그래서 구간을 세로로 더하면 전체 시간을 훌쩍 넘는다. 아래 타임라인 막대가 그 겹침을 보여 준다.

    Args:
        chain: `build_lcel_chain()` 등이 만든 Runnable.
        question: 질문 문자열.

    Returns:
        {"answer", "steps", "total_ms", "llm_ms"} — steps 는 [(시작ms, 종료ms, 단계명, 요약)].
        `llm_ms` 는 **이벤트 종류(`on_chat_model_end`)로** 집어낸 모델 구간이다.
        이름으로 찾으면 안 된다 — `ChatPromptTemplate` 도 "Chat" 으로 시작하고,
        모델 클래스명은 공급자마다 다르다(`ChatOpenAI`/`ChatNVIDIA`/`ChatOllama`…).

    Note:
        비동기 함수다. 노트북에서는 `await trace_lcel_events(...)` 로 호출한다
        (동기 `stream_events()` 는 langchain-core 1.x 에서 RunnableSequence 를
        지원하지 않아 async 를 쓴다).
    """
    import time

    started = time.perf_counter()
    steps: List[tuple] = []
    opened: Dict[str, float] = {}   # run_id → 시작 시각(종료 때 구간을 재려고 기록)
    token_count = 0
    llm_ms = 0.0
    answer = ""

    print(f"[LCEL 실행 추적] {question}")
    print("=" * 96)
    print(f"  {'시작':>7} {'종료':>8}  {'단계':<24} 내놓은 것")
    print("-" * 96)

    async for event in chain.astream_events(question, version="v2"):
        kind, name = event["event"], event.get("name", "")
        if name in _TRACE_SKIP:
            continue
        if kind.endswith("_stream"):          # 토큰 조각은 세기만 한다
            if kind == "on_chat_model_stream":
                token_count += 1
            continue

        now = time.perf_counter()
        if kind.endswith("_start"):
            opened[event["run_id"]] = now     # 종료 때 짝을 맞추려고 기록만 해 둔다
            continue
        if not kind.endswith("_end"):
            continue

        summary = _brief(event.get("data", {}).get("output"))
        if name == "StrOutputParser":
            answer = str(event.get("data", {}).get("output", ""))
        if kind == "on_chat_model_end" and token_count:
            summary = f"토큰 {token_count}개 → " + summary

        begin = (opened.get(event["run_id"], started) - started) * 1000
        end = (now - started) * 1000
        if kind == "on_chat_model_end":   # 이름이 아니라 이벤트 종류로 모델을 집는다
            llm_ms = end - begin
        label = name if len(name) <= 24 else name[:23] + "…"
        steps.append((begin, end, name, summary))
        print(f"  {begin:>5.0f}ms {end:>6.0f}ms  {label:<24} {summary}")

    print("-" * 96)
    _print_timeline(steps)
    flat = answer.strip().replace("\n", " ")
    print(f"\n  최종 답변: {flat[:78]}{'…' if len(flat) > 78 else ''}")
    total_ms = max((end for _, end, _, _ in steps), default=0.0)
    return {"answer": answer, "steps": steps, "total_ms": total_ms, "llm_ms": llm_ms}


def _print_timeline(steps: List[tuple], width: int = 46) -> None:
    """단계별 실행 구간을 막대로 그린다 — '겹쳐서 돈다'를 글보다 빠르게 보여 준다."""
    if not steps:
        return
    total = max(end for _, end, _, _ in steps)
    if total <= 0:
        return
    print(f"  [타임라인] 총 {total:.0f}ms — 막대가 겹치는 구간은 동시에 돌고 있다는 뜻이다")
    for begin, end, name, _ in steps:
        left = int(begin / total * width)
        right = max(left + 1, int(end / total * width))   # 짧은 단계도 최소 한 칸
        bar = " " * left + "█" * (right - left) + "·" * (width - right)
        label = name if len(name) <= 22 else name[:21] + "…"
        print(f"    {label:<22} |{bar}| {end - begin:>5.0f}ms")


def build_stepwise_chain(store, llm, k: int = 3,
                         prompt: Optional["ChatPromptTemplate"] = None):
    """중간값을 **결과 dict 에 쌓아 두는** 체인 — `RunnablePassthrough.assign()` 관용구.

    `assign()` 은 "지금까지의 dict 를 그대로 흘려보내면서 키를 하나 더 붙인다"는 뜻이다.
    그래서 마지막에 받아 보면 question·hits·context·messages·answer 가 전부 남아 있다.

    `trace_lcel_events()` 가 **관찰**(체인을 안 고침)이라면 이쪽은 **설계**다 —
    중간값이 계속 필요하다면(로그 적재·근거 표시·평가) 애초에 이렇게 짓는다.

    Returns:
        질문 → {"question","hits","context","messages","answer"} 를 돌려주는 Runnable.
    """
    if not _LC_AVAILABLE:
        raise ImportError("langchain-core 가 필요합니다 (uv pip install langchain)")
    retrieve = make_retriever(store, k=k)
    prompt = prompt or build_rag_prompt()

    return (
        RunnableLambda(lambda q: {"question": q})
        # 각 assign 이 앞 단계 dict 에 키를 하나씩 덧붙인다 — 이전 값은 지워지지 않는다
        | RunnablePassthrough.assign(hits=lambda x: retrieve(x["question"]))
        | RunnablePassthrough.assign(context=lambda x: format_docs(x["hits"]))
        | RunnablePassthrough.assign(
            messages=lambda x: prompt.invoke({"context": x["context"],
                                              "question": x["question"]}).messages)
        | RunnablePassthrough.assign(
            answer=lambda x: to_text(llm.invoke(x["messages"]).content))
    )


def print_stepwise(result: Dict, max_chars: int = 300) -> None:
    """`build_stepwise_chain()` 결과에 쌓인 중간값을 단계 순서대로 펼쳐 본다."""
    print(f"[1] question  {result['question']}")
    print(f"\n[2] hits      검색 {len(result['hits'])}건")
    for i, hit in enumerate(result["hits"], 1):
        meta = hit.get("metadata", {})
        print(f"      {i}. ({hit['score']:.3f}) {meta.get('title','')} › {meta.get('section','')}")
    context = result["context"]
    print(f"\n[3] context   {len(context)}자 — 검색 결과를 프롬프트용 문자열로 합친 것")
    print(f"      {context[:max_chars].replace(chr(10), ' ')}…")
    print(f"\n[4] messages  LLM 에 실제로 건너간 메시지 {len(result['messages'])}개")
    for message in result["messages"]:
        body = to_text(message.content).replace("\n", " ")
        print(f"      {type(message).__name__:<14} {len(body):>4}자  {body[:72]}…")
    print(f"\n[5] answer    {result['answer'].strip()}")


def print_chain_result(result: Dict, max_chars: int = 400) -> None:
    """build_traced_chain 결과(답변 + 근거)를 보기 좋게 출력한다."""
    print(f"질문: {result['question']}")
    print(f"\n검색된 근거 {len(result['hits'])}건:")
    for i, hit in enumerate(result["hits"], 1):
        meta = hit.get("metadata", {})
        print(f"  [{i}] ({hit['score']:.3f}) {meta.get('title','')} › "
              f"{meta.get('section','')}")
    answer = result["answer"]
    print(f"\n답변:\n{answer[:max_chars]}{'…' if len(answer) > max_chars else ''}")
    print("-" * 84)


# =============================================================================
# 3. RAG Agent — 검색을 '도구'로 제공
# =============================================================================
def make_search_tool(store, k: int = 3, name: str = "search_personal_docs"):
    """벡터 검색을 LangChain 도구(@tool)로 감싼다.

    도구 설명(docstring)이 곧 LLM 이 읽는 사용 설명서다. 언제 써야 하는지를
    구체적으로 적어야 에이전트가 엉뚱한 때 호출하지 않는다.

    Args:
        store: 검색할 벡터 저장소.
        k: 한 번 호출에 돌려줄 청크 수.
        name: 도구 이름.

    Returns:
        LangChain StructuredTool.
    """
    from langchain_core.tools import StructuredTool

    def search_personal_docs(query: str) -> str:
        """개인 문서(회의록·업무보고·학습노트·규정·메모)를 검색한다.

        사용자의 개인 기록에 대한 질문이면 반드시 이 도구를 먼저 호출하라.
        찾는 내용이 안 나오면 검색어를 바꿔 다시 호출해도 된다.

        Args:
            query: 찾고 싶은 내용을 담은 한국어 검색어.
        """
        return format_docs(store.search(query, k=k))

    return StructuredTool.from_function(
        func=search_personal_docs, name=name,
        description=(
            "개인 문서(회의록·업무보고·학습노트·사내규정·개인메모)를 시맨틱 검색한다. "
            "사용자의 기록·일정·결정사항·수치를 묻는 질문이면 반드시 먼저 호출할 것. "
            "결과가 부족하면 검색어를 바꿔 여러 번 호출해도 된다."
        ),
    )


AGENT_SYSTEM_PROMPT = """당신은 사용자의 개인 문서를 검색해 답하는 비서입니다.

- 개인 기록에 관한 질문이면 반드시 search_personal_docs 도구를 먼저 호출하세요.
- 한 번의 검색으로 부족하면 검색어를 바꿔 다시 호출하세요.
- 검색으로 찾은 내용만 근거로 답하고, 없으면 없다고 말하세요.
- 한국어로 간결하게 답하세요."""


def build_rag_agent(llm, store, k: int = 3, system: str = AGENT_SYSTEM_PROMPT):
    """검색 도구를 쥐어 준 RAG 에이전트를 만든다(LangChain 1.x `create_agent`).

    LCEL 체인과 달리 **검색 횟수와 검색어를 LLM 이 스스로 정한다.**
    도구 호출을 지원하는 모델이 필요하다(`.env` 의 공급자에 따라 다름).

    Args:
        llm: 도구 호출이 가능한 BaseChatModel.
        store: 검색할 벡터 저장소.
        k: 검색당 청크 수.
        system: 시스템 프롬프트.

    Returns:
        LangGraph 에이전트 앱(`invoke({"messages": [...]})`).
    """
    from langchain.agents import create_agent

    return create_agent(llm, [make_search_tool(store, k=k)], system_prompt=system)


def invoke_agent(agent, question: str) -> "tuple[str, List[str]]":
    """에이전트를 실행하고 (답변, 실제로 사용한 검색어 목록) 을 돌려준다.

    검색어 목록이 비어 있으면 **모델이 검색을 건너뛰고 자기 지식으로 답한 것** 이다.
    RAG 에서는 이것이 곧 환각 위험 신호이므로 반드시 확인해야 한다.
    """
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result["messages"]
    queries = [call["args"].get("query", "")
               for message in messages
               for call in (getattr(message, "tool_calls", None) or [])]
    return to_text(messages[-1].content), queries


def run_agent(agent, question: str, verbose: bool = True) -> str:
    """에이전트를 실행하고 **도구 호출 과정** 까지 보여 준다.

    Args:
        agent: build_rag_agent 가 만든 앱.
        question: 사용자 질문.
        verbose: True 면 검색어와 호출 횟수를 출력.

    Returns:
        최종 답변 문자열(to_text 정규화).
    """
    answer, queries = invoke_agent(agent, question)
    if verbose:
        print(f"질문: {question}")
        for i, query in enumerate(queries, 1):
            print(f"  🔍 검색 {i}회차: \"{query}\"")
        if not queries:
            print("  ⚠️ 검색 0회 — 모델이 도구를 쓰지 않고 자체 지식으로 답했다(환각 위험)")
        print(f"\n답변:\n{answer}")
        print("-" * 84)
    return answer


def compare_chain_and_agent(store, llm, questions: Sequence[str], k: int = 3) -> None:
    """같은 질문을 LCEL 체인과 Agent 에 각각 물어 차이를 비교한다.

    단순 질문에서는 결과가 비슷하고 체인이 훨씬 빠르다.
    여러 문서를 엮어야 하는 질문에서 Agent 의 반복 검색이 값을 한다.
    """
    import time

    chain = build_lcel_chain(store, llm, k=k)
    try:
        agent = build_rag_agent(llm, store, k=k)
    except Exception as e:
        print(f"에이전트 생성 실패(도구 호출 미지원 공급자일 수 있음): {str(e)[:120]}")
        agent = None

    for question in questions:
        print(f"\n{'=' * 84}\n질문: {question}\n")

        started = time.perf_counter()
        chain_answer = to_text(chain.invoke(question))
        chain_sec = time.perf_counter() - started
        print(f"[LCEL 체인] {chain_sec:.1f}s (검색 1회 고정)")
        print(f"  {chain_answer[:260]}")

        if agent is None:
            continue
        started = time.perf_counter()
        try:
            agent_answer, queries = invoke_agent(agent, question)
            agent_sec = time.perf_counter() - started
            print(f"\n[RAG Agent] {agent_sec:.1f}s / 검색 {len(queries)}회 "
                  + (f"({', '.join(repr(q) for q in queries)})" if queries
                     else "⚠️ 검색을 건너뜀 → 문서가 아닌 자체 지식으로 답할 위험"))
            print(f"  {agent_answer[:260]}")
        except Exception as e:
            print(f"\n[RAG Agent] 실패: {str(e)[:140]}")
