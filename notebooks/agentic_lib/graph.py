"""
graph — LangGraph 기반 상태 그래프 패턴 (모듈 4: 실행)
=====================================================

week12-14 모듈4 노트북(`M05_1_action.ipynb`)의 **길고 반복되는
LangGraph 구현**을 분리한 모듈입니다. 노트북은 '개념과 시연'에 집중하고,
상태(State)·노드(Node)·엣지(Edge) 구성은 이 모듈의 빌더 함수로 재사용합니다.

구성:
    [Plan-Execute-Evaluate 루프 그래프]
        AgentState                          plan→execute→evaluate 공유 상태(TypedDict)
        build_plan_execute_app(llm, ...)    계획·실행·평가·재계획 루프 그래프 컴파일

    [실제 LLM ReAct 에이전트]
        build_react_agent(llm, tools, ...)  langgraph.prebuilt.create_react_agent 래퍼

    [멀티 에이전트 연구 시스템]
        ResearchAgentState                  연구→분석→작성→검토 공유 상태(TypedDict)
        build_research_app(llm)             멀티 에이전트 그래프 컴파일

설계 메모:
    - 모든 그래프 노드는 `llm` 을 캡처하는 **클로저**로 만들고, `build_*` 함수가
      컴파일된 그래프(app)를 돌려줍니다(전역 의존 제거). `llm=None` 이면 LLM 없이
      키워드/템플릿 기반 시뮬레이션으로 동작하므로 오프라인에서도 실행됩니다.
    - 모든 LLM 응답은 `bootstrap.invoke_text()`/`to_text()` 로 정규화합니다
      (Gemini list 형식·qwen3 <think> 블록 흡수).
    - `langgraph` 는 빌더 함수 안에서 **지연 import** 합니다(미설치 환경에서도 모듈 로드 가능).
"""

import operator
from datetime import datetime
from typing import TypedDict, Annotated, List, Dict, Optional, Literal

from .bootstrap import to_text, invoke_text


# ─────────────────────────────────────────────────────────────────────────────
# 1. Plan-Execute-Evaluate 루프 그래프
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    """plan→execute→evaluate 루프를 도는 에이전트의 공유 상태.

    Attributes:
        messages: 대화/이벤트 기록(노드가 append, operator.add 로 누적).
        task: 현재 처리할 태스크 설명.
        plan: 실행 계획(단계 문자열 목록).
        current_step: 현재 실행 중인 단계 인덱스.
        results: 단계별 실행 결과.
        final_answer: 최종 답변(완료 시).
        iteration: 재계획 반복 횟수.
        quality_score: 평가 노드가 매긴 품질 점수(0~1).
        status: 'planning' | 'executing' | 'evaluating' | 'done' | 'failed'.
    """
    messages: Annotated[List[Dict], operator.add]
    task: str
    plan: List[str]
    current_step: int
    results: List[str]
    final_answer: Optional[str]
    iteration: int
    quality_score: float
    status: str


def _keyword_plan(task: str) -> List[str]:
    """LLM 없이 태스크 키워드만으로 단순 계획을 세운다(오프라인 폴백)."""
    plan: List[str] = []
    if "검색" in task or "찾아" in task:
        plan.append("웹 검색 수행")
    if "계산" in task or any(op in task for op in ["+", "-", "*", "/"]):
        plan.append("수학 계산 수행")
    if "요약" in task or "정리" in task:
        plan.append("결과 요약 및 정리")
    if "저장" in task or "파일" in task:
        plan.append("결과 파일 저장")
    if not plan:
        plan = ["태스크 분석", "처리 수행", "결과 정리"]
    return plan


def build_plan_execute_app(llm=None, with_memory: bool = True):
    """plan→execute→evaluate→(재계획 루프) 그래프를 컴파일해 반환한다.

    노드는 `llm` 을 캡처하는 클로저로 구성된다. `llm` 이 주어지면 planner 가 LLM 으로
    계획을 세우고(응답은 to_text 로 정규화), 실패하거나 None 이면 키워드 기반으로
    동작한다. executor/evaluator 는 교육용 시뮬레이션이다(품질 점수 휴리스틱).

    Args:
        llm: 계획 수립에 사용할 LangChain BaseChatModel(없으면 키워드 기반).
        with_memory: True 면 MemorySaver 체크포인터를 붙여 thread 별 상태를 보존한다.

    Returns:
        컴파일된 LangGraph 앱(`app.invoke(state, config=...)` 로 실행).
    """
    # langgraph 는 미설치 환경 보호를 위해 함수 안에서 지연 import 한다.
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver

    def planner_node(state: AgentState) -> AgentState:
        """계획 수립 노드 — LLM(있으면) 또는 키워드로 실행 단계를 만든다."""
        task = state["task"]
        iteration = state.get("iteration", 0)
        print(f"\n[Planner] 태스크: '{task}' (시도 {iteration + 1})")

        plan: Optional[List[str]] = None
        if llm is not None:
            try:
                # LLM 에게 한 줄에 한 단계씩 계획을 요청하고 to_text 로 정규화한다.
                raw = invoke_text(
                    llm,
                    f"다음 태스크를 3~5개의 실행 단계로 나눠 한 줄에 하나씩만 적어줘. "
                    f"번호/설명 없이 단계 제목만.\n태스크: {task}",
                )
                plan = [ln.strip(" -*0123456789.") for ln in raw.splitlines() if ln.strip()][:5]
            except Exception as e:
                print(f"  (LLM 계획 실패 → 키워드 기반으로 대체: {e})")
                plan = None
        if not plan:
            plan = _keyword_plan(task)

        plan.append("최종 답변 생성")
        print(f"  계획: {plan}")
        return {
            "plan": plan,
            "current_step": 0,
            "results": [],
            "status": "executing",
            "iteration": iteration + 1,
            "messages": [{"role": "system", "content": f"계획 수립 완료: {plan}"}],
        }

    def executor_node(state: AgentState) -> AgentState:
        """실행 노드 — 현재 단계를 수행한다(시뮬레이션)."""
        plan = state["plan"]
        step = state["current_step"]
        results = state.get("results", [])

        if step >= len(plan):
            return {"status": "evaluating"}

        current_action = plan[step]
        print(f"\n[Executor] 단계 {step + 1}/{len(plan)}: {current_action}")

        if "웹 검색" in current_action or "검색" in current_action:
            result = f"검색 결과: {state['task']}에 관한 최신 정보 3건 발견"
        elif "계산" in current_action:
            result = "계산 완료: 수식 처리 결과 도출"
        elif "요약" in current_action or "정리" in current_action:
            result = f"요약: {len(results)}개 중간 결과를 통합"
        elif "저장" in current_action:
            result = "파일 저장 완료: output.txt"
        elif "최종 답변" in current_action:
            result = f"'{state['task']}'에 대한 답변 생성 완료"
        else:
            result = f"{current_action} 완료"

        print(f"  결과: {result}")
        next_step = step + 1
        new_status = "evaluating" if next_step >= len(plan) else "executing"
        return {
            "results": results + [result],
            "current_step": next_step,
            "status": new_status,
            "messages": [{"role": "assistant", "content": f"Step {step + 1}: {result}"}],
        }

    def evaluator_node(state: AgentState) -> AgentState:
        """평가 노드 — 결과 품질을 점수화하고 최종 답변 생성 또는 재계획을 결정한다."""
        results = state.get("results", [])
        task = state["task"]
        print(f"\n[Evaluator] {len(results)}개 결과 평가 중...")

        # 품질 점수: 단계가 많을수록 높게(교육용 휴리스틱; 실전은 LLM-as-a-Judge).
        score = min(0.6 + len(results) * 0.1, 1.0)
        print(f"  품질 점수: {score:.2f}")

        if score >= 0.8:
            final_answer = f"[완료] '{task}' 처리 결과:\n" + "\n".join(f"  - {r}" for r in results)
            print("  상태: 완료 (품질 충족)")
            return {"quality_score": score, "final_answer": final_answer, "status": "done"}
        print("  상태: 품질 부족 → 재계획")
        return {"quality_score": score, "status": "planning"}

    def route_after_executor(state: AgentState) -> Literal["executor", "evaluator"]:
        """실행 후 라우팅 — 모든 단계가 끝났으면 평가로, 아니면 계속 실행."""
        if state["status"] == "evaluating":
            return "evaluator"
        return "executor"

    def route_after_evaluator(state: AgentState):
        """평가 후 라우팅 — 완료/최대시도면 종료, 아니면 재계획(루프)."""
        if state["status"] == "done":
            return END
        if state.get("iteration", 0) >= 3:  # 최대 3회 재시도
            print("\n[Evaluator] 최대 재시도 횟수 초과 → 강제 종료")
            return END
        return "planner"

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_conditional_edges("executor", route_after_executor)
    workflow.add_conditional_edges("evaluator", route_after_evaluator)

    if with_memory:
        return workflow.compile(checkpointer=MemorySaver())
    return workflow.compile()


# ─────────────────────────────────────────────────────────────────────────────
# 2. 실제 LLM ReAct 에이전트 (langgraph.prebuilt)
# ─────────────────────────────────────────────────────────────────────────────
def build_react_agent(llm, tools=None, prompt: str = None):
    """`create_react_agent` 로 도구 사용 ReAct 에이전트를 만든다.

    도구를 지정하지 않으면 공통 `tools` 모듈의 기본 도구(계산기/시간/검색/날씨 +
    파일 IO)를 사용한다. `llm` 이 None 이면 None 을 돌려준다(연결 실패 시 시연만).

    Args:
        llm: LangChain BaseChatModel(도구 호출을 지원하는 모델 권장).
        tools: 사용할 LangChain 도구 목록. None 이면 tools.AGENT_TOOLS + FILE_TOOLS.
        prompt: 시스템 프롬프트. None 이면 기본 한국어 안내문.

    Returns:
        컴파일된 ReAct 에이전트(app) 또는 llm 이 없으면 None.
    """
    if llm is None:
        return None
    # langgraph 와 공통 도구는 함수 안에서 지연 import 한다(순환/미설치 보호).
    from langgraph.prebuilt import create_react_agent
    from . import tools as tools_mod

    if tools is None:
        tools = tools_mod.AGENT_TOOLS + tools_mod.FILE_TOOLS
    prompt = prompt or "당신은 도구를 사용할 수 있는 AI 에이전트입니다. 항상 한국어로 답변하세요."
    return create_react_agent(model=llm, tools=tools, prompt=prompt)


# ─────────────────────────────────────────────────────────────────────────────
# 3. 멀티 에이전트 연구 시스템 (연구→분석→작성→검토 루프)
# ─────────────────────────────────────────────────────────────────────────────
class ResearchAgentState(TypedDict):
    """멀티 에이전트 연구 파이프라인의 공유 상태.

    Attributes:
        topic: 조사 주제.
        research_results: 수집된 정보(연구 에이전트가 append).
        analysis: 분석 에이전트의 인사이트.
        report: 작성 에이전트의 보고서 본문.
        review_feedback: 검토 에이전트의 피드백.
        is_approved: 검토 승인 여부.
        revision_count: 보고서 개정 횟수.
    """
    topic: str
    research_results: Annotated[List[str], operator.add]
    analysis: Optional[str]
    report: Optional[str]
    review_feedback: Optional[str]
    is_approved: bool
    revision_count: int


def build_research_app(llm=None):
    """연구→분석→작성→검토 멀티 에이전트 그래프를 컴파일해 반환한다.

    각 에이전트 노드는 `llm` 을 캡처하는 클로저다. `llm` 이 주어지면 분석/작성
    에이전트가 LLM 을 호출(응답은 to_text 정규화)하고, 없거나 실패하면 템플릿
    기반으로 동작한다. 검토 에이전트는 첫 회 수정 요청 → 이후 승인하는
    교육용 휴리스틱이다.

    Args:
        llm: 분석·작성에 사용할 LangChain BaseChatModel(없으면 템플릿).

    Returns:
        컴파일된 LangGraph 앱(`app.invoke(state)` 로 실행).
    """
    from langgraph.graph import StateGraph, END, START

    def researcher_agent(state: ResearchAgentState) -> ResearchAgentState:
        """연구 에이전트 — 여러 소스에서 정보를 수집한다(시뮬레이션)."""
        topic = state["topic"]
        print(f"\n[연구 에이전트] '{topic}' 조사 시작")
        results = [
            f"{topic} - 기술 동향: LLM 기반 에이전트 급성장",
            f"{topic} - 시장 분석: 글로벌 AI 에이전트 시장 $50B 규모 예상",
            f"{topic} - 주요 플레이어: OpenAI, Anthropic, Google DeepMind",
        ]
        print(f"  수집된 정보: {len(results)}건")
        return {"research_results": results}

    def analyst_agent(state: ResearchAgentState) -> ResearchAgentState:
        """분석 에이전트 — 수집 정보를 분석해 인사이트를 만든다(LLM 있으면 활용)."""
        results = state["research_results"]
        print(f"\n[분석 에이전트] {len(results)}개 결과 분석")

        analysis = None
        if llm is not None:
            try:
                joined = "\n".join(f"- {r}" for r in results)
                analysis = invoke_text(
                    llm,
                    f"다음 수집 정보를 바탕으로 핵심 트렌드·주목점·리스크를 간결히 분석해줘.\n{joined}",
                )
            except Exception as e:
                print(f"  (LLM 분석 실패 → 템플릿으로 대체: {e})")
                analysis = None
        if not analysis:
            analysis = (
                "## 분석 결과\n"
                f"- 수집 데이터: {len(results)}건\n"
                "- 핵심 트렌드: AI 에이전트 기술이 빠르게 성숙 단계로 진입\n"
                "- 주목할 점: 멀티모달 + 자율 에이전트 결합이 핵심 경쟁력\n"
                "- 리스크: 거버넌스와 안전성 확보가 상업화의 관건"
            )
        print("  분석 완료")
        return {"analysis": analysis}

    def writer_agent(state: ResearchAgentState) -> ResearchAgentState:
        """작성 에이전트 — 분석 결과를 보고서로 정리한다."""
        topic = state["topic"]
        analysis = state["analysis"]
        results = state["research_results"]
        revision = state.get("revision_count", 0)
        print(f"\n[작성 에이전트] 보고서 작성 (개정 {revision}회)")

        body = "\n".join("- " + r for r in results)
        report = (
            f"# {topic} 분석 보고서\n"
            f"작성일: {datetime.now().strftime('%Y-%m-%d')}\n"
            f"개정: {revision}회\n\n"
            f"## 수집 정보\n{body}\n\n"
            f"{analysis}\n\n"
            "## 결론\n"
            "AI 에이전트 기술은 실용화 단계에 접어들 것으로 전망됩니다."
        )
        print(f"  보고서 작성 완료 ({len(report)} 문자)")
        return {"report": report}

    def reviewer_agent(state: ResearchAgentState) -> ResearchAgentState:
        """검토 에이전트 — 첫 회는 수정 요청, 이후 승인(교육용 휴리스틱)."""
        revision = state.get("revision_count", 0)
        print("\n[검토 에이전트] 보고서 검토 중...")
        if revision == 0:
            feedback = "보고서에 구체적인 수치 데이터와 출처가 필요합니다. 수정 요청."
            approved = False
            print(f"  결과: 수정 요청 - {feedback}")
        else:
            feedback = "보고서 품질이 기준을 충족합니다. 승인."
            approved = True
            print("  결과: 승인")
        return {"review_feedback": feedback, "is_approved": approved, "revision_count": revision + 1}

    def route_after_review(state: ResearchAgentState):
        """검토 후 라우팅 — 승인 또는 최대 개정(3회) 시 종료, 아니면 재작성."""
        if state["is_approved"] or state.get("revision_count", 0) >= 3:
            return END
        return "writer_agent"

    workflow = StateGraph(ResearchAgentState)
    workflow.add_node("researcher_agent", researcher_agent)
    workflow.add_node("analyst_agent", analyst_agent)
    workflow.add_node("writer_agent", writer_agent)
    workflow.add_node("reviewer_agent", reviewer_agent)
    workflow.add_edge(START, "researcher_agent")
    workflow.add_edge("researcher_agent", "analyst_agent")
    workflow.add_edge("analyst_agent", "writer_agent")
    workflow.add_edge("writer_agent", "reviewer_agent")
    workflow.add_conditional_edges("reviewer_agent", route_after_review)
    return workflow.compile()
