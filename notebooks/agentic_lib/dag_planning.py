"""
dag_planning — 고급 계획 수립·실행(Advanced Planning) 구현
=========================================================

week12-14 계획 실습(`M05_2_planning.ipynb`)의 **길고 반복되는 핵심
구현**을 노트북에서 분리한 모듈입니다. 노트북은 '개념과 시연'에 집중하고, 무거운
스키마·실행 엔진·LangGraph 상태 기계는 이 모듈을 import 해서 재사용합니다.

기본 `planning.py` 의 단순 `TaskPlanner` 와 역할이 겹치는 부분은 그쪽을 재사용하고
(예: 순차 실행 개념), 이 모듈은 그보다 한 단계 위인 **의존성 그래프(DAG) 실행**과
**LangGraph 기반 동적 계획**을 다룹니다.

구성:
    [구조화 플래닝]
        Priority / PlanTask / ExecutionPlan      Pydantic 계획 스키마
        create_execution_plan(llm, goal)         LLM 으로 구조화 계획 생성
        print_plan(plan)                         계획을 보기 좋게 출력

    [의존성 그래프 실행]
        TaskStatus / ICON                        DAG 실행 상태 enum·아이콘
        DAGExecutor                              위상 정렬 기반 태스크 실행 엔진
        visualize_dag(plan, statuses)            DAG 구조 텍스트 시각화

    [Plan-and-Execute / 재계획 — LangGraph]
        SimplePlan / PlanExecuteState / ReplanDecision / ReplanState
        build_plan_execute_app(llm, llm_with_tools, tool_map)
        build_replan_app(llm, llm_with_tools, tool_map)

    [계층적 플래닝]
        SubGoalTask / SubGoal / HierarchicalPlan
        create_hierarchical_plan(llm, goal) / print_hierarchical_plan(hplan)
        execute_hierarchical_plan(hplan, llm, tool_map)

    [멀티 에이전트 플래닝 — LangGraph]
        CritiqueResult / MultiAgentState
        build_multi_agent_app(llm, llm_with_tools, tool_map)

설계 메모:
    - LangGraph 노드는 `llm` / `llm_with_tools` / `tool_map` 을 캡처하는 **클로저**로
      만들고, `build_*_app(...)` 가 컴파일된 그래프를 돌려줍니다(전역 의존 제거).
    - 모든 LLM 응답은 `bootstrap.to_text()` 로 정규화합니다(Gemini list·qwen3 <think> 흡수).
    - `langgraph` 는 빌더 함수 안에서 **지연 import** 합니다(미설치 환경에서도 모듈 로드 가능).
"""

from enum import Enum
from typing import List, Optional, Dict, TypedDict, Annotated
import operator
import os

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from .bootstrap import to_text, cap_tool_calls


# ─────────────────────────────────────────────────────────────────────────────
# 1. 구조화 플래닝 — Pydantic 계획 스키마
# ─────────────────────────────────────────────────────────────────────────────
class Priority(str, Enum):
    """태스크 우선순위."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class PlanTask(BaseModel):
    """실행 계획의 태스크 1개(LLM 구조화 출력 대상)."""
    id: int = Field(description="고유 태스크 번호 (1부터)")
    title: str = Field(description="태스크 제목 (15자 이내)")
    description: str = Field(description="수행할 구체적인 작업 내용")
    tool: Optional[str] = Field(
        None, description="사용할 도구 (calculator/get_current_time/web_search/write_document/null)")
    tool_input: Optional[str] = Field(None, description="도구에 전달할 입력값")
    depends_on: List[int] = Field(default=[], description="먼저 완료되어야 하는 태스크 번호 목록")
    priority: Priority = Field(Priority.MEDIUM, description="우선순위")
    estimated_minutes: int = Field(5, description="예상 소요 시간(분)")


class ExecutionPlan(BaseModel):
    """목표 1개에 대한 전체 실행 계획."""
    goal: str = Field(description="달성하려는 최종 목표")
    background: str = Field(description="계획 수립 배경 및 전제")
    tasks: List[PlanTask] = Field(description="실행 태스크 목록")
    success_criteria: str = Field(description="성공 기준")
    total_estimated_minutes: int = Field(description="전체 예상 소요 시간(분)")


# 구조화 플래너에 주입하는 시스템 프롬프트(사용 가능한 도구를 명시).
PLANNING_SYSTEM_PROMPT = (
    "당신은 전문 프로젝트 매니저입니다. 사용자의 목표를 분석하여 "
    "실행 가능한 단계별 계획을 수립하세요.\n"
    "사용 가능한 도구: calculator, get_current_time, web_search, write_document"
)


def create_execution_plan(llm, goal: str) -> ExecutionPlan:
    """목표를 입력받아 구조화된 실행 계획(ExecutionPlan)을 생성한다.

    Args:
        llm: LangChain BaseChatModel.
        goal: 자연어 목표 문자열.

    Returns:
        검증된 ExecutionPlan 객체.
    """
    planner_llm = llm.with_structured_output(ExecutionPlan)
    messages = [
        SystemMessage(content=PLANNING_SYSTEM_PROMPT),
        HumanMessage(content=f"목표: {goal}"),
    ]
    return planner_llm.invoke(messages)


def print_plan(plan: ExecutionPlan) -> None:
    """ExecutionPlan 을 우선순위 아이콘과 함께 보기 좋게 출력한다."""
    print(f"[목표] {plan.goal}")
    print(f"[배경] {plan.background[:80]}")
    print(f"[성공 기준] {plan.success_criteria[:80]}")
    print(f"[예상 시간] 총 {plan.total_estimated_minutes}분")
    print(f"\n[실행 계획] {len(plan.tasks)}개 태스크:")
    for t in plan.tasks:
        dep = f" → 의존: {t.depends_on}" if t.depends_on else ""
        tool_str = f" [{t.tool}]" if t.tool else ""
        pri = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(t.priority.value, "")
        print(f"  Task {t.id:2d} {pri} [{t.estimated_minutes}분] {t.title}{tool_str}{dep}")
        print(f"         {t.description[:70]}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. 의존성 그래프(DAG) 실행 엔진
# ─────────────────────────────────────────────────────────────────────────────
class TaskStatus(str, Enum):
    """DAG 실행 중 태스크의 상태.

    참고: `planning.TaskStatus` 는 단순 순차 플래너용(대기/진행중/완료/실패)이고,
    여기에는 의존 실패 시 건너뛰는 SKIPPED 가 추가되어 DAG 실행에 특화됩니다.
    """
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


# 상태별 출력 아이콘.
ICON = {
    TaskStatus.PENDING: "⏳",
    TaskStatus.RUNNING: "🔄",
    TaskStatus.DONE: "✅",
    TaskStatus.FAILED: "❌",
    TaskStatus.SKIPPED: "⏭️",
}


class DAGExecutor:
    """의존성 그래프(DAG) 기반 태스크 실행 엔진.

    각 태스크의 `depends_on` 을 보고 선행 태스크가 모두 완료(DONE)된 것만 실행하며,
    의존 태스크가 실패/스킵되면 해당 태스크도 건너뜁니다(SKIPPED). 도구가 지정되면
    도구를, 아니면 LLM 을 호출해 태스크를 수행합니다.
    """

    def __init__(self, plan: ExecutionPlan, llm, tool_map: dict):
        """실행기를 초기화한다.

        Args:
            plan: 실행할 ExecutionPlan.
            llm: 도구 없는 태스크를 처리할 LangChain BaseChatModel.
            tool_map: {도구이름: LangChain 도구} 매핑.
        """
        self.plan = plan
        self.llm = llm
        self.tool_map = tool_map
        self.statuses: Dict[int, TaskStatus] = {
            t.id: TaskStatus.PENDING for t in plan.tasks
        }
        self.results: Dict[int, str] = {}
        self.task_map: Dict[int, PlanTask] = {t.id: t for t in plan.tasks}

    def _is_ready(self, task: PlanTask) -> bool:
        """모든 의존 태스크가 완료(DONE)되어 실행 준비가 됐는지 확인한다."""
        return all(
            self.statuses.get(dep) == TaskStatus.DONE
            for dep in task.depends_on
        )

    def _has_failed_dependency(self, task: PlanTask) -> bool:
        """의존 태스크 중 실패/스킵된 것이 있는지 확인한다."""
        return any(
            self.statuses.get(dep) in (TaskStatus.FAILED, TaskStatus.SKIPPED)
            for dep in task.depends_on
        )

    def _execute_task(self, task: PlanTask) -> str:
        """태스크를 실행한다: 도구가 있으면 도구 호출, 없으면 LLM 처리.

        선행 태스크 결과를 컨텍스트로 묶어 LLM 프롬프트에 전달한다.
        """
        # 이전(의존) 태스크 결과를 컨텍스트로 구성
        context_parts = []
        for dep_id in task.depends_on:
            if dep_id in self.results:
                dep_task = self.task_map[dep_id]
                context_parts.append(
                    f"[Task {dep_id} 결과: {dep_task.title}]\n{self.results[dep_id]}"
                )
        context = "\n\n".join(context_parts)

        # 도구가 지정된 경우 → 도구 호출
        if task.tool and task.tool in self.tool_map:
            tool_input = task.tool_input or task.description
            return self.tool_map[task.tool].invoke(tool_input)

        # 도구 없음 → LLM 으로 처리
        prompt_parts = [f"목표: {self.plan.goal}", f"현재 태스크: {task.description}"]
        if context:
            prompt_parts.append(f"참고 이전 결과:\n{context}")
        prompt_parts.append("위 태스크를 실행하고 결과를 반환하세요. 간결하게.")

        response = self.llm.invoke([HumanMessage(content="\n\n".join(prompt_parts))])
        return to_text(response.content)

    def run(self, verbose: bool = True) -> Dict[int, str]:
        """모든 태스크를 의존성 순서에 따라 실행하고 결과 딕셔너리를 반환한다.

        준비된 태스크가 더 이상 진행되지 않으면(순환 의존성/데드락) 경고 후 중단한다.

        Args:
            verbose: 진행 상황 출력 여부.

        Returns:
            {태스크 id: 결과 문자열} 매핑.
        """
        if verbose:
            print(f"[목표] {self.plan.goal}")
            print("=" * 60)

        remaining = list(self.task_map.keys())
        max_iter = len(remaining) * 2

        for _iteration in range(max_iter):
            if not remaining:
                break

            progress = False
            for task_id in list(remaining):
                task = self.task_map[task_id]

                # 의존 태스크 실패 → 건너뜀
                if self._has_failed_dependency(task):
                    self.statuses[task_id] = TaskStatus.SKIPPED
                    remaining.remove(task_id)
                    if verbose:
                        print(f"{ICON[TaskStatus.SKIPPED]} Task {task_id}: {task.title} (의존 태스크 실패)")
                    progress = True
                    continue

                # 아직 준비 안 됨(선행 미완료)
                if not self._is_ready(task):
                    continue

                # 실행
                self.statuses[task_id] = TaskStatus.RUNNING
                if verbose:
                    tool_tag = f" [{task.tool}]" if task.tool else ""
                    print(f"{ICON[TaskStatus.RUNNING]} Task {task_id}: {task.title}{tool_tag}")

                try:
                    result = self._execute_task(task)
                    self.results[task_id] = result
                    self.statuses[task_id] = TaskStatus.DONE
                    if verbose:
                        print(f"   {ICON[TaskStatus.DONE]} 결과: {str(result)[:90]}")
                except Exception as e:
                    self.statuses[task_id] = TaskStatus.FAILED
                    if verbose:
                        print(f"   {ICON[TaskStatus.FAILED]} 오류: {e}")

                remaining.remove(task_id)
                progress = True

            if not progress:  # 데드락 감지
                if verbose:
                    print(f"경고: 순환 의존성 또는 데드락 감지. 남은 태스크: {remaining}")
                break

        done = sum(1 for s in self.statuses.values() if s == TaskStatus.DONE)
        total = len(self.statuses)
        if verbose:
            print(f"\n완료: {done}/{total} 태스크")

        return self.results


def visualize_dag(plan: ExecutionPlan, statuses: dict = None) -> None:
    """DAG 구조를 위상 정렬해 '실행 레이어'별로 텍스트 시각화한다.

    Args:
        plan: 시각화할 ExecutionPlan.
        statuses: {태스크 id: TaskStatus} (선택). 주면 상태 아이콘을 함께 표시한다.
    """
    print(f"[목표] {plan.goal}")
    print(f"총 {len(plan.tasks)}개 태스크\n")

    task_map = {t.id: t for t in plan.tasks}

    # 실행 레이어별 분류(위상 정렬): 선행이 모두 끝난 태스크를 같은 레이어로 묶는다.
    layers = []
    remaining_ids = set(t.id for t in plan.tasks)
    done_ids = set()

    while remaining_ids:
        layer = [
            tid for tid in remaining_ids
            if all(dep in done_ids for dep in task_map[tid].depends_on)
        ]
        if not layer:  # 순환 의존성 → 더 진행 불가
            break
        layers.append(layer)
        done_ids.update(layer)
        remaining_ids -= set(layer)

    # 레이어별 출력
    for level, layer in enumerate(layers):
        prefix = f"레이어 {level + 1}"
        items = []
        for tid in layer:
            t = task_map[tid]
            st = ""
            if statuses:
                st = ICON.get(statuses.get(tid, TaskStatus.PENDING), "")
            tool_str = f"[{t.tool}]" if t.tool else ""
            items.append(f"{st}T{tid}:{t.title[:12]}{tool_str}")
        print(f"  {prefix}: {'  |  '.join(items)}")

        if level < len(layers) - 1:
            print("          " + "   ↓   " * len(layer))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plan-and-Execute / 동적 재계획 — LangGraph 상태 기계
# ─────────────────────────────────────────────────────────────────────────────
class SimplePlan(BaseModel):
    """LLM 이 반환하는 단순 단계 목록(문자열 리스트)."""
    steps: List[str] = Field(description="순서대로 실행할 작업 목록 (5개 이내)")


class PlanExecuteState(TypedDict):
    """Plan-and-Execute 그래프의 상태.

    Attributes:
        goal: 달성 목표.
        plan: 남은 태스크 목록(앞에서부터 하나씩 소비).
        executed: 완료된 태스크+결과(누적; operator.add 로 append).
        final_answer: 최종 종합 답변.
    """
    goal: str
    plan: List[str]
    executed: Annotated[List[dict], operator.add]
    final_answer: str


class ReplanDecision(BaseModel):
    """재계획 노드의 판단 결과."""
    needs_replan: bool = Field(description="계획 수정이 필요한지 여부")
    new_tasks: List[str] = Field(default=[], description="추가할 새 태스크 목록")
    reason: str = Field(description="재계획 판단 이유")
    is_goal_achieved: bool = Field(description="목표가 이미 달성되었는지")


class ReplanState(TypedDict):
    """동적 재계획 그래프의 상태(PlanExecuteState + 재계획 횟수)."""
    goal: str
    plan: List[str]
    executed: Annotated[List[dict], operator.add]
    replan_count: int
    final_answer: str


def _make_plan_node(llm):
    """목표를 단계별 계획(SimplePlan)으로 분해하는 plan 노드를 생성한다."""
    def plan_node(state) -> dict:
        structured = llm.with_structured_output(SimplePlan)
        prompt = [
            SystemMessage(content=(
                "주어진 목표를 달성하기 위한 단계별 실행 계획을 수립하세요.\n"
                "각 단계는 명확하고 실행 가능해야 합니다. 5단계 이내로."
            )),
            HumanMessage(content=f"목표: {state['goal']}"),
        ]
        result = structured.invoke(prompt)
        print(f"[플래너] {len(result.steps)}단계 계획 수립:")
        for i, step in enumerate(result.steps, 1):
            print(f"  {i}. {step}")
        return {"plan": result.steps}
    return plan_node


def _make_execute_node(llm, llm_with_tools, tool_map):
    """계획의 첫 태스크를 도구와 함께 실행하는 execute 노드를 생성한다."""
    def execute_node(state) -> dict:
        current = state["plan"][0]
        remaining = state["plan"][1:]

        # 이전 결과를 컨텍스트로 구성
        context = "\n".join(
            f"  - {r['task']}: {r['result'][:60]}" for r in state["executed"]
        )
        sys_content = f"목표: {state['goal']}"
        if context:
            sys_content += "\n완료된 작업:\n" + context

        prompt = [
            SystemMessage(content=sys_content),
            HumanMessage(content=(
                f"현재 태스크: {current}\n\n"
                "이 태스크를 실행하고 결과를 간결하게 반환하세요. 필요하면 도구를 사용하세요."
            )),
        ]

        # 도구 사용 루프(최대 3회)
        messages = prompt[:]
        result_text = "완료"
        for _ in range(3):
            resp = llm_with_tools.invoke(messages)
            resp = cap_tool_calls(resp)   # 단일 도구 서버(NVIDIA build)면 첫 호출만 남김
            messages.append(resp)
            if not resp.tool_calls:
                result_text = to_text(resp.content)
                break
            for tc in resp.tool_calls:
                print(f"  [실행자 → 도구] {tc['name']}({tc['args']})")
                tool_result = tool_map[tc["name"]].invoke(tc["args"])
                messages.append(ToolMessage(content=str(tool_result), tool_call_id=tc["id"]))
        else:
            result_text = to_text(messages[-1].content) if messages else "완료"

        print(f"[실행자] \"{current[:40]}\" → {str(result_text)[:70]}")
        return {
            "plan": remaining,
            "executed": [{"task": current, "result": str(result_text)}],
        }
    return execute_node


def _make_synthesize_node(llm):
    """실행 결과 전체를 종합해 최종 답변을 만드는 synthesize 노드를 생성한다."""
    def synthesize_node(state) -> dict:
        results_text = "\n".join(
            f"{i + 1}. [{r['task']}]\n   결과: {r['result'][:100]}"
            for i, r in enumerate(state["executed"])
        )
        prompt = [
            SystemMessage(content="실행된 모든 작업의 결과를 종합하여 목표 달성 보고서를 작성하세요."),
            HumanMessage(content=f"목표: {state['goal']}\n\n실행 결과:\n{results_text}"),
        ]
        response = llm.invoke(prompt)
        print("\n[종합자] 최종 답변 생성 완료")
        return {"final_answer": to_text(response.content)}
    return synthesize_node


def _route_after_execute(state) -> str:
    """남은 계획이 있으면 execute, 없으면 synthesize 로 분기한다."""
    return "execute" if state["plan"] else "synthesize"


def build_plan_execute_app(llm, llm_with_tools, tool_map):
    """Plan-and-Execute LangGraph 앱을 구성해 컴파일된 그래프를 반환한다.

    흐름: START → plan → execute(반복) → synthesize → END

    Args:
        llm: 계획/종합용 LangChain BaseChatModel.
        llm_with_tools: 도구가 바인딩된 LLM(실행 노드용).
        tool_map: {도구이름: 도구} 매핑.

    Returns:
        컴파일된 LangGraph 앱(.invoke 가능).
    """
    from langgraph.graph import StateGraph, START, END

    plan_node = _make_plan_node(llm)
    execute_node = _make_execute_node(llm, llm_with_tools, tool_map)
    synthesize_node = _make_synthesize_node(llm)

    builder = StateGraph(PlanExecuteState)
    builder.add_node("plan", plan_node)
    builder.add_node("execute", execute_node)
    builder.add_node("synthesize", synthesize_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "execute")
    builder.add_conditional_edges("execute", _route_after_execute, {
        "execute": "execute",
        "synthesize": "synthesize",
    })
    builder.add_edge("synthesize", END)
    return builder.compile()


def _make_replan_node(llm, max_replans: int = 2):
    """실행 결과를 평가해 새 태스크를 추가/종료를 결정하는 replan 노드를 생성한다."""
    def replan_node(state) -> dict:
        if state["replan_count"] >= max_replans:
            print(f"  [재계획] 최대 재계획 횟수({max_replans}) 도달 → 종합으로 이동")
            return {"plan": [], "replan_count": state["replan_count"]}

        results_text = "\n".join(
            f"- {r['task']}: {r['result'][:80]}" for r in state["executed"]
        )
        structured = llm.with_structured_output(ReplanDecision)
        prompt = [
            SystemMessage(content="AI 에이전트 플래너입니다. 실행 결과를 평가하고 추가 작업이 필요한지 판단하세요."),
            HumanMessage(content=(
                f"목표: {state['goal']}\n\n"
                f"완료된 작업:\n{results_text}\n\n"
                f"남은 계획: {state['plan']}\n\n"
                "목표를 달성하기 위해 추가 작업이 필요한가요?"
            )),
        ]
        decision = structured.invoke(prompt)
        print(f"  [재계획] 수정 필요: {decision.needs_replan} | 이유: {decision.reason[:60]}")

        if decision.is_goal_achieved:
            return {"plan": [], "replan_count": state["replan_count"]}

        if decision.needs_replan and decision.new_tasks:
            new_plan = decision.new_tasks + state["plan"]
            print(f"  [재계획] 새 태스크 {len(decision.new_tasks)}개 추가: {decision.new_tasks[:2]}")
            return {"plan": new_plan, "replan_count": state["replan_count"] + 1}

        return {"replan_count": state["replan_count"]}
    return replan_node


def _route_after_replan(state) -> str:
    """추가된 계획이 있으면 execute, 없으면 synthesize 로 분기한다."""
    return "execute" if state["plan"] else "synthesize"


def build_replan_app(llm, llm_with_tools, tool_map, max_replans: int = 2):
    """동적 재계획 LangGraph 앱을 구성해 컴파일된 그래프를 반환한다.

    흐름: START → plan → execute → replan → (추가 필요?) execute / synthesize → END

    Args:
        llm: 계획/종합/재계획용 LLM.
        llm_with_tools: 도구 바인딩 LLM(실행 노드용).
        tool_map: {도구이름: 도구} 매핑.
        max_replans: 무한 루프 방지를 위한 최대 재계획 횟수.

    Returns:
        컴파일된 LangGraph 앱.
    """
    from langgraph.graph import StateGraph, START, END

    plan_node = _make_plan_node(llm)
    execute_node = _make_execute_node(llm, llm_with_tools, tool_map)
    replan_node = _make_replan_node(llm, max_replans=max_replans)
    synthesize_node = _make_synthesize_node(llm)

    rb = StateGraph(ReplanState)
    rb.add_node("plan", plan_node)
    rb.add_node("execute", execute_node)
    rb.add_node("replan", replan_node)
    rb.add_node("synthesize", synthesize_node)

    rb.add_edge(START, "plan")
    rb.add_edge("plan", "execute")
    # 실행 후에는 곧장 종합하지 않고 재계획 노드로 보내 추가 작업 필요성을 점검한다.
    rb.add_conditional_edges("execute", _route_after_execute, {
        "execute": "execute",
        "synthesize": "replan",
    })
    rb.add_conditional_edges("replan", _route_after_replan, {
        "execute": "execute",
        "synthesize": "synthesize",
    })
    rb.add_edge("synthesize", END)
    return rb.compile()


# ─────────────────────────────────────────────────────────────────────────────
# 4. 계층적 플래닝(Hierarchical Planning)
# ─────────────────────────────────────────────────────────────────────────────
class SubGoalTask(BaseModel):
    """서브목표 안의 구체적 태스크."""
    id: str = Field(description="태스크 ID (예: A-1, B-2)")
    description: str = Field(description="구체적인 작업 내용")
    tool: Optional[str] = Field(None, description="사용할 도구")
    estimated_minutes: int = Field(5)


class SubGoal(BaseModel):
    """최종 목표를 구성하는 서브목표(여러 태스크를 묶음)."""
    id: str = Field(description="서브목표 ID (A, B, C...)")
    title: str = Field(description="서브목표 제목")
    objective: str = Field(description="이 서브목표의 달성 기준")
    tasks: List[SubGoalTask] = Field(description="세부 태스크 목록")
    depends_on: List[str] = Field(default=[], description="의존하는 서브목표 ID")


class HierarchicalPlan(BaseModel):
    """목표 → 서브목표 → 태스크의 2계층 분해 계획."""
    goal: str = Field(description="최종 목표")
    sub_goals: List[SubGoal] = Field(description="서브목표 목록 (3개 이내)")
    integration_task: str = Field(description="모든 서브목표 완료 후 통합 작업")


HIERARCHICAL_SYSTEM_PROMPT = (
    "복잡한 목표를 계층적으로 분해하는 전문 기획자입니다.\n"
    "목표를 2~3개의 서브목표로 나누고, 각 서브목표를 2~3개의 구체적 태스크로 분해하세요.\n"
    "사용 가능한 도구: calculator, get_current_time, web_search, write_document"
)


def create_hierarchical_plan(llm, goal: str) -> HierarchicalPlan:
    """목표를 2계층(서브목표/태스크)으로 분해한 HierarchicalPlan 을 생성한다."""
    hier_planner = llm.with_structured_output(HierarchicalPlan)
    prompt = [
        SystemMessage(content=HIERARCHICAL_SYSTEM_PROMPT),
        HumanMessage(content=f"목표: {goal}"),
    ]
    return hier_planner.invoke(prompt)


def print_hierarchical_plan(hplan: HierarchicalPlan) -> None:
    """HierarchicalPlan 을 계층 구조로 보기 좋게 출력한다."""
    print(f"\n[최종 목표] {hplan.goal}\n")
    for sg in hplan.sub_goals:
        dep = f" (의존: {sg.depends_on})" if sg.depends_on else ""
        print(f"[서브목표 {sg.id}] {sg.title}{dep}")
        print(f"  달성 기준: {sg.objective[:70]}")
        for t in sg.tasks:
            tool_str = f" [{t.tool}]" if t.tool else ""
            print(f"    - {t.id}: {t.description[:60]}{tool_str}")
        print()
    print(f"[통합 작업] {hplan.integration_task}")


def execute_hierarchical_plan(hplan: HierarchicalPlan, llm, tool_map: dict) -> dict:
    """서브목표 의존성 순서대로 계층적 계획을 실행한다.

    각 서브목표 안의 태스크를 순차 실행(도구 우선, 실패 시 LLM 폴백)하고, 서브목표
    단위로 결과를 요약한 뒤, 마지막에 통합 작업으로 최종 보고서를 만든다.

    Args:
        hplan: 실행할 HierarchicalPlan.
        llm: 처리/요약/통합용 LLM.
        tool_map: {도구이름: 도구} 매핑.

    Returns:
        {'sub_goal_results': {서브목표 id: 요약}, 'final': 최종 통합 결과}.
    """
    sg_results: dict = {}   # 서브목표별 요약 결과
    sg_done: set = set()
    sg_map = {sg.id: sg for sg in hplan.sub_goals}

    print(f"[목표] {hplan.goal}\n")

    # 서브목표 실행(의존성 순서)
    remaining_sgs = list(sg_map.keys())
    while remaining_sgs:
        for sg_id in list(remaining_sgs):
            sg = sg_map[sg_id]
            if not all(dep in sg_done for dep in sg.depends_on):
                continue

            print(f"━━ 서브목표 [{sg.id}]: {sg.title} ━━")
            task_results = []

            # 서브목표 내 태스크 순차 실행
            for t in sg.tasks:
                print(f"  🔄 {t.id}: {t.description[:50]}")

                raw = None
                if t.tool and t.tool in tool_map:
                    try:
                        raw = tool_map[t.tool].invoke(t.description)
                    except Exception:
                        # write_document 처럼 인자가 여러 개인 도구는 단일 문자열 호출이
                        # 실패하므로 LLM 처리로 폴백한다.
                        raw = None
                if raw is None:
                    context = "\n".join(f"  - {r}" for r in task_results)
                    prompt_content = (
                        f"서브목표: {sg.title}\n"
                        f"현재 태스크: {t.description}\n"
                    )
                    if context:
                        prompt_content += "이전 결과:\n" + context + "\n"
                    prompt_content += "이 태스크를 실행하고 결과를 한 문단으로 반환하세요."
                    raw = to_text(llm.invoke([HumanMessage(content=prompt_content)]).content)

                result_str = str(raw)[:80]
                task_results.append(f"{t.id}: {result_str}")
                print(f"     ✅ {result_str}")

            # 서브목표 결과 요약
            sg_summary = to_text(llm.invoke([
                SystemMessage(content=f"서브목표 [{sg.title}] 의 작업 결과를 2문장으로 요약하세요."),
                HumanMessage(content="\n".join(task_results)),
            ]).content)
            sg_results[sg_id] = sg_summary
            sg_done.add(sg_id)
            print(f"  📋 서브목표 완료: {str(sg_summary)[:80]}\n")
            remaining_sgs.remove(sg_id)
            break

    # 통합 작업
    print(f"━━ 통합 작업: {hplan.integration_task} ━━")
    all_results = "\n".join(f"[{sid}] {res}" for sid, res in sg_results.items())
    final = to_text(llm.invoke([
        SystemMessage(content="모든 서브목표의 결과를 통합하여 최종 보고서를 작성하세요."),
        HumanMessage(content=(
            f"목표: {hplan.goal}\n\n결과:\n{all_results}\n\n작업: {hplan.integration_task}"
        )),
    ]).content)

    print(f"\n[최종 통합 결과]\n{str(final)[:300]}")
    return {"sub_goal_results": sg_results, "final": final}


# ─────────────────────────────────────────────────────────────────────────────
# 5. 멀티 에이전트 플래닝 — 역할 분담 협업(LangGraph)
# ─────────────────────────────────────────────────────────────────────────────
class CritiqueResult(BaseModel):
    """검증자 에이전트의 품질 평가 결과."""
    score: int = Field(description="품질 점수 0-10")
    passed: bool = Field(description="품질 기준 통과 여부 (8점 이상)")
    feedback: str = Field(description="구체적인 피드백")
    improvements: List[str] = Field(default=[], description="개선 사항 목록")


class MultiAgentState(TypedDict):
    """멀티 에이전트(플래너/실행자/검증자/재작성자/최종화) 그래프 상태."""
    goal: str
    plan: List[str]
    executed: Annotated[List[dict], operator.add]
    draft: str                          # 실행자가 생성한 초안
    critique: Optional[CritiqueResult]  # 검증자 피드백
    revision_count: int
    final_output: str


def build_multi_agent_app(llm, llm_with_tools, tool_map, max_revisions: int = 2):
    """역할 분담형 멀티 에이전트 LangGraph 앱을 구성해 반환한다.

    흐름: planner → executor → critic → (통과 못 함?) rewriter → critic / finalizer → END

    Args:
        llm: 각 에이전트의 추론용 LLM.
        llm_with_tools: 도구 바인딩 LLM(실행자용).
        tool_map: {도구이름: 도구} 매핑.
        max_revisions: 재작성 최대 횟수(무한 루프 방지).

    Returns:
        컴파일된 LangGraph 앱.
    """
    from langgraph.graph import StateGraph, START, END

    # ── 플래너 에이전트 ──
    def planner_agent(state) -> dict:
        """전략적 관점에서 최적 실행 계획을 수립한다."""
        class AgentPlan(BaseModel):
            steps: List[str] = Field(description="실행 단계 (4개 이내)")
            strategy: str = Field(description="전략적 접근 방식")

        structured = llm.with_structured_output(AgentPlan)
        result = structured.invoke([
            SystemMessage(content=(
                "당신은 전략적 사고를 하는 플래너입니다.\n"
                "목표를 분석하고 최적의 실행 계획을 수립하세요.\n"
                "사용 가능 도구: web_search, calculator, write_document"
            )),
            HumanMessage(content=f"목표: {state['goal']}"),
        ])
        print(f"[플래너] 전략: {result.strategy[:60]}")
        print(f"[플래너] {len(result.steps)}단계 계획: {result.steps}")
        return {"plan": result.steps}

    # ── 실행자 에이전트 ──
    def executor_agent(state) -> dict:
        """계획을 실행하고 결과를 종합한 초안(draft)을 생성한다."""
        if not state["plan"]:
            return {}

        all_results = []
        remaining = list(state["plan"])

        for task in remaining:
            critique = state.get("critique")
            feedback = critique.feedback if critique else "없음"
            messages = [
                SystemMessage(content=f"목표: {state['goal']}\n검증자 피드백: {feedback}"),
                HumanMessage(content=f"태스크: {task}\n결과를 반환하세요."),
            ]
            # 도구 사용 루프(최대 3회)
            tool_messages = messages[:]
            for _ in range(3):
                resp = llm_with_tools.invoke(tool_messages)
                resp = cap_tool_calls(resp)   # 단일 도구 서버(NVIDIA build)면 첫 호출만 남김
                tool_messages.append(resp)
                if not resp.tool_calls:
                    content = to_text(resp.content)
                    all_results.append(f"[{task[:30]}] {content[:80]}")
                    print(f"  [실행자] {task[:40]} → {content[:60]}")
                    break
                for tc in resp.tool_calls:
                    print(f"  [실행자 → {tc['name']}] {tc['args']}")
                    tr = tool_map[tc["name"]].invoke(tc["args"])
                    tool_messages.append(ToolMessage(content=str(tr), tool_call_id=tc["id"]))

        # 결과를 종합해 초안 생성
        draft = to_text(llm.invoke([
            SystemMessage(content="수집된 정보를 바탕으로 목표에 맞는 종합 초안을 작성하세요."),
            HumanMessage(content=f"목표: {state['goal']}\n\n수집 결과:\n" + "\n".join(all_results)),
        ]).content)
        print(f"  [실행자] 초안 작성 완료 ({len(draft)}자)")
        return {
            "plan": [],
            "executed": [{"task": t, "result": r} for t, r in zip(remaining, all_results)],
            "draft": draft,
        }

    # ── 검증자 에이전트 ──
    def critic_agent(state) -> dict:
        """초안의 품질을 평가하고 개선 사항을 제시한다."""
        structured = llm.with_structured_output(CritiqueResult)
        result = structured.invoke([
            SystemMessage(content=(
                "당신은 엄격한 품질 검토자입니다.\n"
                "초안이 목표를 충족하는지 평가하고 점수(0-10)와 피드백을 제공하세요.\n"
                "8점 미만이면 passed=False 로 설정하세요."
            )),
            HumanMessage(content=f"목표: {state['goal']}\n\n초안:\n{state['draft'][:500]}"),
        ])
        print(f"  [검증자] 점수: {result.score}/10 | 통과: {result.passed}")
        print(f"  [검증자] 피드백: {result.feedback[:80]}")
        return {"critique": result}

    # ── 재작성자 에이전트 ──
    def rewriter_agent(state) -> dict:
        """검증자 피드백을 반영하여 초안을 재작성한다."""
        critique = state["critique"]
        improvements = "\n".join(f"- {imp}" for imp in critique.improvements)
        response = to_text(llm.invoke([
            SystemMessage(content="검증자의 피드백을 반영하여 초안을 개선하세요."),
            HumanMessage(content=(
                f"목표: {state['goal']}\n\n"
                f"현재 초안:\n{state['draft'][:400]}\n\n"
                f"개선 필요 사항:\n{improvements}"
            )),
        ]).content)
        print("  [재작성자] 개선 완료")
        return {"draft": response, "revision_count": state["revision_count"] + 1}

    # ── 최종화 에이전트 ──
    def finalizer_agent(state) -> dict:
        """검증된 초안을 최종 결과물로 다듬는다."""
        critique_text = ""
        if state.get("critique"):
            critique_text = f"\n개선사항: {', '.join(state['critique'].improvements)}"
        response = to_text(llm.invoke([
            SystemMessage(content=f"초안을{critique_text} 반영하여 최종 완성본을 작성하세요."),
            HumanMessage(content=f"목표: {state['goal']}\n\n초안:\n{state['draft']}"),
        ]).content)
        print(f"  [최종화] 최종 결과물 완성 ({len(response)}자)")
        return {"final_output": response}

    # ── 라우팅: 통과하지 못했고 재작성 여유가 있으면 다시 작성 ──
    def route_after_critic(state) -> str:
        critique = state.get("critique")
        if critique and not critique.passed and state["revision_count"] < max_revisions:
            print(f"  [라우팅] 재작성 필요 (시도 {state['revision_count'] + 1}/{max_revisions})")
            return "rewrite"
        return "finalize"

    # ── 그래프 구성 ──
    mb = StateGraph(MultiAgentState)
    mb.add_node("planner", planner_agent)
    mb.add_node("executor", executor_agent)
    mb.add_node("critic", critic_agent)
    mb.add_node("rewriter", rewriter_agent)
    mb.add_node("finalizer", finalizer_agent)

    mb.add_edge(START, "planner")
    mb.add_edge("planner", "executor")
    mb.add_edge("executor", "critic")
    mb.add_conditional_edges("critic", route_after_critic, {
        "rewrite": "rewriter",
        "finalize": "finalizer",
    })
    mb.add_edge("rewriter", "critic")   # 재작성 후 다시 검증
    mb.add_edge("finalizer", END)
    return mb.compile()
