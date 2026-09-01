"""
governance — 모듈 2(통제): 가드레일 · 트레이싱 · 안전 에이전트
=============================================================

모듈 2(거버넌스, M03_* 노트북들)에서 반복적으로 등장하던 길고 일관된
거버넌스/트레이싱 구현을 분리해 둔 모듈입니다. 노트북은 '개념과 시연'에
집중하고, 클래스 구현 세부는 여기서 import 해서 재사용합니다.

    GuardrailResult       가드레일 판정 결과 enum (통과/차단/경고)
    GuardrailCheck        한 번의 가드레일 검사 결과(판정 + 사유 + 수정본)
    InputGuardrail        사용자 입력 검사(프롬프트 주입/민감 주제/PII/길이)
    OutputGuardrail       AI 응답 검사(기밀 노출/독성 콘텐츠)
    AgentTracer           Chain of Thought 실행 추적기(입력/사고/행동/관찰/최종답변)
    SafeTraceableAgent    가드레일 + 트레이싱을 결합한 LangChain 기반 안전 에이전트
    SelfCorrectingAgent   LLM 자기 비평으로 품질 기준까지 반복 개선하는 자기 수정 에이전트

note:
    LLM 을 호출하는 구현(SafeTraceableAgent, SelfCorrectingAgent)은 LLM 인스턴스를 **생성자로 주입**받고,
    응답은 `bootstrap.to_text()` 로 정규화해 공급자별 content 형식 차이(Gemini 의
    list, qwen3 의 <think> 등)를 흡수합니다.
"""

import math
import re
import time
import uuid
import json
import logging
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .bootstrap import to_text, invoke_text, bind_tools, cap_tool_calls


# ============================================================
# 가드레일 결과 표현
# ============================================================
class GuardrailResult(Enum):
    """가드레일 판정 결과."""
    PASS = "통과"
    BLOCK = "차단"
    WARN = "경고"


@dataclass
class GuardrailCheck:
    """한 번의 가드레일 검사 결과.

    Attributes:
        result: 판정(PASS/BLOCK/WARN).
        reason: 판정 사유(사람이 읽는 설명).
        modified_text: 입력을 마스킹/절단한 경우의 수정본(없으면 None).
    """
    result: GuardrailResult
    reason: str
    modified_text: Optional[str] = None


# ============================================================
# 입력 가드레일
# ============================================================
class InputGuardrail:
    """입력 가드레일: 사용자 입력을 검사한다.

    프롬프트 주입 시도, 민감/유해 주제, 개인정보(PII), 과도한 길이를
    정규식으로 점검해 통과/차단/경고 중 하나로 판정한다.
    """

    # 프롬프트 주입(역할 탈취·지시 무시) 패턴
    INJECTION_PATTERNS = [
        r"ignore previous instructions",
        r"ignore all instructions",
        r"you are now",
        r"forget everything",
        r"jailbreak",
        r"system prompt",
        r"이전 지시사항 무시",
        r"모든 규칙 무시",
        r"역할극",
    ]

    # 민감/유해 주제 패턴
    SENSITIVE_TOPICS = [
        r"폭발물", r"무기 제조", r"해킹 방법",
        r"개인정보 훔치기", r"사기 방법",
    ]

    # 개인정보(PII) 패턴 — 감지 시 마스킹 처리
    PII_PATTERNS = {
        "주민등록번호": r"\d{6}-[1-4]\d{6}",
        "신용카드번호": r"\d{4}-\d{4}-\d{4}-\d{4}",
        "전화번호": r"010-\d{4}-\d{4}",
    }

    def check(self, user_input: str) -> GuardrailCheck:
        """입력 텍스트를 검사해 GuardrailCheck 를 반환한다.

        Args:
            user_input: 사용자 입력 문자열.

        Returns:
            판정/사유/수정본을 담은 GuardrailCheck.
        """
        lower_input = user_input.lower()

        # 1. 프롬프트 주입 검사 → 차단
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, lower_input, re.IGNORECASE):
                return GuardrailCheck(
                    result=GuardrailResult.BLOCK,
                    reason=f"프롬프트 주입 시도 감지: '{pattern}' 패턴",
                )

        # 2. 민감한 주제 검사 → 차단
        for pattern in self.SENSITIVE_TOPICS:
            if re.search(pattern, user_input, re.IGNORECASE):
                return GuardrailCheck(
                    result=GuardrailResult.BLOCK,
                    reason=f"민감한 주제 감지: '{pattern}'",
                )

        # 3. PII 검사 → 경고(마스킹된 수정본 제공)
        for pii_type, pattern in self.PII_PATTERNS.items():
            if re.search(pattern, user_input):
                masked = re.sub(pattern, f"[{pii_type} 마스킹됨]", user_input)
                return GuardrailCheck(
                    result=GuardrailResult.WARN,
                    reason=f"{pii_type} 감지됨 - 마스킹 처리",
                    modified_text=masked,
                )

        # 4. 길이 검사 → 경고(5000자로 절단)
        if len(user_input) > 5000:
            return GuardrailCheck(
                result=GuardrailResult.WARN,
                reason="입력이 너무 깁니다. 5000자로 제한됩니다.",
                modified_text=user_input[:5000],
            )

        return GuardrailCheck(result=GuardrailResult.PASS, reason="모든 검사 통과")


# ============================================================
# 출력 가드레일
# ============================================================
class OutputGuardrail:
    """출력 가드레일: AI 응답을 검사한다.

    기밀 정보(시스템 프롬프트 등) 노출과 독성/유해 콘텐츠를 점검한다.
    """

    HALLUCINATION_INDICATORS = [
        "실제로 이것이 사실인지 확실하지 않지만",
        "잘못된 정보일 수 있습니다",
    ]

    TOXIC_PATTERNS = [
        r"욕설", r"혐오", r"차별",
    ]

    CONFIDENTIAL_PATTERNS = [
        r"시스템 프롬프트",
        r"내 지시사항은",
        r"내부 규칙:",
    ]

    def check(self, ai_output: str) -> GuardrailCheck:
        """AI 출력 텍스트를 검사해 GuardrailCheck 를 반환한다.

        Args:
            ai_output: 모델이 생성한 응답 문자열.

        Returns:
            판정/사유를 담은 GuardrailCheck.
        """
        # 1. 기밀 정보 노출 검사 → 차단
        for pattern in self.CONFIDENTIAL_PATTERNS:
            if re.search(pattern, ai_output, re.IGNORECASE):
                return GuardrailCheck(
                    result=GuardrailResult.BLOCK,
                    reason="기밀 정보 노출 시도 감지",
                )

        # 2. 독성 콘텐츠 검사 → 차단
        for pattern in self.TOXIC_PATTERNS:
            if re.search(pattern, ai_output, re.IGNORECASE):
                return GuardrailCheck(
                    result=GuardrailResult.BLOCK,
                    reason=f"부적절한 콘텐츠 감지: '{pattern}'",
                )

        return GuardrailCheck(result=GuardrailResult.PASS, reason="출력 검사 통과")


# ============================================================
# 에이전트 실행 추적기 (Chain of Thought 로깅)
# ============================================================
class AgentTracer:
    """에이전트 실행 추적기 (Chain of Thought 로그).

    사용자 입력 → 사고(Thought) → 행동(Action) → 관찰(Observation) →
    최종 답변(Final Answer) 의 각 단계를 구조화된 이벤트로 기록하고,
    동시에 사람이 읽을 수 있는 로그를 스트림으로 출력한다.
    """

    def __init__(self, session_id: str = None):
        """추적기를 초기화한다.

        Args:
            session_id: 세션 식별자. None 이면 8자리 임의 ID 를 생성한다.
        """
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.traces: List[Dict] = []
        self.current_span: Optional[Dict] = None
        self._setup_logger()

    def _setup_logger(self):
        """세션별 구조화 로거를 설정한다(핸들러 중복 방지)."""
        self.logger = logging.getLogger(f"AgentTracer.{self.session_id}")
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_user_input(self, user_input: str, guardrail_result: GuardrailCheck = None):
        """사용자 입력과 입력 가드레일 판정을 기록한다."""
        entry = {
            "type": "user_input",
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "content": user_input[:200],
            "guardrail": {
                "result": guardrail_result.result.value if guardrail_result else "skipped",
                "reason": guardrail_result.reason if guardrail_result else None,
            } if guardrail_result else None,
        }
        self.traces.append(entry)
        self.logger.info(
            f"INPUT | guardrail={entry.get('guardrail', {}).get('result', 'N/A')} | {user_input[:80]}"
        )
        return entry

    def log_thought(self, thought: str, step: int):
        """에이전트의 사고 과정(Thought)을 기록한다."""
        entry = {
            "type": "thought",
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "content": thought,
        }
        self.traces.append(entry)
        self.logger.debug(f"THOUGHT[{step}] | {thought[:80]}")
        return entry

    def log_action(self, tool_name: str, tool_input: Any, step: int) -> str:
        """도구 호출(Action)을 기록하고 action_id 를 반환한다."""
        action_id = str(uuid.uuid4())[:8]
        entry = {
            "type": "action",
            "action_id": action_id,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "tool": tool_name,
            "input": str(tool_input)[:200],
        }
        self.traces.append(entry)
        self.logger.info(f"ACTION[{step}] | tool={tool_name} | input={str(tool_input)[:60]}")
        return action_id

    def log_observation(self, action_id: str, result: Any, elapsed_ms: float, step: int):
        """도구 실행 결과(Observation)와 소요 시간을 기록한다."""
        entry = {
            "type": "observation",
            "action_id": action_id,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "result": str(result)[:500],
            "elapsed_ms": round(elapsed_ms, 2),
        }
        self.traces.append(entry)
        self.logger.info(f"OBS[{step}] | elapsed={elapsed_ms:.1f}ms | {str(result)[:60]}")
        return entry

    def log_final_answer(self, answer: str, total_steps: int,
                         guardrail_result: GuardrailCheck = None):
        """최종 답변과 출력 가드레일 판정을 기록한다."""
        entry = {
            "type": "final_answer",
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "total_steps": total_steps,
            "answer": answer[:500],
            "output_guardrail": {
                "result": guardrail_result.result.value,
                "reason": guardrail_result.reason,
            } if guardrail_result else None,
        }
        self.traces.append(entry)
        self.logger.info(f"ANSWER | steps={total_steps} | {answer[:80]}")
        return entry

    def get_trace_summary(self) -> Dict:
        """수집된 추적 이벤트를 요약(도구 호출 수/평균 지연/차단 수 등)한다."""
        actions = [t for t in self.traces if t["type"] == "action"]
        obs = [t for t in self.traces if t["type"] == "observation"]
        blocked = [
            t for t in self.traces
            if t.get("guardrail", {}) and t["guardrail"].get("result") == "차단"
        ]

        return {
            "session_id": self.session_id,
            "total_events": len(self.traces),
            "tool_calls": len(actions),
            "tools_used": list(set(a["tool"] for a in actions)),
            "guardrail_blocks": len(blocked),
            "avg_obs_latency_ms": (
                sum(o["elapsed_ms"] for o in obs) / len(obs) if obs else 0
            ),
        }

    def save_trace(self, filepath: str):
        """추적 데이터(이벤트 + 요약)를 JSON 파일로 저장한다."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": self.session_id,
                "traces": self.traces,
                "summary": self.get_trace_summary(),
            }, f, ensure_ascii=False, indent=2)
        print(f"추적 데이터 저장: {filepath}")


# ============================================================
# 안전하고 추적 가능한 에이전트
# ============================================================
class SafeTraceableAgent:
    """안전하고 추적 가능한 에이전트 (LangChain 기반, 다중 공급자 지원).

    입력 가드레일 → LLM 도구 호출 루프(트레이싱) → 출력 가드레일 순으로
    동작한다. LLM 인스턴스를 주입받아 어떤 공급자(ollama/google 등)든 동일하게
    사용하며, 응답은 to_text() 로 정규화한다.
    """

    def __init__(self, llm=None, session_id: str = None):
        """에이전트를 초기화한다.

        Args:
            llm: LangChain BaseChatModel. None 이면 utils.get_llm() 으로 기본 공급자 사용.
            session_id: 추적 세션 식별자.
        """
        from langchain.tools import tool as lc_tool

        if llm is None:
            # 주입이 없으면 .env 의 기본 공급자(ollama/qwen3:8b)로 생성
            import utils
            llm = utils.get_llm()
        self.llm = llm
        self.input_guard = InputGuardrail()
        self.output_guard = OutputGuardrail()
        self.tracer = AgentTracer(session_id or str(uuid.uuid4())[:8])

        @lc_tool
        def calculator(expression: str) -> str:
            """수학 계산을 수행합니다."""
            try:
                result = eval(expression, {"__builtins__": {}}, {"math": math})
                return f"{expression} = {result}"
            except Exception as e:
                return f"계산 오류: {e}"

        @lc_tool
        def get_weather(city: str) -> str:
            """도시의 날씨를 조회합니다."""
            db = {"서울": "맑음 22°C", "부산": "구름 25°C", "제주": "비 20°C"}
            return f"{city}: {db.get(city, '데이터 없음')}"

        @lc_tool
        def get_current_datetime(dummy: str = "") -> str:
            """현재 날짜와 시간을 반환합니다."""
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.tools = [calculator, get_weather, get_current_datetime]
        self.tools_dict = {t.name: t for t in self.tools}
        # 단일 도구 서버(NVIDIA build)에는 parallel_tool_calls=False 를 유도해 순차 호출
        self.llm_with_tools = bind_tools(self.llm, self.tools)

    def _execute_tool(self, tool_name: str, tool_input: Dict) -> str:
        """허가된 도구만 실행한다(미등록 도구는 보안 차단)."""
        if tool_name not in self.tools_dict:
            return f"[보안] 허가되지 않은 도구: {tool_name}"
        try:
            return str(self.tools_dict[tool_name].invoke(tool_input))
        except Exception as e:
            return f"도구 실행 오류: {e}"

    def run(self, user_input: str) -> Dict[str, Any]:
        """입력→도구루프→출력 가드레일을 거쳐 사용자 요청을 처리한다.

        Args:
            user_input: 사용자 입력 문자열.

        Returns:
            status('success'/'blocked'/'output_blocked'/'error')와 결과를 담은 dict.
        """
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        print(f"\n{'='*60}")
        print(f"[세션 {self.tracer.session_id}] 실행 시작")
        print(f"사용자 입력: {user_input}")
        print(f"{'='*60}")

        # 1) 입력 가드레일
        input_check = self.input_guard.check(user_input)
        self.tracer.log_user_input(user_input, input_check)

        if input_check.result == GuardrailResult.BLOCK:
            print(f"입력 가드레일 차단: {input_check.reason}")
            return {"status": "blocked", "reason": input_check.reason, "answer": None}

        processed_input = input_check.modified_text or user_input
        if input_check.result == GuardrailResult.WARN:
            print(f"입력 경고 처리: {input_check.reason}")

        messages = [
            SystemMessage(content="당신은 안전하고 도움이 되는 AI 에이전트입니다. 항상 한국어로 답변하세요."),
            HumanMessage(content=processed_input),
        ]
        step = 0

        # 2) 도구 호출 루프 (각 단계 트레이싱)
        try:
            while step < 10:
                step += 1
                response = self.llm_with_tools.invoke(messages)
                response = cap_tool_calls(response)   # 단일 도구 서버면 첫 호출만 남김

                if response.tool_calls:
                    messages.append(response)
                    for tc in response.tool_calls:
                        self.tracer.log_thought(f"{tc['name']} 도구가 필요하다고 판단", step)
                        action_id = self.tracer.log_action(tc["name"], tc["args"], step)
                        t0 = time.time()
                        result = self._execute_tool(tc["name"], tc["args"])
                        self.tracer.log_observation(action_id, result, (time.time() - t0) * 1000, step)
                        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                else:
                    # 3) 최종 답변 — to_text 로 공급자 무관 정규화(<think>/list 제거)
                    final_text = to_text(response.content)
                    output_check = self.output_guard.check(final_text)
                    self.tracer.log_final_answer(final_text, step, output_check)

                    if output_check.result == GuardrailResult.BLOCK:
                        print(f"출력 가드레일 차단: {output_check.reason}")
                        return {"status": "output_blocked", "reason": output_check.reason}

                    print(f"\n최종 답변: {final_text}")
                    summary = self.tracer.get_trace_summary()
                    print(f"\n추적 요약: 총 {step}단계, 도구 {summary['tool_calls']}회 호출")
                    return {"status": "success", "answer": final_text, "trace_summary": summary}

        except Exception as e:
            return {"status": "error", "reason": str(e)}


# ============================================================
# 자기 수정(Self-Correction) 에이전트
# ============================================================
class SelfCorrectingAgent:
    """자기 수정 에이전트 — 생성 → 자기 비평 → 개선 루프를 실제 LLM 으로 수행한다.

    비평(critique)은 루브릭을 받은 LLM 이 JSON({quality_score, issues, suggestions})
    으로 돌려준다. 이 '순수 자기채점' 방식은 점수가 후해지는 편향이 있으므로,
    외부 검증기(verifier)를 주입하면 점수·문제점을 객관 신호로 대체해 비평을
    근거지을(grounding) 수 있다.
    참고: 'Self-Refine: Iterative Refinement with Self-Feedback' (arXiv:2303.17651)
    """

    # 자기채점의 기준. 채점 축을 명시해야 '느낌 점수'가 아니라 재현 가능한 평가가 된다.
    RUBRIC = (
        "다음 4개 기준을 각각 0~1 로 채점하고 그 평균을 quality_score 로 삼아라.\n"
        "  1) 구체성: 추상적 일반론이 아니라 실제 이름·수치·예시가 있는가\n"
        "  2) 구조: 단계/항목이 논리적 순서로 정리되었는가\n"
        "  3) 실행가능성: 읽는 사람이 그대로 따라할 수 있는가\n"
        "  4) 정확성: 사실 오류나 근거 없는 단정이 없는가"
    )

    def __init__(self, llm=None, critic_llm=None, max_iterations: int = 3, verifier=None):
        """에이전트를 초기화한다.

        Args:
            llm: 생성·개선에 쓸 LangChain BaseChatModel. None 이면 utils.get_llm()
                (temperature=0.5 — 반복마다 답이 달라질 다양성 확보).
            critic_llm: 비평 전용 모델. None 이면 llm 을 그대로 쓴다. 채점을 안정시키려면
                temperature=0 인 모델을 따로 넘기는 것이 좋다.
            max_iterations: 최대 개선 반복 횟수(비용 상한).
            verifier: 선택적 검증기 `f(response) -> (score, issues)`. 주어지면 LLM 자기채점
                대신 이 객관 신호로 점수·문제점을 덮어쓴다.
        """
        if llm is None:
            import utils
            llm = utils.get_llm(temperature=0.5)
        self.llm = llm
        self.critic_llm = critic_llm or llm
        self.max_iterations = max_iterations
        self.verifier = verifier

    def generate(self, task: str) -> str:
        """초안(초기 답변)을 LLM 으로 생성한다."""
        return invoke_text(self.llm, f"[과제]\n{task}\n\n간결하게 답하라.")

    def critique(self, task: str, response: str) -> Dict:
        """자기 비평: LLM 이 루브릭에 따라 점수·문제점·개선제안을 낸다.

        verifier 가 주입되어 있으면 점수와 문제점은 검증기의 객관 신호로 교체된다
        (개선 제안은 여전히 LLM 이 낸 것을 쓴다).

        Returns:
            {"quality_score": float, "issues": List[str], "suggestions": List[str]}
        """
        prompt = (
            f"너는 엄격한 평가자다. 아래 답변을 평가하라.\n\n"
            f"[과제]\n{task}\n\n[답변]\n{response}\n\n{self.RUBRIC}\n\n"
            "답변이 완벽하지 않다면 issues 를 최소 1개 반드시 적어라.\n"
            '다른 말 없이 JSON 만 출력하라: '
            '{"quality_score": 0.0, "issues": ["..."], "suggestions": ["..."]}'
        )
        crit = self._parse_critique(invoke_text(self.critic_llm, prompt))

        if self.verifier is not None:
            # 객관 검증기가 있으면 자기채점 대신 검증 결과로 점수·문제점을 근거짓는다
            score, issues = self.verifier(response)
            crit["quality_score"], crit["issues"] = score, issues
        return crit

    def refine(self, task: str, response: str, critique: Dict) -> str:
        """비평(문제점+제안)을 반영해 답변을 LLM 으로 다시 쓴다."""
        if not critique["issues"]:
            return response  # 고칠 점이 없으면 그대로 둔다

        guide = "\n".join(f"- {x}" for x in critique["issues"] + critique["suggestions"])
        return invoke_text(self.llm, (
            f"[과제]\n{task}\n\n[직전 답변]\n{response}\n\n[반드시 고칠 점]\n{guide}\n\n"
            "위 지침을 모두 반영한 개선 답변만 출력하라. 메타 설명은 하지 마라."
        ))

    @staticmethod
    def _parse_critique(text: str) -> Dict:
        """LLM 비평 응답에서 JSON 을 뽑아 표준 형태로 정규화한다.

        공급자에 따라 ```json 펜스·앞뒤 설명이 섞여 나오므로 첫 JSON 블록만 취하고,
        파싱에 실패해도 루프가 죽지 않도록 보수적인 폴백을 돌려준다.
        """
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                data = json.loads(match.group(0))
                score = float(data.get("quality_score", 0.5))
                return {
                    "quality_score": max(0.0, min(1.0, score)),   # 0~1 범위로 클램프
                    "issues": [str(x) for x in data.get("issues", []) or []],
                    "suggestions": [str(x) for x in data.get("suggestions", []) or []],
                }
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        return {
            "quality_score": 0.5,   # 판단 불가 → 중간 점수로 두어 개선을 한 번 더 시도하게 한다
            "issues": ["비평 JSON 파싱 실패 — 답변을 한 번 더 다듬는다"],
            "suggestions": [],
        }

    def run(self, task: str, target_quality: float = 0.85, verbose: bool = True) -> Dict:
        """생성 → (비평 → 개선) 루프를 목표 품질 또는 최대 반복까지 수행한다.

        Args:
            task: 처리할 작업 설명.
            target_quality: 도달 목표 품질 점수(0~1).
            verbose: 반복마다 점수·문제점을 출력할지 여부.

        Returns:
            최종 답변/반복 횟수/최종 점수/이력을 담은 dict.
        """
        if verbose:
            print(f"\n=== 자기 수정 에이전트 (목표 {target_quality}, 최대 {self.max_iterations}회) ===")
            print(f"과제: {task}")

        response = self.generate(task)
        crit = self.critique(task, response)                  # 초안에 대한 첫 비평
        history = [{"iteration": 0, "response": response,
                    "score": crit["quality_score"], "critique": crit}]

        for i in range(1, self.max_iterations + 1):
            if verbose:
                print(f"\n[반복 {i}] 품질 점수: {crit['quality_score']:.2f}")
                if crit["issues"]:
                    # 제안은 문제점이 있을 때만 의미가 있다(refine 은 issues 가 없으면 건너뛴다)
                    print(f"  문제점: {', '.join(crit['issues'])}")
                    if crit["suggestions"]:
                        print(f"  개선 방향: {', '.join(crit['suggestions'])}")

            if crit["quality_score"] >= target_quality:
                if verbose:
                    print(f"  ✅ 목표 품질({target_quality}) 달성 — 반복 종료")
                break

            response = self.refine(task, response, crit)      # 비평 반영 재작성
            crit = self.critique(task, response)              # 개선본을 다시 비평
            history.append({"iteration": i, "response": response,
                            "score": crit["quality_score"], "critique": crit})
            if verbose:
                print(f"  → 개선본: {response[:80].replace(chr(10), ' ')}...")
        else:
            if verbose:
                print("\n  ⏹ 최대 반복 도달 — 자기 수정은 개선을 돕지만 완벽을 보장하지 않는다")

        if verbose:
            trail = " → ".join(f"{h['score']:.2f}" for h in history)
            print(f"\n[점수 추이] {trail}  (개선 {len(history)-1}회)")

        return {"final_response": response, "iterations": len(history),
                "final_score": crit["quality_score"], "history": history}
