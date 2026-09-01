"""
automation — OpenClaw 스타일 업무 자동화 & 최종 통합 시스템 (모듈 4: 실행)
========================================================================

week12-14 모듈4 노트북의 **업무 자동화/최종 통합** 구현을 분리한 모듈입니다.

구성:
    WorkflowAutomationAgent   파일 시스템·데이터를 다루는 선언적 워크플로우 실행기
                              (OpenClaw 스타일). 모든 파일 입출력은 안전을 위해
                              **notebooks/workspace/ 하위로만 제한**된다(경로 가드).
    FinalAgentSystem          4개 모듈(연결·통제·지식·실행)을 통합한 최종 파이프라인.
                              llm 을 주입하면 응답 생성에 활용한다(없으면 시뮬레이션).

설계 메모:
    - 실제 파일을 만들고 지우는 OpenClaw 류 에이전트는 경로 탈출(`..`) 시 시스템
      파일을 건드릴 위험이 있어, 작업 폴더를 `notebooks/workspace/` 루트로 고정하고
      모든 경로를 그 하위로 강제한다(`_safe_path`).
    - LLM 응답은 `bootstrap.invoke_text()`/`to_text()` 로 정규화한다.
"""

import csv
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from .bootstrap import invoke_text

# 이 패키지(agentic_lib)의 부모가 notebooks/ 이므로, 그 아래 workspace/ 를 작업 루트로 쓴다.
_NOTEBOOKS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKSPACE_ROOT = os.path.join(_NOTEBOOKS_DIR, "workspace")


class WorkflowAutomationAgent:
    """OpenClaw 스타일 업무 자동화 에이전트(파일/데이터 액션 + 선언적 워크플로우).

    안전을 위해 모든 파일 입출력은 `notebooks/workspace/` 하위로만 허용된다.
    작업 폴더 밖(절대경로·`..` 탈출)을 가리키면 ValueError 를 던진다.
    """

    def __init__(self, workspace: str = None):
        """에이전트를 생성한다(작업 폴더는 notebooks/workspace 하위로 고정).

        Args:
            workspace: 작업 하위 폴더명(선택). None 이면 notebooks/workspace 자체.
                       workspace/ 밖을 가리키면 ValueError.
        """
        base = os.path.abspath(_WORKSPACE_ROOT)
        if workspace is None:
            target = base
        elif os.path.isabs(workspace):
            target = os.path.abspath(workspace)
        else:
            target = os.path.abspath(os.path.join(base, workspace))
        if not (target == base or target.startswith(base + os.sep)):
            raise ValueError("작업 폴더는 notebooks/workspace 하위만 허용됩니다.")
        self.workspace = target
        os.makedirs(self.workspace, exist_ok=True)
        self.action_log: List[Dict] = []

    def _safe_path(self, filename: str) -> str:
        """파일명을 작업 폴더 하위 절대경로로 변환한다(경로 탈출 방지)."""
        target = os.path.abspath(os.path.join(self.workspace, filename))
        if not (target == self.workspace or target.startswith(self.workspace + os.sep)):
            raise ValueError("작업 폴더 밖의 경로에는 접근할 수 없습니다.")
        return target

    def _log_action(self, action: str, params: Dict, result: Any) -> None:
        """수행한 액션을 감사 로그에 기록한다(결과는 200자까지)."""
        self.action_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "params": params,
            "result": str(result)[:200],
        })

    # ── 파일 시스템 액션 ──
    def create_file(self, filename: str, content: str) -> str:
        """작업 폴더에 파일을 새로 만든다."""
        path = self._safe_path(filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        result = f"파일 생성: {os.path.relpath(path, _NOTEBOOKS_DIR)}"
        self._log_action("create_file", {"filename": filename}, result)
        return result

    def read_file(self, filename: str) -> str:
        """작업 폴더의 파일을 읽어 내용을 반환한다."""
        path = self._safe_path(filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self._log_action("read_file", {"filename": filename}, content[:100])
            return content
        except FileNotFoundError:
            return f"파일 없음: {filename}"

    def list_files(self) -> List[str]:
        """작업 폴더의 파일 목록을 반환한다."""
        files = os.listdir(self.workspace)
        self._log_action("list_files", {}, files)
        return files

    def append_to_file(self, filename: str, content: str) -> str:
        """작업 폴더의 파일에 내용을 덧붙인다(없으면 새로 만든다)."""
        path = self._safe_path(filename)
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + content)
        result = f"파일 추가 완료: {os.path.relpath(path, _NOTEBOOKS_DIR)}"
        self._log_action("append_file", {"filename": filename}, result)
        return result

    # ── 데이터 처리 액션 ──
    def process_csv(self, data: List[Dict], filename: str) -> str:
        """딕셔너리 리스트를 CSV 파일로 저장한다."""
        if not data:
            return "데이터 없음"
        path = self._safe_path(filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
        result = f"CSV 저장: {os.path.relpath(path, _NOTEBOOKS_DIR)} ({len(data)}행)"
        self._log_action("process_csv", {"rows": len(data), "filename": filename}, result)
        return result

    def generate_report(self, title: str, sections: Dict[str, str]) -> str:
        """제목과 섹션(딕셔너리)으로 마크다운 보고서 파일을 생성한다."""
        lines = [f"# {title}", f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
        for section_title, content in sections.items():
            lines.append(f"## {section_title}")
            lines.append(content)
            lines.append("")
        report_content = "\n".join(lines)
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        return self.create_file(filename, report_content)

    # ── 워크플로우 실행 ──
    def execute_workflow(self, workflow_spec: Dict) -> Dict:
        """선언적 워크플로우(steps 목록)를 순서대로 실행한다.

        각 step 은 {action, params, description?, output_key?} 형식이며, params 값이
        '$키' 형태면 이전 step 결과(results)를 참조해 치환한다.

        Args:
            workflow_spec: {"name": ..., "steps": [...]} 형식의 워크플로우 명세.

        Returns:
            {output_key: 결과} 딕셔너리.
        """
        results: Dict[str, Any] = {}
        print(f"\n=== 워크플로우 실행: {workflow_spec['name']} ===")
        for step in workflow_spec["steps"]:
            action = step["action"]
            params = dict(step.get("params", {}))  # 원본 명세를 건드리지 않도록 복사
            print(f"\n  단계: {step.get('description', action)}")

            # '$키' 참조를 이전 단계 결과로 치환
            for key, val in params.items():
                if isinstance(val, str) and val.startswith("$"):
                    params[key] = results.get(val[1:], val)

            if hasattr(self, action):
                try:
                    result = getattr(self, action)(**params)
                    results[step.get("output_key", action)] = result
                    print(f"    결과: {str(result)[:80]}")
                except Exception as e:
                    results[step.get("output_key", action)] = f"오류: {e}"
                    print(f"    오류: {e}")
            else:
                print(f"    알 수 없는 액션: {action}")
        return results

    def get_audit_log(self) -> str:
        """감사 로그를 JSON 문자열로 반환한다."""
        return json.dumps(self.action_log, ensure_ascii=False, indent=2)


class FinalAgentSystem:
    """15주 과정 전체(연결·통제·지식·실행)를 통합한 최종 에이전트 시스템.

    파이프라인 각 단계를 시뮬레이션하며 처리량/품질 지표를 누적한다. `llm` 을
    주입하면 최종 출력 생성에 활용하고(응답은 to_text 정규화), 없으면 템플릿을 쓴다.

    구성 요소:
        - 모듈 1: 도구 연동 (Tool Use + MCP + A2A)
        - 모듈 2: 안전성 (Guardrails + Logging)
        - 모듈 3: 지식 (RAG + Graph + Memory)
        - 모듈 4: 실행 (LangGraph + Evaluation)
    """

    def __init__(self, config: Dict, llm=None):
        """시스템을 생성한다.

        Args:
            config: 시스템 설정 딕셔너리(예: provider, max_iterations).
            llm: 출력 생성에 사용할 LangChain BaseChatModel(없으면 템플릿).
        """
        self.config = config
        self.llm = llm
        self.system_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "blocked_requests": 0,
            "total_tool_calls": 0,
            "avg_quality_score": 0.0,
            "scores_list": [],
        }

    def process_request(self, user_input: str) -> Dict:
        """완전한 에이전트 파이프라인(가드레일→지식→실행→도구→평가)을 시뮬레이션한다."""
        self.system_metrics["total_requests"] += 1
        result = {"input": user_input, "pipeline": [], "output": None, "metrics": {}}
        print(f"\n[시스템] 요청 처리: '{user_input[:60]}'")

        # 1. 입력 가드레일 (모듈 2)
        result["pipeline"].append("입력 가드레일 통과")
        print("  [1] 입력 가드레일: 통과")
        # 2. 메모리/지식 검색 (모듈 3)
        result["pipeline"].append("메모리/지식 검색")
        print("  [2] 지식 검색: 완료")
        # 3. LangGraph 실행 (모듈 4)
        result["pipeline"].append("LangGraph 실행")
        print("  [3] LangGraph 워크플로우: 실행")
        # 4. 도구 실행 (모듈 1)
        tool_calls = 2  # 시뮬레이션
        self.system_metrics["total_tool_calls"] += tool_calls
        result["pipeline"].append(f"도구 실행 {tool_calls}회")
        print(f"  [4] 도구 실행: {tool_calls}회")

        # 5. 출력 생성 (llm 있으면 활용) + 평가
        output = None
        if self.llm is not None:
            try:
                output = invoke_text(self.llm, f"다음 요청에 간결히 답하세요: {user_input}")
            except Exception as e:
                print(f"    (LLM 출력 실패 → 템플릿으로 대체: {e})")
                output = None
        if not output:
            output = f"'{user_input}'에 대한 AI 에이전트 시스템의 종합적인 답변입니다."
        quality = 0.85
        result["pipeline"].append(f"품질 평가: {quality:.2f}")
        print(f"  [5] 품질 평가: {quality:.2f}")
        # 6. 출력 가드레일
        result["pipeline"].append("출력 가드레일 통과")
        print("  [6] 출력 가드레일: 통과")

        # 메트릭 업데이트
        self.system_metrics["successful_requests"] += 1
        self.system_metrics["scores_list"].append(quality)
        self.system_metrics["avg_quality_score"] = (
            sum(self.system_metrics["scores_list"]) / len(self.system_metrics["scores_list"])
        )
        result["output"] = output
        result["metrics"] = {"quality": quality, "tool_calls": tool_calls}
        return result

    def get_system_report(self) -> str:
        """누적 지표를 바탕으로 시스템 성능 보고서 문자열을 생성한다."""
        m = self.system_metrics
        success_rate = (m["successful_requests"] / m["total_requests"] * 100
                        if m["total_requests"] > 0 else 0)
        return f"""
=== 최종 에이전트 시스템 성능 보고서 ===
생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}

[처리량]
  총 요청: {m['total_requests']}건
  성공: {m['successful_requests']}건 ({success_rate:.1f}%)
  차단: {m['blocked_requests']}건

[품질]
  평균 품질 점수: {m['avg_quality_score']:.3f}
  총 도구 호출: {m['total_tool_calls']}회

[구성 모듈]
  - 모듈 1: Tool Use + MCP + A2A
  - 모듈 2: NeMo Guardrails + Logging
  - 모듈 3: RAG + Graph + Memory
  - 모듈 4: LangGraph + RAGAS
""".strip()
