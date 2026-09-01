"""
evaluation — RAG 평가(RAGAS 스타일) & LLM-as-a-Judge (모듈 4: 실행)
=================================================================

week12-14 모듈4 노트북의 **평가 구현**을 분리한 모듈입니다. RAGAS 패키지는
무겁고(여러 무거운 의존성) 일부 환경에서 설치가 까다로워, 핵심 지표를 외부
의존성 없이 **자체 구현**합니다. 또한 상위 LLM 으로 응답을 채점하는
LLM-as-a-Judge 패턴을 제공합니다.

구성:
    RAGSample        평가 표본(question/answer/contexts/ground_truth)
    RAGEvaluator     RAGAS 스타일 지표(단어 중첩 휴리스틱):
                       faithfulness / answer_relevancy / context_recall / context_precision
    LLMJudge         judge_llm 을 주입받아 응답을 1~5점으로 채점(출력 to_text 정규화)

설계 메모:
    - RAGEvaluator 는 LLM 없이 동작하는 결정론적 휴리스틱이라 오프라인·재현 가능합니다.
      (실제 RAGAS 는 각 문장의 지지 여부를 LLM 으로 판단합니다.)
    - LLMJudge 는 `judge_llm` 을 주입받고, 응답은 `bootstrap.to_text()` 로 정규화한 뒤
      JSON 을 파싱합니다. 파싱 실패/모델 없음 시 규칙 기반 폴백으로 점수를 냅니다.
"""

import json
import re
from dataclasses import dataclass
from typing import List, Dict, Optional

from .bootstrap import to_text


@dataclass
class RAGSample:
    """RAG 평가 표본 1개.

    Attributes:
        question: 사용자 질문.
        answer: 시스템이 생성한 답변.
        contexts: 검색되어 답변에 사용된 컨텍스트 문서들.
        ground_truth: 정답(기준 답변).
    """
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str


def _words(text: str) -> set:
    """텍스트를 소문자 단어 집합으로 변환한다(지표 계산용 토큰화)."""
    return set(re.findall(r"\w+", text.lower()))


class RAGEvaluator:
    """RAGAS 스타일 RAG 평가기(단어 중첩 기반 휴리스틱, LLM 불필요).

    각 점수는 0~1 범위이며, 실제 RAGAS 지표의 직관을 단어 집합 연산으로 근사한다.
    """

    def faithfulness_score(self, answer: str, contexts: List[str]) -> float:
        """Faithfulness — 답변 단어 중 컨텍스트가 뒷받침하는 비율(환각 방지 지표)."""
        if not contexts or not answer:
            return 0.0
        answer_words = _words(answer)
        if not answer_words:
            return 0.0
        context_words = set()
        for ctx in contexts:
            context_words |= _words(ctx)
        return len(answer_words & context_words) / len(answer_words)

    def answer_relevancy_score(self, question: str, answer: str) -> float:
        """Answer Relevancy — 질문 단어가 답변에 반영된 비율(답변 적합성 지표)."""
        if not question or not answer:
            return 0.0
        q_words = _words(question)
        if not q_words:
            return 0.0
        return len(q_words & _words(answer)) / len(q_words)

    def context_recall_score(self, contexts: List[str], ground_truth: str) -> float:
        """Context Recall — 정답에 필요한 단어가 컨텍스트에 포함된 비율(검색 재현율)."""
        if not contexts or not ground_truth:
            return 0.0
        gt_words = _words(ground_truth)
        if not gt_words:
            return 0.0
        context_words = set()
        for ctx in contexts:
            context_words |= _words(ctx)
        return len(gt_words & context_words) / len(gt_words)

    def context_precision_score(self, question: str, contexts: List[str]) -> float:
        """Context Precision — 검색된 컨텍스트 중 질문과 관련된 문서의 비율(검색 정밀도)."""
        if not contexts or not question:
            return 0.0
        q_words = _words(question)
        if not q_words:
            return 0.0
        relevant = 0
        for ctx in contexts:
            # 질문 단어의 20% 이상이 겹치면 '관련 있음' 으로 본다.
            if len(q_words & _words(ctx)) / len(q_words) > 0.2:
                relevant += 1
        return relevant / len(contexts)

    def evaluate(self, sample: RAGSample) -> Dict[str, float]:
        """표본 1개에 대해 네 지표와 종합 점수(ragas_score)를 계산한다."""
        faithfulness = self.faithfulness_score(sample.answer, sample.contexts)
        relevancy = self.answer_relevancy_score(sample.question, sample.answer)
        recall = self.context_recall_score(sample.contexts, sample.ground_truth)
        precision = self.context_precision_score(sample.question, sample.contexts)
        ragas_score = (faithfulness + relevancy + recall + precision) / 4
        return {
            "faithfulness": round(faithfulness, 3),
            "answer_relevancy": round(relevancy, 3),
            "context_recall": round(recall, 3),
            "context_precision": round(precision, 3),
            "ragas_score": round(ragas_score, 3),
        }

    def batch_evaluate(self, samples: List[RAGSample]) -> Dict:
        """여러 표본을 평가하고 지표별 평균/최소/최대 통계를 함께 반환한다."""
        all_scores = [self.evaluate(s) for s in samples]
        metrics = [
            "faithfulness", "answer_relevancy",
            "context_recall", "context_precision", "ragas_score",
        ]
        summary = {}
        for metric in metrics:
            scores = [s[metric] for s in all_scores]
            summary[metric] = {
                "mean": round(sum(scores) / len(scores), 3),
                "min": round(min(scores), 3),
                "max": round(max(scores), 3),
            }
        return {"per_sample": all_scores, "summary": summary}


# 기본 평가 기준(LLMJudge 가 사용). '이름: 설명' 형식.
DEFAULT_CRITERIA = [
    "정확성: 답변이 사실에 기반한가",
    "완성도: 질문을 충분히 다루었는가",
    "명확성: 이해하기 쉽게 설명했는가",
    "간결성: 불필요한 내용 없이 핵심만 담았는가",
]


class LLMJudge:
    """LLM-as-a-Judge — 상위 모델이 에이전트 응답을 채점한다(참고: AgentBench).

    `judge_llm` 을 주입받아 사용하며, 응답은 `to_text()` 로 정규화한 뒤 JSON 을
    파싱한다. 모델이 없거나 파싱에 실패하면 규칙 기반 폴백으로 점수를 낸다.
    """

    def __init__(self, judge_llm=None):
        """평가기를 생성한다.

        Args:
            judge_llm: 채점에 사용할 LangChain BaseChatModel(없으면 규칙 기반 폴백).
        """
        self.judge_llm = judge_llm
        self.evaluation_log: List[Dict] = []

    def evaluate_response(self, question: str, response: str, criteria: List[str] = None) -> Dict:
        """질문-응답을 기준별로 채점해 {"scores", "total", "summary"} 를 반환한다."""
        criteria = criteria or DEFAULT_CRITERIA

        if self.judge_llm:
            from langchain_core.messages import HumanMessage
            criteria_str = "\n".join(f"- {c}" for c in criteria)
            prompt = f"""다음 AI 에이전트 응답을 평가해주세요.

질문: {question}
응답: {response}

평가 기준:
{criteria_str}

각 기준별로 1-5점 점수와 이유를 제시하고, 총점과 종합 의견을 JSON 형식으로 반환하세요.
형식: {{"scores": {{"기준명": {{"score": N, "reason": "..."}}}}, "total": N, "summary": "..."}}"""
            try:
                resp = self.judge_llm.invoke([HumanMessage(content=prompt)])
                # 공급자 무관하게 평문으로 정규화(Gemini list·qwen3 <think> 흡수)
                content = to_text(resp.content)
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    evaluation = json.loads(json_match.group())
                    self.evaluation_log.append({"question": question, "evaluation": evaluation})
                    return evaluation
            except Exception as e:
                print(f"LLM Judge 오류: {e}")

        # 폴백: 규칙 기반 채점(길이/질문 단어 포함 여부로 간단 가점)
        scores = {}
        for criterion in criteria:
            name = criterion.split(":")[0]
            score = 3
            if len(response) > 100:
                score = min(score + 1, 5)
            if any(w in response for w in question.split()):
                score = min(score + 1, 5)
            scores[name] = {"score": score, "reason": "규칙 기반 평가"}
        total = sum(v["score"] for v in scores.values()) / len(scores) * 20
        return {"scores": scores, "total": round(total), "summary": "규칙 기반 자동 평가"}

    def comparative_evaluation(self, question: str, response_a: str, response_b: str) -> Dict:
        """두 응답(A/B)을 채점해 승자와 점수 차를 반환한다."""
        eval_a = self.evaluate_response(question, response_a)
        eval_b = self.evaluate_response(question, response_b)
        winner = "A" if eval_a["total"] >= eval_b["total"] else "B"
        return {
            "question": question,
            "response_a_score": eval_a["total"],
            "response_b_score": eval_b["total"],
            "winner": winner,
            "difference": abs(eval_a["total"] - eval_b["total"]),
        }
