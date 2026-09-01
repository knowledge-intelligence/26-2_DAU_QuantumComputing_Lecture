"""
agent_memory — 지식 에이전트의 4계층 메모리 + Deep-Knowledge Agent
==================================================================

M04_4(지식 시리즈 4편)에서 쓰는, **인지 구조를 본뜬** 에이전트 메모리 구현입니다.
사람의 기억 분류(작업/단기/일화/의미)를 그대로 계층으로 나눠, 각 계층이 무엇을
얼마나 오래 들고 있는지 눈으로 확인할 수 있게 했습니다.

구성:
    ShortTermMemory      단기 기억 — 최근 N턴만 남기는 슬라이딩 윈도우(deque)
    EpisodicMemory       일화 기억 — '겪은 일'을 벡터 DB(ChromaDB)에 저장하고 유사도로 회상
    SemanticMemory       의미 기억 — '사실'을 신뢰도와 함께 명시적으로 저장/검색
    AgentMemorySystem    위 셋 + 작업 메모리(working memory)를 하나로 묶은 메모리 시스템
    DeepKnowledgeAgent   Hybrid RAG + 메모리 + 추론 체인 = '사고 → 검색 → 추론 → 생성' 에이전트

계층을 나누는 이유(핵심):
    - 단기 기억은 **싸지만 금방 잊는다** — 매 요청 프롬프트에 통째로 들어가므로 토큰 비용이 곧 길이다.
    - 일화 기억은 **비싸지만 오래 간다** — 임베딩·벡터 검색 비용을 내고 '예전 그 대화'를 되살린다.
    - 의미 기억은 **정확하지만 누가 넣어 줘야 한다** — 사실은 요약이 아니라 확정된 지식이다.
    한 계층으로 다 하려 들면 비용이 터지거나(전부 단기) 맥락을 잃는다(전부 벡터).

LLM 호출 규약(rag.py 와 동일):
    - 이 모듈은 LLM 을 직접 만들지 않는다. 생성자/메서드 인자로 '주입'받는다.
    - 모든 LLM 응답은 bootstrap.to_text() 로 정규화한다(Gemini list content·qwen3 <think> 흡수).
"""

import hashlib
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from .bootstrap import to_text


class ShortTermMemory:
    """단기 기억 — 현재 대화 맥락을 유지하는 슬라이딩 윈도우 버퍼.

    deque(maxlen) 이라 정해진 크기를 넘으면 가장 오래된 메시지가 **자동으로 밀려난다**.
    잊는 것이 버그가 아니라 설계다 — 프롬프트 길이(=비용)를 상수로 묶는 장치다.

    agentic_lib.memory.ConversationMemory 와 역할이 겹치지만, 이쪽은 턴(user+assistant 쌍)
    단위로 세고 타임스탬프를 남긴다는 점이 다르다.
    """

    def __init__(self, max_turns: int = 10):
        """버퍼를 생성한다. user/assistant 한 쌍을 1턴으로 보고 2배 크기로 보관한다.

        Args:
            max_turns: 보관할 최대 대화 턴 수.
        """
        self.messages: deque = deque(maxlen=max_turns * 2)  # user + assistant
        self.max_turns = max_turns

    def add(self, role: str, content: str) -> None:
        """메시지를 타임스탬프와 함께 추가한다(한도를 넘으면 오래된 것이 밀려난다)."""
        self.messages.append({"role": role, "content": content,
                              "timestamp": datetime.now().isoformat()})

    def get_messages(self) -> List[Dict]:
        """role/content 만 추린 메시지 목록을 반환한다."""
        return [{"role": m["role"], "content": m["content"]}
                for m in self.messages]

    def get_context_string(self) -> str:
        """최근 대화를 '[role] content' 줄 형태의 문자열로 합쳐 반환한다."""
        return "\n".join([
            f"[{m['role']}] {m['content']}"
            for m in self.messages
        ])


class EpisodicMemory:
    """일화(에피소딕) 기억 — 겪은 일을 벡터 DB 에 저장하고 유사도로 회상한다.

    단기 기억에서 밀려난 대화도 여기에 남아 있으면 '예전에 이런 이야기를 했다'를
    되살릴 수 있다. 저장 단위는 하나의 사건(보통 Q/A 한 쌍)이며,
    중요도(importance)와 태그를 메타데이터로 붙여 나중에 선별·필터링할 수 있게 한다.
    """

    def __init__(self, collection):
        """ChromaDB 컬렉션을 받아 에피소드 저장소로 사용한다.

        Args:
            collection: 임베딩 함수가 붙은 ChromaDB 컬렉션.
        """
        self.collection = collection
        self.episode_count = 0

    def save_episode(self, content: str, importance: float = 0.5,
                     tags: Optional[List[str]] = None, verbose: bool = True) -> str:
        """중요 에피소드를 임베딩하여 저장하고 episode_id 를 반환한다.

        Args:
            content: 저장할 사건 내용(예: "Q: ... A: ...").
            importance: 중요도(0~1). 나중에 회상 결과를 선별하는 근거로 쓴다.
            tags: 분류용 태그 목록.
            verbose: True 면 저장 사실을 출력한다.
        """
        self.episode_count += 1
        episode_id = f"episode_{self.episode_count:04d}"

        self.collection.add(
            documents=[content],
            ids=[episode_id],
            metadatas=[{
                "type": "episode",
                "importance": importance,
                "tags": ",".join(tags or []),
                "timestamp": datetime.now().isoformat()
            }]
        )
        if verbose:
            print(f"에피소드 저장: [{episode_id}] {content[:60]}...")
        return episode_id

    def recall(self, query: str, n_results: int = 3) -> List[Dict]:
        """쿼리와 유사한 에피소드를 벡터 검색으로 회상한다(실패 시 빈 리스트).

        컬렉션이 비어 있거나 조회에 실패해도 에이전트 전체가 멈추면 안 되므로
        예외를 흡수하고 빈 결과를 돌려준다 — 기억은 '있으면 좋은 것'이지 필수가 아니다.
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"type": "episode"}
            )
            return [
                {"content": doc, "metadata": meta, "distance": dist}
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )
            ]
        except Exception:
            return []


class SemanticMemory:
    """의미(시맨틱) 기억 — 사실/지식을 신뢰도와 함께 명시적으로 저장한다.

    일화 기억이 "언제 무슨 일이 있었나"라면, 의미 기억은 "무엇이 참인가"다.
    대화에서 자동으로 쌓이지 않고 **누군가 학습(learn)시켜야** 한다는 점이 핵심이며,
    그래서 신뢰도(confidence)와 출처(source)를 함께 들고 다닌다.
    """

    def __init__(self):
        self.facts: Dict[str, Dict] = {}

    def learn(self, fact: str, confidence: float = 1.0, source: Optional[str] = None) -> str:
        """새로운 사실을 학습(저장)하고 fact_id 를 반환한다.

        같은 문장을 다시 학습하면 내용 해시가 같으므로 **덮어써서 중복이 쌓이지 않는다.**

        Args:
            fact: 사실 문장.
            confidence: 신뢰도(0~1). 검색 결과 정렬 기준.
            source: 출처 표기(선택).
        """
        fact_id = hashlib.md5(fact.encode()).hexdigest()[:8]
        self.facts[fact_id] = {
            "content": fact,
            "confidence": confidence,
            "source": source,
            "learned_at": datetime.now().isoformat(),
            "recall_count": 0
        }
        return fact_id

    def search(self, keyword: str) -> List[Dict]:
        """키워드로 사실을 검색한다(회상 횟수 누적, 신뢰도 내림차순 정렬).

        의미 검색이 아니라 **부분 문자열 매칭** 이다 — 계층별 특성을 대비시키기 위한
        의도적인 단순화이며, 그래서 '그래프 DB'로 물으면 'Neo4j' 사실을 못 찾는다.
        """
        results = []
        for fact_id, fact_data in self.facts.items():
            if keyword.lower() in fact_data["content"].lower():
                fact_data["recall_count"] += 1
                results.append({"id": fact_id, **fact_data})
        return sorted(results, key=lambda x: x["confidence"], reverse=True)


class AgentMemorySystem:
    """4계층 메모리 시스템 — 작업 + 단기 + 일화 + 의미 기억을 하나로 묶는다.

    에이전트 입장에서는 계층이 몇 개든 상관없고, 필요한 것은 두 가지뿐이다.
        perceive() / respond()      — 대화를 흘려 넣는다(어느 계층에 남길지는 시스템이 정한다)
        get_relevant_context()      — 지금 질문에 필요한 기억만 뽑아 하나의 문자열로 받는다
    이 경계 덕분에 계층을 바꿔도(예: 일화 기억을 Chroma → Qdrant) 에이전트 코드는 그대로다.
    """

    def __init__(self, collection):
        """에피소딕 메모리에 쓸 ChromaDB 컬렉션을 받아 하위 메모리들을 초기화한다."""
        self.short_term = ShortTermMemory(max_turns=10)
        self.episodic = EpisodicMemory(collection)
        self.semantic = SemanticMemory()
        self.working_memory: Dict = {}  # 현재 태스크 컨텍스트(지금 처리 중인 것만)

    def perceive(self, user_input: str) -> None:
        """사용자 입력을 단기 기억과 작업 메모리에 반영한다."""
        self.short_term.add("user", user_input)
        self.working_memory["last_input"] = user_input

    def respond(self, response: str, important: bool = False) -> None:
        """에이전트 응답을 저장하고, 중요하면 일화 기억으로도 남긴다.

        `important` 가 곧 **단기에서 끝낼지, 장기로 넘길지** 를 가르는 스위치다.
        모든 대화를 벡터 DB 에 넣으면 비용도 잡음도 함께 늘어난다.
        """
        self.short_term.add("assistant", response)
        if important:
            context = f"Q: {self.working_memory.get('last_input', '')} A: {response}"
            self.episodic.save_episode(context, importance=0.8)

    def get_relevant_context(self, query: str, n_episodes: int = 2, n_facts: int = 3,
                             episode_chars: int = 100) -> str:
        """쿼리와 관련된 단기/일화/의미 기억을 모아 하나의 컨텍스트 문자열로 반환한다.

        Args:
            query: 현재 질문.
            n_episodes: 회상할 일화 기억 개수.
            n_facts: 포함할 사실 개수.
            episode_chars: 일화 1건당 프롬프트에 넣을 최대 글자 수
                (표시용 자르기가 아니라 **프롬프트 예산** 이다).
        """
        parts = []

        # 최근 대화 (단기 기억)
        recent = self.short_term.get_context_string()
        if recent:
            parts.append(f"[최근 대화]\n{recent}")

        # 관련 에피소드 (일화 기억 = 장기)
        episodes = self.episodic.recall(query, n_results=n_episodes)
        if episodes:
            ep_text = "\n".join([e["content"][:episode_chars] for e in episodes])
            parts.append(f"[관련 과거 경험]\n{ep_text}")

        # 사실 (의미 기억)
        facts = self.semantic.search(query)
        if facts:
            fact_text = "\n".join([f["content"] for f in facts[:n_facts]])
            parts.append(f"[관련 지식]\n{fact_text}")

        return "\n\n".join(parts)

    def stats(self) -> Dict[str, Any]:
        """계층별 보유량을 요약해 반환한다(실습에서 '무엇이 어디에 남았는지' 확인용)."""
        return {
            "단기(메시지)": len(self.short_term.messages),
            "단기(한도)": self.short_term.messages.maxlen,
            "일화(저장 건수)": self.episodic.episode_count,
            "의미(사실 수)": len(self.semantic.facts),
            "작업(키)": list(self.working_memory.keys()),
        }


class DeepKnowledgeAgent:
    """복합 지식 추론 에이전트 — Hybrid RAG + 4계층 메모리 + 추론 체인.

    한 번의 질문을 네 단계로 나눠 처리하고, 각 단계를 `reasoning_chain` 에 남긴다.

        think()     어떤 소스를 봐야 하는지 전략을 한 문장으로 세운다
        retrieve()  Hybrid RAG(벡터+그래프) 검색 + 메모리 회상을 한 번에 모은다
        reason()    모은 근거가 무엇인지 요약해 기록한다(추적 가능성의 핵심)
        generate()  근거 + 추론 메모를 LLM 에 넘겨 최종 답변을 만든다

    단계를 쪼개는 이유는 성능이 아니라 **관찰 가능성** 이다. 답이 틀렸을 때
    검색이 틀렸는지(retrieve) 종합이 틀렸는지(generate) 구분할 수 있어야 고칠 수 있다.
    """

    def __init__(self, hybrid_rag, memory_system: AgentMemorySystem, llm=None):
        """구성요소를 주입받는다.

        Args:
            hybrid_rag: 검색을 담당하는 rag.HybridRAG.
            memory_system: 대화/지식 기억을 담당하는 AgentMemorySystem.
            llm: 최종 답변 생성용 LLM(없으면 폴백 요약 문자열을 반환한다).
        """
        self.hybrid_rag = hybrid_rag
        self.memory = memory_system
        self.llm = llm
        self.reasoning_chain: List[str] = []

    def think(self, question: str) -> str:
        """질문 해결 전략(어떤 소스를 검색할지)을 한 문장 사고로 기록한다."""
        thought = f"'{question}'를 이해하기 위해 지식 그래프와 벡터 DB를 검색해야 한다"
        self.reasoning_chain.append(thought)
        return thought

    def retrieve(self, query: str) -> Dict:
        """Hybrid RAG 검색 결과와 메모리 컨텍스트를 합쳐 반환한다."""
        rag_result = self.hybrid_rag.search(query)
        memory_context = self.memory.get_relevant_context(query)
        return {**rag_result, "memory_context": memory_context}

    def reason(self, question: str, retrieved: Dict) -> str:
        """검색 결과가 무엇이었는지 요약한 추론 메모를 기록한다."""
        reasoning = f"""
지식 그래프에서 발견된 엔티티: {retrieved['graph_entities']}
벡터 검색 결과: {len(retrieved['vector_results'])}개
기억에서 관련 컨텍스트: {'있음' if retrieved['memory_context'] else '없음'}

이 정보들을 종합하여 '{question}'에 답할 수 있다."""
        self.reasoning_chain.append(reasoning)
        return reasoning

    def generate_answer(self, question: str, llm=None, verbose: bool = True) -> Dict:
        """전체 파이프라인을 실행하고 답변/추론체인/출처를 담은 딕셔너리를 반환한다.

        Args:
            question: 사용자 질문.
            llm: 이번 호출에만 쓸 LLM(없으면 생성자 주입분을 쓴다).
            verbose: True 면 단계별 진행 상황을 출력한다.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"질문: {question}")
            print(f"{'='*60}")

        self.reasoning_chain = []
        thought = self.think(question)
        if verbose:
            print(f"\n[사고] {thought}")

        self.memory.perceive(question)
        retrieved = self.retrieve(question)
        reasoning = self.reason(question, retrieved)
        if verbose:
            print(f"[추론] {reasoning.strip()}")

        model = llm or self.llm
        if model is not None:
            from langchain_core.messages import HumanMessage, SystemMessage
            full_context = f"""
{retrieved['combined_context']}

=== 과거 기억 ===
{retrieved.get('memory_context', '없음')}

=== 추론 과정 ===
{reasoning}
"""
            response = model.invoke([
                SystemMessage(content="당신은 지식 그래프와 벡터 DB를 활용하는 심층 지식 에이전트입니다."),
                HumanMessage(content=f"{full_context}\n\n질문: {question}"),
            ])
            answer = to_text(response.content)  # 공급자 무관 정규화(<think>/list 처리)
        else:
            answer = (f"[Deep Knowledge Agent] 벡터 {len(retrieved['vector_results'])}개, "
                      f"그래프 {len(retrieved['graph_entities'])}개 엔티티 종합 답변")

        # 최종 답변은 중요한 사건이므로 일화 기억에도 남긴다
        self.memory.respond(answer, important=True)
        if verbose:
            # 답변은 자르지 않는다 — 표/목록형 응답이 끊기면 내용이 통째로 사라진다
            print(f"\n[최종 답변] {answer}")

        return {
            "question": question,
            "answer": answer,
            "reasoning_chain": self.reasoning_chain,
            "sources": [
                {"type": "vector", "count": len(retrieved["vector_results"])},
                {"type": "graph", "entities": retrieved["graph_entities"]},
            ],
        }
