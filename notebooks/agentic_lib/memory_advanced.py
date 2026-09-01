"""
memory_advanced — 고급 에이전트 메모리 패턴
============================================

week09-11 메모리 실습 노트북에서 길고 반복적으로 등장하던 '고급 메모리' 구현을
한곳에 모은 모듈입니다. 노트북은 개념·실행 흐름에 집중하고, 클래스 구현은 여기서
import 해서 재사용합니다(기본 단기/장기 메모리는 sibling 모듈 `memory.py` 참조).

제공 클래스/함수:
    BufferMemory            모든 대화를 그대로 누적(맥락 완전 보존, 토큰 폭발)
    SlidingWindowMemory     최근 N개 메시지만 유지(deque 슬라이딩 윈도우)
    SummaryMemory           오래된 대화를 LLM 으로 요약·압축
    SemanticMemoryStore     LLM 이 직접 관련 기억을 선별하는 시맨틱 메모리
    FullMemoryAgent         단기+장기+시맨틱 메모리를 통합한 에이전트
    make_store_memory_tools LangGraph Store 에 저장/조회하는 도구 한 쌍 팩토리
                            (store·user_id 를 클로저로 고정 — user_id 는 도구 스키마에 없다)

모든 LLM 응답은 `bootstrap.to_text()` 로 정규화합니다. 즉 Gemini 의 list 형 content,
qwen3 의 <think>...</think> 추론 블록을 자동으로 깔끔한 문자열로 통일합니다.
"""

from collections import deque
import datetime

from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage,
)
from langchain_core.tools import tool

from .bootstrap import to_text

# 단기 메모리 클래스들이 공통으로 앞에 붙이는 기본 시스템 메시지.
# 노트북에서 `DEFAULT_SYSTEM as SYSTEM` 으로 가져와 LangGraph 노드 등에서도 재사용한다.
DEFAULT_SYSTEM = SystemMessage(
    content='당신은 사용자 정보를 기억하는 친절한 AI 어시스턴트입니다.'
)


class BufferMemory:
    """버퍼 메모리 — 모든 대화 메시지를 그대로 누적 유지한다.

    가장 단순한 단기 기억 형태다. 맥락을 빠짐없이 보존하지만, 대화가 길어질수록
    토큰이 선형으로 증가(토큰 폭발)하는 단점이 있다.
    """

    def __init__(self, system: SystemMessage = None):
        """버퍼를 생성한다.

        Args:
            system: 맨 앞에 붙일 시스템 메시지. None 이면 DEFAULT_SYSTEM.
        """
        self.system = system or DEFAULT_SYSTEM
        self.messages = []

    def add(self, message) -> None:
        """메시지(HumanMessage/AIMessage 등)를 버퍼에 추가한다."""
        self.messages.append(message)

    def get(self) -> list:
        """시스템 메시지 + 누적된 전체 메시지를 LLM 입력 형태로 반환한다."""
        return [self.system] + self.messages

    def token_estimate(self) -> int:
        """누적 토큰 수를 대략 추정한다(문자 4개당 1토큰 가정)."""
        total_chars = sum(len(m.content) for m in self.messages)
        return total_chars // 4


class SlidingWindowMemory:
    """슬라이딩 윈도우 메모리 — 최근 max_messages 개 메시지만 유지한다.

    `collections.deque(maxlen=N)` 으로 구현한다. 새 메시지가 들어오면 가장 오래된
    메시지가 자동으로 밀려나, 토큰 사용량을 예측 가능한 범위로 묶어 둔다.
    """

    def __init__(self, max_messages: int = 6, system: SystemMessage = None):
        """윈도우 버퍼를 생성한다.

        Args:
            max_messages: 유지할 최대 메시지 수(초과 시 오래된 것부터 폐기).
            system: 맨 앞에 붙일 시스템 메시지. None 이면 DEFAULT_SYSTEM.
        """
        self.system = system or DEFAULT_SYSTEM
        self.messages = deque(maxlen=max_messages)

    def add(self, message) -> None:
        """메시지를 추가한다(가득 차면 가장 오래된 메시지가 자동 제거)."""
        self.messages.append(message)

    def get(self) -> list:
        """시스템 메시지 + 현재 윈도우에 남은 메시지를 반환한다."""
        return [self.system] + list(self.messages)

    def __len__(self) -> int:
        """현재 윈도우에 보관 중인 메시지 수."""
        return len(self.messages)


class SummaryMemory:
    """요약 메모리 — 오래된 대화를 LLM 으로 요약·압축한다.

    버퍼 메모리(맥락 보존)와 슬라이딩 윈도우(토큰 절약)의 장점을 결합한다.
    최근 max_recent 개는 원본 그대로 유지하고, 그보다 오래된 메시지는 누적해
    두었다가 일정량이 쌓이면 LLM 으로 한 번에 요약해 `summary` 에 통합한다.
    """

    def __init__(self, llm, max_recent: int = 4, system: SystemMessage = None):
        """요약 메모리를 생성한다.

        Args:
            llm: 요약에 사용할 LangChain BaseChatModel.
            max_recent: 원본 그대로 유지할 최근 메시지 수.
            system: 맨 앞에 붙일 시스템 메시지. None 이면 DEFAULT_SYSTEM.
        """
        self.llm = llm
        self.system = system or DEFAULT_SYSTEM
        self.summary = ''                        # 압축된 과거 요약문
        self.recent = deque(maxlen=max_recent)   # 최근 원본 메시지
        self._pending_summary = []               # 요약 전 대기 중인 오래된 메시지

    def add(self, message) -> None:
        """메시지를 추가한다. 윈도우에서 밀려난 메시지는 요약 대기열로 보낸다."""
        if len(self.recent) == self.recent.maxlen:
            # 가장 오래된 메시지를 요약 대기열(pending)로 이동
            oldest = self.recent[0]
            self._pending_summary.append(oldest)
        self.recent.append(message)

        # 대기열이 일정량 쌓이면 요약 실행
        if len(self._pending_summary) >= 4:
            self._run_summary()

    def _run_summary(self) -> None:
        """대기 중인 오래된 메시지들을 LLM 으로 요약해 기존 요약에 통합한다."""
        if not self._pending_summary:
            return
        conv_text = '\n'.join(
            f"[{'사용자' if isinstance(m, HumanMessage) else 'AI'}] {m.content}"
            for m in self._pending_summary
        )
        prompt = [
            SystemMessage(content=(
                '다음 대화에서 사용자의 핵심 정보(이름·직업·거주지·취미 등)를 '
                '불릿 포인트로 요약하세요. 기존 요약이 있으면 통합하세요.\n'
                f'기존 요약: {self.summary or "없음"}'
            )),
            HumanMessage(content=conv_text)
        ]
        # to_text 로 공급자 무관 정규화(<think> 제거 포함)
        self.summary = to_text(self.llm.invoke(prompt).content)
        self._pending_summary.clear()
        print(f'  [요약 갱신] {self.summary[:80]}...')

    def get_context(self) -> list:
        """시스템 메시지 + (있으면)요약문 + 최근 원본 메시지를 반환한다."""
        msgs = [self.system]
        if self.summary:
            msgs.append(SystemMessage(content=f'과거 대화 요약:\n{self.summary}'))
        msgs.extend(list(self.recent))
        return msgs


class SemanticMemoryStore:
    """시맨틱 메모리 — LLM 이 질문과 관련 있는 기억만 직접 선별하는 저장소.

    임베딩/벡터 DB 없이도 '관련 기억 검색' 개념을 보여주기 위해, 저장된 기억 목록을
    LLM 에 보여주고 질문과 가장 관련 있는 항목 번호를 고르게 한다(추가 패키지 불필요).
    실제 대규모 운영에서는 ChromaDB/FAISS 같은 임베딩 벡터 검색을 사용한다.
    """

    def __init__(self, llm):
        """시맨틱 메모리 저장소를 생성한다.

        Args:
            llm: 관련성 판단에 사용할 LangChain BaseChatModel.
        """
        self.llm = llm
        self.memories = []   # 각 항목: {'id', 'content', 'timestamp', 'tags'}
        self._next_id = 1

    def remember(self, content: str, tags: list = None) -> int:
        """새 기억을 저장하고 부여된 id 를 반환한다.

        Args:
            content: 기억할 내용.
            tags: 분류용 태그 목록(선택).
        """
        entry = {
            'id': self._next_id,
            'content': content,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            'tags': tags or [],
        }
        self.memories.append(entry)
        self._next_id += 1
        print(f'  [기억 저장 #{entry["id"]}] {content[:60]}')
        return entry['id']

    def search(self, query: str, top_k: int = 3) -> list:
        """질문과 관련 있는 기억을 LLM 으로 선별해 최대 top_k 개 반환한다."""
        if not self.memories:
            return []

        memories_text = '\n'.join(
            f'[#{m["id"]}] {m["content"]}' for m in self.memories
        )
        prompt = [
            SystemMessage(content=(
                f'다음 기억 목록에서 질문과 가장 관련 있는 것을 {top_k}개 선택하세요.\n'
                '응답 형식: 번호 목록만 (예: #1, #3, #5)'
            )),
            HumanMessage(content=f'질문: {query}\n\n기억 목록:\n{memories_text}')
        ]
        # to_text 로 정규화한 응답에서 '#id' 가 언급된 기억만 추린다
        answer = to_text(self.llm.invoke(prompt).content)
        selected = [m for m in self.memories if f'#{m["id"]}' in answer]
        return selected[:top_k]

    def get_context(self, query: str) -> str:
        """질문 관련 기억들을 컨텍스트 문자열로 묶어 반환한다(없으면 안내 문구)."""
        relevant = self.search(query)
        if not relevant:
            return '관련 기억 없음'
        return '\n'.join(f'- {m["content"]}' for m in relevant)


class FullMemoryAgent:
    """통합 메모리 에이전트 — 단기(슬라이딩 윈도우) + 장기/시맨틱 기억을 결합한다.

    매 턴마다 (1) 시맨틱 검색으로 관련 장기 기억을 끌어오고, (2) 단기 대화 히스토리와
    함께 LLM 에 전달해 답하고, (3) 대화에서 기억할 만한 사실을 LLM 으로 추출해 장기
    기억에 저장한다.
    """

    def __init__(self, llm, user_id: str, window_size: int = 6):
        """통합 메모리 에이전트를 생성한다.

        Args:
            llm: 추론·사실추출에 사용할 LangChain BaseChatModel.
            user_id: 사용자 식별자.
            window_size: 단기 기억(슬라이딩 윈도우)에 유지할 메시지 수.
        """
        self.llm = llm
        self.user_id = user_id
        self.short_term = deque(maxlen=window_size)       # 단기: 슬라이딩 윈도우
        self.long_term = SemanticMemoryStore(llm=llm)     # 장기: 시맨틱 메모리
        self._system = SystemMessage(
            content='당신은 사용자를 잘 아는 개인 AI 어시스턴트입니다. 한국어로 답변하세요.'
        )

    def _extract_facts(self, user_msg: str, ai_reply: str) -> list:
        """대화에서 기억할 만한 사용자 사실을 LLM 으로 추출해 리스트로 반환한다."""
        prompt = [
            SystemMessage(content=(
                '다음 대화에서 사용자에 대해 기억해야 할 구체적 사실이 있으면 추출하세요.\n'
                '없으면 "없음" 이라고만 하세요.\n'
                '있으면 각 사실을 한 줄씩 나열하세요. (예: 이름: 홍길동)'
            )),
            HumanMessage(content=f'사용자: {user_msg}\nAI: {ai_reply}')
        ]
        content = to_text(self.llm.invoke(prompt).content)
        if '없음' in content:
            return []
        return [line.strip() for line in content.strip().split('\n') if line.strip()]

    def chat(self, user_input: str) -> str:
        """한 턴을 처리한다: 관련 장기 기억 검색 → LLM 응답 → 단기/장기 기억 갱신."""
        print(f'\n[사용자] {user_input}')

        # 1. 시맨틱 검색으로 관련 장기 기억 추출
        context = self.long_term.get_context(user_input)
        if context != '관련 기억 없음':
            print(f'  [장기 기억 검색] {context[:80]}')

        # 2. 메시지 구성(장기 기억 컨텍스트 + 단기 히스토리 + 현재 입력)
        messages = [self._system]
        if context != '관련 기억 없음':
            messages.append(SystemMessage(
                content=f'이 사용자에 대해 알고 있는 정보:\n{context}'
            ))
        messages.extend(list(self.short_term))
        messages.append(HumanMessage(content=user_input))

        # 3. LLM 호출(응답은 to_text 로 정규화)
        ai_reply = to_text(self.llm.invoke(messages).content)

        # 4. 단기 기억 갱신
        self.short_term.append(HumanMessage(content=user_input))
        self.short_term.append(AIMessage(content=ai_reply))

        # 5. 장기 기억 갱신(중요한 사실 추출 후 저장)
        for fact in self._extract_facts(user_input, ai_reply):
            self.long_term.remember(fact)

        print(f'[AI] {ai_reply[:130]}')
        return ai_reply


def make_store_memory_tools(store, user_id: str):
    """LangGraph Store 에 저장/조회하는 기억 도구 한 쌍을 생성하는 팩토리.

    store 와 user_id 를 **둘 다 클로저로 고정**한다. 특히 user_id 를 도구 인자로
    노출하지 않는 것이 핵심이다 — 노출하면 모델이 사용자 ID 를 스스로 지어내
    (예: 이름을 그대로 넣어) 엉뚱한 네임스페이스에 저장하거나, 대화 이력이 비어 있는
    새 스레드에서 "사용자 ID 를 알려 주세요"라고 되묻는다. **누구의 기억을 읽고 쓸지는
    모델이 아니라 애플리케이션이 정하는 권한 경계다.**

    Args:
        store: LangGraph BaseStore(예: InMemoryStore) 인스턴스.
        user_id: 기억을 격리할 네임스페이스 키. 실제 네임스페이스는 ('user_memory', user_id).

    Returns:
        [save_user_info, recall_user_info] LangChain 도구 리스트.
    """
    namespace = ('user_memory', user_id)

    @tool
    def save_user_info(key: str, value: str) -> str:
        '''사용자 정보를 장기 기억에 저장합니다. key: 항목명(예: 거주지), value: 내용'''
        existing = store.get(namespace, key)
        if existing and existing.value.get('content') == value:
            # 같은 값이면 다시 쓰지 않는다 — 작은 모델이 같은 저장을 반복하는 루프를 끊어 준다
            return f'이미 저장되어 있습니다(변경 없음): [{key}] {value}'
        data = dict(existing.value) if existing else {}
        data['content'] = value
        store.put(namespace, key, data)
        return f'저장 완료: [{key}] {value}'

    @tool
    def recall_user_info(key: str = '') -> str:
        '''사용자 정보를 장기 기억에서 조회합니다. key 를 생략하면 저장된 정보를 모두 반환합니다.'''
        if key:
            item = store.get(namespace, key)
            if item:
                return f'[{key}] {item.value["content"]}'
            return f'{key} 에 대한 정보가 없습니다.'
        items = store.search(namespace)
        if not items:
            return '저장된 정보가 없습니다.'
        return '\n'.join(f'[{i.key}] {i.value["content"]}' for i in items)

    return [save_user_info, recall_user_info]
