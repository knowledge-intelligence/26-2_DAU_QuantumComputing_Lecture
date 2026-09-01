"""
llama_rag — LlamaIndex 기반 RAG / GraphRAG (+ Neo4j 연동)
==========================================================

LangChain 과 함께 RAG 양대 프레임워크인 **LlamaIndex** 를 이 강의의 규약에 맞게 붙입니다.

핵심 문제: LlamaIndex 는 자체 LLM/임베딩 추상화를 갖고 있어서 그대로 쓰면
`.env` 의 `LLM_PROVIDER` 규약이 깨집니다(공급자를 코드에 박게 됨).
그래서 **어댑터 두 개** 를 만들어 이 강의의 공급자를 그대로 주입합니다.

    utils.get_llm()            →  LangChainLLM        (LlamaIndex CustomLLM)
    embeddings.get_embedder()  →  InjectedEmbedding   (LlamaIndex BaseEmbedding)

이렇게 하면 `llama-index-llms-*` / `llama-index-embeddings-*` 패키지를 하나도
설치하지 않고도 ollama·google·nvidia·openrouter 어디서나 동일하게 동작합니다.

제공 기능:
    configure()                   Settings 에 어댑터를 꽂아 전역 설정
    to_documents()                doc_prep.PersonalDoc → LlamaIndex Document
    build_vector_index()          VectorStoreIndex (기본 RAG)
    build_property_graph_index()  PropertyGraphIndex (GraphRAG). Neo4j 저장 지원
    ask()                         질의 후 답변 + 근거 노드 출력
"""

from typing import Dict, List, Optional, Sequence

from .bootstrap import to_text

try:
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.llms import (
        CompletionResponse,
        CompletionResponseGen,
        CustomLLM,
        LLMMetadata,
    )
    from llama_index.core.llms.callbacks import llm_completion_callback
    _LLAMA_AVAILABLE = True
except Exception:  # ModuleNotFoundError 등
    BaseEmbedding = object  # type: ignore
    CustomLLM = object      # type: ignore
    _LLAMA_AVAILABLE = False


def _require_llama() -> None:
    """LlamaIndex 미설치 시 설치 명령을 안내한다."""
    if not _LLAMA_AVAILABLE:
        raise ImportError(
            "llama-index-core 가 필요합니다 — uv pip install llama-index-core")


# =============================================================================
# 1. 어댑터 — 이 강의의 LLM/임베딩을 LlamaIndex 규격으로 감싼다
# =============================================================================
class LangChainLLM(CustomLLM):
    """LangChain BaseChatModel 을 LlamaIndex LLM 으로 감싸는 어댑터.

    LlamaIndex 는 `complete()` / `stream_complete()` 두 개만 있으면 동작한다.
    응답은 bootstrap.to_text() 로 정규화해 공급자별 차이(<think>, list content)를 흡수한다.
    """

    # pydantic 모델이라 필드를 선언해 둔다(LlamaIndex 의 LLM 은 pydantic 기반)
    context_window: int = 8192
    num_output: int = 1024
    model_name: str = "injected-langchain-llm"

    def __init__(self, llm, **kwargs):
        """LangChain 모델을 받아 감싼다.

        Args:
            llm: `utils.get_llm()` 이 돌려준 BaseChatModel.
        """
        super().__init__(**kwargs)
        # pydantic 이 관리하지 않는 속성은 object.__setattr__ 로 직접 넣는다
        object.__setattr__(self, "_llm", llm)

    @property
    def metadata(self) -> "LLMMetadata":
        """LlamaIndex 가 컨텍스트 길이 계산에 쓰는 메타데이터."""
        return LLMMetadata(context_window=self.context_window,
                           num_output=self.num_output,
                           model_name=self.model_name)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> "CompletionResponse":
        """프롬프트 하나를 넣고 완성 텍스트를 받는다."""
        response = self._llm.invoke(prompt)
        return CompletionResponse(text=to_text(response.content))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs) -> "CompletionResponseGen":
        """스트리밍은 쓰지 않으므로 완성본을 한 번에 흘려보낸다."""
        text = self.complete(prompt).text
        yield CompletionResponse(text=text, delta=text)


class InjectedEmbedding(BaseEmbedding):
    """agentic_lib.embeddings 의 임베더를 LlamaIndex 임베딩으로 감싸는 어댑터.

    질의/문서를 구분해 인코딩하므로 비대칭 모델(e5, nv-embedqa 등)도 제대로 동작한다.
    """

    def __init__(self, embedder, **kwargs):
        """임베더를 받아 감싼다.

        Args:
            embedder: `embeddings.get_embedder()` 결과.
        """
        super().__init__(**kwargs)
        object.__setattr__(self, "_embedder", embedder)

    @classmethod
    def class_name(cls) -> str:
        return "InjectedEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """질의 임베딩(kind="query")."""
        return self._embedder.encode([query], kind="query")[0].tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        """문서 임베딩(kind="passage")."""
        return self._embedder.encode([text], kind="passage")[0].tolist()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """문서 배치 임베딩 — 온라인 API 호출 수를 줄인다."""
        return [v.tolist() for v in self._embedder.encode(texts, kind="passage")]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """비동기 인터페이스(동기 구현을 그대로 사용)."""
        return self._get_query_embedding(query)


def configure(llm, embedder, chunk_size: int = 400, chunk_overlap: int = 80):
    """LlamaIndex 전역 Settings 에 이 강의의 LLM/임베딩을 꽂는다.

    Args:
        llm: LangChain BaseChatModel(`utils.get_llm()`).
        embedder: `embeddings.get_embedder()` 결과.
        chunk_size: LlamaIndex 기본 노드 분할 크기.
        chunk_overlap: 노드 간 겹침.

    Returns:
        설정된 Settings 객체(확인용).
    """
    _require_llama()
    from llama_index.core import Settings

    # LlamaIndex 는 내부적으로 asyncio.run() 을 쓰는데, Jupyter 는 이미 이벤트 루프가
    # 돌고 있어 충돌한다("asyncio.run() cannot be called from a running event loop").
    # nest_asyncio 가 루프 중첩을 허용해 준다(llama-index-core 의 의존성이라 항상 있다).
    try:
        import nest_asyncio

        nest_asyncio.apply()
    except Exception:
        pass  # 일반 파이썬 스크립트에서는 필요 없다

    Settings.llm = LangChainLLM(llm)
    Settings.embed_model = InjectedEmbedding(embedder)
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap
    print(f"LlamaIndex 설정 완료 — LLM: {type(llm).__name__} / "
          f"임베딩: {embedder.name} ({embedder.dim}차원) / 청크: {chunk_size}")
    return Settings


# =============================================================================
# 2. 문서 변환
# =============================================================================
def to_documents(docs: Sequence) -> List:
    """doc_prep.PersonalDoc(또는 dict) 목록을 LlamaIndex Document 로 바꾼다.

    메타데이터를 그대로 실어 보내야 나중에 출처 표시와 필터가 가능하다.

    Args:
        docs: PersonalDoc 목록 또는 {"text","metadata"} dict 목록.

    Returns:
        llama_index.core.Document 목록.
    """
    _require_llama()
    from llama_index.core import Document

    documents = []
    for doc in docs:
        if hasattr(doc, "body"):          # doc_prep.PersonalDoc
            documents.append(Document(
                text=doc.body,
                doc_id=doc.doc_id,
                metadata={"title": doc.title, "category": doc.category,
                          "created": doc.created, "author": doc.author,
                          "tags": ",".join(doc.tags), "source": doc.path},
            ))
        else:                              # 일반 dict
            documents.append(Document(text=doc.get("text", ""),
                                      metadata=doc.get("metadata", {})))
    return documents


# =============================================================================
# 3. 인덱스 — Vector / PropertyGraph
# =============================================================================
def build_vector_index(documents: Sequence):
    """LlamaIndex 기본 RAG 인덱스(VectorStoreIndex)를 만든다.

    LangChain 으로 직접 조립했던 것(청킹 → 임베딩 → 저장 → 검색)을
    LlamaIndex 는 이 한 줄로 처리한다 — 추상화 수준의 차이를 보여 주는 대목이다.
    """
    _require_llama()
    from llama_index.core import VectorStoreIndex

    return VectorStoreIndex.from_documents(list(documents), show_progress=False)


def build_property_graph_index(documents: Sequence, neo4j: Optional[Dict] = None,
                               max_triplets_per_chunk: int = 8,
                               show_progress: bool = False):
    """**GraphRAG** 인덱스(PropertyGraphIndex)를 만든다.

    LLM 이 문서에서 (주어, 관계, 목적어) 트리플을 뽑아 지식 그래프를 자동 구축한다.
    Vector RAG 가 "비슷한 청크"를 찾는다면, 이쪽은 "연결된 엔티티"를 따라간다.

    Args:
        documents: to_documents() 결과.
        neo4j: {"uri","user","password"} 를 주면 Neo4j 에 그래프를 저장한다.
            None 이면 메모리 내 SimplePropertyGraphStore 를 쓴다.
        max_triplets_per_chunk: 청크당 최대 추출 트리플 수(많을수록 느리다).
        show_progress: 추출 진행률 표시.

    Returns:
        PropertyGraphIndex.
    """
    _require_llama()
    from llama_index.core import PropertyGraphIndex
    from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

    graph_store = None
    if neo4j:
        from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

        graph_store = Neo4jPropertyGraphStore(
            username=neo4j["user"], password=neo4j["password"], url=neo4j["uri"],
        )
        print(f"Neo4j 그래프 저장소 연결: {neo4j['uri']}")

    return PropertyGraphIndex.from_documents(
        list(documents),
        property_graph_store=graph_store,
        kg_extractors=[SimpleLLMPathExtractor(
            max_paths_per_chunk=max_triplets_per_chunk, num_workers=1)],
        show_progress=show_progress,
    )


#: SimpleLLMPathExtractor 기본 프롬프트의 영어 few-shot 예시가 그대로 새어 나온 흔적.
#: 작은 모델일수록 예시를 베껴 오므로 표시할 때 걸러 준다.
_TEMPLATE_ARTIFACTS = {"subject", "object", "predicate", "philz", "coffee shop",
                       "entity", "relation"}


def get_triplets(index) -> List[tuple]:
    """PropertyGraphIndex 에서 (주어, 관계, 목적어) 트리플을 모두 가져온다.

    `get_triplets()` 는 필터 인자가 없으면 빈 목록을 돌려주므로,
    먼저 엔티티 이름을 모두 모은 뒤 그것으로 조회한다.
    """
    store = index.property_graph_store
    entity_names = sorted({n.name for n in store.get()
                           if hasattr(n, "name") and n.name})
    if not entity_names:
        return []
    # 같은 트리플이 여러 엔티티 조회에 중복해서 걸리므로 (주어,관계,목적어)로 중복 제거
    seen, unique = set(), []
    for subject, relation, obj in store.get_triplets(entity_names=entity_names):
        key = (subject.name, relation.label, obj.name)
        if key not in seen:
            seen.add(key)
            unique.append((subject, relation, obj))
    return unique


def print_extracted_triplets(index, limit: int = 20,
                             drop_artifacts: bool = True) -> None:
    """LLM 이 문서에서 자동 추출한 지식 그래프를 출력한다.

    Args:
        index: PropertyGraphIndex.
        limit: 출력할 최대 개수.
        drop_artifacts: 프롬프트 예시가 새어 나온 트리플(Subject→Object 등)을 숨길지.
    """
    try:
        triplets = get_triplets(index)
    except Exception as e:
        print(f"트리플 조회 실패: {str(e)[:120]}")
        return

    junk = [t for t in triplets
            if t[0].name.strip().lower() in _TEMPLATE_ARTIFACTS
            or t[2].name.strip().lower() in _TEMPLATE_ARTIFACTS]
    shown = [t for t in triplets if t not in junk] if drop_artifacts else triplets

    print(f"LLM 이 자동 추출한 관계 {len(shown)}개 (상위 {min(limit, len(shown))}개)")
    for subject, relation, obj in shown[:limit]:
        print(f"  ({subject.name}) -[{relation.label}]-> ({obj.name})")
    if junk and drop_artifacts:
        print(f"\n※ 프롬프트 예시가 그대로 복사된 트리플 {len(junk)}개는 숨겼습니다"
              f"(예: {junk[0][0].name} → {junk[0][2].name}).")
        print("   작은 모델일수록 추출 프롬프트의 few-shot 예시를 베껴 오는 현상이 흔합니다 —"
              " GraphRAG 품질은 LLM 성능에 크게 좌우됩니다.")


# =============================================================================
# 4. 질의
# =============================================================================
#: PropertyGraphIndex 의 검색 결과 노드는 LlamaIndex 가
#:     "Here are some facts extracted from the provided text:" + 트리플들 + 원본 청크
#: 형태로 조립한다(sub_retrievers/base.py 의 DEFAULT_PREAMBLE).
#: 이 문구가 미리보기 앞자리를 다 차지해 근거 노드가 전부 똑같아 보이므로 표시할 때 걷어낸다.
_FALLBACK_PREAMBLE = "Here are some facts extracted from the provided text:"


def _graph_preamble() -> str:
    """LlamaIndex 가 그래프 노드 앞에 붙이는 정형 문구를 가져온다(버전 차이 흡수)."""
    try:
        from llama_index.core.indices.property_graph.sub_retrievers.base import (
            DEFAULT_PREAMBLE,
        )
        return DEFAULT_PREAMBLE.strip()
    except Exception:
        return _FALLBACK_PREAMBLE


def _node_snippet(node, width: int = 70) -> str:
    """근거 노드 미리보기 문자열 — 정형 문구를 떼고 공백을 정리한다.

    떼고 나면 GraphRAG 가 실제로 근거로 삼은 **트리플**이 앞에 오므로,
    노드마다 무엇이 달랐는지가 눈에 보인다.
    """
    text = node.get_content().lstrip()
    preamble = _graph_preamble()
    if text.startswith(preamble):
        text = text[len(preamble):]
    return " ".join(text.split())[:width]  # 줄바꿈·중복 공백 정리 후 자르기


def ask(index, question: str, top_k: int = 3, verbose: bool = True) -> str:
    """인덱스에 질문하고 답변 + 근거 노드를 보여 준다.

    Args:
        index: VectorStoreIndex 또는 PropertyGraphIndex.
        question: 질문.
        top_k: 검색할 노드 수.
        verbose: 근거 노드를 함께 출력할지.

    Returns:
        답변 문자열.
    """
    engine = index.as_query_engine(similarity_top_k=top_k)
    response = engine.query(question)

    if verbose:
        print(f"질문: {question}")
        nodes = getattr(response, "source_nodes", []) or []
        print(f"\n근거 노드 {len(nodes)}건:")
        for i, node in enumerate(nodes, 1):
            meta = node.node.metadata or {}
            title = meta.get("title", meta.get("file_name", "제목없음"))
            snippet = _node_snippet(node.node)
            score = f"{node.score:.3f}" if node.score is not None else "  -  "
            print(f"  [{i}] ({score}) {title}: {snippet}…")
        print(f"\n답변:\n{response}")
        print("-" * 84)
    return str(response)


def compare_vector_and_graph(vector_index, graph_index, questions: Sequence[str],
                             top_k: int = 3) -> None:
    """같은 질문을 Vector RAG 와 GraphRAG 에 각각 던져 답변을 비교한다.

    단일 사실 조회는 Vector 가 빠르고 충분하다.
    여러 문서에 흩어진 사실을 이어야 하는 질문에서 Graph 쪽이 값을 한다.
    """
    import time

    for question in questions:
        print(f"\n{'=' * 84}\n질문: {question}\n")
        for label, index in [("Vector RAG", vector_index), ("GraphRAG", graph_index)]:
            started = time.perf_counter()
            try:
                answer = ask(index, question, top_k=top_k, verbose=False)
                print(f"[{label}] {time.perf_counter() - started:.1f}s")
                print(f"  {answer[:300]}\n")
            except Exception as e:
                print(f"[{label}] 실패: {str(e)[:140]}\n")
