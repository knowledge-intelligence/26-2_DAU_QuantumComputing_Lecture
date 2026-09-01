"""
rag — 지식 검색(RAG)·지식 그래프 공통 구현
==========================================

모듈 3(9-11주차, 지식) 노트북에서 반복되던 RAG/지식 그래프 구현을 한곳에 모았습니다.
노트북은 '개념'과 '실행'에 집중하고, 길고 반복되는 구현 세부는 이 모듈을 import 해서 씁니다.

구성:
    chunk_document          utils.chunk_text 래퍼(문서를 단어 기준 청크로 분할)
    VectorRAG               ChromaDB 컬렉션을 감싼 벡터 검색/생성(LLM 주입)
    KnowledgeGraph          순수 Python(dict) 기반 지식 그래프 — Neo4j 없이 개념 학습용
    Neo4jKnowledgeGraph     Neo4j 기반 지식 그래프(미설치/미연결 시 친절히 안내 후 폴백)
    Neo4jVectorStore        Neo4j 네이티브 벡터 인덱스(5.11+) + 그래프 확장 검색 = Graph RAG
    HybridRAG               Vector RAG + Graph RAG 결합(LLM 주입)

LLM 호출 규약(중요):
    - 이 모듈은 LLM 인스턴스를 직접 만들지 않는다. 생성자나 메서드 인자로 '주입'받는다.
      (노트북이 utils.get_llm() 으로 만든 공급자 무관 모델을 그대로 넘긴다)
    - 모든 LLM 응답은 bootstrap.to_text() 로 정규화한다.
      (Gemini 의 list[dict], qwen3 의 <think> 블록 등을 깔끔한 문자열로 통일)
"""

from collections import deque
from typing import Dict, List, Optional

import utils  # notebooks/utils.py — chunk_text / cosine_similarity 재사용
from .bootstrap import to_text

# Neo4j 드라이버는 선택 의존성이다. 설치되어 있지 않아도 다른 기능은 동작해야 하므로
# import 실패를 흡수하고, 실제 사용 시점에 친절히 안내한다.
try:
    from neo4j import GraphDatabase  # type: ignore
    _NEO4J_AVAILABLE = True
except Exception:  # ModuleNotFoundError 등
    GraphDatabase = None  # type: ignore
    _NEO4J_AVAILABLE = False


def chunk_document(text: str, chunk_size: int = 50, overlap: int = 10) -> List[str]:
    """문서 텍스트를 겹치는 단어 청크로 분할한다(utils.chunk_text 래퍼).

    Args:
        text: 원본 문서 본문.
        chunk_size: 청크당 단어 수.
        overlap: 인접 청크 간 겹치는 단어 수.

    Returns:
        청크 문자열 리스트.
    """
    return utils.chunk_text(text, chunk_size=chunk_size, overlap=overlap)


class VectorRAG:
    """벡터 RAG — ChromaDB 컬렉션을 감싸 유사도 검색과 답변 생성을 담당한다.

    임베딩·인덱싱은 ChromaDB 가 처리하고, 이 클래스는 검색 결과 정리와
    (주입된 LLM 을 사용한) 답변 생성을 맡는다.
    """

    def __init__(self, collection, llm=None):
        """벡터 RAG 를 생성한다.

        Args:
            collection: 문서가 적재된 ChromaDB 컬렉션.
            llm: 답변 생성에 쓸 LangChain BaseChatModel(없으면 검색만, 생성은 폴백).
        """
        self.collection = collection
        self.llm = llm

    def search(self, query: str, n_results: int = 3) -> List[Dict]:
        """쿼리와 가장 유사한 청크를 검색한다.

        Args:
            query: 검색 질의.
            n_results: 반환할 상위 결과 수.

        Returns:
            {content, title, source, similarity} 딕셔너리 리스트(유사도 = 1 - 거리).
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append({
                "content": doc,
                "title": meta["title"],
                "source": meta["source"],
                "similarity": 1 - dist,
            })
        return retrieved

    def generate(self, query: str, llm=None, n_results: int = 3, verbose: bool = True) -> str:
        """검색 결과를 컨텍스트로 LLM 답변을 생성한다.

        Args:
            query: 사용자 질문.
            llm: 이번 호출에만 쓸 LLM(없으면 생성자 주입 LLM 사용).
            n_results: 검색할 청크 수.
            verbose: True 면 검색된 청크 요약을 출력한다.

        Returns:
            to_text 로 정규화된 답변 문자열(LLM 이 없으면 컨텍스트 요약 폴백).
        """
        retrieved = self.search(query, n_results=n_results)
        context = "\n\n".join([
            f"[{r['title']}] {r['content']} (유사도: {r['similarity']:.3f})"
            for r in retrieved
        ])

        if verbose:
            print(f"\n=== RAG 검색 결과 (쿼리: '{query}') ===")
            for r in retrieved:
                print(f"  [{r['similarity']:.3f}] [{r['title']}] {r['content'][:80]}...")

        model = llm or self.llm
        if model is None:
            return (f"[컨텍스트 기반 답변]\n검색된 {len(retrieved)}개 문서 기반:\n"
                    f"{context[:200]}")

        prompt = f"""다음 참고 문서를 바탕으로 질문에 답변하세요.

참고 문서:
{context}

질문: {query}

위 문서에 기반하여 정확하고 간결하게 한국어로 답변하세요. 문서에 없는 내용은 추측하지 마세요."""

        from langchain_core.messages import HumanMessage
        response = model.invoke([HumanMessage(content=prompt)])
        return to_text(response.content)  # 공급자 무관 정규화


class KnowledgeGraph:
    """순수 Python(dict) 기반 지식 그래프 — Neo4j 연동 전 개념 이해용.

    엔티티는 노드(id → {type, properties}), 관계는 엣지({from, relation, to, properties})로
    표현한다. 이웃 탐색·multi-hop 탐색·RAG 컨텍스트 변환을 지원한다.
    """

    def __init__(self):
        self.nodes: Dict[str, Dict] = {}  # id -> {type, properties}
        self.edges: List[Dict] = []       # {from, to, relation, properties}

    def add_node(self, node_id: str, node_type: str, **properties) -> "KnowledgeGraph":
        """노드(엔티티)를 추가한다. 메서드 체이닝을 위해 self 를 반환한다."""
        self.nodes[node_id] = {"type": node_type, "properties": properties}
        return self

    def add_edge(self, from_id: str, relation: str, to_id: str, **properties) -> "KnowledgeGraph":
        """엣지(관계)를 추가한다. 메서드 체이닝을 위해 self 를 반환한다."""
        self.edges.append({
            "from": from_id,
            "relation": relation,
            "to": to_id,
            "properties": properties,
        })
        return self

    def query_neighbors(self, node_id: str, relation: str = None,
                        direction: str = "out") -> List[Dict]:
        """노드의 직접 이웃(1홉)을 탐색한다.

        엣지는 방향이 있으므로(`from -[relation]-> to`) **어느 방향을 볼지** 를 정해야 한다.
        문서의 태그처럼 항상 화살표를 '받기만' 하는 노드는 `direction="out"` 으로는
        이웃이 0개로 나온다 — 이때는 `"in"` 이나 `"both"` 를 써야 한다.

        Args:
            node_id: 시작 노드 id.
            relation: 지정하면 해당 관계의 엣지만 남긴다(None 이면 전체).
            direction: `"out"`(나가는 엣지, 기본) · `"in"`(들어오는 엣지) ·
                `"both"`(양쪽 = Cypher 의 무방향 `-[]-` 에 해당).

        Returns:
            `{node, relation, direction, properties}` 딕셔너리 리스트.
            `direction` 은 그 이웃이 나가는 엣지(`"out"`)로 닿았는지 들어오는 엣지(`"in"`)로
            닿았는지를 알려 주므로, 화살표를 올바른 방향으로 출력할 수 있다.
        """
        if direction not in ("out", "in", "both"):
            raise ValueError("direction 은 'out' / 'in' / 'both' 중 하나여야 한다")

        results = []
        for edge in self.edges:
            if relation is not None and edge["relation"] != relation:
                continue
            # 이 엣지에서 '반대편 노드'가 무엇이고 어느 방향으로 닿았는지 정한다
            if direction in ("out", "both") and edge["from"] == node_id:
                other, edge_direction = edge["to"], "out"
            elif direction in ("in", "both") and edge["to"] == node_id:
                other, edge_direction = edge["from"], "in"
            else:
                continue
            neighbor = self.nodes.get(other, {})
            results.append({
                "node": other,
                "relation": edge["relation"],
                "direction": edge_direction,
                "properties": neighbor.get("properties", {}),
            })
        return results

    def multi_hop_query(self, start_node: str, max_hops: int = 2,
                        direction: str = "out") -> List[List]:
        """시작 노드에서 `max_hops` 홉 이내로 닿는 경로를 BFS 로 수집한다(다단계 추론).

        **홉(hop)** 은 '엣지를 한 번 건너는 것'이다. 1홉은 직접 이웃, 2홉은 이웃의 이웃이며,
        `max_hops=2` 는 엣지를 최대 두 번까지 건넌 경로만 모은다는 뜻이다.
        가까운 홉부터 넓혀 가는 BFS 라서 각 노드는 **가장 짧은 경로로 한 번만** 등장한다
        (DFS 로 하면 1홉이면 닿을 노드가 우연히 2홉 경로로 먼저 발견되어 가려질 수 있다).

        Args:
            start_node: 출발 노드 id.
            max_hops: 건널 수 있는 최대 엣지 수.
            direction: 엣지 방향 규약 — `query_neighbors()` 와 동일(`"out"`/`"in"`/`"both"`).

        Returns:
            경로 리스트. 각 경로는 `[(relation, node), ...]` 형태이고 길이가 곧 홉 수다.
        """
        visited = {start_node}
        paths: List[List] = []
        queue = deque([(start_node, [])])  # (현재 노드, 여기까지의 경로)

        while queue:
            node, path = queue.popleft()
            if len(path) >= max_hops:  # 이미 max_hops 만큼 건넜으면 더 나아가지 않는다
                continue
            for n in self.query_neighbors(node, direction=direction):
                if n["node"] in visited:
                    continue
                visited.add(n["node"])
                new_path = path + [(n["relation"], n["node"])]
                paths.append(new_path)
                queue.append((n["node"], new_path))

        return paths

    def to_context(self, node_ids: List[str]) -> str:
        """지정한 노드들의 설명과 이웃 관계를 RAG 컨텍스트 문자열로 변환한다.

        태그처럼 '받는 쪽'인 노드도 근거를 갖도록 **양방향**(`direction="both"`) 으로 이웃을
        모은다. 들어오는 엣지는 `문서 -[mentions]-> 태그` 처럼 원래 방향 그대로 적어
        LLM 이 관계를 거꾸로 읽지 않게 한다.
        """
        lines = []
        for node_id in node_ids:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                props = node["properties"]
                lines.append(f"{node_id} ({node['type']}): {props.get('description', '')}")
                for n in self.query_neighbors(node_id, direction="both"):
                    if n["direction"] == "out":
                        lines.append(f"  {node_id} -[{n['relation']}]-> {n['node']}")
                    else:
                        lines.append(f"  {n['node']} -[{n['relation']}]-> {node_id}")
        return "\n".join(lines)


class Neo4jKnowledgeGraph:
    """Neo4j 기반 지식 그래프.

    Neo4j 드라이버 미설치 또는 서버 미연결 시 예외를 던지지 않고 connected=False 로
    두어, 노트북이 로컬 KnowledgeGraph 로 자연스럽게 폴백할 수 있게 한다.
    """

    def __init__(self, uri: str, user: str, password: str):
        """드라이버를 만들고 연결을 점검한다(실패해도 예외를 던지지 않음).

        Args:
            uri: 예) bolt://localhost:7687
            user: 사용자명(예: neo4j)
            password: 비밀번호
        """
        self.uri = uri
        self.connected = False
        if not _NEO4J_AVAILABLE:
            print("[Neo4j] 'neo4j' 드라이버가 설치되어 있지 않습니다 "
                  "(CMD:  uv pip install neo4j). 로컬 KnowledgeGraph 모드로 전환합니다.")
            return
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            print(f"Neo4j 연결 성공: {uri}")
            self.connected = True
        except Exception as e:
            print(f"Neo4j 연결 실패: {e}")
            print("로컬 KnowledgeGraph 모드로 전환합니다.")

    def close(self) -> None:
        """드라이버 연결을 닫는다(연결되어 있을 때만)."""
        if self.connected:
            self.driver.close()

    def setup_schema(self) -> None:
        """유니크 제약(인덱스)을 설정한다."""
        if not self.connected:
            return
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT tech_name IF NOT EXISTS "
                        "FOR (t:Technology) REQUIRE t.name IS UNIQUE")
            session.run("CREATE CONSTRAINT company_name IF NOT EXISTS "
                        "FOR (c:Company) REQUIRE c.name IS UNIQUE")
        print("Neo4j 스키마 설정 완료")

    def ingest_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """Python KnowledgeGraph 의 노드/관계를 Neo4j 에 적재한다."""
        if not self.connected:
            print("Neo4j 미연결: 로컬 그래프 사용")
            return
        with self.driver.session() as session:
            # 노드 생성(MERGE 로 멱등성 보장)
            for node_id, node_data in kg.nodes.items():
                session.run(
                    f"MERGE (n:{node_data['type']} {{name: $name}}) SET n += $props",
                    name=node_id,
                    props=node_data["properties"],
                )
            # 관계 생성(관계 타입은 대문자_언더스코어로 정규화)
            for edge in kg.edges:
                relation = edge["relation"].upper().replace(" ", "_")
                session.run(
                    f"""
                    MATCH (a {{name: $from_name}}), (b {{name: $to_name}})
                    MERGE (a)-[r:{relation}]->(b)
                    """,
                    from_name=edge["from"],
                    to_name=edge["to"],
                )
        print(f"Neo4j에 {len(kg.nodes)}개 노드, {len(kg.edges)}개 관계 저장 완료")

    def cypher_query(self, query: str, params: dict = None) -> List[Dict]:
        """임의의 Cypher 쿼리를 실행하고 결과를 dict 리스트로 반환한다."""
        if not self.connected:
            return []
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]

    def entity_search(self, entity_name: str) -> List[Dict]:
        """엔티티와 그 직접 관계를 조회한다."""
        query = """
        MATCH (n {name: $name})
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n.name as entity, n.description as description,
               type(r) as relation, m.name as related_entity
        """
        return self.cypher_query(query, {"name": entity_name})

    def multi_hop_search(self, start: str, end_type: str, max_hops: int = 3) -> List[Dict]:
        """start 노드에서 end_type 라벨까지 최대 max_hops 경로를 탐색한다."""
        query = f"""
        MATCH path = (start {{name: $start}})-[*1..{max_hops}]->(end:{end_type})
        RETURN [node in nodes(path) | node.name] as path_nodes,
               length(path) as hops
        ORDER BY hops
        LIMIT 5
        """
        return self.cypher_query(query, {"start": start})


class HybridRAG:
    """Vector RAG + Graph RAG 결합 시스템.

    벡터 검색 결과와 지식 그래프 컨텍스트를 하나로 합쳐 LLM 에 제공한다.
    LLM 은 주입받으며, 응답은 to_text 로 정규화한다.
    """

    def __init__(self, vector_rag: VectorRAG, knowledge_graph: KnowledgeGraph, llm=None):
        """하이브리드 RAG 를 생성한다.

        Args:
            vector_rag: 벡터 검색을 담당하는 VectorRAG 인스턴스.
            knowledge_graph: 그래프 컨텍스트를 제공하는 KnowledgeGraph.
            llm: 답변 생성용 LLM(없으면 폴백 요약).
        """
        self.vector_rag = vector_rag
        self.kg = knowledge_graph
        self.llm = llm

    def extract_entities(self, query: str) -> List[str]:
        """쿼리 문자열에 이름이 등장하는 그래프 노드를 엔티티로 추출한다(간이 NER)."""
        return [nid for nid in self.kg.nodes if nid.lower() in query.lower()]

    def search(self, query: str, n_vector: int = 3) -> Dict:
        """벡터 검색 + 그래프 엔티티 컨텍스트를 합쳐 반환한다."""
        vector_results = self.vector_rag.search(query, n_results=n_vector)
        entities = self.extract_entities(query)
        graph_context = self.kg.to_context(entities) if entities else ""
        vector_context = "\n".join([
            f"[Vector:{r['title']}] {r['content']}" for r in vector_results
        ])
        combined = (f"=== 벡터 검색 결과 ===\n{vector_context}\n\n"
                    f"=== 지식 그래프 컨텍스트 ===\n{graph_context or '관련 엔티티 없음'}")
        return {
            "query": query,
            "vector_results": vector_results,
            "graph_entities": entities,
            "combined_context": combined,
        }

    def generate_answer(self, query: str, llm=None, verbose: bool = True) -> str:
        """벡터+그래프 컨텍스트로 LLM 답변을 생성한다(to_text 정규화)."""
        search_result = self.search(query)
        if verbose:
            print("=== Hybrid RAG 검색 ===")
            print(f"쿼리: {query}")
            print(f"발견된 엔티티: {search_result['graph_entities']}")

        model = llm or self.llm
        if model is None:
            return (f"[하이브리드 RAG] 벡터 {len(search_result['vector_results'])}개 + "
                    f"그래프 {len(search_result['graph_entities'])}개 엔티티 기반 답변")

        from langchain_core.messages import HumanMessage
        response = model.invoke([HumanMessage(content=f"""다음 컨텍스트를 바탕으로 질문에 답하세요.

{search_result['combined_context']}

질문: {query}

벡터 검색과 지식 그래프 정보를 모두 활용하여 한국어로 답변하세요.""")])
        return to_text(response.content)  # 공급자 무관 정규화


class Neo4jVectorStore:
    """Neo4j 를 **벡터 데이터베이스 + 그래프 DB** 로 동시에 쓰는 저장소.

    Neo4j 5.11 부터 네이티브 벡터 인덱스를 지원하므로, 문서 노드에 임베딩을
    속성으로 저장해 두면 Cypher 한 줄로 시맨틱 검색을 할 수 있다.
    여기에 그래프 관계(:MENTIONS 등)를 얹으면
    "벡터로 후보를 찾고 → 관계로 근거를 확장하는" Graph RAG 가 된다.

    그래프 구조:
        (:Document {doc_id, text, embedding})-[:MENTIONS]->(:Topic {name})
        (:Topic)-[관계]->(:Topic)

    연결 실패 시 예외를 던지지 않고 connected=False 로 두어, 노트북이
    로컬 폴백 경로로 자연스럽게 넘어갈 수 있게 한다.
    """

    def __init__(self, uri: str, user: str, password: str,
                 database: str = "neo4j", index_name: str = "doc_embeddings"):
        """드라이버를 만들고 연결을 점검한다(실패해도 예외 없음).

        Args:
            uri: 예) bolt://localhost:7687
            user: 사용자명(기본 neo4j)
            password: 비밀번호
            database: 사용할 데이터베이스명
            index_name: 생성/조회할 벡터 인덱스 이름
        """
        self.uri = uri
        self.database = database
        self.index_name = index_name
        self.dim = 0
        self.connected = False
        self.driver = None
        if not _NEO4J_AVAILABLE:
            print("[Neo4j] 'neo4j' 드라이버가 없습니다 (CMD:  uv pip install neo4j).")
            return
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            self.connected = True
            print(f"Neo4j 연결 성공: {uri}")
        except Exception as e:
            print(f"Neo4j 연결 실패: {e}")
            print("→ Docker 로 기동(CMD, APOC 포함):  docker run -d --name neo4j "
                  "-p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password "
                  '-e "NEO4J_PLUGINS=[\\"apoc\\"]" neo4j:5-community')

    # ---- 기본 실행 ---------------------------------------------------------
    def run(self, cypher: str, **params) -> List[Dict]:
        """임의의 Cypher 를 실행하고 결과를 dict 리스트로 반환한다.

        Args:
            cypher: 실행할 Cypher 문.
            **params: `$name` 형태 파라미터(문자열 포매팅 대신 항상 이걸 쓴다).
        """
        if not self.connected:
            return []
        with self.driver.session(database=self.database) as session:
            return [dict(record) for record in session.run(cypher, **params)]

    def close(self) -> None:
        """드라이버 연결을 닫는다."""
        if self.connected and self.driver is not None:
            self.driver.close()

    def clear(self) -> None:
        """실습 데이터(Document/Topic)를 모두 지운다 — 반복 실행을 멱등하게 만든다."""
        self.run("MATCH (n) WHERE n:Document OR n:Topic DETACH DELETE n")
        print("기존 Document/Topic 노드 삭제 완료")

    # ---- 스키마 / 인덱스 ---------------------------------------------------
    def setup_schema(self, dim: int, similarity: str = "cosine") -> None:
        """유니크 제약과 **벡터 인덱스** 를 만든다(이미 있으면 건너뜀).

        Args:
            dim: 임베딩 차원. 인덱스 생성 시 고정되므로 임베더를 바꾸면 인덱스도 다시 만들어야 한다.
            similarity: 'cosine' 또는 'euclidean'.
        """
        if not self.connected:
            return
        self.dim = dim
        self.run("CREATE CONSTRAINT doc_id IF NOT EXISTS "
                 "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE")
        self.run("CREATE CONSTRAINT topic_name IF NOT EXISTS "
                 "FOR (t:Topic) REQUIRE t.name IS UNIQUE")
        # 인덱스 이름/차원은 Cypher 파라미터로 넘길 수 없어 f-string 을 쓴다(내부 상수라 안전)
        self.run(f"""
        CREATE VECTOR INDEX {self.index_name} IF NOT EXISTS
        FOR (d:Document) ON (d.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {int(dim)},
            `vector.similarity_function`: '{similarity}'
        }}}}
        """)
        self.run("CALL db.awaitIndexes(60000)")  # 인덱스가 ONLINE 이 될 때까지 대기
        print(f"스키마 준비 완료 — 벡터 인덱스 '{self.index_name}' (dim={dim}, {similarity})")

    def drop_vector_index(self) -> None:
        """벡터 인덱스를 삭제한다(차원이 다른 임베더로 바꿀 때 사용)."""
        self.run(f"DROP INDEX {self.index_name} IF EXISTS")

    # ---- 적재 --------------------------------------------------------------
    def add_documents(self, texts: List[str], embeddings,
                      topics: Optional[List[List[str]]] = None) -> None:
        """문서 텍스트 + 임베딩(+주제 태그)을 Neo4j 에 적재한다.

        Args:
            texts: 문서 본문 리스트.
            embeddings: (len(texts), dim) 임베딩 배열(numpy 또는 리스트).
            topics: 문서별 주제 태그 리스트. 주면 (:Document)-[:MENTIONS]->(:Topic) 을 만든다.
        """
        if not self.connected:
            print("Neo4j 미연결 — 적재를 건너뜁니다")
            return
        for i, text in enumerate(texts):
            vector = embeddings[i]
            vector = vector.tolist() if hasattr(vector, "tolist") else list(vector)
            # 벡터 속성은 db.create.setNodeVectorProperty 로 넣어야 인덱스가 인식한다
            self.run("""
                MERGE (d:Document {doc_id: $doc_id})
                SET d.text = $text
                WITH d
                CALL db.create.setNodeVectorProperty(d, 'embedding', $vector)
                """, doc_id=f"doc-{i}", text=text, vector=vector)
            for topic in (topics[i] if topics else []):
                self.run("""
                    MATCH (d:Document {doc_id: $doc_id})
                    MERGE (t:Topic {name: $topic})
                    MERGE (d)-[:MENTIONS]->(t)
                    """, doc_id=f"doc-{i}", topic=topic)
        print(f"문서 {len(texts)}개 적재 완료" + (" (주제 태그 포함)" if topics else ""))

    def add_topic_links(self, links: List[tuple]) -> None:
        """주제 사이의 관계 (from, RELATION, to) 를 만든다(multi-hop 탐색용)."""
        if not self.connected:
            return
        for from_topic, relation, to_topic in links:
            rel = relation.upper().replace(" ", "_")
            self.run(f"""
                MERGE (a:Topic {{name: $a}})
                MERGE (b:Topic {{name: $b}})
                MERGE (a)-[:{rel}]->(b)
                """, a=from_topic, b=to_topic)
        print(f"주제 관계 {len(links)}개 생성 완료")

    # ---- 검색 --------------------------------------------------------------
    def similarity_search(self, query_embedding, k: int = 3) -> List[Dict]:
        """벡터 인덱스로 시맨틱 검색한다(Cypher: db.index.vector.queryNodes).

        Args:
            query_embedding: 질의 임베딩(1차원).
            k: 반환할 문서 수.

        Returns:
            [{doc_id, text, score}] — score 는 코사인 유사도(0~1).
        """
        vector = query_embedding.tolist() if hasattr(query_embedding, "tolist") \
            else list(query_embedding)
        return self.run("""
            CALL db.index.vector.queryNodes($index, $k, $vector)
            YIELD node, score
            RETURN node.doc_id AS doc_id, node.text AS text, score
            ORDER BY score DESC
            """, index=self.index_name, k=k, vector=vector)

    def graph_expanded_search(self, query_embedding, k: int = 2, hops: int = 2) -> Dict:
        """**벡터로 씨앗을 찾고 → 그래프로 확장**하는 Graph RAG 검색.

        1) 벡터 인덱스로 가장 가까운 문서 k개(씨앗)를 찾는다.
        2) 씨앗 문서가 언급한 주제에서 최대 hops 단계까지 관련 주제를 넓힌다.
        3) 넓힌 주제를 언급하는 다른 문서들을 근거로 추가한다.

        Args:
            query_embedding: 질의 임베딩.
            k: 씨앗 문서 수.
            hops: 주제 그래프에서 확장할 홉 수.

        Returns:
            {"seeds": [...], "topics": [...], "expanded": [...]}
        """
        seeds = self.similarity_search(query_embedding, k=k)
        if not seeds:
            return {"seeds": [], "topics": [], "expanded": []}
        seed_ids = [s["doc_id"] for s in seeds]

        # 씨앗 문서 → 주제 → (최대 hops) 관련 주제
        topics = self.run(f"""
            MATCH (d:Document)-[:MENTIONS]->(t:Topic)
            WHERE d.doc_id IN $ids
            OPTIONAL MATCH (t)-[*1..{int(hops)}]-(related:Topic)
            WITH collect(DISTINCT t.name) + collect(DISTINCT related.name) AS names
            UNWIND names AS name
            RETURN DISTINCT name ORDER BY name
            """, ids=seed_ids)
        topic_names = [t["name"] for t in topics if t["name"]]

        # 확장된 주제를 언급하는 (씨앗이 아닌) 다른 문서들
        expanded = self.run("""
            MATCH (d:Document)-[:MENTIONS]->(t:Topic)
            WHERE t.name IN $topics AND NOT d.doc_id IN $ids
            RETURN d.doc_id AS doc_id, d.text AS text,
                   collect(DISTINCT t.name) AS via_topics,
                   count(DISTINCT t) AS overlap
            ORDER BY overlap DESC
            """, topics=topic_names, ids=seed_ids)
        return {"seeds": seeds, "topics": topic_names, "expanded": expanded}

    def stats(self) -> Dict:
        """적재된 Document/Topic/관계 개수를 요약한다."""
        rows = self.run("""
            OPTIONAL MATCH (d:Document) WITH count(DISTINCT d) AS docs
            OPTIONAL MATCH (t:Topic) WITH docs, count(DISTINCT t) AS topics
            OPTIONAL MATCH ()-[r]->() RETURN docs, topics, count(r) AS rels
            """)
        return rows[0] if rows else {}
