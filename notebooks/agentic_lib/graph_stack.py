"""
graph_stack — Graph RAG 확장 스택: 다른 그래프 DB · Text2Cypher · 커뮤니티 요약
==============================================================================

M04_3 의 2~5장이 **Neo4j + LlamaIndex** 한 조합을 깊게 판다면, 이 모듈은 그 옆의
선택지들을 실습 가능한 형태로 모읍니다.

구성:
    GRAPH_DB_MATRIX / OTHER_DB_SNIPPETS / print_other_db_snippets()
        DB 별 선택 기준 데이터와, 실습에서 띄우지 않는 DB(NebulaGraph·ArangoDB)의 연결 코드
    BoltGraph
        Bolt 프로토콜 공용 클라이언트 — **Neo4j 와 Memgraph 를 같은 코드로** 다룬다
    benchmark_graph_db()
        같은 적재/탐색 작업을 두 DB 에 돌려 지연을 비교
    build_cypher_qa_chain() / ask_cypher()
        LangChain `GraphCypherQAChain` — 자연어 질문을 **Cypher 로 번역** 해 그래프에 던진다
    detect_communities() / summarize_communities() / global_answer()
        Microsoft GraphRAG 의 핵심 아이디어(**커뮤니티 요약 → 전역 질문**)를
        우리 스택(LlamaIndex 그래프 + `.env` 공급자)으로 재현

LLM 호출 규약(rag.py 와 동일):
    - LLM 을 직접 만들지 않고 **주입** 받는다(`utils.get_llm()` 결과를 그대로 넘긴다).
    - 모든 응답은 `bootstrap.to_text()` 로 정규화한다.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .bootstrap import to_text

try:
    from neo4j import GraphDatabase
    _BOLT_AVAILABLE = True
except Exception:  # 드라이버 미설치
    GraphDatabase = None  # type: ignore
    _BOLT_AVAILABLE = False


# =============================================================================
# 1. 그래프 DB 선택 기준 — 무엇을 언제 고르나
# =============================================================================
#: 그래프 DB 별 선택 기준. 표로 출력하지는 않고(노트북이 산문으로 설명한다),
#: `print_other_db_snippets()` 가 각 DB 의 `pick_when` 을 머리말로 쓴다.
#: `hands_on` 은 이 노트북에서 **직접 실행** 하는지 여부다.
GRAPH_DB_MATRIX: List[Dict] = [
    {
        "name": "Neo4j",
        "query": "Cypher",
        "strength": "생태계·문서 최다, 5.11+ 네이티브 벡터 인덱스",
        "watch_out": "JVM 기반이라 메모리를 많이 쓴다, 커뮤니티판은 단일 DB",
        "pick_when": "표준을 따르고 싶을 때 · 벡터와 그래프를 한 DB 에 둘 때",
        "hands_on": "2·4장에서 실행",
    },
    {
        "name": "Memgraph",
        "query": "Cypher (Neo4j 호환)",
        "strength": "C++ 인메모리 — 쓰기/탐색 지연이 낮아 실시간 처리에 유리",
        "watch_out": "인메모리라 RAM 이 곧 용량 한계, APOC 대신 MAGE 를 쓴다",
        "pick_when": "그래프가 자주 갱신되는 실시간 RAG · 낮은 지연이 필요할 때",
        "hands_on": "6장에서 실행(같은 Cypher 를 그대로 재사용)",
    },
    {
        "name": "NebulaGraph",
        "query": "nGQL (+ openCypher 일부)",
        "strength": "샤딩 기반 분산 — 수십억 노드/엣지 규모까지 확장",
        "watch_out": "meta·storage·graph 3개 서비스를 띄워야 해 준비 비용이 크다",
        "pick_when": "단일 노드로 감당이 안 되는 초대규모 지식 그래프",
        "hands_on": "미실행(컨테이너 3개 필요) — 연결 스니펫만 소개",
    },
    {
        "name": "ArangoDB",
        "query": "AQL",
        "strength": "멀티 모델 — 문서·키값·그래프를 한 엔진에서 함께 다룬다",
        "watch_out": "Cypher 가 아니라 AQL 이라 기존 쿼리를 다시 써야 한다",
        "pick_when": "문서 기반 RAG 와 GraphRAG 를 한 저장소로 합치고 싶을 때",
        "hands_on": "미실행 — 연결 스니펫만 소개",
    },
]


# =============================================================================
# 2. Bolt 공용 클라이언트 — Neo4j 와 Memgraph 를 같은 코드로
# =============================================================================
class BoltGraph:
    """Bolt 프로토콜 그래프 DB 클라이언트(Neo4j·Memgraph 공용).

    Memgraph 는 Neo4j 와 **같은 Bolt 프로토콜 + 같은 Cypher** 를 쓰므로,
    드라이버도 쿼리도 바꾸지 않고 접속 주소만 바꾸면 된다 — 이 클래스가 그 사실을 보여 준다.

    미연결 시 예외를 던지지 않고 `connected=False` 로 두어 노트북이 건너뛸 수 있게 한다.
    """

    def __init__(self, uri: str, user: str = "", password: str = "",
                 label: str = "graph-db", quiet: bool = False):
        """드라이버를 만들고 연결을 점검한다(실패해도 예외를 던지지 않음).

        Args:
            uri: 예) `bolt://localhost:7687`(Neo4j) / `bolt://localhost:7688`(Memgraph).
            user: 사용자명. Memgraph 는 기본적으로 인증이 없어 빈 문자열을 쓴다.
            password: 비밀번호(화면에 출력하지 않는다).
            label: 출력에 쓸 표시 이름.
            quiet: True 면 성공/실패 메시지를 찍지 않는다(기동 대기 루프용).
        """
        self.uri, self.label = uri, label
        self.driver = None
        self.connected = False

        if not _BOLT_AVAILABLE:
            if not quiet:
                print("neo4j 드라이버가 없습니다 — uv pip install neo4j")
            return
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            self.connected = True
            if not quiet:
                print(f"{label} 연결 성공: {uri}")
        except Exception as e:
            if not quiet:
                print(f"{label} 연결 실패({uri}): {str(e)[:120]}")

    # ------------------------------------------------------------------ 기본
    def run(self, cypher: str, **params) -> List[Dict]:
        """Cypher 를 실행하고 결과를 dict 리스트로 돌려준다."""
        if not self.connected:
            return []
        with self.driver.session() as session:
            return [record.data() for record in session.run(cypher, **params)]

    def version(self) -> str:
        """서버 버전 문자열(두 DB 의 조회 방식이 달라 순서대로 시도한다)."""
        for cypher, key in [("SHOW VERSION;", None),                       # Memgraph
                            ("CALL dbms.components() YIELD versions "
                             "RETURN versions[0] AS v", "v")]:             # Neo4j
            try:
                rows = self.run(cypher)
                if rows:
                    return str(list(rows[0].values())[0] if key is None else rows[0][key])
            except Exception:
                continue
        return "unknown"

    def clear(self) -> None:
        """이 실습이 만든 노드만 지운다(반복 실행을 멱등하게).

        `MATCH (n) DETACH DELETE n` 로 전체를 지우지 않는 이유: Neo4j 에는 2장(문서 벡터)과
        4장(PropertyGraphIndex)의 데이터가 이미 들어 있다. 그래서 6장은 `:GDoc`/`:GTopic`
        이라는 **전용 라벨** 만 쓰고 그것만 정리한다 — 실무에서도 같은 DB 를 나눠 쓸 때
        이렇게 네임스페이스를 분리한다.
        """
        self.run("MATCH (n) WHERE n:GDoc OR n:GTopic DETACH DELETE n")

    def stats(self) -> Dict[str, int]:
        """이 실습 네임스페이스(`:GDoc`/`:GTopic`)의 노드/관계 개수를 센다."""
        nodes = self.run("MATCH (n) WHERE n:GDoc OR n:GTopic RETURN count(n) AS c")
        rels = self.run("MATCH (a)-[r]->(b) WHERE (a:GDoc OR a:GTopic) "
                        "AND (b:GDoc OR b:GTopic) RETURN count(r) AS c")
        return {"nodes": nodes[0]["c"] if nodes else 0,
                "rels": rels[0]["c"] if rels else 0}

    def close(self) -> None:
        """드라이버 연결을 닫는다."""
        if self.driver:
            self.driver.close()

    # -------------------------------------------------------------- 실습 적재
    #: 문서-태그 그래프를 만드는 Cypher. **Neo4j 와 Memgraph 에 글자 그대로 같이 쓴다.**
    #: (라벨만 6장 전용 `:GDoc`/`:GTopic` 이고, 문법은 2장에서 쓴 것과 동일하다)
    LOAD_CYPHER = """
        UNWIND $rows AS row
        MERGE (d:GDoc {doc_id: row.doc_id})
          SET d.title = row.title, d.category = row.category
        WITH d, row
        UNWIND row.tags AS tag
        MERGE (t:GTopic {name: tag})
        MERGE (d)-[:MENTIONS]->(t)
    """

    LINK_CYPHER = """
        UNWIND $links AS link
        MERGE (a:GTopic {name: link.source})
        MERGE (b:GTopic {name: link.target})
        MERGE (a)-[r:RELATES {label: link.relation}]->(b)
    """

    def load_document_graph(self, docs: Sequence, topic_links: Sequence[Tuple]) -> Dict[str, int]:
        """개인 문서(doc_prep.PersonalDoc)와 태그 관계를 그래프로 적재한다.

        Args:
            docs: `doc_prep.load_markdown_docs()` 결과.
            topic_links: `(source, relation, target)` 튜플 목록.

        Returns:
            적재 후 노드/관계 개수.
        """
        rows = [{"doc_id": d.doc_id, "title": d.title,
                 "category": d.category, "tags": list(d.tags)} for d in docs]
        links = [{"source": s, "relation": r, "target": t} for s, r, t in topic_links]
        self.run(self.LOAD_CYPHER, rows=rows)
        self.run(self.LINK_CYPHER, links=links)
        return self.stats()

    def two_hop_paths(self, start: str, limit: int = 5) -> List[Dict]:
        """태그 하나에서 출발하는 1~2홉 경로(두 DB 공통 Cypher).

        `size(relationships(path))` 를 쓰는 이유: Neo4j 의 `length(path)` 는 Memgraph 에서
        동작이 다를 수 있어, 양쪽에서 똑같이 도는 표현을 골랐다.
        """
        return self.run("""
            MATCH path = (a:GTopic {name: $start})-[:RELATES*1..2]-(b:GTopic)
            WHERE a <> b
            RETURN [x IN nodes(path) | x.name] AS names,
                   size(relationships(path)) AS hops
            ORDER BY hops
            LIMIT $limit
        """, start=start, limit=limit)


def benchmark_graph_db(store: "BoltGraph", docs: Sequence,
                       topic_links: Sequence[Tuple], rounds: int = 20) -> Optional[Dict]:
    """같은 적재/탐색 작업을 돌려 지연을 잰다(DB 간 감을 잡기 위한 용도).

    Args:
        store: 측정 대상 `BoltGraph`.
        docs: 적재할 문서.
        topic_links: 태그 관계.
        rounds: 탐색 쿼리 반복 횟수.

    Returns:
        `{"db","write_ms","read_ms","nodes","rels"}` 또는 미연결 시 None.

    Note:
        데이터가 수십 건 규모라 **절대 성능 비교가 아니다.** 컨테이너 상태·캐시·워밍업에
        따라 흔들리므로 "자릿수 감" 정도로만 읽어야 한다.
    """
    if not store.connected:
        return None

    store.clear()
    started = time.perf_counter()
    store.load_document_graph(docs, topic_links)
    write_ms = (time.perf_counter() - started) * 1000

    store.two_hop_paths("권한")  # 워밍업 — 첫 실행의 파싱/플랜 비용을 제외한다
    started = time.perf_counter()
    for _ in range(rounds):
        store.two_hop_paths("권한")
    read_ms = (time.perf_counter() - started) * 1000 / rounds

    stats = store.stats()
    return {"db": store.label, "write_ms": write_ms, "read_ms": read_ms, **stats}


def print_benchmark(results: Sequence[Optional[Dict]]) -> None:
    """benchmark_graph_db() 결과들을 표로 출력한다."""
    rows = [r for r in results if r]
    if not rows:
        print("측정 결과 없음 — 연결된 DB 가 없습니다")
        return
    print(f"{'DB':<12}{'적재(ms)':>12}{'2홉 탐색(ms)':>16}{'노드':>8}{'관계':>8}")
    print("-" * 56)
    for r in rows:
        print(f"{r['db']:<12}{r['write_ms']:>12.1f}{r['read_ms']:>16.2f}"
              f"{r['nodes']:>8}{r['rels']:>8}")
    print("\n※ 데이터가 작아 절대 비교가 아니다 — 워밍업·캐시에 따라 흔들린다.")


#: 실습에서 띄우지 않는 DB 의 연결 코드(개념만 보여 준다).
OTHER_DB_SNIPPETS: Dict[str, str] = {
    "NebulaGraph": """# uv pip install nebula3-python
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

pool = ConnectionPool()
pool.init([("127.0.0.1", 9669)], Config())      # graphd 주소
session = pool.get_session("root", "nebula")
session.execute("USE personal_docs")
# nGQL — Cypher 가 아니다(GO / FETCH / LOOKUP 문법)
print(session.execute("GO 2 STEPS FROM '권한' OVER RELATES YIELD dst(edge)"))""",
    "ArangoDB": """# uv pip install python-arango
from arango import ArangoClient

db = ArangoClient(hosts="http://localhost:8529").db("_system", "root", "password")
graph = db.create_graph("docs")                  # 문서 컬렉션과 같은 DB 안에 그래프를 둔다
# AQL — 문서 조회와 그래프 순회를 한 쿼리로 섞을 수 있는 것이 이 DB 의 강점
db.aql.execute('''
    FOR t IN 1..2 OUTBOUND 'topics/권한' GRAPH 'docs'
        RETURN t.name
''')""",
}


def print_other_db_snippets() -> None:
    """실습에서 띄우지 않는 DB 들의 연결 코드를 출력한다."""
    for name, code in OTHER_DB_SNIPPETS.items():
        entry = next(d for d in GRAPH_DB_MATRIX if d["name"] == name)
        print(f"===== {name} — {entry['pick_when']} =====")
        print(code)
        print()


# =============================================================================
# 3. Text2Cypher — 자연어 질문을 Cypher 로 번역 (LangChain GraphCypherQAChain)
# =============================================================================
#: 2장에서 만든 깨끗한 스키마만 LLM 에게 보여 준다.
#: 4장의 PropertyGraphIndex 가 같은 DB 에 `entity`/`Chunk`/`__Node__` 를 잔뜩 만들어 두어,
#: 이것을 걸러 주지 않으면 LLM 이 엉뚱한 라벨로 Cypher 를 쓴다.
CLEAN_SCHEMA_TYPES = ["Document", "Topic", "MENTIONS",
                      "PART_OF", "CONSTRAINS", "DEFINES", "NEEDS", "REPORTS"]

#: Cypher 생성 프롬프트 — 임베딩 반환 금지 등 실무 제약을 걸어 둔다.
#: ⚠️ `PromptTemplate` 이 `{...}` 를 변수로 읽으므로, 예시 안의 중괄호는 `{{ }}` 로 이스케이프한다.
#: 실제 변수는 `{schema}` 와 `{question}` 둘뿐이다.
CYPHER_PROMPT_TEMPLATE = """당신은 Neo4j Cypher 전문가입니다.
아래 스키마만 사용해 질문에 답하는 **읽기 전용** Cypher 를 한 개 작성하세요.

규칙:
- 스키마에 없는 라벨/관계/속성을 지어내지 마세요.
- 질문에 나온 **값(태그 이름 등)은 원문 그대로** 쓰세요. 한국어를 영어로 번역하면 매칭에 실패합니다.
  예: '프로젝트' 태그 → `{{name: '프로젝트'}}` (O) / `{{name: 'project'}}` (X)
- CREATE / MERGE / DELETE / SET 은 절대 쓰지 마세요.
- `embedding` 속성은 값이 매우 크므로 절대 RETURN 하지 마세요.
- 사람이 읽을 수 있는 값(doc_id, title, name 등)을 RETURN 하세요.
- **가변 길이 경로(N홉) 문법을 정확히 지키세요.** 관계 타입은 콜론 하나로 `|` 로만 잇고,
  `*1..2` 는 **대괄호 안, 관계 목록 바로 뒤** 에 붙입니다.
      O:  MATCH (a)-[:PART_OF|CONSTRAINS*1..2]->(b)
      X:  MATCH (a)-[:PART_OF|:CONSTRAINS*1..2]->(b)     <- 콜론을 반복하면 문법 오류
      X:  MATCH (a)-[:PART_OF|CONSTRAINS]*1..2->(b)      <- *는 대괄호 밖에 올 수 없다
      X:  MATCH (a)-[r:PART_OF|CONSTRAINS*1..2]->(b)     <- 가변 길이에는 변수(r)를 붙이지 않는다
- 설명·마크다운·머리말 없이 Cypher 문만 출력하세요. **출력의 첫 단어는 반드시 `MATCH` 입니다.**

스키마:
{schema}

질문: {question}
Cypher:"""

#: 답변 생성 프롬프트. 기본값이 영어라 그대로 두면 한국어 질문에도 영어로 답하거나
#: 근거가 있는데도 "I don't know" 를 내는 일이 있어, 한국어로 명시해 준다.
QA_PROMPT_TEMPLATE = """당신은 그래프 DB 조회 결과를 사람이 읽을 문장으로 옮기는 조수입니다.

아래 '조회 결과'는 질문에 답하기 위해 실제 그래프에서 가져온 **정확한 정보** 입니다.
이 결과만 근거로 한국어로 간결하게 답하세요.

- 결과가 비어 있을 때만 "조회 결과가 없습니다" 라고 답하세요.
- 결과가 있으면 그 값을 그대로 활용해 답하세요. 임의로 부정하지 마세요.
- 결과에 없는 내용을 지어내지 마세요.

조회 결과: {context}

질문: {question}
답변:"""


def build_cypher_qa_chain(llm, uri: str, user: str, password: str,
                          include_types: Optional[List[str]] = None,
                          top_k: int = 5, verbose: bool = True):
    """자연어 → Cypher → 그래프 조회 → 답변 체인을 만든다.

    Vector RAG 가 "비슷한 텍스트"를 찾는다면, 이 체인은 질문을 **쿼리 언어로 번역** 해
    그래프에 직접 묻는다. 집계("몇 개냐")나 정확한 관계 조회에 강하다.

    Args:
        llm: `utils.get_llm()` 결과(공급자 무관).
        uri/user/password: Neo4j 접속 정보.
        include_types: LLM 에게 보여 줄 라벨/관계 화이트리스트.
            None 이면 `CLEAN_SCHEMA_TYPES` 를 쓴다.
        top_k: 그래프 조회 결과 중 LLM 에 넘길 최대 행 수.
        verbose: 생성된 Cypher 를 화면에 출력할지.

    Returns:
        `(chain, graph)` 튜플. 실패 시 `(None, None)`.
    """
    try:
        from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
    except ImportError:
        print("langchain-neo4j 가 필요합니다 — uv pip install langchain-neo4j")
        return None, None

    from langchain_core.prompts import PromptTemplate

    try:
        graph = Neo4jGraph(url=uri, username=user, password=password,
                           refresh_schema=True)
    except Exception as e:
        print(f"Neo4jGraph 연결 실패: {str(e)[:140]}")
        return None, None

    chain = GraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        cypher_prompt=PromptTemplate.from_template(CYPHER_PROMPT_TEMPLATE),
        qa_prompt=PromptTemplate.from_template(QA_PROMPT_TEMPLATE),
        include_types=include_types if include_types is not None else CLEAN_SCHEMA_TYPES,
        validate_cypher=True,        # 관계 방향 오류를 자동 교정한다
        return_intermediate_steps=True,
        top_k=top_k,
        verbose=verbose,
        # 생성된 쿼리를 그대로 실행하므로 LangChain 이 명시적 동의를 요구한다.
        # 읽기 전용 프롬프트 + 화이트리스트 스키마로 위험을 줄이지만, 운영에서는
        # 읽기 전용 계정으로 접속하는 것이 정석이다.
        allow_dangerous_requests=True,
    )
    return chain, graph


def print_chain_schema(chain, limit: int = 900) -> None:
    """체인이 **실제로 LLM 에게 보여 주는** 스키마를 출력한다.

    `graph.schema` 는 DB 전체 스키마(4장이 만든 `entity`/`Chunk`/`__Node__` 포함)지만,
    `chain.graph_schema` 는 `include_types` 화이트리스트가 적용된 뒤의 것이다.
    Text2Cypher 정확도는 **이 문자열의 품질** 에 거의 그대로 비례한다.
    """
    if chain is None:
        print("체인이 없습니다")
        return
    schema = chain.graph_schema
    print(schema[:limit] + ("…" if len(schema) > limit else ""))


def ask_cypher(chain, question: str, verbose: bool = True) -> Dict:
    """Text2Cypher 체인에 질문하고 생성된 Cypher·조회 결과·답변을 보여 준다.

    Returns:
        `{"question","cypher","context","answer"}`. 실패 시 `answer` 에 오류 요약.
    """
    if chain is None:
        return {"question": question, "cypher": "", "context": [], "answer": "체인 없음"}

    print(f"\n질문: {question}")
    try:
        out = chain.invoke({"query": question})
    except Exception as e:
        print(f"  실패: {str(e)[:200]}")
        return {"question": question, "cypher": "", "context": [],
                "answer": f"실패: {str(e)[:200]}"}

    cypher, context = "", []
    for step in out.get("intermediate_steps", []):
        cypher = step.get("query", cypher)
        context = step.get("context", context)

    if verbose:
        print(f"  생성된 Cypher: {' '.join(str(cypher).split())}")
        print(f"  조회 결과 {len(context)}행: {str(context)[:200]}")
        print(f"  답변: {out.get('result', '')}")
    return {"question": question, "cypher": cypher,
            "context": context, "answer": out.get("result", "")}


# =============================================================================
# 4. Microsoft GraphRAG 스타일 — 커뮤니티 요약과 전역(Global) 질문
# =============================================================================
@dataclass
class Community:
    """지식 그래프에서 찾아낸 하나의 주제 클러스터."""

    community_id: int
    members: List[str] = field(default_factory=list)      # 엔티티 이름들
    triplets: List[Tuple[str, str, str]] = field(default_factory=list)
    summary: str = ""                                     # LLM 요약(나중에 채운다)

    def as_text(self, limit: int = 30) -> str:
        """LLM 에 넘길 관계 목록 문자열."""
        return "\n".join(f"({s}) -[{r}]-> ({o})" for s, r, o in self.triplets[:limit])


def triplets_from_graph_index(index) -> List[Tuple[str, str, str]]:
    """LlamaIndex PropertyGraphIndex 에서 (주어, 관계, 목적어) 문자열 트리플을 뽑는다."""
    from . import llama_rag

    return [(s.name, r.label, o.name) for s, r, o in llama_rag.get_triplets(index)]


def detect_communities(triplets: Sequence[Tuple[str, str, str]],
                       min_size: int = 2, seed: int = 42) -> List[Community]:
    """트리플 집합을 **커뮤니티(주제 클러스터)** 로 나눈다.

    Microsoft GraphRAG 는 Leiden 알고리즘으로 그래프를 계층적 커뮤니티로 나눈 뒤
    각 커뮤니티를 요약해 둔다. 여기서는 이미 설치된 networkx 의 **Louvain** 으로
    같은 아이디어를 한 단계(비계층)만 재현한다.

    Args:
        triplets: (주어, 관계, 목적어) 목록.
        min_size: 이보다 작은 커뮤니티는 버린다(잡음 제거).
        seed: Louvain 은 난수를 쓰므로 재현성을 위해 고정한다.

    Returns:
        멤버 수가 많은 순으로 정렬된 Community 목록.
    """
    try:
        import networkx as nx
    except ImportError:
        print("networkx 가 필요합니다 — uv pip install networkx")
        return []

    graph = nx.Graph()  # 커뮤니티 탐지는 방향을 보지 않는다
    for subject, relation, obj in triplets:
        graph.add_edge(subject, obj, label=relation)
    if graph.number_of_nodes() == 0:
        return []

    groups = nx.community.louvain_communities(graph, seed=seed)

    communities: List[Community] = []
    for i, members in enumerate(sorted(groups, key=len, reverse=True)):
        if len(members) < min_size:
            continue
        member_set = set(members)
        inside = [t for t in triplets if t[0] in member_set and t[2] in member_set]
        communities.append(Community(community_id=i,
                                     members=sorted(member_set),
                                     triplets=inside))
    return communities


def print_communities(communities: Sequence[Community], limit: int = 5,
                      members_shown: int = 8) -> None:
    """탐지된 커뮤니티를 요약해 출력한다."""
    print(f"커뮤니티 {len(communities)}개 탐지 (상위 {min(limit, len(communities))}개)\n")
    for community in communities[:limit]:
        members = ", ".join(community.members[:members_shown])
        more = f" 외 {len(community.members) - members_shown}개" if len(
            community.members) > members_shown else ""
        print(f"[C{community.community_id}] 엔티티 {len(community.members)}개 / "
              f"관계 {len(community.triplets)}개")
        print(f"   {members}{more}")
        if community.summary:
            print(f"   요약: {community.summary}")
        print()


_SUMMARY_PROMPT = """다음은 문서에서 자동 추출한 지식 그래프의 한 클러스터입니다.
이 클러스터가 **무엇에 관한 것인지** 한국어 두 문장 이내로 요약하세요.
관계 목록에 없는 내용은 지어내지 마세요.

관계 목록:
{relations}

요약:"""


def summarize_communities(llm, communities: Sequence[Community],
                          limit: int = 5, verbose: bool = True) -> List[Community]:
    """각 커뮤니티를 LLM 으로 요약한다(Microsoft GraphRAG 의 community report 단계).

    Args:
        llm: 주입된 LangChain 모델.
        communities: `detect_communities()` 결과.
        limit: 요약할 커뮤니티 수(LLM 호출이 그만큼 발생한다).
        verbose: 진행 상황 출력.

    Returns:
        `summary` 가 채워진 Community 목록(입력 객체를 그대로 갱신한다).
    """
    from langchain_core.messages import HumanMessage

    targets = list(communities)[:limit]
    for community in targets:
        if llm is None:
            community.summary = f"(LLM 없음) 엔티티 {len(community.members)}개 클러스터"
            continue
        prompt = _SUMMARY_PROMPT.format(relations=community.as_text())
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            community.summary = to_text(response.content).strip()
        except Exception as e:
            community.summary = f"(요약 실패: {str(e)[:80]})"
        if verbose:
            print(f"[C{community.community_id}] {community.summary[:120]}")
    return targets


_GLOBAL_PROMPT = """아래는 문서 모음 전체에서 뽑아낸 **주제 클러스터별 요약** 입니다.
이 요약들만 근거로 질문에 한국어로 답하세요. 없는 내용은 지어내지 마세요.

{summaries}

질문: {question}
답변:"""


def global_answer(llm, communities: Sequence[Community], question: str,
                  verbose: bool = True) -> str:
    """커뮤니티 요약들을 근거로 **전역(Global) 질문** 에 답한다.

    "이 문서 모음의 주요 주제가 뭐야?" 같은 질문은 특정 청크를 찾는 문제가 아니라
    **전체를 조망** 해야 하는 문제라, 벡터 검색(top-k 청크)으로는 원리적으로 약하다.
    커뮤니티 요약을 미리 만들어 두면 그 요약들만 모아 답할 수 있다 —
    이것이 Microsoft GraphRAG 의 Global Search 아이디어다.

    Args:
        llm: 주입된 LangChain 모델.
        communities: `summarize_communities()` 로 요약이 채워진 커뮤니티.
        question: 전역 질문.
        verbose: 답변 출력 여부.

    Returns:
        답변 문자열.
    """
    from langchain_core.messages import HumanMessage

    summaries = "\n\n".join(
        f"[클러스터 {c.community_id}] (엔티티 {len(c.members)}개)\n{c.summary}"
        for c in communities if c.summary)
    if not summaries:
        return "요약된 커뮤니티가 없습니다 — summarize_communities() 를 먼저 실행하세요."

    if llm is None:
        return f"(LLM 없음) 커뮤니티 {len(communities)}개 요약 기반 답변"

    response = llm.invoke([HumanMessage(
        content=_GLOBAL_PROMPT.format(summaries=summaries, question=question))])
    answer = to_text(response.content).strip()
    if verbose:
        print(f"\n질문(전역): {question}\n답변:\n{answer}")
    return answer
