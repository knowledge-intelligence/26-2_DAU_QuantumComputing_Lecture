# M04_3_graph_rag.ipynb — 환경 설치·구축·실행 가이드

> 9~11주차 모듈 3(지식) **3/4편**: **지식 그래프**(`KnowledgeGraph`) →
> **Neo4j 벡터 인덱스**(`Neo4jVectorStore`) → **LlamaIndex RAG · GraphRAG**(`PropertyGraphIndex`)
> → **Hybrid RAG**(Vector + Graph) → **다른 그래프 DB**(Memgraph) · **Text2Cypher** ·
> **커뮤니티 요약(Microsoft GraphRAG 식)**. LLM 은 `.env` 의 `LLM_PROVIDER` 를 그대로 사용합니다.

## 0. 전제

- Windows 11 + **CMD(`cmd.exe`)** + Python **3.11**(`uv` 관리)
- 공통 1회 준비는 [README.md](README.md) 를 먼저 따라 하세요.
- [2편 M04_2](M04_2_vector_rag.md) 와 **같은 개인 문서** 를 씁니다(노트북이 자동 생성).
- **Neo4j 는 Docker Desktop for Windows** 로 띄웁니다. 없으면 1·3장만 실행되고 2·4·7장은 건너뜁니다.
- **6장은 Memgraph 컨테이너를 추가로 띄웁니다**(포트 7688). 없으면 Neo4j 쪽 결과만 나옵니다.

## 1. 이 노트북이 필요로 하는 것

| 구분 | 내용 |
|---|---|
| Python 패키지 | `chromadb`, `sentence-transformers`, `numpy`, `neo4j`, `llama-index-core`, `llama-index-graph-stores-neo4j`, **`langchain-neo4j`**(7장), **`networkx`**(8장, llama-index 의존성으로 이미 설치됨) |
| 임베딩 | `paraphrase-multilingual-MiniLM-L12-v2` (오프라인) |
| 그래프 DB | **Neo4j 5.11+ + APOC 플러그인**(2·4·7장) · **Memgraph**(6장) — 둘 다 Docker |
| LLM | `.env` 의 `LLM_PROVIDER`. **4·7·8장은 LLM 이 트리플/Cypher/요약을 만들므로 모델 품질이 결과를 좌우** |

구현 위치:

| 모듈 | 역할 |
|---|---|
| [`agentic_lib/rag.py`](../agentic_lib/rag.py) | `KnowledgeGraph`(순수 Python), `Neo4jVectorStore`(벡터 인덱스+그래프 확장), `HybridRAG` |
| [`agentic_lib/llama_rag.py`](../agentic_lib/llama_rag.py) | LlamaIndex 어댑터(`LangChainLLM`·`InjectedEmbedding`), `build_vector_index()`, `build_property_graph_index()` |
| [`agentic_lib/graph_stack.py`](../agentic_lib/graph_stack.py) | **6~8장**: `BoltGraph`(Neo4j·Memgraph 공용), `benchmark_graph_db()`, `build_cypher_qa_chain()`/`ask_cypher()`, `detect_communities()`/`summarize_communities()`/`global_answer()` |
| [`agentic_lib/doc_prep.py`](../agentic_lib/doc_prep.py) | 2편과 공유하는 개인 문서 로딩·청킹 |

## 2. 추가 패키지 설치 (CMD)

```bat
REM 벡터 검색 + 임베딩 + Neo4j 드라이버
uv pip install chromadb sentence-transformers numpy neo4j

REM LlamaIndex (GraphRAG)
uv pip install llama-index-core llama-index-graph-stores-neo4j

REM 7장 Text2Cypher (LangChain GraphCypherQAChain)
uv pip install langchain-neo4j
```

> ⚠️ `llama-index-graph-stores-neo4j` 는 `neo4j` 드라이버를 **5.x 로 고정** 합니다
> (6.x 가 설치되어 있으면 5.x 로 내려갑니다). Neo4j 5 서버와는 정상 동작하므로 문제없습니다.

## 3. Neo4j 준비 — **APOC 플러그인 필수**

LlamaIndex 의 `Neo4jPropertyGraphStore` 는 스키마 조회에 `apoc.meta.data` 를 호출합니다.
APOC 없이 띄우면 4장에서 `ProcedureNotFound` 오류가 납니다.

```bat
REM CMD 한 줄 — 큰따옴표 이스케이프에 주의(CMD 가 따옴표를 벗겨내지 않도록)
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password -e "NEO4J_PLUGINS=[\"apoc\"]" neo4j:5-community

REM 상태 확인
docker ps --filter name=neo4j

REM APOC 이 실제로 올라갔는지 확인
docker inspect neo4j --format "{{range .Config.Env}}{{println .}}{{end}}" | findstr PLUGINS
REM  → NEO4J_PLUGINS=["apoc"]   (대괄호 안에 따옴표가 있어야 정상)
```

브라우저 콘솔: http://localhost:7474 (neo4j / password)

`notebooks/.env` (미설정 시 아래 기본값 사용):

```ini
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

> 노트북의 기동 셀은 **이미 떠 있으면 재사용** 하고, 없으면 위 명령을 `utils.run_cmd()` 로
> 실행한 뒤 Bolt 포트가 열릴 때까지 최대 90초 기다립니다.

## 3-1. Memgraph 준비 (6장) — 포트 7688

Memgraph 는 **Neo4j 와 같은 Bolt 프로토콜 + 같은 Cypher** 를 씁니다. 6장은 같은 드라이버·같은
쿼리로 두 DB 를 모두 다뤄 그 사실을 확인합니다. Neo4j 의 7687 과 겹치지 않게 **7688** 로 매핑합니다.

```bat
REM 인증 없음이 기본값이라 사용자/비밀번호는 빈 문자열로 접속한다
docker run -d --name memgraph -p 7688:7687 memgraph/memgraph:latest

REM 확인
docker ps --filter name=memgraph
```

`notebooks/.env` (미설정 시 기본값 사용):

```ini
MEMGRAPH_URI=bolt://localhost:7688
```

> 6장은 `:GDoc`/`:GTopic` **전용 라벨** 만 쓰고 그것만 지웁니다 — 같은 Neo4j 안의
> 2·4장 데이터(문서 벡터·PropertyGraphIndex)를 건드리지 않기 위해서입니다.
>
> ⚠️ 컨테이너 로그에 `vm.max_map_count is too low` 경고가 보일 수 있습니다.
> 대규모 데이터가 아니면 실습에는 영향이 없습니다.

## 4. 실행 순서

1. Jupyter 에서 커널을 **`Agentic AI Tutorial (uv)`** 로 선택
2. 위에서부터 **순서대로** 실행
   - 0장 setup + 개인 문서 로딩
   - 1장 **지식 그래프 원리** — 태그를 엔티티로 삼아 그래프 구성, 방향(direction)·multi-hop 탐색
   - 2장 **Neo4j 벡터 인덱스** — 스키마/인덱스 → 적재 → Cypher 시맨틱 검색 →
     그래프 확장 검색 → 집계·경로·공통이웃 Cypher
   - 3장 **LlamaIndex RAG** — 공급자 무관 어댑터 → `VectorStoreIndex`
   - 4장 **GraphRAG** — `PropertyGraphIndex` 로 트리플 자동 추출 → Neo4j 저장 → Vector 와 비교 →
     **4-1절** Louvain 커뮤니티 요약 → **전역(Global) 질문**(Microsoft GraphRAG 아이디어)
   - 5장 **Hybrid RAG** — 벡터 + 그래프 컨텍스트 결합
   - 6장 **그래프 DB 갈아타기** — Memgraph 기동 → 같은 코드로 Neo4j·Memgraph 적재/탐색 →
     NebulaGraph·ArangoDB 연결 스니펫
   - 7장 **Text2Cypher** — `GraphCypherQAChain` 으로 자연어 질문을 Cypher 로 번역(집계 질문에 강함)
3. ⏱️ **4장은 오래 걸립니다** — 청크마다 LLM 을 호출하므로 문서 3개에 **1~2분**,
   4-1절 커뮤니티 요약에 **30초~1분** 이 더 붙습니다. 7장도 LLM 을 여러 번 부릅니다.

> 💡 실습 중 `agentic_lib` 를 직접 고쳤다면 **커널을 재시작하고 위에서부터 다시 실행**하세요.
> 파이썬은 이미 import 한 모듈을 캐시하므로, 재시작 없이는 커널이 옛 코드를 계속 씁니다.

## 5. 자주 겪는 문제

| 증상 | 원인/해결 |
|---|---|
| `there is no procedure with the name apoc.meta.data` | **APOC 미설치**. 3절 명령으로 컨테이너를 다시 만드세요(`docker rm -f neo4j` 후 재실행) |
| `docker inspect` 결과가 `NEO4J_PLUGINS=[apoc]` | CMD 가 따옴표를 벗겼습니다. `-e "NEO4J_PLUGINS=[\"apoc\"]"` 형태로 정확히 입력 |
| `asyncio.run() cannot be called from a running event loop` | 라이브러리가 `nest_asyncio` 를 적용합니다. 뜬다면 `uv pip install nest-asyncio` |
| `Neo4j 연결 실패` | Docker 미실행/기동 중. `docker ps --filter name=neo4j` 확인 후 셀 재실행 |
| Neo4j 차원 불일치 오류 | 인덱스 차원 ≠ 임베더 차원. 노트북이 `drop_vector_index()` 후 재생성합니다 |
| 추출된 트리플이 이상함(`Subject → Object`) | 작은 모델이 추출 프롬프트의 예시를 베낀 것. 라이브러리가 걸러서 표시하고 개수를 알려 줍니다 |
| GraphRAG 답변 품질이 낮음 | 그래프 품질 = 추출 LLM 품질. 더 큰 모델(`google`, `ollama` qwen3:8b)로 바꿔 보세요 |
| `User Safety: safe` 같은 무의미한 답변 | `openrouter/free` 라우팅 결과. 모델이 고정된 공급자로 바꾸세요 |
| 4장이 너무 느림 | 정상입니다. `documents[:3]` 을 `[:2]` 로 줄이거나 `max_triplets_per_chunk` 를 낮추세요 |
| 라이브러리를 고쳤는데 반영이 안 됨 (`unexpected keyword argument` 등) | 커널이 옛 모듈을 캐시한 것. **커널 재시작 후 위에서부터 다시 실행**하세요 |
| 6장 `Memgraph 연결 실패` | 컨테이너 미기동 또는 포트 충돌. `docker ps --filter name=memgraph` 확인, 7688 이 비어 있는지 점검 |
| 6장 Memgraph 인증 오류 | Memgraph 는 **인증 없음**이 기본입니다. 사용자/비밀번호를 빈 문자열로 두세요 |
| 7장 `langchain-neo4j 가 필요합니다` | `uv pip install langchain-neo4j` (노트북 셀이 자동 설치도 시도합니다) |
| 7장에서 엉뚱한 라벨로 Cypher 생성 | 스키마에 4장 잡음(`entity`/`Chunk`/`__Node__`)이 섞인 것. `include_types` 화이트리스트가 적용됐는지 `print_chain_schema()` 로 확인 |
| 7장 `SyntaxError ... colon in the separation of alternative relationship types` | LLM 이 `[:A\|:B*1..2]` 를 생성. Neo4j 5 는 가변 길이와 함께 쓴 `\|:` 를 거부한다 → 콜론은 한 번만(`[:A\|B*1..2]`). 라이브러리 프롬프트에 규칙이 들어 있다 |
| 7장 조회 결과가 0건인데 데이터는 있음 | LLM 이 한국어 값을 영어로 번역(`{name:'project'}`)한 것. 프롬프트의 "값은 원문 그대로" 규칙이 적용됐는지 확인 |
| 7장 답변이 `I don't know` / 영어 | 기본 QA 프롬프트가 영어라서 생기는 현상. 라이브러리가 한국어 `QA_PROMPT_TEMPLATE` 로 교체해 둔다 |
| 프롬프트 수정 후 `Input to PromptTemplate is missing variables` | 예시 안의 `{...}` 를 LangChain 이 변수로 읽은 것. `{{ }}` 로 이스케이프한다 |
| 7장 `allow_dangerous_requests` 경고 | LLM 생성 쿼리를 실행하므로 LangChain 이 명시적 동의를 요구합니다. 운영에서는 **읽기 전용 계정**을 쓰세요 |
| 4-1절 커뮤니티가 1개뿐 | 추출된 그래프가 빈약한 것(추출 LLM 품질). 더 큰 모델로 4장을 다시 돌리거나 `documents[:3]` 을 늘리세요 |

## 6. 명령 요약 (복붙용)

```bat
REM 패키지
uv pip install chromadb sentence-transformers numpy neo4j llama-index-core llama-index-graph-stores-neo4j langchain-neo4j

REM Neo4j (APOC 포함) — 한 줄
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password -e "NEO4J_PLUGINS=[\"apoc\"]" neo4j:5-community

REM Memgraph (6장) — 인증 없음, 7688 로 매핑
docker run -d --name memgraph -p 7688:7687 memgraph/memgraph:latest

REM .env 준비 (notebooks/ 에서 1회)
cd notebooks && copy .env.example .env

uv run jupyter notebook

REM 실습 후 정리
docker stop neo4j memgraph       REM 중지(데이터 유지)
docker rm -f neo4j memgraph      REM 삭제(데이터도 삭제)
```
