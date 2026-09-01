"""
vector_stores — ChromaDB · FAISS · Qdrant 통일 인터페이스
=========================================================

같은 청크·같은 임베딩을 **세 가지 벡터 데이터베이스** 에 넣고 똑같이 검색해 보는
실습용 어댑터입니다. LLM 을 `utils.get_llm()` 으로, 임베딩을 `embeddings.get_embedder()` 로
통일한 것과 같은 규약을 벡터 DB 에도 적용합니다.

    store = get_vector_store("faiss", embedder)
    store.add(chunks)                                  # doc_prep.Chunk 목록
    hits = store.search("질문", k=3, where={"category": "회의규정"})

세 DB 의 성격:
    ChromaDB  임베딩 함수 내장, 파일 영속화가 쉬움. 소규모 프로토타입의 기본값.
    FAISS     Meta 의 벡터 인덱스 라이브러리. DB 가 아니라 **인덱스** 라서 가장 빠르지만
              메타데이터 저장·필터링은 애플리케이션이 직접 해야 한다(여기서는 후처리 필터).
    Qdrant    Rust 기반 벡터 DB. 메타데이터 필터가 인덱스 단계에 통합되어 있고
              `:memory:` 모드로 서버 없이도 동일 API 를 쓸 수 있다.

설계 규약:
    - 임베딩은 **바깥에서 주입** 한다(세 DB 가 완전히 같은 벡터를 쓰도록).
    - `search()` 는 항상 `[{"id","text","score","metadata"}]` 를 돌려준다.
      score 는 **코사인 유사도(1에 가까울수록 유사)** 로 통일한다.
    - 설치되지 않은 백엔드는 예외 대신 `available()` 로 미리 걸러낸다.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


def _import_ok(module: str) -> bool:
    """모듈 import 가능 여부(설치 확인용)."""
    try:
        __import__(module)
        return True
    except Exception:
        return False


#: 백엔드 이름 → (필요 패키지, 설치 명령, 한 줄 설명)
STORE_REGISTRY: Dict[str, Dict[str, str]] = {
    "chroma": {"module": "chromadb", "install": "uv pip install chromadb",
               "note": "임베딩 함수 내장 · 파일 영속화 쉬움 · 프로토타입 기본값"},
    "faiss": {"module": "faiss", "install": "uv pip install faiss-cpu",
              "note": "DB 가 아닌 인덱스 · 가장 빠름 · 메타데이터는 직접 관리"},
    "qdrant": {"module": "qdrant_client", "install": "uv pip install qdrant-client",
               "note": "메타데이터 필터가 인덱스에 통합 · :memory: 모드 지원"},
}


def available_stores(verbose: bool = True) -> Dict[str, bool]:
    """설치되어 바로 쓸 수 있는 벡터 DB 백엔드를 점검한다."""
    result = {name: _import_ok(spec["module"]) for name, spec in STORE_REGISTRY.items()}
    if verbose:
        print(f"{'백엔드':<10} {'사용가능':<8} {'설치 명령':<34} 특징")
        print("-" * 104)
        for name, spec in STORE_REGISTRY.items():
            print(f"{name:<10} {'✅' if result[name] else '❌':<8} "
                  f"{spec['install']:<34} {spec['note']}")
    return result


@dataclass
class BaseVectorStore:
    """세 백엔드가 공유하는 인터페이스.

    Attributes:
        name: 백엔드 이름(chroma/faiss/qdrant).
        embedder: agentic_lib.embeddings 의 임베더(바깥에서 주입).
        collection: 컬렉션(인덱스) 이름.
    """

    name: str
    embedder: object
    collection: str = "personal_docs"

    def __post_init__(self):
        """백엔드 초기화 훅. 베이스에 정의해 둬야 dataclass 가 생성한 __init__ 이
        이를 호출하고, 실제로는 하위 클래스의 구현이 실행된다."""

    def add(self, chunks: Sequence) -> float:
        """청크 목록을 적재하고 소요 시간(초)을 돌려준다."""
        raise NotImplementedError

    def search(self, query: str, k: int = 3,
               where: Optional[Dict[str, object]] = None) -> List[Dict]:
        """질의와 가장 가까운 청크 k개를 돌려준다.

        Args:
            query: 자연어 질의.
            k: 반환 개수.
            where: 메타데이터 완전일치 필터(예: {"category": "회의록"}).

        Returns:
            [{"id","text","score","metadata"}] — score 는 코사인 유사도.
        """
        raise NotImplementedError

    def count(self) -> int:
        """적재된 청크 수."""
        raise NotImplementedError

    # --- 공통 헬퍼 ---------------------------------------------------------
    def _encode_docs(self, texts: Sequence[str]) -> np.ndarray:
        """문서용 임베딩(정규화되어 있으므로 코사인 = 내적)."""
        return self.embedder.encode(list(texts), kind="passage")

    def _encode_query(self, text: str) -> np.ndarray:
        """질의용 임베딩(비대칭 모델을 위해 kind 를 구분한다)."""
        return self.embedder.encode([text], kind="query")[0]


class ChromaStore(BaseVectorStore):
    """ChromaDB 어댑터 — 인메모리 클라이언트 + 사전 계산 임베딩 사용."""

    def __post_init__(self):
        """컬렉션을 새로 만든다(같은 이름이 있으면 지우고 재생성)."""
        import chromadb

        self._client = chromadb.Client()
        try:
            self._client.delete_collection(self.collection)
        except Exception:
            pass  # 없으면 그만
        # 임베딩 함수를 지정하지 않고, 우리가 만든 벡터를 직접 넣는다
        # → 세 백엔드가 완전히 동일한 벡터를 쓰게 되어 비교가 공정해진다
        self._coll = self._client.create_collection(
            name=self.collection, metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def add(self, chunks: Sequence) -> float:
        started = time.perf_counter()
        vectors = self._encode_docs([c.text for c in chunks])
        self._coll.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[dict(c.metadata) for c in chunks],
            embeddings=vectors.tolist(),
        )
        return time.perf_counter() - started

    def search(self, query: str, k: int = 3,
               where: Optional[Dict[str, object]] = None) -> List[Dict]:
        result = self._coll.query(
            query_embeddings=[self._encode_query(query).tolist()],
            n_results=k, where=where or None,
        )
        hits = []
        for i in range(len(result["ids"][0])):
            # Chroma 는 '거리'를 준다. cosine space 이므로 유사도 = 1 - 거리
            hits.append({
                "id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "score": 1.0 - float(result["distances"][0][i]),
                "metadata": result["metadatas"][0][i],
            })
        return hits

    def count(self) -> int:
        return self._coll.count()


class FaissStore(BaseVectorStore):
    """FAISS 어댑터 — 내적(IP) 인덱스 + 파이썬 쪽 메타데이터 관리.

    FAISS 는 순수 벡터 인덱스라 메타데이터 개념이 없다. 그래서 필터는
    **검색 후 걸러내는 방식(post-filter)** 으로 구현한다. 필터가 강할수록
    k 를 넉넉히 뽑아야 결과가 비지 않는다 — 이것이 FAISS 의 실무적 약점이다.
    """

    def __post_init__(self):
        """빈 인덱스와 메타데이터 저장소를 준비한다."""
        self._index = None
        self._payloads: List[Dict] = []

    def add(self, chunks: Sequence) -> float:
        import faiss

        started = time.perf_counter()
        vectors = self._encode_docs([c.text for c in chunks]).astype(np.float32)
        # 벡터가 L2 정규화되어 있으므로 내적(IndexFlatIP) = 코사인 유사도
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        self._payloads = [{"id": c.chunk_id, "text": c.text,
                           "metadata": dict(c.metadata)} for c in chunks]
        return time.perf_counter() - started

    def search(self, query: str, k: int = 3,
               where: Optional[Dict[str, object]] = None,
               overfetch: int = 10) -> List[Dict]:
        """FAISS 검색 — 필터가 있으면 k 의 `overfetch` 배를 뽑아 후처리로 걸러낸다.

        Args:
            query: 자연어 질의.
            k: 최종 반환 개수.
            where: 메타데이터 완전일치 필터.
            overfetch: **FAISS 전용** — 필터가 있을 때 k 의 몇 배를 뽑을지.
                기본 10. 1 로 두면 과다인출 없이 딱 k 개만 뽑으므로,
                필터에서 걸러진 만큼 결과가 비어 버린다(교육용 실험에 쓴다).

        Returns:
            [{"id","text","score","metadata"}] — 최대 k 개(못 채울 수 있다).
        """
        if self._index is None:
            return []
        # 후처리 필터가 결과를 깎아내므로 넉넉히 뽑아 둔다
        fetch = min(len(self._payloads), k * overfetch if where else k)
        query_vec = self._encode_query(query).astype(np.float32).reshape(1, -1)
        scores, indices = self._index.search(query_vec, fetch)

        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            payload = self._payloads[int(idx)]
            if where and any(payload["metadata"].get(key) != value
                             for key, value in where.items()):
                continue  # post-filter
            hits.append({"id": payload["id"], "text": payload["text"],
                         "score": float(score), "metadata": payload["metadata"]})
            if len(hits) >= k:
                break
        return hits

    def count(self) -> int:
        return len(self._payloads)


class QdrantStore(BaseVectorStore):
    """Qdrant 어댑터 — `:memory:` 모드(서버 불필요) + 인덱스 통합 필터."""

    def __post_init__(self):
        """인메모리 클라이언트를 만든다(운영에서는 URL 을 주면 그대로 동작)."""
        from qdrant_client import QdrantClient

        self._client = QdrantClient(":memory:")
        self._dim = 0

    def add(self, chunks: Sequence) -> float:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        started = time.perf_counter()
        vectors = self._encode_docs([c.text for c in chunks])
        self._dim = int(vectors.shape[1])
        self._client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
        )
        # Qdrant 의 point id 는 정수/UUID 만 허용 → 순번을 id 로 쓰고 원래 id 는 payload 에
        self._client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=i, vector=vectors[i].tolist(),
                                payload={"chunk_id": c.chunk_id, "text": c.text,
                                         **c.metadata})
                    for i, c in enumerate(chunks)],
        )
        return time.perf_counter() - started

    def search(self, query: str, k: int = 3,
               where: Optional[Dict[str, object]] = None) -> List[Dict]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = None
        if where:
            # 필터가 인덱스 단계에서 적용된다 → 결과가 비지 않고 k 개가 채워진다
            query_filter = Filter(must=[
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in where.items()
            ])
        response = self._client.query_points(
            collection_name=self.collection,
            query=self._encode_query(query).tolist(),
            limit=k, query_filter=query_filter, with_payload=True,
        )
        hits = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = payload.pop("text", "")
            chunk_id = payload.pop("chunk_id", str(point.id))
            hits.append({"id": chunk_id, "text": text,
                         "score": float(point.score), "metadata": payload})
        return hits

    def count(self) -> int:
        return int(self._client.count(self.collection).count)


_STORE_CLASSES = {"chroma": ChromaStore, "faiss": FaissStore, "qdrant": QdrantStore}


def get_vector_store(kind: str, embedder, collection: str = "personal_docs") -> BaseVectorStore:
    """백엔드 이름으로 벡터 저장소를 만든다(임베딩은 주입).

    Args:
        kind: "chroma" / "faiss" / "qdrant".
        embedder: agentic_lib.embeddings 의 임베더.
        collection: 컬렉션(인덱스) 이름.

    Raises:
        ValueError: 알 수 없는 백엔드.
        ImportError: 해당 패키지가 설치되어 있지 않음.
    """
    kind = kind.strip().lower()
    if kind not in _STORE_CLASSES:
        raise ValueError(f"알 수 없는 벡터 DB: {kind} (가능: {', '.join(_STORE_CLASSES)})")
    spec = STORE_REGISTRY[kind]
    if not _import_ok(spec["module"]):
        raise ImportError(f"{spec['module']} 미설치 — {spec['install']}")
    return _STORE_CLASSES[kind](name=kind, embedder=embedder, collection=collection)


# =============================================================================
# 비교 유틸
# =============================================================================
def compare_search(stores: Sequence[BaseVectorStore], query: str, k: int = 3,
                   where: Optional[Dict[str, object]] = None) -> None:
    """같은 질의를 여러 백엔드에 던져 결과와 속도를 나란히 출력한다."""
    print(f"질의: {query}" + (f"   (필터: {where})" if where else ""))
    print("=" * 100)
    for store in stores:
        started = time.perf_counter()
        hits = store.search(query, k=k, where=where)
        elapsed = (time.perf_counter() - started) * 1000
        print(f"[{store.name}]  {elapsed:.1f} ms  ({len(hits)}건)")
        for rank, hit in enumerate(hits, 1):
            meta = hit["metadata"]
            body = hit["text"].replace("\n", " ")
            print(f"   {rank}. [{hit['score']:.3f}] {meta.get('title','')} › "
                  f"{meta.get('section','')}")
            print(f"      {body[:76]}{'…' if len(body) > 76 else ''}")
        if not hits:
            print("   (결과 없음)")
        print()


def faiss_overfetch_demo(faiss_store: BaseVectorStore, query: str,
                         where: Dict[str, object], k: int = 2,
                         multipliers: Sequence[int] = (1, 2, 5, 10),
                         compare_store: Optional[BaseVectorStore] = None) -> None:
    """FAISS 후처리 필터가 **과다인출(overfetch)에 의존한다** 는 것을 눈으로 보인다.

    FAISS 는 필터를 모른 채 상위 N개를 뽑고, 그 다음에 조건에 맞지 않는 것을 버린다.
    따라서 **필터를 통과하는 청크가 전체 순위에서 몇 등에 있는지** 가 전부다.
    정답이 12위에 있는데 2개만 뽑으면 결과는 빈손이 된다.

    Args:
        faiss_store: FAISS 스토어(`overfetch` 인자를 받는다).
        query: 자연어 질의.
        where: 메타데이터 필터(이 실습의 핵심 — 필터가 없으면 배수는 무의미).
        k: 최종적으로 받고 싶은 결과 개수.
        multipliers: 시험해 볼 과다인출 배수 목록.
        compare_store: 비교용 스토어(chroma/qdrant) — 필터가 인덱스에 통합된 쪽.
    """
    total = faiss_store.count()
    print(f"질의: {query}")
    print(f"필터: {where}   요청 k={k}   전체 청크 {total}개")
    print("=" * 100)

    # --- 1단계: 필터 없이 전체 순위 — 정답이 몇 등에 있는지가 모든 것을 결정한다 ---
    ranking = faiss_store.search(query, k=total)  # 필터 없음 = 순수 벡터 순위
    passing = []  # 필터를 통과하는 청크의 (순위, 점수, 메타)
    print("[1단계] 필터를 빼고 전체 순위를 본다 — 조건에 맞는 청크가 몇 등인가")
    print(f"  {'순위':>4} {'점수':>7}  {'분류':<8} {'통과':<5} 제목 › 섹션")
    print("  " + "-" * 96)
    last_printed = 0
    for rank, hit in enumerate(ranking, 1):
        meta = hit["metadata"]
        ok = all(meta.get(key) == value for key, value in where.items())
        if ok:
            passing.append((rank, hit["score"], meta))
        # 순위가 길면 지루하므로 통과한 것과 앞쪽 몇 개만 보여 준다
        if not (ok or rank <= 3):
            continue
        if rank > last_printed + 1:  # 건너뛴 구간이 있으면 표시한다
            print(f"  {'…':>4}")
        print(f"  {rank:>4} {hit['score']:>7.3f}  {meta.get('category',''):<8} "
              f"{'✅' if ok else '❌':<5} {meta.get('title','')} › {meta.get('section','')}")
        last_printed = rank

    if not passing:
        print("\n  ⚠️ 필터를 통과하는 청크가 없다 — 이 실험은 필터 조건을 바꿔야 한다.")
        return
    ranks = [rank for rank, _, _ in passing]
    need = ranks[min(k, len(ranks)) - 1]  # k개를 채우려면 여기까지 뽑아야 한다
    print(f"\n  → 조건을 통과하는 청크는 {len(passing)}개, 전체 순위로 {ranks} 위.")
    print(f"     k={k} 를 채우려면 최소 **{need}개** 를 뽑아야 한다"
          f"(= {need}위까지 훑어야 한다).")

    # --- 2단계: 배수를 바꿔 가며 실제로 몇 건이 돌아오는지 ---
    print(f"\n[2단계] overfetch 배수를 바꿔 가며 같은 검색을 반복한다")
    print(f"  {'배수':>5} {'뽑는 개수':>9} {'결과':>9}  상태")
    print("  " + "-" * 96)
    for mult in multipliers:
        fetch = min(total, k * mult)  # FaissStore.search 내부와 같은 계산
        hits = faiss_store.search(query, k=k, where=where, overfetch=mult)
        if not hits:
            status = "❌ 결과 0건 — 자료가 없는 것처럼 보이지만 실은 못 뽑은 것이다"
        elif len(hits) < k:
            status = f"⚠️ {k}건 중 {len(hits)}건만 — 조용히 누락된다(오류도 안 난다)"
        else:
            status = "✅ 요청한 만큼 채움"
        if fetch >= total:  # 배수가 전체 청크 수를 넘으면 사실상 전수 검색이다
            status += " (전체를 다 훑음 — 인덱스의 의미가 사라진 상태)"
        print(f"  {'×' + str(mult):>5} {str(fetch) + '개':>9} "
              f"{str(len(hits)) + '건/' + str(k) + '건':>9}  {status}")

    # --- 3단계: 필터가 인덱스에 통합된 백엔드와 비교 ---
    if compare_store is not None:
        hits = compare_store.search(query, k=k, where=where)
        print(f"\n[3단계] 같은 조건을 {compare_store.name} 에 던지면")
        print(f"  {len(hits)}건/{k}건 — 배수라는 개념 자체가 없다. "
              f"조건에 맞는 것 중에서 상위 k개를 인덱스가 직접 채워 준다.")


def evaluate_chunking(chunk_sets: Dict[str, Sequence], embedder,
                      gold: Sequence[tuple], k: int = 3,
                      backend: str = "faiss") -> Dict[str, Dict[str, float]]:
    """청킹 전략별로 **검색 정확도** 를 측정한다 — 감이 아니라 숫자로 고른다.

    각 전략의 청크를 같은 벡터 DB 에 넣고 골드셋 질문을 던져,
    정답 문서/섹션이 상위에 오는 비율을 잰다.

    Args:
        chunk_sets: {전략 이름: doc_prep.Chunk 목록}.
        embedder: 임베더(모든 전략에 동일하게 적용).
        gold: [(질문, 정답 doc_id, 정답 섹션 키워드)] 목록.
        k: 상위 몇 개까지 볼지.
        backend: 평가에 쓸 벡터 DB(기본 faiss — 가장 빠름).

    Returns:
        {전략: {"chunks","avg_len","doc_top1","sec_top1","sec_topk"}}.
        섹션 정보를 잃는 전략(fixed/recursive)은 sec_* 가 None 이다.
    """
    print(f"{'전략':<24} {'청크':>5} {'평균길이':>8} {'문서 Top-1':>10} "
          f"{'섹션 Top-1':>10} {'섹션 Top-' + str(k):>10}")
    print("-" * 76)

    report: Dict[str, Dict[str, float]] = {}
    for label, chunks in chunk_sets.items():
        store = get_vector_store(backend, embedder, collection=f"eval_{label}")
        store.add(chunks)
        # 이 전략이 섹션 메타데이터를 갖고 있는지(고정/재귀 분할은 잃어버린다)
        has_section = any(c.metadata.get("section") for c in chunks)

        doc_hit = sec_top1 = sec_topk = 0
        for question, gold_doc, gold_section in gold:
            hits = store.search(question, k=k)
            if hits and hits[0]["metadata"].get("doc_id") == gold_doc:
                doc_hit += 1
            if has_section:
                if hits and gold_section in str(hits[0]["metadata"].get("section", "")):
                    sec_top1 += 1
                if any(gold_section in str(h["metadata"].get("section", ""))
                       for h in hits):
                    sec_topk += 1

        n = len(gold)
        lengths = [len(c.text) for c in chunks] or [0]
        row = {"chunks": len(chunks), "avg_len": sum(lengths) / len(lengths),
               "doc_top1": doc_hit / n,
               "sec_top1": sec_top1 / n if has_section else None,
               "sec_topk": sec_topk / n if has_section else None}
        report[label] = row
        sec1 = f"{row['sec_top1']:>9.0%}" if has_section else f"{'구조없음':>8}"
        seck = f"{row['sec_topk']:>9.0%}" if has_section else f"{'구조없음':>8}"
        print(f"{label:<24} {row['chunks']:>5} {row['avg_len']:>8.0f} "
              f"{row['doc_top1']:>9.0%} {sec1} {seck}")

    print("-" * 76)
    print("※ '구조없음' = 섹션 메타데이터를 만들지 못하는 전략(고정·재귀 분할).")
    print("   문서는 맞게 찾아도 **어느 대목인지 짚어 줄 수 없어** 출처 표시·재확인이 어렵다.")
    return report


def rank_agreement(stores: Sequence[BaseVectorStore], queries: Sequence[str],
                   k: int = 3) -> None:
    """여러 질의에 대해 백엔드들이 같은 청크를 찾아내는지 확인한다.

    같은 벡터·같은 유사도 함수를 쓰므로 **결과가 일치하는 것이 정상** 이다.
    어긋난다면 임베딩이나 거리 척도 설정이 다르다는 신호다.
    """
    print(f"{'질의':<34} {'Top-1 청크 id':<44} 일치")
    print("-" * 88)
    for query in queries:
        top_ids = [(store.name, (store.search(query, k=k) or [{"id": "-"}])[0]["id"])
                   for store in stores]
        unique = {tid for _, tid in top_ids}
        label = query if len(query) <= 32 else query[:31] + "…"
        first = top_ids[0][1]
        print(f"{label:<34} {first:<44} {'✅' if len(unique) == 1 else '❌'}")
        if len(unique) > 1:   # 어긋난 경우에만 백엔드별로 펼쳐 보여준다
            for name, tid in top_ids:
                print(f"{'':<34}   └ {name}: {tid}")
    print("\n※ 세 백엔드가 같은 벡터·같은 코사인 척도를 쓰므로 Top-1 은 일치해야 정상이다.")
    print("※ 어긋난다면 임베딩이 다르거나 거리 척도(cosine/L2) 설정이 다르다는 신호다.")
