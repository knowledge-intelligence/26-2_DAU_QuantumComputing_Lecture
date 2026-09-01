"""
embeddings — 임베딩 공급자 추상화(오프라인 로컬 + 온라인 무료 API)
====================================================================

M04 시리즈(지식)에서 쓰는 텍스트 임베딩을 **공급자 무관 인터페이스**로 통일합니다.
LLM 이 `utils.get_llm()` 하나로 통일된 것처럼, 임베딩도 `get_embedder()` 하나로 통일합니다.

지원 공급자:
    local       sentence-transformers (오프라인, 무료, 최초 1회만 다운로드)   ★ 기본
    google      Google AI Studio  gemini-embedding-001      (무료 티어)
    nvidia      build.nvidia.com  nvidia/nv-embedqa-e5-v5   (무료 크레딧)
    openrouter  openrouter.ai     ...-embed-...:free        (무료 모델)
    ollama      로컬 Ollama 서버의 임베딩 모델(OpenAI 호환 /v1/embeddings)
    openai      OpenAI text-embedding-3-small (유료 — 비교용으로만 언급)

설계 규약:
    - 모든 임베더는 `encode(texts, kind=...) -> np.ndarray (n, dim)` 을 제공한다.
    - 기본적으로 **L2 정규화** 된 벡터를 돌려준다 → 코사인 유사도 = 내적.
      (Gemini 는 3072 미만 차원으로 줄이면 정규화가 풀리므로 이 처리가 특히 중요하다)
    - 검색용 비대칭 모델(NVIDIA nv-embedqa 계열)을 위해 `kind="query" | "passage"` 를 받는다.
    - 키가 없거나 네트워크가 막히면 예외 대신 친절한 안내 + 폴백이 가능하도록
      `available_providers()` 로 미리 사용 가능 여부를 조회할 수 있다.

사용 예 (노트북 셀):
    from agentic_lib import embeddings as emb

    local  = emb.get_embedder("local")            # 오프라인
    online = emb.get_embedder("openrouter")       # 온라인 무료
    vecs   = online.encode(["문장1", "문장2"], kind="passage")
    rows   = emb.benchmark([local, online], emb.SAMPLE_CORPUS, emb.SAMPLE_QUERIES)
    emb.print_benchmark_table(rows)
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# requests 는 sentence-transformers/huggingface_hub 의 의존성이라 사실상 항상 있지만,
# 온라인 공급자를 쓰지 않는 환경에서도 import 가 깨지지 않도록 방어한다.
try:
    import requests  # type: ignore
    _REQUESTS_AVAILABLE = True
except Exception:  # ModuleNotFoundError 등
    requests = None  # type: ignore
    _REQUESTS_AVAILABLE = False


# =============================================================================
# 공급자별 기본 모델 레지스트리
# =============================================================================
# .env 로 덮어쓸 수 있게 환경변수 이름을 함께 적어 둔다.
# dim 은 "기대 차원"으로 표시용이며, 실제 차원은 첫 호출 결과에서 확정한다.
EMBED_REGISTRY: Dict[str, Dict] = {
    "local": {
        "env": "LOCAL_EMBED_MODEL",
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384,
        "cost": "무료(오프라인)",
        "note": "최초 1회 다운로드 후 네트워크 불필요",
    },
    "google": {
        "env": "GOOGLE_EMBED_MODEL",
        "model": "gemini-embedding-001",
        "dim": 768,  # 기본 3072 → MRL 로 768 까지 축소 가능(EMBED_DIM 으로 조절)
        "cost": "무료 티어",
        "note": "taskType 지원, outputDimensionality 로 차원 조절(축소 시 재정규화 필요)",
    },
    "nvidia": {
        "env": "NVIDIA_EMBED_MODEL",
        "model": "nvidia/nemotron-3-embed-1b",
        "dim": 2048,
        "cost": "무료 크레딧",
        "note": "다국어 검색 모델 — input_type(query/passage) 필수. 영어 전용 대안: nvidia/nv-embedqa-e5-v5",
    },
    "openrouter": {
        "env": "OPENROUTER_EMBED_MODEL",
        "model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        "dim": 2048,
        "cost": "무료(:free)",
        "note": "OpenAI 호환 /v1/embeddings, 여러 공급자를 한 키로 라우팅",
    },
    "ollama": {
        "env": "OLLAMA_EMBED_MODEL",
        "model": "bge-m3",
        "dim": 1024,
        "cost": "무료(로컬 서버)",
        "note": "ollama pull bge-m3 필요. OpenAI 호환 /v1/embeddings",
    },
    "openai": {
        "env": "OPENAI_EMBED_MODEL",
        "model": "text-embedding-3-small",
        "dim": 1536,
        "cost": "유료",
        "note": "비교 기준으로만 언급(무료 아님)",
    },
}

# OpenAI 호환 `/v1/embeddings` 를 쓰는 공급자들의 (base_url 환경변수, 기본값, 키 환경변수)
_OPENAI_COMPAT: Dict[str, Tuple[str, str, str]] = {
    "nvidia":     ("NVIDIA_EMBED_BASE_URL",     "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
    "openrouter": ("OPENROUTER_BASE_URL",       "https://openrouter.ai/api/v1",        "OPENROUTER_API_KEY"),
    "openai":     ("OPENAI_BASE_URL",           "https://api.openai.com/v1",           "OPENAI_API_KEY"),
    "ollama":     ("OLLAMA_BASE_URL",           "http://localhost:11434/v1",           ""),  # 키 불필요
}

GOOGLE_EMBED_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"


# =============================================================================
# 유틸
# =============================================================================
def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """행 단위로 L2 정규화한다(정규화 후에는 코사인 유사도 = 내적).

    Args:
        vectors: (n, dim) 형태의 임베딩 배열.

    Returns:
        같은 shape 의 정규화된 배열(0 벡터는 그대로 둔다).
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # 0 나눗셈 방지
    return vectors / norms


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """두 임베딩 집합 사이의 코사인 유사도 행렬 (len(a), len(b)) 을 구한다."""
    return l2_normalize(a) @ l2_normalize(b).T


def rank_by_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray,
                       docs: Sequence[str], top_k: int = 3) -> List[Tuple[float, int, str]]:
    """쿼리 벡터와 가장 가까운 문서를 (유사도, 인덱스, 본문) 으로 정렬해 돌려준다."""
    sims = cosine_sim_matrix(np.asarray(query_vec).reshape(1, -1), doc_vecs)[0]
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i), docs[i]) for i in order]


# =============================================================================
# 임베더 구현
# =============================================================================
@dataclass
class BaseEmbedder:
    """모든 임베더의 공통 인터페이스.

    Attributes:
        provider: 공급자 키(local/google/nvidia/openrouter/ollama/openai).
        model: 실제 사용하는 모델 이름.
        dim: 임베딩 차원(첫 encode 이후 실제 값으로 갱신).
        online: 네트워크 API 사용 여부(오프라인 로컬이면 False).
    """

    provider: str
    model: str
    dim: int = 0
    online: bool = True

    @property
    def name(self) -> str:
        """표·로그에 쓰는 짧은 표시 이름."""
        return f"{self.provider}:{self.model.split('/')[-1]}"

    def encode(self, texts: Sequence[str], kind: str = "passage",
               normalize: bool = True, batch_size: int = 16) -> np.ndarray:
        """텍스트 목록을 임베딩 배열 (n, dim) 로 변환한다.

        Args:
            texts: 임베딩할 문자열 목록.
            kind: "query"(질의) 또는 "passage"(문서). 비대칭 모델에서만 의미가 있다.
            normalize: True 면 L2 정규화(코사인 = 내적).
            batch_size: 한 번의 요청에 담을 문장 수.
        """
        raise NotImplementedError

    def encode_one(self, text: str, kind: str = "query") -> np.ndarray:
        """문자열 하나를 1차원 벡터로 임베딩한다(편의 함수)."""
        return self.encode([text], kind=kind)[0]


@dataclass
class LocalEmbedder(BaseEmbedder):
    """sentence-transformers 기반 오프라인 임베더.

    최초 1회 모델을 내려받은 뒤에는 네트워크 없이 동작하며, 비용이 들지 않는다.
    """

    #: 비대칭 모델용 입력 접두사. e5 계열은 "query: " / "passage: " 를 요구한다.
    query_prefix: str = ""
    passage_prefix: str = ""
    _model: object = field(default=None, repr=False)

    def __post_init__(self):
        """모델을 로드하고 실제 임베딩 차원을 확정한다."""
        from sentence_transformers import SentenceTransformer  # 지연 import(무거움)

        self._model = SentenceTransformer(self.model)
        self.online = False
        # sentence-transformers 5.x 에서 메서드 이름이 바뀌어 둘 다 지원한다
        get_dim = getattr(self._model, "get_embedding_dimension", None) \
            or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    @property
    def name(self) -> str:
        """표시 이름 — 접두사를 쓰는 모델이면 그 사실을 함께 드러낸다."""
        base = f"{self.provider}:{self.model.split('/')[-1]}"
        return base + ("+prefix" if self.query_prefix or self.passage_prefix else "")

    def encode(self, texts: Sequence[str], kind: str = "passage",
               normalize: bool = True, batch_size: int = 16) -> np.ndarray:
        """로컬에서 임베딩을 계산한다.

        접두사가 설정된 모델(e5 계열 등)은 kind 에 따라 "query: "/"passage: " 를
        **입력 문자열 앞에 직접 붙여서** 인코딩한다. 이 접두사를 빼먹으면
        모델이 학습된 방식과 달라져 검색 품질이 눈에 띄게 떨어진다.
        """
        prefix = self.query_prefix if kind == "query" else self.passage_prefix
        inputs = [prefix + t for t in texts] if prefix else list(texts)
        vecs = self._model.encode(inputs, batch_size=batch_size,
                                  show_progress_bar=False)
        vecs = np.asarray(vecs, dtype=np.float32)
        return l2_normalize(vecs) if normalize else vecs


@dataclass
class OpenAICompatEmbedder(BaseEmbedder):
    """OpenAI 호환 `POST /v1/embeddings` 를 쓰는 온라인 임베더.

    NVIDIA build / OpenRouter / OpenAI / 로컬 Ollama 가 모두 같은 규격을 쓰므로
    base_url 과 키만 바꿔 끼우면 동일한 코드로 동작한다.
    """

    base_url: str = ""
    api_key: str = ""
    timeout: int = 60
    #: nv-embedqa 계열처럼 query/passage 를 구분해야 하는 모델이면 True
    needs_input_type: bool = False

    def _payload(self, batch: List[str], kind: str) -> dict:
        """공급자별 요청 본문을 만든다."""
        body = {"model": self.model, "input": batch, "encoding_format": "float"}
        if self.needs_input_type:
            # NVIDIA NeMo Retriever 계열은 질의/문서를 다르게 인코딩한다(비대칭 검색 모델)
            body["input_type"] = "query" if kind == "query" else "passage"
            body["truncate"] = "END"  # 최대 길이 초과 시 뒤를 자른다
        return body

    def encode(self, texts: Sequence[str], kind: str = "passage",
               normalize: bool = True, batch_size: int = 16) -> np.ndarray:
        """HTTP 로 임베딩을 요청한다(배치 단위로 나눠 호출)."""
        if not _REQUESTS_AVAILABLE:
            raise RuntimeError("requests 패키지가 필요합니다 (uv pip install requests)")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        out: List[List[float]] = []
        texts = list(texts)
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            resp = requests.post(f"{self.base_url}/embeddings", headers=headers,
                                 json=self._payload(batch, kind), timeout=self.timeout)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"[{self.provider}] 임베딩 실패 HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()["data"]
            # index 순서를 보장하기 위해 정렬 후 사용(공급자에 따라 순서가 섞일 수 있다)
            data = sorted(data, key=lambda d: d.get("index", 0))
            out.extend(d["embedding"] for d in data)

        vecs = np.asarray(out, dtype=np.float32)
        self.dim = int(vecs.shape[1])
        return l2_normalize(vecs) if normalize else vecs


@dataclass
class GoogleEmbedder(BaseEmbedder):
    """Google AI Studio(Gemini) 임베딩 API 임베더.

    OpenAI 규격이 아니라 `models/{model}:batchEmbedContents` 를 사용하며,
    검색 품질을 위해 taskType(RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT)을 구분한다.
    """

    api_key: str = ""
    output_dim: int = 768
    timeout: int = 60

    def encode(self, texts: Sequence[str], kind: str = "passage",
               normalize: bool = True, batch_size: int = 16) -> np.ndarray:
        """batchEmbedContents 로 여러 문장을 한 번에 임베딩한다."""
        if not _REQUESTS_AVAILABLE:
            raise RuntimeError("requests 패키지가 필요합니다 (uv pip install requests)")
        task_type = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
        url = f"{GOOGLE_EMBED_ENDPOINT}/{self.model}:batchEmbedContents"

        out: List[List[float]] = []
        texts = list(texts)
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            body = {"requests": [{
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": t}]},
                "taskType": task_type,
                "outputDimensionality": self.output_dim,
            } for t in batch]}
            resp = requests.post(url, headers={"x-goog-api-key": self.api_key,
                                               "Content-Type": "application/json"},
                                 json=body, timeout=self.timeout)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"[google] 임베딩 실패 HTTP {resp.status_code}: {resp.text[:200]}")
            out.extend(e["values"] for e in resp.json()["embeddings"])

        vecs = np.asarray(out, dtype=np.float32)
        self.dim = int(vecs.shape[1])
        # ⚠️ Gemini 는 3072 미만으로 축소(MRL)하면 벡터가 단위벡터가 아니다 → 반드시 재정규화
        return l2_normalize(vecs) if normalize else vecs


# =============================================================================
# 팩토리 / 가용성 점검
# =============================================================================
def _model_for(provider: str, model: Optional[str]) -> str:
    """공급자의 사용 모델을 (인자 > 환경변수 > 레지스트리 기본값) 순으로 결정한다."""
    if model:
        return model
    spec = EMBED_REGISTRY[provider]
    return os.getenv(spec["env"], "") or spec["model"]


def get_embedder(provider: Optional[str] = None, model: Optional[str] = None,
                 **kwargs) -> BaseEmbedder:
    """공급자 이름으로 임베더를 생성한다(LLM 의 utils.get_llm() 에 대응).

    Args:
        provider: local/google/nvidia/openrouter/ollama/openai. None 이면
            `.env` 의 EMBED_PROVIDER, 그것도 없으면 "local".
        model: 모델 이름 직접 지정(없으면 공급자별 기본값).
        **kwargs: 구현체에 전달할 추가 옵션(예: GoogleEmbedder(output_dim=1536)).

    Returns:
        BaseEmbedder 하위 인스턴스.

    Raises:
        ValueError: 알 수 없는 공급자이거나 필요한 API 키가 없을 때.
    """
    provider = (provider or os.getenv("EMBED_PROVIDER", "local")).strip().lower()
    if provider not in EMBED_REGISTRY:
        raise ValueError(f"알 수 없는 임베딩 공급자: {provider} "
                         f"(가능: {', '.join(EMBED_REGISTRY)})")
    model_name = _model_for(provider, model)

    if provider == "local":
        # e5 계열은 "query: "/"passage: " 접두사가 필수다 → 모델 이름으로 자동 설정
        # (auto_prefix=False 로 끄면 접두사 없이 쓸 때의 품질 저하를 실습할 수 있다)
        auto_prefix = kwargs.pop("auto_prefix", True)
        if auto_prefix and "e5" in model_name.lower():
            kwargs.setdefault("query_prefix", "query: ")
            kwargs.setdefault("passage_prefix", "passage: ")
        return LocalEmbedder(provider=provider, model=model_name, **kwargs)

    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY 가 없습니다 — aistudio.google.com 에서 발급 후 .env 에 추가")
        kwargs.setdefault("output_dim", int(os.getenv("EMBED_DIM", "768")))
        return GoogleEmbedder(provider=provider, model=model_name, api_key=api_key, **kwargs)

    env_url, default_url, env_key = _OPENAI_COMPAT[provider]
    base_url = (os.getenv(env_url, "") or default_url).rstrip("/")
    api_key = os.getenv(env_key, "") if env_key else ""
    if env_key and not api_key:
        raise ValueError(f"{env_key} 가 없습니다 — .env 에 추가하세요")
    kwargs.setdefault("needs_input_type", provider == "nvidia")
    return OpenAICompatEmbedder(provider=provider, model=model_name,
                                base_url=base_url, api_key=api_key, **kwargs)


def available_providers(verbose: bool = True) -> Dict[str, bool]:
    """각 공급자를 지금 바로 쓸 수 있는지(키/패키지 기준) 점검한다.

    실제 네트워크 호출은 하지 않고 전제조건만 본다.

    Args:
        verbose: True 면 표 형태로 출력한다.

    Returns:
        {공급자: 사용가능여부} 딕셔너리.
    """
    result: Dict[str, bool] = {}
    reasons: Dict[str, str] = {}

    for provider, spec in EMBED_REGISTRY.items():
        if provider == "local":
            try:
                import sentence_transformers  # noqa: F401
                result[provider], reasons[provider] = True, "설치됨"
            except Exception:
                result[provider], reasons[provider] = False, "uv pip install sentence-transformers"
        elif provider == "google":
            has = bool(os.getenv("GOOGLE_API_KEY", ""))
            result[provider], reasons[provider] = has, "키 설정됨" if has else "GOOGLE_API_KEY 필요"
        elif provider == "ollama":
            result[provider], reasons[provider] = True, "로컬 서버 필요(ollama pull bge-m3)"
        else:
            env_key = _OPENAI_COMPAT[provider][2]
            has = bool(os.getenv(env_key, ""))
            result[provider], reasons[provider] = has, "키 설정됨" if has else f"{env_key} 필요"

    if verbose:
        print(f"{'공급자':<12} {'사용가능':<8} {'모델':<44} {'차원':>5}  {'비용':<12} 비고")
        print("-" * 118)
        for provider, spec in EMBED_REGISTRY.items():
            mark = "✅" if result[provider] else "❌"
            print(f"{provider:<12} {mark:<8} {_model_for(provider, None):<44} "
                  f"{spec['dim']:>5}  {spec['cost']:<12} {reasons[provider]}")
    return result


# =============================================================================
# 오프라인(sentence-transformers) 모델 카탈로그
# =============================================================================
#: SentenceTransformer 로 바로 로드되는 대표 모델들.
#: dim/params/max_tokens 는 HuggingFace config 에서 확인한 값이다.
LOCAL_MODEL_CATALOG: List[Dict] = [
    {"model": "sentence-transformers/all-MiniLM-L6-v2",
     "dim": 384, "params": "23M", "max_tokens": 512, "lang": "영어 전용",
     "note": "세계에서 가장 많이 쓰이는 임베딩. 가볍지만 한국어는 사실상 불가"},
    {"model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
     "dim": 384, "params": "118M", "max_tokens": 512, "lang": "다국어 50+",
     "note": "⭐ 본 강의 기본값 — 한국어 가능 + 가장 빠름"},
    {"model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
     "dim": 768, "params": "278M", "max_tokens": 512, "lang": "다국어 50+",
     "note": "위 모델의 상위판 — 더 정확하지만 2~3배 느림"},
    {"model": "intfloat/multilingual-e5-small",
     "dim": 384, "params": "118M", "max_tokens": 512, "lang": "다국어 100+",
     "note": "검색 특화. 입력에 'query: '/'passage: ' 접두사를 꼭 붙여야 함"},
    {"model": "intfloat/multilingual-e5-base",
     "dim": 768, "params": "278M", "max_tokens": 512, "lang": "다국어 100+",
     "note": "e5-small 상위판. 접두사 규칙 동일"},
    {"model": "jhgan/ko-sroberta-multitask",
     "dim": 768, "params": "111M", "max_tokens": 512, "lang": "한국어 특화",
     "note": "KLUE 기반 한국어 SBERT — 한국어만 다룬다면 유력 후보"},
    {"model": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
     "dim": 768, "params": "111M", "max_tokens": 512, "lang": "한국어 특화",
     "note": "서울대 KR-SBERT. 문장 유사도(STS)에 강함"},
    {"model": "BAAI/bge-m3",
     "dim": 1024, "params": "568M", "max_tokens": 8192, "lang": "다국어 100+",
     "note": "긴 문서(8k 토큰) 지원. 무겁지만 검색 품질 최상급"},
    {"model": "nlpai-lab/KURE-v1",
     "dim": 1024, "params": "568M", "max_tokens": 8192, "lang": "한국어 검색 특화",
     "note": "bge-m3 를 한국어 검색용으로 추가 학습한 모델"},
]


def print_local_catalog() -> None:
    """SentenceTransformer 로 로드 가능한 대표 모델 목록을 표로 출력한다."""
    print(f"{'모델':<58} {'차원':>5} {'파라미터':>8} {'최대토큰':>8}  {'언어':<14} 비고")
    print("-" * 165)
    for spec in LOCAL_MODEL_CATALOG:
        print(f"{spec['model']:<58} {spec['dim']:>5} {spec['params']:>8} "
              f"{spec['max_tokens']:>8}  {spec['lang']:<14} {spec['note']}")
    print("-" * 165)
    print("사용법:  emb.get_embedder('local', model='<위 모델 이름>')")
    print("※ 모델은 최초 1회 자동 다운로드되어 %USERPROFILE%\\.cache\\huggingface 에 캐시된다.")


def list_hub_models(search: Optional[str] = None, limit: int = 8) -> List[Tuple[str, int]]:
    """HuggingFace Hub 에서 sentence-transformers 모델을 다운로드 순으로 조회한다.

    카탈로그에 없는 모델을 직접 찾아볼 때 쓴다(네트워크 필요).

    Args:
        search: 검색어(예: "korean"). None 이면 전체 인기 순.
        limit: 가져올 개수.

    Returns:
        [(모델 id, 다운로드 수)] — 조회 실패 시 빈 리스트.
    """
    try:
        from huggingface_hub import HfApi

        models = HfApi().list_models(filter="sentence-transformers", search=search,
                                     sort="downloads", limit=limit)
        return [(m.id, int(m.downloads or 0)) for m in models]
    except Exception as e:
        print(f"[HF Hub] 조회 실패(네트워크/버전): {str(e)[:120]}")
        return []


# =============================================================================
# 공급자별 문장 상호 유사도 비교
# =============================================================================
def _as_embedder_list(embedders) -> List[BaseEmbedder]:
    """dict({공급자: 임베더}) 든 list 든 임베더 리스트로 통일한다."""
    return list(embedders.values()) if isinstance(embedders, dict) else list(embedders)


def similarity_matrices(embedders, sentences: Sequence[str],
                        kind: str = "passage") -> Dict[str, np.ndarray]:
    """같은 문장 집합을 여러 임베더로 인코딩해 각각의 상호 유사도 행렬을 만든다.

    오프라인 모델에서 본 유사도 매트릭스를 **온라인 공급자에서도 동일하게** 뽑아
    나란히 비교하기 위한 함수다.

    Args:
        embedders: 임베더 리스트 또는 {공급자: 임베더} 딕셔너리.
        sentences: 비교할 문장들(모든 임베더에 동일하게 사용).
        kind: 인코딩 종류. 문장끼리의 대칭 비교이므로 보통 "passage" 로 통일한다.

    Returns:
        {임베더 이름: (n, n) 코사인 유사도 행렬}. 실패한 임베더는 결과에서 빠진다.
    """
    matrices: Dict[str, np.ndarray] = {}
    for embedder in _as_embedder_list(embedders):
        try:
            vecs = embedder.encode(list(sentences), kind=kind)
            matrices[embedder.name] = cosine_sim_matrix(vecs, vecs)
        except Exception as e:  # 쿼터 초과/네트워크 오류 등은 건너뛴다
            print(f"[{embedder.name}] 유사도 계산 실패(건너뜀): {str(e)[:110]}")
    return matrices


def print_similarity_matrix(sim_matrix: np.ndarray, title: str = "",
                            sentences: Sequence[str] = None) -> None:
    """(n, n) 유사도 행렬을 S0..Sn 격자로 출력한다.

    Args:
        sim_matrix: 코사인 유사도 행렬.
        title: 행렬 위에 붙일 제목(보통 임베더 이름).
        sentences: 주면 행렬 아래에 문장 목록도 함께 출력한다.
    """
    n = len(sim_matrix)
    if title:
        print(f"[{title}]")
    print("      " + "".join(f"  S{j}   " for j in range(n)))
    for i, row in enumerate(sim_matrix):
        print(f"  S{i} " + "".join(f"{v:7.3f}" for v in row))
    for i, sentence in enumerate(sentences or []):
        print(f"  S{i}: {sentence}")


def print_pair_comparison(matrices: Dict[str, np.ndarray],
                          sentences: Sequence[str]) -> None:
    """문장 쌍별 유사도를 공급자 열로 나란히 놓고 비교 출력한다.

    절대값은 모델마다 스케일이 달라 직접 비교할 수 없다. 대신 **쌍의 순서(순위)** 가
    모델 간에 일치하는지를 함께 표시한다 — 이것이 실제 검색 품질과 직결된다.

    Args:
        matrices: similarity_matrices() 결과.
        sentences: 행렬을 만들 때 사용한 문장들.
    """
    if not matrices:
        print("비교할 유사도 행렬이 없습니다")
        return
    names = list(matrices)
    pairs = [(i, j) for i in range(len(sentences)) for j in range(i + 1, len(sentences))]

    # 열 이름: 공급자명이 서로 다르면 공급자명, 같은 공급자가 여럿이면 모델명을 쓴다
    providers = [name.split(":")[0] for name in names]
    columns = providers if len(set(providers)) == len(providers) \
        else [name.split(":")[-1][:12] for name in names]

    header = "문장 쌍  " + "".join(f"{col:>13}" for col in columns)
    print(header)
    print("-" * len(header))
    for i, j in pairs:
        row = f"S{i}–S{j}    " + "".join(f"{matrices[name][i][j]:>13.3f}" for name in names)
        print(row)

    # 쌍을 유사도 내림차순으로 세운 순서가 공급자끼리 같은지 확인한다.
    # 뒤쪽(무관한 쌍들)의 순서는 사실상 잡음이므로, '몇 위까지 일치하는지' 를 함께 본다.
    print("-" * len(header))
    orders = {name: sorted(pairs, key=lambda p: -matrices[name][p[0]][p[1]])
              for name in names}
    reference = orders[names[0]]
    print(f"기준({names[0]}) 순위: " + " > ".join(f"S{i}–S{j}" for i, j in reference))
    for name in names[1:]:
        order = orders[name]
        # 기준과 앞에서부터 몇 개나 같은지(공통 접두 길이)
        agree = 0
        for ref_pair, pair in zip(reference, order):
            if ref_pair != pair:
                break
            agree += 1
        verdict = "✅ 완전 일치" if agree == len(pairs) else f"△ 상위 {agree}쌍까지 일치"
        print(f"  {verdict}  {name}: " + " > ".join(f"S{i}–S{j}" for i, j in order))
    print("\n※ 절대값은 모델마다 스케일이 달라 그대로 비교하면 안 된다"
          "(어떤 모델은 0.9대, 어떤 모델은 0.3대에 몰린다).")
    print("※ 의미 있는 비교 기준은 '쌍의 순위' 와 '유사·비유사 간의 격차' 다.")


# =============================================================================
# 오프라인 ↔ 온라인 비교용 미니 벤치마크
# =============================================================================
#: 한국어 검색 품질 비교용 미니 코퍼스(문서 10개)
SAMPLE_CORPUS: List[str] = [
    "LangGraph는 상태 그래프로 에이전트 워크플로우를 구성하는 프레임워크다.",
    "ChromaDB는 임베딩 벡터를 저장하고 유사도로 검색하는 오픈소스 벡터 데이터베이스다.",
    "Neo4j는 노드와 관계로 지식을 표현하는 그래프 데이터베이스이며 Cypher 질의어를 쓴다.",
    "MCP는 호스트-클라이언트-서버 구조로 LLM 에 외부 도구와 데이터를 연결하는 개방형 프로토콜이다.",
    "RAG는 검색으로 찾은 문서를 프롬프트에 넣어 환각을 줄이는 기법이다.",
    "Ollama는 로컬 PC 에서 오픈 모델을 OpenAI 호환 API 로 서빙하는 실행기다.",
    "NeMo Guardrails는 Colang 규칙으로 대화의 입력과 출력을 통제하는 안전장치다.",
    "ReAct는 추론(Thought)과 행동(Action)을 번갈아 수행하는 에이전트 패턴이다.",
    "벡터 임베딩은 문장의 의미를 고차원 실수 벡터로 바꿔 유사도 계산을 가능하게 한다.",
    "Docker는 애플리케이션과 의존성을 컨테이너 이미지로 묶어 어디서나 동일하게 실행한다.",
]

#: (질문, 정답 문서 인덱스) — 표현이 겹치지 않게 만들어 '의미' 검색 능력을 본다
SAMPLE_QUERIES: List[Tuple[str, int]] = [
    ("에이전트 워크플로우를 그래프로 짜려면 무엇을 쓰나요?", 0),
    ("문장 벡터를 저장해 두고 비슷한 걸 찾아주는 저장소는?", 1),
    ("관계 중심으로 데이터를 다루는 DB 와 그 질의 언어는?", 2),
    ("모델에 외부 툴을 붙이는 표준 규약이 뭔가요?", 3),
    ("모델이 없는 말을 지어내는 문제를 문서 검색으로 줄이는 방법은?", 4),
    ("내 노트북에서 오픈소스 모델을 직접 돌리려면?", 5),
    ("대화 입출력을 규칙으로 막는 안전 장치는?", 6),
    ("생각하고 행동하기를 반복하는 추론 패턴 이름은?", 7),
    ("의미가 비슷한지 숫자로 재려면 텍스트를 어떻게 바꾸나요?", 8),
    ("환경이 달라도 똑같이 실행되게 앱을 포장하는 기술은?", 9),
]

#: SAMPLE_CORPUS 각 문서에 붙는 주제 태그(Neo4j 그래프 실습에서 Topic 노드가 된다)
SAMPLE_TOPICS: List[List[str]] = [
    ["에이전트", "워크플로우"],   # 0 LangGraph
    ["벡터DB", "검색"],           # 1 ChromaDB
    ["그래프DB", "검색"],         # 2 Neo4j
    ["프로토콜", "도구연결"],     # 3 MCP
    ["검색", "환각완화"],         # 4 RAG
    ["로컬LLM", "서빙"],          # 5 Ollama
    ["안전", "규칙"],             # 6 NeMo Guardrails
    ["에이전트", "추론"],         # 7 ReAct
    ["벡터", "검색"],             # 8 임베딩
    ["인프라", "배포"],           # 9 Docker
]

#: 주제 사이의 관계 (from, RELATION, to) — multi-hop 탐색 실습용
SAMPLE_TOPIC_LINKS: List[Tuple[str, str, str]] = [
    ("벡터DB", "STORES", "벡터"),
    ("그래프DB", "STORES", "관계"),
    ("검색", "NEEDS", "벡터"),
    ("환각완화", "USES", "검색"),
    ("에이전트", "USES", "도구연결"),
    ("에이전트", "USES", "추론"),
    ("안전", "APPLIES_TO", "에이전트"),
    ("로컬LLM", "RUNS_ON", "인프라"),
    ("서빙", "RUNS_ON", "인프라"),
    ("워크플로우", "ORCHESTRATES", "에이전트"),
]


@dataclass
class BenchmarkRow:
    """벤치마크 한 줄(공급자 하나의 측정 결과)."""

    name: str
    dim: int
    online: bool
    index_sec: float          # 코퍼스 전체 임베딩(색인) 소요 시간
    query_sec: float          # 질의 1건당 평균 임베딩 시간
    top1: float               # Top-1 정확도
    top3: float               # Top-3 정확도
    mrr: float                # Mean Reciprocal Rank
    error: str = ""           # 실패 시 사유


def benchmark(embedders: Sequence[BaseEmbedder],
              corpus: Sequence[str] = None,
              queries: Sequence[Tuple[str, int]] = None,
              verbose: bool = True) -> List[BenchmarkRow]:
    """여러 임베더의 검색 품질·속도를 같은 데이터로 비교한다.

    측정 항목:
        - 색인 시간: 코퍼스 전체를 임베딩하는 데 걸린 시간
        - 질의 시간: 질의 1건 임베딩 평균 시간(온라인은 네트워크 왕복 포함)
        - Top-1 / Top-3 정확도, MRR: 정답 문서가 상위에 오는 비율

    Args:
        embedders: 비교할 임베더 목록.
        corpus: 검색 대상 문서(기본 SAMPLE_CORPUS).
        queries: (질문, 정답 인덱스) 목록(기본 SAMPLE_QUERIES).
        verbose: 진행 상황 출력 여부.

    Returns:
        BenchmarkRow 리스트(실패한 공급자는 error 가 채워진다).
    """
    corpus = list(corpus or SAMPLE_CORPUS)
    queries = list(queries or SAMPLE_QUERIES)
    rows: List[BenchmarkRow] = []

    for embedder in embedders:
        if verbose:
            print(f"[벤치마크] {embedder.name} ...", flush=True)
        try:
            t0 = time.perf_counter()
            doc_vecs = embedder.encode(corpus, kind="passage")
            index_sec = time.perf_counter() - t0

            hits1 = hits3 = 0
            rr_total = 0.0
            t0 = time.perf_counter()
            for question, gold in queries:
                q_vec = embedder.encode([question], kind="query")[0]
                sims = doc_vecs @ q_vec              # 정규화되어 있으므로 내적 = 코사인
                order = np.argsort(-sims)
                rank = int(np.where(order == gold)[0][0]) + 1
                hits1 += int(rank == 1)
                hits3 += int(rank <= 3)
                rr_total += 1.0 / rank
            query_sec = (time.perf_counter() - t0) / max(len(queries), 1)

            n = len(queries)
            rows.append(BenchmarkRow(
                name=embedder.name, dim=embedder.dim, online=embedder.online,
                index_sec=index_sec, query_sec=query_sec,
                top1=hits1 / n, top3=hits3 / n, mrr=rr_total / n,
            ))
        except Exception as e:  # 키 없음/쿼터 초과/네트워크 오류 등은 건너뛴다
            rows.append(BenchmarkRow(name=embedder.name, dim=embedder.dim,
                                     online=embedder.online, index_sec=0.0, query_sec=0.0,
                                     top1=0.0, top3=0.0, mrr=0.0, error=str(e)[:120]))
            if verbose:
                print(f"  └ 실패(건너뜀): {str(e)[:120]}")
    return rows


# =============================================================================
# 의미 공간 시각화 (온·오프라인 모델 한 장 비교)
# =============================================================================
#: 시각화용 샘플 문장 — 앞 3개는 '날씨', 뒤 2개는 '음식' 주제
VIZ_SENTENCES: List[str] = [
    "오늘 날씨가 정말 좋다",
    "오늘 날씨 맑고 좋다",
    "비가 올 것 같은 날씨다",
    "오늘 저녁은 치킨을 먹자",
    "맛있는 파스타 먹으러 가자",
]

#: VIZ_SENTENCES 의 주제 그룹 번호(같은 번호 = 같은 주제)
VIZ_GROUPS: List[int] = [0, 0, 0, 1, 1]


def use_korean_font() -> None:
    """matplotlib 한글 폰트를 OS 별로 설정한다(폰트가 없으면 조용히 넘어간다)."""
    import platform

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    system = platform.system()
    preferred = {"Windows": ["Malgun Gothic", "Gulim"],
                 "Darwin": ["AppleGothic"]}.get(system, ["NanumGothic", "NanumBarunGothic"])
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            plt.rc("font", family=name)
            break
    plt.rcParams["axes.unicode_minus"] = False  # 한글 폰트에서 마이너스 기호 깨짐 방지


def topic_separation(sim_matrix: np.ndarray, groups: Sequence[int]) -> float:
    """주제 분리도 = (같은 주제 쌍 평균 유사도) − (다른 주제 쌍 평균 유사도).

    값이 클수록 '같은 주제'와 '다른 주제'를 뚜렷하게 갈라낸다는 뜻이다.
    """
    same, diff = [], []
    n = len(groups)
    for i in range(n):
        for j in range(i + 1, n):
            (same if groups[i] == groups[j] else diff).append(float(sim_matrix[i][j]))
    if not same or not diff:
        return 0.0
    return sum(same) / len(same) - sum(diff) / len(diff)


def plot_semantic_space(embedders, sentences: Sequence[str] = None,
                        groups: Sequence[int] = None, base_index: int = 0,
                        save_path: str = "",
                        title: str = "온·오프라인 임베딩 모델 의미 공간 비교") -> Dict[str, float]:
    """여러 임베더의 의미 공간을 **한 장의 그림** 으로 비교한다.

    각 패널은 한 모델의 임베딩을 PCA 로 2차원에 투영하고,
    기준 문장 벡터와 나머지 벡터 사이의 각도(호)와 원래 고차원 코사인 유사도를 표시한다.
    마지막 패널에는 모델별 '주제 분리도'를 막대로 요약한다.

    Args:
        embedders: 임베더 리스트 또는 {공급자: 임베더} 딕셔너리.
        sentences: 비교할 문장들(기본 VIZ_SENTENCES).
        groups: 문장별 주제 그룹 번호(기본 VIZ_GROUPS). 분리도 계산과 색상에 사용.
        base_index: 기준 문장 인덱스(★ 로 표시).
        save_path: 그림을 파일로도 저장할 경로. 기본값은 저장하지 않고
            노트북에 인라인으로만 표시한다.
        title: 전체 제목.

    Returns:
        {임베더 이름: 주제 분리도}. matplotlib 이 없으면 빈 딕셔너리.
    """
    try:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError as e:
        print(f"필수 패키지 없음({e.name}) — 시각화를 건너뜁니다 "
              "(uv pip install matplotlib scikit-learn)")
        return {}

    sentences = list(sentences or VIZ_SENTENCES)
    groups = list(groups or VIZ_GROUPS)[:len(sentences)]
    use_korean_font()

    # --- 모델별 임베딩/유사도/2D 좌표 계산 -----------------------------------
    panels = []
    for embedder in _as_embedder_list(embedders):
        try:
            vecs = embedder.encode(sentences, kind="passage")
            sims = cosine_sim_matrix(vecs, vecs)
            panels.append({
                "name": embedder.name,
                "dim": embedder.dim,
                "online": embedder.online,
                "sims": sims[base_index],                 # 기준 문장과의 유사도
                # 고차원 임베딩을 화면에 그리기 위해 주성분 2개로 축소한다
                "coords": PCA(n_components=2).fit_transform(vecs),
                "separation": topic_separation(sims, groups),
            })
        except Exception as e:
            print(f"[{embedder.name}] 시각화용 임베딩 실패(건너뜀): {str(e)[:110]}")
    if not panels:
        return {}

    # --- 그림 배치 -----------------------------------------------------------
    n_cols = min(3, len(panels) + 1)
    n_rows = -(-(len(panels) + 1) // n_cols)  # 올림 나눗셈
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 5.6 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    # 주제별 색상(같은 주제는 비슷한 계열) — 날씨=파랑/청록, 음식=빨강/주황
    palette = ["#1f77b4", "#2ca02c", "#17becf", "#d62728", "#ff7f0e",
               "#9467bd", "#8c564b", "#e377c2"]

    for ax, panel in zip(axes, panels):
        coords = panel["coords"]
        base = coords[base_index]
        base_angle = np.degrees(np.arctan2(base[1], base[0])) % 360
        # 호 반지름을 문장마다 다르게 해서 겹치지 않게 한다
        span = float(np.abs(coords).max()) or 1.0
        radii = np.linspace(0.35, 0.75, len(sentences)) * span

        ax.grid(True, linestyle="--", alpha=0.4)
        ax.axhline(0, color="gray", linewidth=0.8, alpha=0.7)
        ax.axvline(0, color="gray", linewidth=0.8, alpha=0.7)

        # 방향이 거의 같은 벡터끼리는 라벨이 겹치므로, 뒤에 오는 것을 더 바깥으로 밀어낸다
        point_angles = [np.degrees(np.arctan2(py, px)) % 360 for px, py in coords]
        label_shifts = []   # (바깥쪽 배율, 수직 방향 밀어내기 pt)
        for i, angle in enumerate(point_angles):
            close = sum(1 for prev in point_angles[:i]
                        if min(abs(angle - prev), 360 - abs(angle - prev)) < 20)
            # 겹칠 때마다 더 바깥으로 + 좌우(수직 방향)로도 번갈아 밀어낸다
            label_shifts.append((1.0 + 0.9 * close, ((-1) ** close) * 21 * close))

        for i, (x, y) in enumerate(coords):
            color = palette[i % len(palette)]
            is_base = (i == base_index)

            # 1) 원점 → 문장 벡터 화살표
            ax.quiver(0, 0, x, y, angles="xy", scale_units="xy", scale=1,
                      color=color, alpha=0.75, linewidth=1.5, zorder=2)

            # 2) 기준 벡터와 이루는 2D 각도를 호로 표시
            if not is_base:
                angle_i = np.degrees(np.arctan2(y, x)) % 360
                t1, t2 = min(base_angle, angle_i), max(base_angle, angle_i)
                if t2 - t1 > 180:      # 항상 좁은 쪽 각을 그린다
                    t1, t2 = t2, t1 + 360
                ax.add_patch(patches.Arc((0, 0), radii[i] * 2, radii[i] * 2,
                                         theta1=t1, theta2=t2, color=color,
                                         linestyle="--", linewidth=1.3, alpha=0.8, zorder=3))

            # 3) 좌표 점 + 라벨(문장 · 고차원 코사인 유사도)
            ax.scatter(x, y, color=color, s=210 if is_base else 110,
                       marker="*" if is_base else "o", zorder=4,
                       edgecolor="black", linewidth=0.5)
            # 패널이 여러 개라 문장 전체를 쓰면 라벨이 겹친다 → S번호 + 유사도만 쓰고
            # 문장 원문은 그림 상단 범례에 한 번만 표시한다
            prefix = "★ " if is_base else ""
            label = f"{prefix}S{i} ({panel['sims'][i] * 100:.1f}%)"
            # 라벨을 원점→점 방향으로 바깥쪽에 배치해야 벡터끼리 가까워도 글자가 겹치지 않는다
            length = float(np.hypot(x, y)) or 1.0
            step, perp = label_shifts[i]
            unit_x, unit_y = x / length, y / length          # 원점→점 방향
            off_x = unit_x * 34 * step + (-unit_y) * perp    # 수직 성분 = (-uy, ux)
            off_y = unit_y * 26 * step + unit_x * perp
            ax.annotate(label, (x, y), textcoords="offset points",
                        xytext=(off_x, off_y), fontsize=8.5,
                        ha="left" if off_x > 8 else ("right" if off_x < -8 else "center"),
                        va="bottom" if off_y >= 0 else "top",
                        fontweight="bold" if is_base else "normal",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                  edgecolor=color, alpha=0.9), zorder=5)

        margin = span * 0.62  # 바깥쪽 라벨이 잘리지 않도록 여백을 준다
        ax.set_xlim(coords[:, 0].min() - margin, coords[:, 0].max() + margin)
        ax.set_ylim(coords[:, 1].min() - margin, coords[:, 1].max() + margin)
        kind = "온라인" if panel["online"] else "오프라인"
        ax.set_title(f"[{kind}] {panel['name']}\n{panel['dim']}차원 · 주제 분리도 "
                     f"{panel['separation']:+.3f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("의미 축 1 (PC1)", fontsize=9)
        ax.set_ylabel("의미 축 2 (PC2)", fontsize=9)

    # --- 마지막 패널: 주제 분리도 요약 막대 ----------------------------------
    summary_ax = axes[len(panels)]
    # 한글 폰트에 U+2212(−) 글리프가 없는 경우가 많아 제목·라벨에는 ASCII 하이픈만 쓴다
    names = [f"{p['name'].split(':')[0]}\n{p['name'].split(':')[-1]}" for p in panels]
    values = [p["separation"] for p in panels]
    colors = ["#4C78A8" if p["online"] else "#54A24B" for p in panels]
    summary_ax.barh(names, values, color=colors)
    summary_ax.set_title("주제 분리도 요약\n(같은 주제 평균 - 다른 주제 평균 유사도, 클수록 좋음)",
                         fontsize=10, fontweight="bold")
    summary_ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    summary_ax.tick_params(axis="y", labelsize=7.5)
    summary_ax.set_xlim(0, max(values) * 1.25 if max(values) > 0 else 1)
    for y, value in enumerate(values):
        summary_ax.text(value, y, f" {value:+.3f}", va="center", fontsize=9)
    summary_ax.legend(handles=[patches.Patch(color="#54A24B", label="오프라인"),
                               patches.Patch(color="#4C78A8", label="온라인")],
                      loc="lower right", fontsize=8)

    for ax in axes[len(panels) + 1:]:   # 남는 칸은 숨긴다
        ax.axis("off")

    # 문장 원문은 그림 상단에 한 번만 범례로 표시한다(각 패널은 S번호만 사용)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="*" if i == base_index else "o", linestyle="",
                      color=palette[i % len(palette)],
                      markersize=13 if i == base_index else 9,
                      markeredgecolor="black", markeredgewidth=0.5,
                      label=f"S{i}: {s}" + ("  (기준)" if i == base_index else ""))
               for i, s in enumerate(sentences)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=min(len(sentences), 5), fontsize=9.5, frameon=True)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if save_path:                       # 기본은 저장하지 않는다(노트북 인라인 표시)
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        print(f"그림 저장: {save_path}")
    plt.show()                          # 중간 파일 없이 노트북에 바로 표시
    return {p["name"]: p["separation"] for p in panels}


def print_benchmark_table(rows: Sequence[BenchmarkRow]) -> None:
    """benchmark() 결과를 사람이 읽기 좋은 표로 출력한다."""
    print(f"{'임베더':<34} {'차원':>5} {'구분':<6} {'색인(s)':>8} {'질의(s)':>8} "
          f"{'Top-1':>7} {'Top-3':>7} {'MRR':>7}")
    print("-" * 92)
    for r in rows:
        kind = "온라인" if r.online else "오프라인"
        if r.error:
            print(f"{r.name:<34} {r.dim:>5} {kind:<6} {'-':>8} {'-':>8} "
                  f"{'-':>7} {'-':>7} {'-':>7}   ← 실패: {r.error}")
            continue
        print(f"{r.name:<34} {r.dim:>5} {kind:<6} {r.index_sec:>8.2f} {r.query_sec:>8.3f} "
              f"{r.top1:>7.1%} {r.top3:>7.1%} {r.mrr:>7.3f}")
    print("-" * 92)
    ok = [r for r in rows if not r.error]
    if ok:
        best_q = max(ok, key=lambda r: (r.mrr, -r.query_sec))
        fast = min(ok, key=lambda r: r.query_sec)
        print(f"검색 품질 1위: {best_q.name} (MRR {best_q.mrr:.3f})  |  "
              f"질의 최속: {fast.name} ({fast.query_sec*1000:.0f} ms/건)")
