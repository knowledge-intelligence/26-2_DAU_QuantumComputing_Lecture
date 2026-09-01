"""
doc_prep — 개인 문서의 구조화 · 청크 분할 · 메타데이터 설계
============================================================

RAG 품질의 8할은 검색 이전 단계인 **문서 전처리** 에서 결정됩니다.
이 모듈은 M04_2(Vector RAG) 노트북에서 쓰는 전처리 파이프라인을 담습니다.

    원본 파일(.md)  →  구조 파싱(front matter + 헤더)  →  청크 분할  →  메타데이터 부착
         │                      │                          │                │
    write_sample_docs      load_markdown_docs         chunk_document    build_chunk_metadata
                                                      (4가지 전략 비교)

핵심 개념:
    1. **구조화** — 문서를 통짜 텍스트가 아니라 (front matter + 섹션 트리) 로 읽는다.
    2. **청크 분할** — 검색 단위. 너무 크면 잡음이 섞이고, 너무 작으면 맥락이 끊긴다.
    3. **메타데이터** — 필터링·출처 표시·권한 제어의 근거. 청크마다 반드시 붙인다.

의존성 규약:
    - LangChain 스플리터(`langchain_text_splitters`)는 있으면 쓰고, 없으면
      순수 파이썬 폴백으로 동작한다(강의 환경 어디서나 실행 가능하도록).
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# LangChain 텍스트 스플리터는 선택 의존성으로 다룬다(없으면 폴백).
try:
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )
    _SPLITTERS_AVAILABLE = True
except Exception:  # ModuleNotFoundError 등
    MarkdownHeaderTextSplitter = None  # type: ignore
    RecursiveCharacterTextSplitter = None  # type: ignore
    _SPLITTERS_AVAILABLE = False


# =============================================================================
# 1. 샘플 "개인 문서" — 실제 파일로 떨어뜨려 놓고 로드한다
# =============================================================================
#: 파일명 → 마크다운 본문. YAML 스타일 front matter 로 메타데이터를 표현한다.
SAMPLE_PERSONAL_DOCS: Dict[str, str] = {
    "2026-03-14_회의록_프로젝트킥오프.md": """---
title: 사내 문서 검색 시스템 킥오프 회의록
category: 회의록
tags: [프로젝트, 킥오프, RAG, 보안]
created: 2026-03-14
author: 신성태
---

# 사내 문서 검색 시스템 킥오프 회의록

## 참석자
개발팀 4명, 기획팀 2명이 참석했다. 외부 자문으로 데이터 플랫폼팀 1명이 배석했다.

## 배경
사내 위키와 공유 드라이브에 문서가 흩어져 있어 필요한 정보를 찾는 데 평균 20분이 걸린다.
검색이 키워드 완전 일치 방식이라 표현이 조금만 달라도 결과가 나오지 않는다는 불만이 많았다.

## 결정 사항
1. 1차 목표는 사내 규정 문서와 회의록을 대상으로 하는 시맨틱 검색이다.
2. 임베딩은 사내망에서 동작해야 하므로 오프라인 모델을 우선 검토한다.
3. 문서 권한이 부서별로 다르므로 메타데이터에 부서와 공개 범위를 반드시 넣는다.
4. 6주 안에 프로토타입을 만들어 사내 데모를 진행한다.

## 다음 액션
개발팀은 다음 주까지 문서 수집 파이프라인 초안을 만든다.
기획팀은 부서별 문서 접근 정책을 정리해 공유한다.
""",
    "학습노트_langchain_lcel.md": """---
title: LangChain LCEL 학습 노트
category: 학습노트
tags: [LangChain, LCEL, RAG]
created: 2026-02-28
author: 신성태
---

# LangChain LCEL 학습 노트

## LCEL 이란
LCEL(LangChain Expression Language)은 파이프 연산자로 컴포넌트를 연결하는 선언적 문법이다.
`prompt | llm | parser` 처럼 왼쪽 출력이 오른쪽 입력으로 흘러간다.

## 왜 쓰는가
직접 함수를 이어 붙이는 것과 달리 스트리밍, 배치, 비동기가 공짜로 따라온다.
각 단계가 Runnable 이라는 같은 인터페이스를 따르기 때문에 중간에 무엇을 끼워 넣어도 된다.

## 자주 쓰는 조각
RunnablePassthrough 는 입력을 그대로 흘려보내면서 곁가지로 다른 값을 덧붙일 때 쓴다.
RunnableLambda 는 평범한 파이썬 함수를 체인 안으로 끌어들인다.
StrOutputParser 는 모델 응답 객체에서 문자열만 뽑아낸다.

## 실수했던 것
프롬프트 변수 이름과 딕셔너리 키가 다르면 조용히 빈 문자열이 들어간다.
체인을 만들고 나면 반드시 작은 입력으로 한 번 돌려보고 중간 출력을 확인해야 한다.
""",
    "업무_주간보고_2026-W12.md": """---
title: 2026년 12주차 주간 업무 보고
category: 업무보고
tags: [프로젝트, 주간보고, RAG]
created: 2026-03-20
author: 신성태
---

# 2026년 12주차 주간 업무 보고

## 이번 주 한 일
문서 수집 파이프라인의 첫 버전을 만들었다. 마크다운과 텍스트 파일을 읽어
front matter 를 분리하고 섹션 단위로 쪼개는 데까지 동작한다.
임베딩 모델 세 가지를 같은 데이터로 비교해 표로 정리했다.

## 문제와 해결
청크를 너무 잘게 쪼갰더니 검색 결과가 문장 조각처럼 나와 답변 품질이 떨어졌다.
청크 크기를 키우고 겹침을 두는 방식으로 바꾸니 맥락이 유지되었다.

## 다음 주 계획
벡터 데이터베이스를 세 종류 비교해 우리 환경에 맞는 것을 고른다.
부서별 권한 필터가 검색 단계에서 동작하는지 확인한다.
""",
    "여행_오사카_준비메모.md": """---
title: 오사카 여행 준비 메모
category: 개인메모
tags: [여행, 오사카, 체크리스트]
created: 2026-01-11
author: 신성태
---

# 오사카 여행 준비 메모

## 일정
3박 4일이며 첫날은 저녁 도착이라 숙소 근처만 둘러보기로 했다.
둘째 날은 교토 당일치기, 셋째 날은 시내와 수족관을 돌 계획이다.

## 예약할 것
간사이 공항에서 시내로 가는 특급 열차표를 미리 끊어 두면 줄을 서지 않아도 된다.
인기 있는 식당은 예약 없이 가면 두 시간씩 기다린다.

## 챙길 것
110볼트 변환 어댑터, 우산, 교통카드를 챙긴다.
현금을 쓰는 가게가 아직 많아 엔화를 조금 준비하는 편이 낫다.
""",
    "건강_러닝기록_3월.md": """---
title: 3월 러닝 기록
category: 개인메모
tags: [운동, 러닝, 건강]
created: 2026-03-31
author: 신성태
---

# 3월 러닝 기록

## 총량
한 달 동안 열두 번 달렸고 누적 거리는 96킬로미터였다.
가장 길게 달린 날은 15킬로미터였고 평균 페이스는 킬로미터당 5분 40초였다.

## 몸 상태
둘째 주에 무릎이 시큰거려 사흘을 쉬었다. 신발을 바꾸고 나서는 괜찮아졌다.
아침 공복 러닝은 속이 불편해서 가벼운 식사 후에 나가는 쪽으로 바꿨다.

## 4월 목표
누적 120킬로미터를 목표로 하되 주 3회 이상은 유지한다.
""",
    "규정_문서보안_지침.md": """---
title: 사내 문서 보안 지침
category: 사내규정
tags: [보안, 규정, 권한]
created: 2025-11-05
author: 정보보안팀
---

# 사내 문서 보안 지침

## 등급 구분
문서는 공개, 사내한정, 대외비 세 등급으로 나눈다.
등급은 문서를 만든 사람이 지정하며 지정하지 않으면 사내한정으로 간주한다.

## 검색 시스템 적용 원칙
검색 색인에는 등급 정보를 반드시 함께 저장한다.
사용자가 볼 수 없는 등급의 문서는 검색 결과 자체에 나타나지 않아야 한다.
요약이나 인용 형태로도 노출되어서는 안 된다.

## 위반 시
대외비 문서를 외부로 전송하면 즉시 보고 대상이며 감사 기록이 남는다.
""",
}


def write_sample_docs(directory: str) -> List[str]:
    """샘플 개인 문서를 실제 `.md` 파일로 만든다(이미 있으면 덮어쓴다).

    RAG 실습은 "메모리 위의 문자열"이 아니라 **디스크의 파일** 에서 시작해야
    로딩·인코딩·경로 같은 현실 문제를 같이 다룰 수 있다.

    Args:
        directory: 파일을 만들 폴더(없으면 생성).

    Returns:
        생성된 파일 경로 목록.
    """
    os.makedirs(directory, exist_ok=True)
    paths = []
    for filename, content in SAMPLE_PERSONAL_DOCS.items():
        path = os.path.join(directory, filename)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        paths.append(path)
    return paths


# =============================================================================
# 2. 구조화 — front matter + 섹션 파싱
# =============================================================================
@dataclass
class PersonalDoc:
    """구조화된 개인 문서 하나.

    Attributes:
        doc_id: 파일명에서 만든 문서 식별자.
        path: 원본 파일 경로.
        title/category/author/created: front matter 에서 뽑은 메타데이터.
        tags: 태그 목록.
        body: front matter 를 걷어낸 본문(마크다운).
        sections: [{"heading": "## 배경", "text": "..."}] 형태의 섹션 목록.
    """

    doc_id: str
    path: str
    title: str = ""
    category: str = "미분류"
    author: str = ""
    created: str = ""
    tags: List[str] = field(default_factory=list)
    body: str = ""
    sections: List[Dict[str, str]] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        """본문 글자 수."""
        return len(self.body)


def parse_front_matter(text: str) -> "tuple[Dict[str, object], str]":
    """`---` 로 감싼 YAML 스타일 front matter 를 파싱한다.

    PyYAML 없이도 동작하도록 `key: value` 와 `[a, b]` 리스트만 지원하는
    최소 파서를 직접 구현했다(강의용으로 동작을 눈에 보이게 하려는 의도도 있다).

    Args:
        text: 파일 전체 텍스트.

    Returns:
        (메타데이터 dict, front matter 를 제거한 본문).
    """
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text

    meta: Dict[str, object] = {}
    for line in text[3:end].strip().splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if value.startswith("[") and value.endswith("]"):
            # [a, b, c] → 리스트
            meta[key] = [v.strip() for v in value[1:-1].split(",") if v.strip()]
        else:
            meta[key] = value
    body = text[end + 4:].lstrip("\n")
    return meta, body


def split_sections(markdown: str) -> List[Dict[str, str]]:
    """마크다운 본문을 `##` 헤더 기준 섹션으로 나눈다.

    헤더가 없으면 전체를 하나의 섹션("본문")으로 본다.

    Returns:
        [{"heading": 섹션 제목, "text": 섹션 본문}] 목록.
    """
    sections: List[Dict[str, str]] = []
    current = {"heading": "본문", "text": ""}
    for line in markdown.splitlines():
        header = re.match(r"^(#{1,6})\s+(.*)$", line)
        if header:
            if current["text"].strip():
                sections.append({"heading": current["heading"],
                                 "text": current["text"].strip()})
            current = {"heading": header.group(2).strip(), "text": ""}
        else:
            current["text"] += line + "\n"
    if current["text"].strip():
        sections.append({"heading": current["heading"], "text": current["text"].strip()})
    return sections


def load_markdown_docs(directory: str) -> List[PersonalDoc]:
    """폴더의 `.md` 파일을 읽어 구조화된 PersonalDoc 목록으로 만든다.

    Args:
        directory: 문서 폴더.

    Returns:
        파일명 순으로 정렬된 PersonalDoc 목록.
    """
    docs: List[PersonalDoc] = []
    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith((".md", ".markdown", ".txt")):
            continue
        path = os.path.join(directory, filename)
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        meta, body = parse_front_matter(raw)
        docs.append(PersonalDoc(
            doc_id=os.path.splitext(filename)[0],
            path=path,
            title=str(meta.get("title", os.path.splitext(filename)[0])),
            category=str(meta.get("category", "미분류")),
            author=str(meta.get("author", "")),
            created=str(meta.get("created", "")),
            tags=list(meta.get("tags", [])) if isinstance(meta.get("tags"), list) else [],
            body=body,
            sections=split_sections(body),
        ))
    return docs


def print_doc_structure(docs: Sequence[PersonalDoc]) -> None:
    """구조화 결과(메타데이터 + 섹션 트리)를 표로 출력한다."""
    print(f"{'문서 ID':<34} {'분류':<8} {'작성일':<11} {'글자수':>6} {'섹션':>4}  태그")
    print("-" * 108)
    for doc in docs:
        print(f"{doc.doc_id:<34} {doc.category:<8} {doc.created:<11} "
              f"{doc.char_count:>6} {len(doc.sections):>4}  {', '.join(doc.tags)}")
    print("-" * 108)
    print(f"총 {len(docs)}개 문서 / {sum(len(d.sections) for d in docs)}개 섹션 / "
          f"{sum(d.char_count for d in docs):,}자")


# =============================================================================
# 3. 청크 분할 — 세 가지 전략
# =============================================================================
@dataclass
class Chunk:
    """검색 단위 청크 하나(본문 + 메타데이터)."""

    text: str
    metadata: Dict[str, object]

    @property
    def chunk_id(self) -> str:
        """벡터 DB 에 넣을 고유 id."""
        return f"{self.metadata['doc_id']}#{self.metadata['chunk_index']}"


def _fixed_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """전략 1) 글자 수 기준 고정 분할 — 가장 단순하지만 문장을 자른다."""
    step = max(chunk_size - overlap, 1)
    return [text[i:i + chunk_size] for i in range(0, len(text), step)
            if text[i:i + chunk_size].strip()]


def _recursive_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """전략 2) 재귀 분할 — 문단 → 문장 → 단어 순으로 자연스러운 경계를 찾는다."""
    if _SPLITTERS_AVAILABLE:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", ".", " ", ""],
        )
        return [c for c in splitter.split_text(text) if c.strip()]
    # 폴백: 문단 단위로 모으다가 chunk_size 를 넘으면 끊는다
    chunks, buffer = [], ""
    for para in text.split("\n\n"):
        if len(buffer) + len(para) > chunk_size and buffer:
            chunks.append(buffer.strip())
            buffer = buffer[-overlap:] if overlap else ""
        buffer += para + "\n\n"
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks


def _merge_small(pieces: List[tuple], min_chars: int) -> List[tuple]:
    """너무 짧은 조각을 바로 앞 조각에 붙인다.

    섹션 단위로 자르면 "참석자" 같은 한두 줄짜리 섹션이 독립 청크가 되어 버린다.
    이런 파편은 검색되어도 맥락이 없어 답변에 도움이 안 되고, 정작 필요한 청크를
    상위 k개 밖으로 밀어낸다. 그래서 최소 길이에 못 미치면 앞 조각과 합친다.

    Args:
        pieces: (섹션 제목, 본문) 목록.
        min_chars: 이 길이 미만이면 병합 대상.

    Returns:
        병합된 (섹션 제목, 본문) 목록. 합쳐진 섹션 제목은 " / " 로 잇는다.
    """
    merged: List[tuple] = []
    for heading, text in pieces:
        if merged and len(merged[-1][1]) < min_chars:
            prev_heading, prev_text = merged[-1]
            joined = prev_heading if prev_heading == heading \
                else f"{prev_heading} / {heading}".strip(" /")
            merged[-1] = (joined, f"{prev_text}\n\n{text}")
        else:
            merged.append((heading, text))
    return merged


def chunk_document(doc: PersonalDoc, strategy: str = "header_merged",
                   chunk_size: int = 400, overlap: int = 80,
                   min_chars: int = 0) -> List[Chunk]:
    """문서 하나를 청크 목록으로 나누고 **메타데이터를 부착** 한다.

    Args:
        doc: 구조화된 문서.
        strategy: 청킹 전략.
            - "fixed": 글자 수 고정 분할
            - "recursive": 문단→문장→단어 경계를 찾는 재귀 분할
            - "header": 섹션 경계를 존중하고 섹션 안에서 재귀 분할
            - "header_merged": header + 짧은 섹션 병합 (⭐ 실무 권장)
        chunk_size: 청크 목표 글자 수.
        overlap: 청크 간 겹침 글자 수(맥락 유실 방지).
        min_chars: 병합 기준 최소 길이. 0 이면 전략별 기본값(header_merged=150).

    Returns:
        Chunk 목록. 각 청크는 문서 메타 + 섹션 + 순번을 갖는다.
    """
    pieces: List[tuple] = []   # (섹션 제목, 청크 본문)

    if strategy in ("header", "header_merged"):
        # 섹션 경계를 먼저 존중하고, 섹션이 길면 그 안에서만 재귀 분할한다
        for section in doc.sections:
            for piece in _recursive_split(section["text"], chunk_size, overlap):
                pieces.append((section["heading"], piece))
        if strategy == "header_merged":
            pieces = _merge_small(pieces, min_chars or 150)
    elif strategy == "recursive":
        for piece in _recursive_split(doc.body, chunk_size, overlap):
            pieces.append(("", piece))
    elif strategy == "fixed":
        for piece in _fixed_split(doc.body, chunk_size, overlap):
            pieces.append(("", piece))
    else:
        raise ValueError(f"알 수 없는 청킹 전략: {strategy} "
                         "(fixed/recursive/header/header_merged)")

    total = len(pieces)
    return [Chunk(text=text, metadata=build_chunk_metadata(doc, heading, i, total))
            for i, (heading, text) in enumerate(pieces)]


def build_chunk_metadata(doc: PersonalDoc, section: str, index: int,
                         total: int) -> Dict[str, object]:
    """청크 메타데이터를 설계한 스키마대로 만든다.

    설계 원칙:
        - **출처 추적**: doc_id·title·source 로 답변에 근거를 표시할 수 있어야 한다.
        - **필터링 키**: category·tags·created 는 검색 전 좁히기(pre-filter)에 쓴다.
        - **스칼라만**: 대부분의 벡터 DB 는 리스트를 메타데이터로 못 받는다
          → tags 는 쉼표 문자열로 평탄화한다.
        - **위치 정보**: section·chunk_index 로 원문에서의 자리를 되짚을 수 있다.

    Args:
        doc: 원본 문서.
        section: 이 청크가 속한 섹션 제목("" 가능).
        index: 문서 안에서의 청크 순번.
        total: 문서의 전체 청크 수.

    Returns:
        벡터 DB 에 그대로 넣을 수 있는 스칼라 값 딕셔너리.
    """
    return {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "category": doc.category,
        "author": doc.author,
        "created": doc.created,
        "tags": ",".join(doc.tags),      # 리스트 → 쉼표 문자열(벡터 DB 호환)
        "section": section,
        "chunk_index": index,
        "chunk_total": total,
        "source": os.path.basename(doc.path),
    }


def chunk_all(docs: Sequence[PersonalDoc], strategy: str = "header_merged",
              chunk_size: int = 400, overlap: int = 80,
              min_chars: int = 0) -> List[Chunk]:
    """여러 문서를 한 번에 청킹한다."""
    chunks: List[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, strategy, chunk_size, overlap, min_chars))
    return chunks


# =============================================================================
# 4. 청킹 전략 비교
# =============================================================================
def _sentence_cut_ratio(chunks: Sequence[Chunk]) -> float:
    """문장 중간에서 잘린 청크의 비율 — 낮을수록 좋은 분할이다."""
    if not chunks:
        return 0.0
    cut = sum(1 for c in chunks
              if c.text.strip() and c.text.strip()[-1] not in ".!?。\n:)]”\"'")
    return cut / len(chunks)


def compare_chunking(docs: Sequence[PersonalDoc], chunk_size: int = 400,
                     overlap: int = 80) -> Dict[str, List[Chunk]]:
    """세 가지 청킹 전략을 같은 문서에 적용하고 통계를 비교 출력한다.

    Args:
        docs: 구조화된 문서 목록.
        chunk_size: 목표 청크 크기.
        overlap: 겹침 크기.

    Returns:
        {전략 이름: 청크 목록}.
    """
    labels = {"fixed": "고정 크기", "recursive": "재귀 분할",
              "header": "섹션 인식", "header_merged": "섹션+병합 ⭐"}
    results: Dict[str, List[Chunk]] = {}

    print(f"{'전략':<14} {'청크 수':>7} {'평균길이':>8} {'최소':>6} {'최대':>6} "
          f"{'문장중간절단':>12}  설명")
    print("-" * 110)
    notes = {
        "fixed": "글자 수로만 자름 — 구현은 쉽지만 단어·문장이 끊긴다",
        "recursive": "문단→문장→단어 순으로 경계를 찾음 — 범용 기본값",
        "header": "섹션 경계 존중 — 맥락은 좋지만 짧은 섹션이 파편으로 남는다",
        "header_merged": "짧은 섹션을 앞과 병합 — 파편 제거, 실무 권장값",
    }
    for strategy in ("fixed", "recursive", "header", "header_merged"):
        chunks = chunk_all(docs, strategy, chunk_size, overlap)
        results[strategy] = chunks
        lengths = [len(c.text) for c in chunks] or [0]
        print(f"{labels[strategy]:<14} {len(chunks):>7} {sum(lengths)/len(lengths):>8.0f} "
              f"{min(lengths):>6} {max(lengths):>6} {_sentence_cut_ratio(chunks):>11.0%}"
              f"  {notes[strategy]}")
    print("-" * 110)
    print("※ '문장중간절단' = 청크가 문장부호로 끝나지 않는 비율(낮을수록 자연스러운 분할)")
    print("※ 평균길이가 너무 짧으면(<150자) 검색은 되어도 맥락이 없어 답변 품질이 떨어진다")
    return results


#: 청킹/검색 품질 평가용 골드셋 — (질문, 정답 문서 id, 정답 섹션 키워드)
#: 질문은 문서와 표현이 겹치지 않게 만들어 '의미' 검색이 되는지 본다.
GOLD_QUESTIONS: List[tuple] = [
    ("킥오프 회의에서 결정된 것 중 권한 관련 내용은?",
     "2026-03-14_회의록_프로젝트킥오프", "결정 사항"),
    ("프로토타입 데모까지 기간이 얼마나 되지?",
     "2026-03-14_회의록_프로젝트킥오프", "결정 사항"),
    ("LCEL 에서 자주 하는 실수는?", "학습노트_langchain_lcel", "실수했던 것"),
    ("RunnableLambda 는 언제 쓰나?", "학습노트_langchain_lcel", "자주 쓰는 조각"),
    ("청크 크기 문제를 어떻게 해결했나?", "업무_주간보고_2026-W12", "문제와 해결"),
    ("3월에 총 몇 킬로미터 달렸나?", "건강_러닝기록_3월", "총량"),
    ("무릎이 아팠던 건 언제인가?", "건강_러닝기록_3월", "몸 상태"),
    ("오사카에서 미리 예약해야 하는 것은?", "여행_오사카_준비메모", "예약할 것"),
    ("대외비 문서를 외부로 보내면 어떻게 되나?", "규정_문서보안_지침", "위반 시"),
    ("검색 색인에 등급 정보를 넣어야 하는 이유는?", "규정_문서보안_지침", "검색 시스템 적용 원칙"),
]


def print_chunk_samples(chunks: Sequence[Chunk], n: int = 3,
                        width: int = 90) -> None:
    """청크와 그 메타데이터를 몇 개만 보기 좋게 출력한다."""
    for chunk in list(chunks)[:n]:
        meta = chunk.metadata
        print(f"[{chunk.chunk_id}] {meta['title']} › {meta['section'] or '(섹션 없음)'}")
        print(f"  분류={meta['category']} | 태그={meta['tags']} | 작성일={meta['created']}")
        body = chunk.text.replace("\n", " ")
        print(f"  {body[:width]}{'…' if len(body) > width else ''}")
        print()
