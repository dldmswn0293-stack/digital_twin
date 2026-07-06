"""
② Sentence-Transformers 임베딩 + Chroma
③ BM25 하이브리드 (LangChain EnsembleRetriever)

설계 의도:
- Dense(임베딩) 검색: 의미적 유사성 기반. "이상 조기 탐지"처럼 표현이 달라도
  의미가 통하는 문서를 찾는 데 강함.
- Sparse(BM25) 검색: 키워드 정확 매칭 기반. "GroupKFold", "Focal Loss", "ESC" 같은
  도메인 고유명사/기술 용어는 임베딩보다 BM25가 더 정확히 잡아내는 경우가 많음.
  → 제조 도메인 문서는 특히 약어/부품명/파라미터명이 핵심이라 하이브리드가 필수.
- EnsembleRetriever로 두 결과를 RRF(Reciprocal Rank Fusion)로 결합.
"""

from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever

from config import get_embeddings
from ingest_corpus import build_corpus

PERSIST_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "manufacturing_agent_corpus"

DENSE_WEIGHT = 0.5
SPARSE_WEIGHT = 0.5
TOP_K = 4


def build_vectorstore(persist: bool = True) -> Chroma:
    """청크 -> 임베딩 -> Chroma 인덱싱"""
    chunks = build_corpus()
    if not chunks:
        raise ValueError("코퍼스가 비어 있음. corpus/ 폴더에 문서를 추가하세요.")

    embeddings = get_embeddings()

    if persist:
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(PERSIST_DIR),
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=embeddings, collection_name=COLLECTION_NAME
        )

    return vectorstore, chunks


def load_vectorstore() -> Chroma:
    """이미 저장된 Chroma 컬렉션 로드 (재인덱싱 없이)"""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )


def build_hybrid_retriever(top_k: int = TOP_K) -> EnsembleRetriever:
    """
    Dense(Chroma) + Sparse(BM25) 앙상블 리트리버 생성.
    매 프로세스 시작 시 BM25는 in-memory로 재구축(가벼움),
    Chroma는 persist된 인덱스를 로드하거나 없으면 새로 구축.
    """
    if not PERSIST_DIR.exists() or not any(PERSIST_DIR.iterdir()):
        vectorstore, chunks = build_vectorstore(persist=True)
    else:
        vectorstore = load_vectorstore()
        chunks = build_corpus()  # BM25용 원본 청크는 항상 재계산 (가벼운 연산)

    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = top_k

    hybrid = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[DENSE_WEIGHT, SPARSE_WEIGHT],
    )
    return hybrid


if __name__ == "__main__":
    retriever = build_hybrid_retriever()

    test_queries = [
        "베어링 예지보전에서 데이터 누수를 왜 조심해야 해",
        "ESC O-ring 소재 관련 개선 사례",
        "웨이퍼 결함 클래스 불균형을 어떻게 해결했어",
    ]

    for q in test_queries:
        print(f"\n{'='*60}\n질의: {q}\n{'='*60}")
        results = retriever.invoke(q)
        for r in results:
            print(f"[{r.metadata.get('category')} / {r.metadata.get('source')}]")
            print(r.page_content[:120].replace("\n", " ") + "...\n")
