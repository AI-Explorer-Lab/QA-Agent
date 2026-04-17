import logging
import os
from typing import Dict, Iterable, List

from core.hybrid_ranker import fuse_hybrid_results, keyword_rank_documents
from core.retrieval_deduper import dedupe_ranked_documents
from db_service.faiss_store import list_faiss_documents, search_documents_v2
from db_service.pgvector_store import list_documents_pgvector, search_documents_pgvector

logger = logging.getLogger(__name__)


def _safe_int(value: str, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: str, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _valid_doc(doc: Dict) -> bool:
    if not isinstance(doc, dict):
        return False
    raw_doc = str(doc.get("raw_doc", "") or "").strip()
    if not raw_doc:
        return False
    if doc.get("chunk_id") or doc.get("doc_id"):
        return True
    return False


def _normalize_docs(docs: Iterable[Dict]) -> List[Dict]:
    normalized: List[Dict] = []
    for item in docs:
        if not _valid_doc(item):
            continue
        normalized.append(item)
    return normalized


def _hybrid_from_single_backend(
    query: str,
    k: int,
    dense_docs: List[Dict],
    sparse_candidates: List[Dict],
) -> List[Dict]:
    if not dense_docs and not sparse_candidates:
        return []

    sparse_pool_factor = _safe_int(os.getenv("HYBRID_SPARSE_POOL_FACTOR", "6"), 6)
    sparse_pool = max(1, k * max(1, sparse_pool_factor))
    sparse_algorithm = str(os.getenv("HYBRID_SPARSE_ALGORITHM", "overlap")).strip().lower()
    if sparse_algorithm not in {"overlap", "bm25"}:
        logger.warning(
            "Unsupported HYBRID_SPARSE_ALGORITHM=%s, fallback to overlap.",
            sparse_algorithm,
        )
        sparse_algorithm = "overlap"

    bm25_k1 = _safe_float(os.getenv("HYBRID_BM25_K1", "1.5"), 1.5)
    bm25_b = _safe_float(os.getenv("HYBRID_BM25_B", "0.75"), 0.75)
    if sparse_algorithm == "bm25":
        sparse_min_score = _safe_float(os.getenv("HYBRID_BM25_MIN_SCORE", "0.05"), 0.05)
    else:
        sparse_min_score = _safe_float(os.getenv("HYBRID_SPARSE_MIN_SCORE", "0.05"), 0.05)

    sparse_docs = keyword_rank_documents(
        query=query,
        docs=sparse_candidates,
        k=sparse_pool,
        min_score=sparse_min_score,
        algorithm=sparse_algorithm,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
    )
    fused = fuse_hybrid_results(
        query=query,
        dense_docs=dense_docs,
        sparse_docs=sparse_docs,
        k=k,
    )
    if fused:
        return fused
    return dedupe_ranked_documents(dense_docs + sparse_docs, k)


def search_documents(query: str, k: int) -> List[Dict]:
    backend = os.getenv("VECTOR_STORE_BACKEND", "faiss").strip().lower()
    limit = max(1, int(k))

    dense_pool_factor = _safe_int(os.getenv("HYBRID_DENSE_POOL_FACTOR", "4"), 4)
    dense_k = max(limit, limit * max(1, dense_pool_factor))

    sparse_scan_limit = _safe_int(os.getenv("HYBRID_SPARSE_SCAN_LIMIT", "3000"), 3000)
    sparse_scan_limit = max(limit, sparse_scan_limit)

    if backend == "pgvector":
        dense_docs = _normalize_docs(search_documents_pgvector(query, k=dense_k))
        sparse_candidates = _normalize_docs(list_documents_pgvector(limit=sparse_scan_limit))
        ranked = _hybrid_from_single_backend(query, limit, dense_docs, sparse_candidates)
        if ranked:
            return ranked
        logger.warning("pgvector retrieval is empty after hybrid ranking, fallback to faiss.")
        fallback = _normalize_docs(search_documents_v2(query, limit))
        return dedupe_ranked_documents(fallback, limit)

    if backend in {"hybrid", "both"}:
        dense_faiss = _normalize_docs(search_documents_v2(query, dense_k))
        dense_pg = _normalize_docs(search_documents_pgvector(query, dense_k))
        dense_docs = dense_faiss + dense_pg

        sparse_candidates = _normalize_docs(
            list_faiss_documents(limit=sparse_scan_limit)
            + list_documents_pgvector(limit=sparse_scan_limit)
        )

        ranked = _hybrid_from_single_backend(query, limit, dense_docs, sparse_candidates)
        if ranked:
            return ranked
        return dedupe_ranked_documents(dense_docs, limit)

    # Default backend: faiss
    dense_docs = _normalize_docs(search_documents_v2(query, dense_k))
    sparse_candidates = _normalize_docs(list_faiss_documents(limit=sparse_scan_limit))
    ranked = _hybrid_from_single_backend(query, limit, dense_docs, sparse_candidates)
    if ranked:
        return ranked
    return dedupe_ranked_documents(dense_docs, limit)
