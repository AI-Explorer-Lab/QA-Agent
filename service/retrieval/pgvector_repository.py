from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Iterable, Mapping, Sequence

from .sparse_retriever import SparseBM25Retriever, coarse_tokenize
from .types import ensure_candidate_dict


try:  # pragma: no cover - optional runtime dependency for real pgvector mode.
    from sqlalchemy import create_engine, text
except Exception:  # pragma: no cover
    create_engine = None
    text = None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_vector(values: Sequence[float], dim: int) -> list[float]:
    vector = [0.0] * dim
    for index, value in enumerate(values[:dim]):
        vector[index] = _as_float(value)

    norm = math.sqrt(sum(number * number for number in vector))
    if norm > 0:
        vector = [number / norm for number in vector]
    return vector


def deterministic_embedding(text: str, dim: int = 1024) -> list[float]:
    if int(dim) != 1024:
        raise ValueError("Embedding dimension must be 1024 for this project.")

    tokens = coarse_tokenize(text)
    if not tokens:
        return [0.0] * dim

    vector = [0.0] * dim
    for token in tokens:
        digest = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        position = digest % dim
        sign = 1.0 if ((digest >> 8) & 1) == 0 else -1.0
        vector[position] += sign

    return _normalize_vector(vector, dim)


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0

    size = min(len(a), len(b))
    numerator = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for index in range(size):
        value_a = _as_float(a[index])
        value_b = _as_float(b[index])
        numerator += value_a * value_b
        norm_a += value_a * value_a
        norm_b += value_b * value_b

    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return numerator / math.sqrt(norm_a * norm_b)


class PgvectorRepository:
    """
    Unified repository interface:
    - default backend is pgvector (real SQL mode when database_url is configured)
    - local_dev backend keeps everything in memory for tests
    """

    def __init__(
        self,
        backend: str = "pgvector",
        database_url: str = "",
        embedding_dim: int = 1024,
        local_chunks: Iterable[Mapping[str, Any]] | None = None,
    ) -> None:
        if int(embedding_dim) != 1024:
            raise ValueError("embedding_dim must be 1024.")

        self.embedding_dim = 1024
        self.backend = str(backend or "pgvector").strip().lower()
        self.database_url = str(database_url or "").strip()
        self._local_chunks: list[Dict[str, Any]] = []
        self._sparse_retriever = SparseBM25Retriever()
        self._engine = None

        if local_chunks:
            self.upsert_chunks(local_chunks)

    def upsert_chunks(self, chunks: Iterable[Mapping[str, Any]]) -> int:
        indexed: Dict[str, Dict[str, Any]] = {str(chunk.get("chunk_id") or ""): dict(chunk) for chunk in self._local_chunks}

        count = 0
        for source in chunks:
            chunk = ensure_candidate_dict(source)
            chunk_id = str(chunk.get("chunk_id") or "").strip()
            if not chunk_id:
                basis = str(chunk.get("raw_doc") or chunk.get("content") or count)
                chunk_id = "chunk-" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]
                chunk["chunk_id"] = chunk_id

            chunk.setdefault("collection_name", "default")
            chunk.setdefault("raw_doc", str(chunk.get("raw_doc") or chunk.get("content") or ""))
            chunk.setdefault("chunk_type", str(chunk.get("chunk_type") or "text"))

            embedding = chunk.get("embedding")
            if isinstance(embedding, Sequence) and not isinstance(embedding, (str, bytes)):
                vector = _normalize_vector([_as_float(item) for item in embedding], self.embedding_dim)
            else:
                vector = deterministic_embedding(str(chunk.get("raw_doc") or ""), self.embedding_dim)
            chunk["embedding"] = vector

            indexed[chunk_id] = chunk
            count += 1

        self._local_chunks = list(indexed.values())
        self._sparse_retriever.index_chunks(self._local_chunks)
        return count

    def list_local_chunks(self, collection_name: str = "") -> list[Dict[str, Any]]:
        if not collection_name:
            return [dict(item) for item in self._local_chunks]
        return [
            dict(item)
            for item in self._local_chunks
            if str(item.get("collection_name") or "") == collection_name
        ]

    def dense_search(
        self,
        collection_name: str,
        query_embedding: Sequence[float] | None,
        top_k: int,
        query_text: str = "",
        chunk_type: str | None = None,
    ) -> list[Dict[str, Any]]:
        if self.backend == "pgvector" and self.database_url:
            try:
                return self._dense_search_pgvector(
                    collection_name=collection_name,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    query_text=query_text,
                    chunk_type=chunk_type,
                )
            except Exception:
                pass

        embedding = query_embedding
        if embedding is None:
            embedding = deterministic_embedding(query_text, self.embedding_dim)
        normalized_query = _normalize_vector([_as_float(item) for item in embedding], self.embedding_dim)

        rows: list[Dict[str, Any]] = []
        for chunk in self._local_chunks:
            if collection_name and str(chunk.get("collection_name") or "") != collection_name:
                continue
            if chunk_type and str(chunk.get("chunk_type") or "") != chunk_type:
                continue

            score = _cosine_similarity(normalized_query, chunk.get("embedding") or [])
            payload = dict(chunk)
            payload["dense_score"] = max(0.0, min(1.0, score))
            payload["similarity"] = payload["dense_score"]
            payload["score"] = payload["dense_score"]
            rows.append(payload)

        rows.sort(key=lambda item: float(item.get("dense_score") or 0.0), reverse=True)
        return rows[: max(1, int(top_k))]

    def keyword_search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int,
        chunk_type: str | None = None,
        table_only: bool = False,
    ) -> list[Dict[str, Any]]:
        effective_chunk_type = "table" if table_only else chunk_type

        if self.backend == "pgvector" and self.database_url:
            try:
                return self._keyword_search_pgvector(
                    collection_name=collection_name,
                    query_text=query_text,
                    top_k=top_k,
                    chunk_type=effective_chunk_type,
                )
            except Exception:
                pass

        filtered = []
        for chunk in self._local_chunks:
            if collection_name and str(chunk.get("collection_name") or "") != collection_name:
                continue
            if effective_chunk_type and str(chunk.get("chunk_type") or "") != effective_chunk_type:
                continue
            filtered.append(chunk)

        local_retriever = SparseBM25Retriever(k1=self._sparse_retriever.k1, b=self._sparse_retriever.b)
        local_retriever.index_chunks(filtered)
        result = local_retriever.search(
            query=query_text,
            top_k=top_k,
            collection_name=collection_name,
            chunk_type=effective_chunk_type,
        )
        for row in result:
            row.setdefault("score", float(row.get("bm25_score") or 0.0))
        return result

    def table_search(self, collection_name: str, query_text: str, top_k: int) -> list[Dict[str, Any]]:
        return self.keyword_search(
            collection_name=collection_name,
            query_text=query_text,
            top_k=top_k,
            chunk_type="table",
            table_only=True,
        )

    def _dense_search_pgvector(
        self,
        collection_name: str,
        query_embedding: Sequence[float] | None,
        top_k: int,
        query_text: str,
        chunk_type: str | None,
    ) -> list[Dict[str, Any]]:
        if create_engine is None or text is None:
            raise RuntimeError("sqlalchemy is unavailable")

        if query_embedding is None:
            query_embedding = deterministic_embedding(query_text, self.embedding_dim)

        vector_literal = "[" + ",".join(f"{_as_float(value):.10f}" for value in query_embedding[: self.embedding_dim]) + "]"

        if self._engine is None:
            self._engine = create_engine(self.database_url, pool_pre_ping=True)

        sql = text(
            """
            SELECT
                chunk_id,
                doc_id,
                collection_name,
                doc_source,
                content AS raw_doc,
                metadata_json,
                1 - (embedding <=> CAST(:query_vector AS vector)) AS dense_score
            FROM pdf_chunks
            WHERE (:collection_name = '' OR collection_name = :collection_name)
              AND (:chunk_type = '' OR metadata_json ->> 'chunk_type' = :chunk_type)
            ORDER BY embedding <=> CAST(:query_vector AS vector)
            LIMIT :top_k
            """
        )

        rows: list[Dict[str, Any]] = []
        with self._engine.begin() as connection:
            records = connection.execute(
                sql,
                {
                    "query_vector": vector_literal,
                    "collection_name": collection_name,
                    "chunk_type": chunk_type or "",
                    "top_k": max(1, int(top_k)),
                },
            ).mappings()

            for record in records:
                metadata = record.get("metadata_json") or {}
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except Exception:
                        metadata = {}
                payload = {
                    "chunk_id": record.get("chunk_id"),
                    "doc_id": record.get("doc_id") or "",
                    "collection_name": record.get("collection_name") or collection_name,
                    "doc_source": record.get("doc_source") or "",
                    "raw_doc": record.get("raw_doc") or "",
                    "chunk_type": metadata.get("chunk_type") or "text",
                    "page_idx": metadata.get("page_idx"),
                    "chunk_index": metadata.get("chunk_index"),
                    "heading_path": metadata.get("heading_path") or "",
                    "level1_title": metadata.get("level1_title") or "",
                    "level2_title": metadata.get("level2_title") or "",
                    "level3_title": metadata.get("level3_title") or "",
                    "table_id": metadata.get("table_id") or "",
                    "sub_table_id": metadata.get("sub_table_id") or "",
                    "table_header_text": metadata.get("table_header_text") or "",
                    "table_context_text": metadata.get("table_context_text") or "",
                    "dense_score": max(0.0, min(1.0, _as_float(record.get("dense_score")))),
                }
                payload["similarity"] = payload["dense_score"]
                payload["score"] = payload["dense_score"]
                rows.append(payload)
        return rows

    def _keyword_search_pgvector(
        self,
        collection_name: str,
        query_text: str,
        top_k: int,
        chunk_type: str | None,
    ) -> list[Dict[str, Any]]:
        if create_engine is None or text is None:
            raise RuntimeError("sqlalchemy is unavailable")

        if self._engine is None:
            self._engine = create_engine(self.database_url, pool_pre_ping=True)

        query_tokens = coarse_tokenize(query_text)
        if not query_tokens:
            return []

        sql = text(
            """
            SELECT
                chunk_id,
                doc_id,
                collection_name,
                doc_source,
                content AS raw_doc,
                metadata_json,
                CASE
                    WHEN content ILIKE :query_pattern THEN 1.0
                    ELSE 0.0
                END AS lexical_score
            FROM pdf_chunks
            WHERE (:collection_name = '' OR collection_name = :collection_name)
              AND (:chunk_type = '' OR metadata_json ->> 'chunk_type' = :chunk_type)
              AND content ILIKE :query_pattern
            LIMIT :scan_limit
            """
        )

        scan_limit = max(50, int(top_k) * 6)
        rows: list[Dict[str, Any]] = []
        with self._engine.begin() as connection:
            records = connection.execute(
                sql,
                {
                    "query_pattern": f"%{query_text}%",
                    "collection_name": collection_name,
                    "chunk_type": chunk_type or "",
                    "scan_limit": scan_limit,
                },
            ).mappings()

            for record in records:
                metadata = record.get("metadata_json") or {}
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except Exception:
                        metadata = {}
                content = str(record.get("raw_doc") or "")
                content_tokens = set(coarse_tokenize(content))
                overlap = len(set(query_tokens) & content_tokens)
                score = overlap / max(1, len(set(query_tokens)))
                payload = {
                    "chunk_id": record.get("chunk_id"),
                    "doc_id": record.get("doc_id") or "",
                    "collection_name": record.get("collection_name") or collection_name,
                    "doc_source": record.get("doc_source") or "",
                    "raw_doc": content,
                    "chunk_type": metadata.get("chunk_type") or "text",
                    "page_idx": metadata.get("page_idx"),
                    "chunk_index": metadata.get("chunk_index"),
                    "heading_path": metadata.get("heading_path") or "",
                    "level1_title": metadata.get("level1_title") or "",
                    "level2_title": metadata.get("level2_title") or "",
                    "level3_title": metadata.get("level3_title") or "",
                    "table_id": metadata.get("table_id") or "",
                    "sub_table_id": metadata.get("sub_table_id") or "",
                    "table_header_text": metadata.get("table_header_text") or "",
                    "table_context_text": metadata.get("table_context_text") or "",
                    "bm25_score": max(0.0, min(1.0, score)),
                }
                payload["score"] = payload["bm25_score"]
                rows.append(payload)

        rows.sort(key=lambda item: float(item.get("bm25_score") or 0.0), reverse=True)
        return rows[: max(1, int(top_k))]
