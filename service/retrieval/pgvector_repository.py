from __future__ import annotations

import hashlib
import asyncio
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from .sparse_retriever import SparseBM25Retriever, coarse_tokenize


from sqlalchemy import text

from database import get_async_session
from database.init_db import init_pgvector_schema


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



def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _nullable_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _json_dumps(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), ensure_ascii=False, default=str)


def _vector_literal(values: Sequence[float], dim: int) -> str:
    vector = _normalize_vector([_as_float(value) for value in values], dim)
    return "[" + ",".join(f"{value:.10f}" for value in vector) + "]"


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
        self._schema_ready = False
        self.revision = 0

        if local_chunks:
            self._upsert_chunks_local([self._prepare_chunk(source) for source in local_chunks])

    async def upsert_chunks(self, chunks: Iterable[Mapping[str, Any]]) -> int:
        prepared = [self._prepare_chunk(source) for source in chunks]
        if not prepared:
            return 0
        if self.backend == "pgvector":
            count = await self._upsert_chunks_pgvector(prepared)
        elif self.backend == "local_dev":
            count = self._upsert_chunks_local(prepared)
        else:
            raise RuntimeError(f"Unsupported repository backend: {self.backend}")
        if count:
            self.revision += 1
        return count

    async def get_latest_document_by_source(self, collection_name: str, doc_source: str) -> Dict[str, Any] | None:
        collection = _clean_text(collection_name)
        source = _clean_text(doc_source)
        if not collection or not source:
            return None

        if self.backend == "pgvector":
            await self._ensure_schema_ready()
            sql = text(
                """
                SELECT
                    doc_id,
                    collection_name,
                    doc_source,
                    title,
                    doc_hash,
                    page_count,
                    indexed_at,
                    created_at,
                    updated_at,
                    metadata_json
                FROM pdf_documents
                WHERE collection_name = :collection_name
                  AND doc_source = :doc_source
                ORDER BY indexed_at DESC, id DESC
                LIMIT 1
                """
            )
            async with get_async_session(backend="pgvector", database_url=self.database_url) as session:
                record = (
                    await session.execute(
                    sql,
                    {"collection_name": collection, "doc_source": source},
                    )
                ).mappings().first()
            if record is None:
                return None
            payload = dict(record)
            metadata = payload.get("metadata_json") or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = {}
            payload["metadata_json"] = metadata
            return payload

        # local_dev fallback: best-effort scan in-memory chunks
        for row in reversed(self._local_chunks):
            if str(row.get("collection_name") or "") != collection:
                continue
            if str(row.get("doc_source") or "") != source:
                continue
            return {
                "doc_id": row.get("doc_id") or "",
                "collection_name": collection,
                "doc_source": source,
                "title": row.get("title") or "",
                "doc_hash": row.get("doc_hash") or "",
                "page_count": int(row.get("page_count") or 0),
                "indexed_at": "",
                "created_at": "",
                "updated_at": "",
                "metadata_json": row.get("metadata_json") or {},
            }
        return None

    async def delete_documents_by_source(self, collection_name: str, doc_source: str) -> int:
        collection = _clean_text(collection_name)
        source = _clean_text(doc_source)
        if not collection or not source:
            return 0

        if self.backend == "pgvector":
            await self._ensure_schema_ready()
            sql = text(
                "DELETE FROM pdf_documents WHERE collection_name = :collection_name AND doc_source = :doc_source"
            )
            async with get_async_session(backend="pgvector", database_url=self.database_url) as session:
                result = await session.execute(
                    sql,
                    {"collection_name": collection, "doc_source": source},
                )
                await session.commit()
                return int(getattr(result, "rowcount", 0) or 0)

        if self.backend == "local_dev":
            before = len(self._local_chunks)
            self._local_chunks = [
                row
                for row in self._local_chunks
                if not (
                    str(row.get("collection_name") or "") == collection
                    and str(row.get("doc_source") or "") == source
                )
            ]
            self._sparse_retriever.index_chunks(self._local_chunks)
            return before - len(self._local_chunks)

        raise RuntimeError(f"Unsupported repository backend: {self.backend}")

    async def replace_collection_chunks(self, collection_name: str, chunks: Iterable[Mapping[str, Any]]) -> int:
        collection = _clean_text(collection_name) or "default"
        prepared = [self._prepare_chunk(source) for source in chunks]
        for row in prepared:
            row["collection_name"] = collection
        if self.backend == "pgvector":
            count = await self._replace_collection_pgvector(collection, prepared)
        elif self.backend == "local_dev":
            self._delete_collection_local(collection)
            count = self._upsert_chunks_local(prepared)
        else:
            raise RuntimeError(f"Unsupported repository backend: {self.backend}")
        self.revision += 1
        return count

    async def delete_collection(self, collection_name: str) -> int:
        collection = _clean_text(collection_name)
        if not collection:
            raise ValueError("collection_name is required when deleting indexed chunks.")
        if self.backend == "pgvector":
            count = await self._delete_collection_pgvector(collection)
        elif self.backend == "local_dev":
            count = self._delete_collection_local(collection)
        else:
            raise RuntimeError(f"Unsupported repository backend: {self.backend}")
        self.revision += 1
        return count

    def clear_local(self) -> None:
        self._local_chunks = []
        self._sparse_retriever.index_chunks([])
        self.revision += 1

    async def count_collection_chunks(self, collection_name: str = "") -> int:
        collection = _clean_text(collection_name)
        if self.backend == "pgvector":
            await self._ensure_schema_ready()
            sql = "SELECT COUNT(*) FROM pdf_chunks"
            params: Dict[str, Any] = {}
            if collection:
                sql += " WHERE collection_name = :collection_name"
                params["collection_name"] = collection
            async with get_async_session(backend="pgvector", database_url=self.database_url) as session:
                return int((await session.execute(text(sql), params)).scalar() or 0)
        if not collection:
            return len(self._local_chunks)
        return len([row for row in self._local_chunks if str(row.get("collection_name") or "") == collection])

    def list_local_chunks(self, collection_name: str = "") -> list[Dict[str, Any]]:
        if not collection_name:
            return [dict(item) for item in self._local_chunks]
        return [
            dict(item)
            for item in self._local_chunks
            if str(item.get("collection_name") or "") == collection_name
        ]

    async def dense_search(
        self,
        collection_name: str,
        query_embedding: Sequence[float] | None,
        top_k: int,
        query_text: str = "",
        chunk_type: str | None = None,
    ) -> list[Dict[str, Any]]:
        if self.backend == "pgvector":
            return await self._dense_search_pgvector(
                collection_name=collection_name,
                query_embedding=query_embedding,
                top_k=top_k,
                query_text=query_text,
                chunk_type=chunk_type,
            )

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

    async def keyword_search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int,
        chunk_type: str | None = None,
        table_only: bool = False,
    ) -> list[Dict[str, Any]]:
        effective_chunk_type = "table" if table_only else chunk_type

        if self.backend == "pgvector":
            return await self._keyword_search_pgvector(
                collection_name=collection_name,
                query_text=query_text,
                top_k=top_k,
                chunk_type=effective_chunk_type,
            )

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

    async def keyword_search_many(
        self,
        collection_name: str,
        query_texts: Sequence[str],
        top_k: int,
        chunk_type: str | None = None,
        table_only: bool = False,
    ) -> Dict[str, list[Dict[str, Any]]]:
        effective_chunk_type = "table" if table_only else chunk_type
        queries = [str(query or "").strip() for query in query_texts if coarse_tokenize(str(query or ""))]
        if not queries:
            return {}

        if self.backend == "pgvector":
            candidates = await self._keyword_candidates_pgvector(
                collection_name=collection_name,
                top_k=top_k,
                chunk_type=effective_chunk_type,
            )
        else:
            candidates = []
            for chunk in self._local_chunks:
                if collection_name and str(chunk.get("collection_name") or "") != collection_name:
                    continue
                if effective_chunk_type and str(chunk.get("chunk_type") or "") != effective_chunk_type:
                    continue
                candidates.append(dict(chunk))

        retriever = SparseBM25Retriever(k1=self._sparse_retriever.k1, b=self._sparse_retriever.b)
        retriever.index_chunks(candidates)
        return {
            query: retriever.search(
                query=query,
                top_k=top_k,
                collection_name=collection_name,
                chunk_type=effective_chunk_type,
            )
            for query in queries
        }

    async def table_search(self, collection_name: str, query_text: str, top_k: int) -> list[Dict[str, Any]]:
        return await self.keyword_search(
            collection_name=collection_name,
            query_text=query_text,
            top_k=top_k,
            chunk_type="table",
            table_only=True,
        )

    def _prepare_chunk(self, source: Mapping[str, Any]) -> Dict[str, Any]:
        chunk = dict(source)
        metadata = dict(chunk.get("metadata") or {}) if isinstance(chunk.get("metadata"), Mapping) else {}
        content = _clean_text(chunk.get("raw_doc") or chunk.get("content"))
        collection_name = _clean_text(chunk.get("collection_name")) or "default"
        doc_source = _clean_text(chunk.get("doc_source") or chunk.get("source"))
        doc_id = _clean_text(chunk.get("doc_id"))
        if not doc_id:
            basis = "|".join([collection_name, doc_source, content[:500]])
            doc_id = "doc_" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]
        chunk_id = _clean_text(chunk.get("chunk_id"))
        if not chunk_id:
            basis = "|".join([doc_id, content, str(chunk.get("chunk_index") or "")])
            chunk_id = "chunk_" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]

        chunk_type = _clean_text(chunk.get("chunk_type") or metadata.get("chunk_type")) or "text"
        page_idx = _nullable_int(chunk.get("page_idx") if chunk.get("page_idx") is not None else metadata.get("page_idx"))
        chunk_index = _as_int(chunk.get("chunk_index") if chunk.get("chunk_index") is not None else metadata.get("chunk_index"), 0)
        page_range = _clean_text(chunk.get("page_range") or metadata.get("page_range"))
        heading_path = _clean_text(chunk.get("heading_path") or metadata.get("heading_path"))
        level1_title = _clean_text(chunk.get("level1_title") or metadata.get("level1_title"))
        level2_title = _clean_text(chunk.get("level2_title") or metadata.get("level2_title"))
        level3_title = _clean_text(chunk.get("level3_title") or metadata.get("level3_title"))
        table_id = _clean_text(chunk.get("table_id") or metadata.get("table_id"))
        sub_table_id = _clean_text(chunk.get("sub_table_id") or metadata.get("sub_table_id"))
        table_header_text = _clean_text(chunk.get("table_header_text") or metadata.get("table_header_text"))
        table_context_text = _clean_text(chunk.get("table_context_text") or metadata.get("table_context_text"))
        search_text = "\n".join(part for part in [content, heading_path, table_header_text, table_context_text] if part)

        embedding = chunk.get("embedding")
        if isinstance(embedding, Sequence) and not isinstance(embedding, (str, bytes)):
            vector = _normalize_vector([_as_float(item) for item in embedding], self.embedding_dim)
        else:
            vector = deterministic_embedding(content, self.embedding_dim)

        row = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "collection_name": collection_name,
            "doc_source": doc_source,
            "title": _clean_text(chunk.get("title")) or (Path(doc_source).stem if doc_source else ""),
            "doc_hash": _clean_text(chunk.get("doc_hash") or metadata.get("doc_hash")),
            "page_count": _as_int(chunk.get("page_count") or metadata.get("page_count"), 0),
            "page_idx": page_idx,
            "page_range": page_range,
            "chunk_type": chunk_type,
            "chunk_index": chunk_index,
            "heading_path": heading_path,
            "level1_title": level1_title,
            "level2_title": level2_title,
            "level3_title": level3_title,
            "table_id": table_id,
            "sub_table_id": sub_table_id,
            "table_header_text": table_header_text,
            "table_context_text": table_context_text,
            "search_text": search_text or content,
            "content": content,
            "raw_doc": content,
            "embedding": vector,
            "source_channels": list(chunk.get("source_channels") or []),
        }
        metadata.update({key: value for key, value in row.items() if key != "embedding"})
        row["metadata_json"] = metadata
        return row

    def _upsert_chunks_local(self, chunks: Iterable[Mapping[str, Any]]) -> int:
        indexed: Dict[str, Dict[str, Any]] = {str(chunk.get("chunk_id") or ""): dict(chunk) for chunk in self._local_chunks}
        count = 0
        for source in chunks:
            chunk = dict(source)
            indexed[str(chunk.get("chunk_id") or "")] = chunk
            count += 1
        self._local_chunks = list(indexed.values())
        self._sparse_retriever.index_chunks(self._local_chunks)
        return count

    def _delete_collection_local(self, collection_name: str) -> int:
        before = len(self._local_chunks)
        self._local_chunks = [row for row in self._local_chunks if str(row.get("collection_name") or "") != collection_name]
        self._sparse_retriever.index_chunks(self._local_chunks)
        return before - len(self._local_chunks)

    async def _ensure_schema_ready(self) -> None:
        if self.backend != "pgvector":
            raise RuntimeError("PG session requested for a non-pgvector repository.")
        if not self.database_url:
            raise RuntimeError("PGVECTOR_DATABASE_URL is empty. Configure storage.pgvector.database_url in config/app.yaml.")
        if not self._schema_ready:
            await init_pgvector_schema(database_url=self.database_url)
            self._schema_ready = True

    async def _upsert_chunks_pgvector(self, chunks: Sequence[Mapping[str, Any]]) -> int:
        await self._ensure_schema_ready()
        doc_sql = text("""
            INSERT INTO pdf_documents (doc_id, collection_name, doc_source, title, doc_hash, page_count, metadata_json, indexed_at, created_at, updated_at)
            VALUES (:doc_id, :collection_name, :doc_source, :title, :doc_hash, :page_count, CAST(:metadata_json AS jsonb), NOW(), NOW(), NOW())
            ON CONFLICT (doc_id) DO UPDATE SET
                collection_name = EXCLUDED.collection_name,
                doc_source = EXCLUDED.doc_source,
                title = EXCLUDED.title,
                doc_hash = EXCLUDED.doc_hash,
                page_count = GREATEST(pdf_documents.page_count, EXCLUDED.page_count),
                metadata_json = EXCLUDED.metadata_json,
                indexed_at = NOW(),
                updated_at = NOW()
        """)
        chunk_sql = text("""
            INSERT INTO pdf_chunks (
                chunk_id, doc_id, collection_name, doc_source, page_idx, page_range, chunk_type, chunk_index,
                heading_path, level1_title, level2_title, level3_title,
                table_id, sub_table_id, table_header_text, table_context_text,
                search_text, content, metadata_json, embedding, created_at, updated_at
            ) VALUES (
                :chunk_id, :doc_id, :collection_name, :doc_source, :page_idx, :page_range, :chunk_type, :chunk_index,
                :heading_path, :level1_title, :level2_title, :level3_title,
                :table_id, :sub_table_id, :table_header_text, :table_context_text,
                :search_text, :content, CAST(:metadata_json AS jsonb), CAST(:embedding AS vector), NOW(), NOW()
            )
            ON CONFLICT (chunk_id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                collection_name = EXCLUDED.collection_name,
                doc_source = EXCLUDED.doc_source,
                page_idx = EXCLUDED.page_idx,
                page_range = EXCLUDED.page_range,
                chunk_type = EXCLUDED.chunk_type,
                chunk_index = EXCLUDED.chunk_index,
                heading_path = EXCLUDED.heading_path,
                level1_title = EXCLUDED.level1_title,
                level2_title = EXCLUDED.level2_title,
                level3_title = EXCLUDED.level3_title,
                table_id = EXCLUDED.table_id,
                sub_table_id = EXCLUDED.sub_table_id,
                table_header_text = EXCLUDED.table_header_text,
                table_context_text = EXCLUDED.table_context_text,
                search_text = EXCLUDED.search_text,
                content = EXCLUDED.content,
                metadata_json = EXCLUDED.metadata_json,
                embedding = EXCLUDED.embedding,
                updated_at = NOW()
        """)
        docs: Dict[str, Dict[str, Any]] = {}
        chunk_rows: list[Dict[str, Any]] = []
        for chunk in chunks:
            row = dict(chunk)
            page_count = _as_int(row.get("page_count"), 0)
            if page_count <= 0 and row.get("page_idx") is not None:
                page_count = _as_int(row.get("page_idx"), -1) + 1
            doc_payload = {
                "doc_id": row["doc_id"],
                "collection_name": row["collection_name"],
                "doc_source": row["doc_source"],
                "title": row.get("title") or "",
                "doc_hash": row.get("doc_hash") or "",
                "page_count": max(0, page_count),
                "metadata_json": _json_dumps({"doc_source": row["doc_source"], "doc_hash": row.get("doc_hash") or ""}),
            }
            current = docs.get(str(row["doc_id"]))
            if current is None or doc_payload["page_count"] > current["page_count"]:
                docs[str(row["doc_id"])] = doc_payload
            row["metadata_json"] = _json_dumps(row.get("metadata_json") or {})
            row["embedding"] = _vector_literal(row.get("embedding") or [], self.embedding_dim)
            chunk_rows.append(row)
        async with get_async_session(backend="pgvector", database_url=self.database_url) as session:
            for doc in docs.values():
                await session.execute(doc_sql, doc)
            for row in chunk_rows:
                await session.execute(chunk_sql, row)
            await session.commit()
        return len(chunk_rows)

    async def _replace_collection_pgvector(self, collection_name: str, chunks: Sequence[Mapping[str, Any]]) -> int:
        await self._ensure_schema_ready()
        async with get_async_session(backend="pgvector", database_url=self.database_url) as session:
            await session.execute(text("DELETE FROM pdf_chunks WHERE collection_name = :collection_name"), {"collection_name": collection_name})
            await session.execute(text("DELETE FROM pdf_documents WHERE collection_name = :collection_name"), {"collection_name": collection_name})
            await session.commit()
        if not chunks:
            return 0
        return await self._upsert_chunks_pgvector(chunks)

    async def _delete_collection_pgvector(self, collection_name: str) -> int:
        await self._ensure_schema_ready()
        async with get_async_session(backend="pgvector", database_url=self.database_url) as session:
            result = await session.execute(text("DELETE FROM pdf_chunks WHERE collection_name = :collection_name"), {"collection_name": collection_name})
            await session.execute(text("DELETE FROM pdf_documents WHERE collection_name = :collection_name"), {"collection_name": collection_name})
            await session.commit()
        return int(result.rowcount or 0)

    async def _dense_search_pgvector(
        self,
        collection_name: str,
        query_embedding: Sequence[float] | None,
        top_k: int,
        query_text: str,
        chunk_type: str | None,
    ) -> list[Dict[str, Any]]:
        if query_embedding is None:
            query_embedding = deterministic_embedding(query_text, self.embedding_dim)

        vector_literal = _vector_literal(query_embedding, self.embedding_dim)
        await self._ensure_schema_ready()

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
        async with get_async_session(backend="pgvector", database_url=self.database_url) as session:
            records = (
                await session.execute(
                    sql,
                    {
                        "query_vector": vector_literal,
                        "collection_name": collection_name,
                        "chunk_type": chunk_type or "",
                        "top_k": max(1, int(top_k)),
                    },
                )
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

    async def _keyword_search_pgvector(
        self,
        collection_name: str,
        query_text: str,
        top_k: int,
        chunk_type: str | None,
    ) -> list[Dict[str, Any]]:
        await self._ensure_schema_ready()

        if not coarse_tokenize(query_text):
            return []

        candidates = await self._keyword_candidates_pgvector(
            collection_name=collection_name,
            top_k=top_k,
            chunk_type=chunk_type,
        )

        retriever = SparseBM25Retriever(k1=self._sparse_retriever.k1, b=self._sparse_retriever.b)
        retriever.index_chunks(candidates)
        return retriever.search(
            query=query_text,
            top_k=top_k,
            collection_name=collection_name,
            chunk_type=chunk_type,
        )

    async def _keyword_candidates_pgvector(
        self,
        collection_name: str,
        top_k: int,
        chunk_type: str | None,
    ) -> list[Dict[str, Any]]:
        await self._ensure_schema_ready()

        sql = text(
            """
            SELECT
                chunk_id,
                doc_id,
                collection_name,
                doc_source,
                chunk_type,
                search_text,
                content AS raw_doc,
                metadata_json
            FROM pdf_chunks
            WHERE (:collection_name = '' OR collection_name = :collection_name)
              AND (:chunk_type = '' OR chunk_type = :chunk_type)
            ORDER BY id DESC
            LIMIT :scan_limit
            """
        )

        scan_limit = max(200, int(top_k) * 40)
        candidates: list[Dict[str, Any]] = []
        async with get_async_session(backend="pgvector", database_url=self.database_url) as session:
            records = (
                await session.execute(
                    sql,
                    {
                        "collection_name": collection_name,
                        "chunk_type": chunk_type or "",
                        "scan_limit": scan_limit,
                    },
                )
            ).mappings()

            for record in records:
                metadata = record.get("metadata_json") or {}
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except Exception:
                        metadata = {}
                content = str(record.get("raw_doc") or "")
                search_text = str(record.get("search_text") or content)
                candidates.append({
                    "chunk_id": record.get("chunk_id"),
                    "doc_id": record.get("doc_id") or "",
                    "collection_name": record.get("collection_name") or collection_name,
                    "doc_source": record.get("doc_source") or "",
                    "raw_doc": content,
                    "content": content,
                    "search_text": search_text,
                    "chunk_type": record.get("chunk_type") or metadata.get("chunk_type") or "text",
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
                })

        return candidates




