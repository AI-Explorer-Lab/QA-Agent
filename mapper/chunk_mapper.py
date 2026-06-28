"""Chunk mapper with pgvector and local_dev backend support."""

from __future__ import annotations

import json
import asyncio
from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import text

from database import get_async_session, get_storage_backend
from domain import Chunk
from mapper.local_dev_repository import LocalDevRepository


class ChunkMapper:
    def __init__(
        self,
        backend: str | None = None,
        database_url: str | None = None,
        local_repository: LocalDevRepository | None = None,
        embedding_dim: int = 1024,
    ) -> None:
        self.backend = (backend or get_storage_backend()).strip()
        self.local_repository = local_repository
        self.embedding_dim = max(1, int(embedding_dim))

        if self.backend == "local_dev":
            self.local_repository = local_repository or LocalDevRepository(database_url)
            self.database_url = database_url
        else:
            self.database_url = database_url

    @staticmethod
    def _now_utc() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _to_json(data: dict) -> str:
        return json.dumps(data, ensure_ascii=True, default=str)

    @staticmethod
    def _parse_vector_text(vector_text: str) -> list[float]:
        if not vector_text:
            return []
        raw = vector_text.strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        if not raw:
            return []
        result: list[float] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                result.append(float(part))
            except ValueError:
                continue
        return result

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        normalized = [float(value) for value in embedding]
        if len(normalized) >= self.embedding_dim:
            return normalized[: self.embedding_dim]
        return normalized + [0.0] * (self.embedding_dim - len(normalized))

    @staticmethod
    def _to_vector_literal(embedding: list[float]) -> str:
        return "[" + ",".join(str(value) for value in embedding) + "]"

    async def upsert_chunks(self, chunks: Iterable[Chunk]) -> int:
        records = list(chunks)
        if not records:
            return 0

        if self.backend == "local_dev":
            assert self.local_repository is not None
            return await asyncio.to_thread(self.local_repository.upsert_chunks, records)

        statement = text(
            """
            INSERT INTO pdf_chunks (
                chunk_id,
                doc_id,
                collection_name,
                doc_source,
                page_idx,
                page_range,
                chunk_type,
                chunk_index,
                heading_path,
                level1_title,
                level2_title,
                level3_title,
                table_id,
                sub_table_id,
                table_header_text,
                table_context_text,
                search_text,
                content,
                metadata_json,
                embedding,
                created_at,
                updated_at
            ) VALUES (
                :chunk_id,
                :doc_id,
                :collection_name,
                :doc_source,
                :page_idx,
                :page_range,
                :chunk_type,
                :chunk_index,
                :heading_path,
                :level1_title,
                :level2_title,
                :level3_title,
                :table_id,
                :sub_table_id,
                :table_header_text,
                :table_context_text,
                :search_text,
                :content,
                CAST(:metadata_json AS JSONB),
                CAST(:embedding_literal AS vector),
                :created_at,
                :updated_at
            )
            ON CONFLICT (chunk_id) DO UPDATE SET
                doc_id=EXCLUDED.doc_id,
                collection_name=EXCLUDED.collection_name,
                doc_source=EXCLUDED.doc_source,
                page_idx=EXCLUDED.page_idx,
                page_range=EXCLUDED.page_range,
                chunk_type=EXCLUDED.chunk_type,
                chunk_index=EXCLUDED.chunk_index,
                heading_path=EXCLUDED.heading_path,
                level1_title=EXCLUDED.level1_title,
                level2_title=EXCLUDED.level2_title,
                level3_title=EXCLUDED.level3_title,
                table_id=EXCLUDED.table_id,
                sub_table_id=EXCLUDED.sub_table_id,
                table_header_text=EXCLUDED.table_header_text,
                table_context_text=EXCLUDED.table_context_text,
                search_text=EXCLUDED.search_text,
                content=EXCLUDED.content,
                metadata_json=EXCLUDED.metadata_json,
                embedding=EXCLUDED.embedding,
                updated_at=EXCLUDED.updated_at
            """
        )

        now_utc = self._now_utc()
        async with get_async_session(backend="pgvector", database_url=self.database_url) as session:
            for chunk in records:
                payload = chunk.model_dump(mode="json")
                normalized_embedding = self._normalize_embedding(payload.get("embedding", []))
                search_text = str(payload.get("search_text") or payload.get("content", ""))

                await session.execute(
                    statement,
                    {
                        "chunk_id": payload["chunk_id"],
                        "doc_id": payload["doc_id"],
                        "collection_name": payload["collection_name"],
                        "doc_source": payload["doc_source"],
                        "page_idx": payload.get("page_idx"),
                        "page_range": payload.get("page_range", ""),
                        "chunk_type": payload.get("chunk_type", "text"),
                        "chunk_index": int(payload.get("chunk_index", 0)),
                        "heading_path": payload.get("heading_path", ""),
                        "level1_title": payload.get("level1_title", ""),
                        "level2_title": payload.get("level2_title", ""),
                        "level3_title": payload.get("level3_title", ""),
                        "table_id": payload.get("table_id", ""),
                        "sub_table_id": payload.get("sub_table_id", ""),
                        "table_header_text": payload.get("table_header_text", ""),
                        "table_context_text": payload.get("table_context_text", ""),
                        "search_text": search_text,
                        "content": payload.get("content", ""),
                        "metadata_json": self._to_json(payload.get("metadata", {})),
                        "embedding_literal": self._to_vector_literal(normalized_embedding),
                        "created_at": payload.get("created_at") or now_utc,
                        "updated_at": payload.get("updated_at") or now_utc,
                    },
                )
            await session.commit()
        return len(records)

    async def upsert_chunk(self, chunk: Chunk) -> None:
        await self.upsert_chunks([chunk])

    async def list_by_collection(
        self,
        collection_name: str,
        doc_id: str | None = None,
        limit: int = 1000,
    ) -> list[Chunk]:
        if self.backend == "local_dev":
            assert self.local_repository is not None
            return await asyncio.to_thread(
                self.local_repository.list_chunks_by_collection,
                collection_name,
                doc_id,
                limit,
            )

        if doc_id:
            query = text(
                """
                SELECT
                    chunk_id,
                    doc_id,
                    collection_name,
                    doc_source,
                    page_idx,
                    page_range,
                    chunk_type,
                    chunk_index,
                    heading_path,
                    level1_title,
                    level2_title,
                    level3_title,
                    table_id,
                    sub_table_id,
                    table_header_text,
                    table_context_text,
                    search_text,
                    content,
                    metadata_json,
                    embedding::text AS embedding_text,
                    created_at,
                    updated_at
                FROM pdf_chunks
                WHERE collection_name = :collection_name
                  AND doc_id = :doc_id
                ORDER BY chunk_index ASC
                LIMIT :limit
                """
            )
            params = {
                "collection_name": collection_name,
                "doc_id": doc_id,
                "limit": max(1, int(limit)),
            }
        else:
            query = text(
                """
                SELECT
                    chunk_id,
                    doc_id,
                    collection_name,
                    doc_source,
                    page_idx,
                    page_range,
                    chunk_type,
                    chunk_index,
                    heading_path,
                    level1_title,
                    level2_title,
                    level3_title,
                    table_id,
                    sub_table_id,
                    table_header_text,
                    table_context_text,
                    search_text,
                    content,
                    metadata_json,
                    embedding::text AS embedding_text,
                    created_at,
                    updated_at
                FROM pdf_chunks
                WHERE collection_name = :collection_name
                ORDER BY doc_id ASC, chunk_index ASC
                LIMIT :limit
                """
            )
            params = {
                "collection_name": collection_name,
                "limit": max(1, int(limit)),
            }

        async with get_async_session(backend="pgvector", database_url=self.database_url) as session:
            rows = (await session.execute(query, params)).mappings().all()

        chunks: list[Chunk] = []
        for row in rows:
            chunks.append(
                Chunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    collection_name=row["collection_name"],
                    doc_source=row["doc_source"],
                    page_idx=row["page_idx"],
                    page_range=row["page_range"],
                    chunk_type=row["chunk_type"],
                    chunk_index=int(row["chunk_index"] or 0),
                    heading_path=row["heading_path"],
                    level1_title=row["level1_title"],
                    level2_title=row["level2_title"],
                    level3_title=row["level3_title"],
                    table_id=row["table_id"],
                    sub_table_id=row["sub_table_id"],
                    table_header_text=row["table_header_text"],
                    table_context_text=row["table_context_text"],
                    search_text=row["search_text"],
                    content=row["content"],
                    metadata=row["metadata_json"] or {},
                    embedding=self._parse_vector_text(row["embedding_text"] or ""),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
        return chunks
