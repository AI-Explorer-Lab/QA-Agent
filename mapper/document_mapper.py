"""Document mapper with pgvector and local_dev backend support."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Iterable, Optional

from sqlalchemy import text

from database import get_sqlalchemy_engine, get_storage_backend
from domain import Document
from mapper.local_dev_repository import LocalDevRepository


class DocumentMapper:
    def __init__(
        self,
        backend: Optional[str] = None,
        database_url: Optional[str] = None,
        local_repository: Optional[LocalDevRepository] = None,
    ) -> None:
        self.backend = (backend or get_storage_backend()).strip()
        self.local_repository = local_repository

        if self.backend == "local_dev":
            self.local_repository = local_repository or LocalDevRepository(database_url)
            self.engine = None
        else:
            self.engine = get_sqlalchemy_engine(backend="pgvector", database_url=database_url)

    @staticmethod
    def _now_utc() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _to_json(data: dict) -> str:
        return json.dumps(data, ensure_ascii=True, default=str)

    def upsert_documents(self, documents: Iterable[Document]) -> int:
        records = list(documents)
        if not records:
            return 0

        if self.backend == "local_dev":
            assert self.local_repository is not None
            return self.local_repository.upsert_documents(records)

        statement = text(
            """
            INSERT INTO pdf_documents (
                doc_id,
                collection_name,
                doc_source,
                title,
                doc_hash,
                page_count,
                metadata_json,
                indexed_at,
                created_at,
                updated_at
            ) VALUES (
                :doc_id,
                :collection_name,
                :doc_source,
                :title,
                :doc_hash,
                :page_count,
                CAST(:metadata_json AS JSONB),
                :indexed_at,
                :created_at,
                :updated_at
            )
            ON CONFLICT (doc_id) DO UPDATE SET
                collection_name=EXCLUDED.collection_name,
                doc_source=EXCLUDED.doc_source,
                title=EXCLUDED.title,
                doc_hash=EXCLUDED.doc_hash,
                page_count=EXCLUDED.page_count,
                metadata_json=EXCLUDED.metadata_json,
                indexed_at=EXCLUDED.indexed_at,
                updated_at=EXCLUDED.updated_at
            """
        )

        now_utc = self._now_utc()
        with self.engine.begin() as connection:
            for document in records:
                payload = document.model_dump(mode="json")
                connection.execute(
                    statement,
                    {
                        "doc_id": payload["doc_id"],
                        "collection_name": payload["collection_name"],
                        "doc_source": payload["doc_source"],
                        "title": payload.get("title", ""),
                        "doc_hash": payload.get("doc_hash", ""),
                        "page_count": int(payload.get("page_count", 0)),
                        "metadata_json": self._to_json(payload.get("metadata", {})),
                        "indexed_at": payload.get("indexed_at") or now_utc,
                        "created_at": payload.get("created_at") or now_utc,
                        "updated_at": payload.get("updated_at") or now_utc,
                    },
                )
        return len(records)

    def upsert_document(self, document: Document) -> None:
        self.upsert_documents([document])

    def list_by_collection(self, collection_name: str, limit: int = 1000) -> list[Document]:
        if self.backend == "local_dev":
            assert self.local_repository is not None
            return self.local_repository.list_documents_by_collection(collection_name, limit=limit)

        query = text(
            """
            SELECT
                doc_id,
                collection_name,
                doc_source,
                title,
                doc_hash,
                page_count,
                metadata_json,
                indexed_at,
                created_at,
                updated_at
            FROM pdf_documents
            WHERE collection_name = :collection_name
            ORDER BY updated_at DESC
            LIMIT :limit
            """
        )

        with self.engine.begin() as connection:
            rows = connection.execute(
                query,
                {"collection_name": collection_name, "limit": max(1, int(limit))},
            ).mappings().all()

        return [
            Document(
                doc_id=row["doc_id"],
                collection_name=row["collection_name"],
                doc_source=row["doc_source"],
                title=row["title"],
                doc_hash=row["doc_hash"],
                page_count=int(row["page_count"] or 0),
                metadata=row["metadata_json"] or {},
                indexed_at=row["indexed_at"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def get_by_doc_id(self, doc_id: str) -> Optional[Document]:
        if self.backend == "local_dev":
            assert self.local_repository is not None
            connection = self.local_repository._connect()
            try:
                cursor = connection.cursor()
                cursor.execute(
                    """
                    SELECT
                        doc_id,
                        collection_name,
                        doc_source,
                        title,
                        doc_hash,
                        page_count,
                        metadata_json,
                        indexed_at,
                        created_at,
                        updated_at
                    FROM pdf_documents
                    WHERE doc_id = ?
                    LIMIT 1
                    """,
                    (doc_id,),
                )
                row = cursor.fetchone()
            finally:
                connection.close()

            if row is None:
                return None

            return Document(
                doc_id=row["doc_id"],
                collection_name=row["collection_name"],
                doc_source=row["doc_source"],
                title=row["title"],
                doc_hash=row["doc_hash"],
                page_count=int(row["page_count"] or 0),
                metadata=self.local_repository._from_json(row["metadata_json"], {}),
                indexed_at=row["indexed_at"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

        query = text(
            """
            SELECT
                doc_id,
                collection_name,
                doc_source,
                title,
                doc_hash,
                page_count,
                metadata_json,
                indexed_at,
                created_at,
                updated_at
            FROM pdf_documents
            WHERE doc_id = :doc_id
            LIMIT 1
            """
        )

        with self.engine.begin() as connection:
            row = connection.execute(query, {"doc_id": doc_id}).mappings().first()

        if row is None:
            return None

        return Document(
            doc_id=row["doc_id"],
            collection_name=row["collection_name"],
            doc_source=row["doc_source"],
            title=row["title"],
            doc_hash=row["doc_hash"],
            page_count=int(row["page_count"] or 0),
            metadata=row["metadata_json"] or {},
            indexed_at=row["indexed_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
