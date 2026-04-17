import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text, create_engine, func, text
from sqlalchemy.orm import declarative_base, sessionmaker

from core.config_loader import load_runtime_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = PROJECT_ROOT / ".vendor"
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

try:
    from pgvector.sqlalchemy import Vector
except Exception:  # pragma: no cover
    Vector = None

from embedding_service.embedding_processor import embedding

logger = logging.getLogger(__name__)

Base = declarative_base()
load_runtime_env()

_REMOVED_TABLE_METADATA_FIELDS = {
    "table_anchor_text",
    "table_anchor_confidence",
    "table_context_chunk_id",
    "table_context_tokens",
    "table_context_absorbed",
}

_ENGINE = None
_ENGINE_URL = ""
_SESSION_LOCAL = None
_SESSION_URL = ""


def _safe_int(value: str, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _get_database_url() -> str:
    load_runtime_env()
    return os.getenv("PGVECTOR_DATABASE_URL", "").strip()


def _get_embedding_dim() -> int:
    load_runtime_env()
    return _safe_int(os.getenv("PGVECTOR_EMBEDDING_DIM", "1024"), 1024)


if Vector is not None:

    class PgVectorChunk(Base):
        __tablename__ = "rag_chunks"

        id = Column(Integer, primary_key=True, autoincrement=True)
        chunk_id = Column(String(255), unique=True, index=True, nullable=False)
        doc_id = Column(String(255), index=True, nullable=False)
        doc_source = Column(String(1024), nullable=False)
        content = Column(Text, nullable=False)
        metadata_json = Column(JSON, default=dict)
        embedding = Column(Vector(_get_embedding_dim()), nullable=False)
        created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def _get_engine():
    global _ENGINE, _ENGINE_URL
    database_url = _get_database_url()
    if not database_url:
        return None

    if _ENGINE is not None and _ENGINE_URL == database_url:
        return _ENGINE

    try:
        _ENGINE = create_engine(database_url, pool_pre_ping=True)
        _ENGINE_URL = database_url
        return _ENGINE
    except Exception as exc:
        logger.warning("PGVector engine initialization skipped: %s", exc)
        _ENGINE = None
        _ENGINE_URL = ""
        return None


def _get_session_local():
    global _SESSION_LOCAL, _SESSION_URL
    engine = _get_engine()
    if engine is None:
        return None

    if _SESSION_LOCAL is not None and _SESSION_URL == _ENGINE_URL:
        return _SESSION_LOCAL

    _SESSION_LOCAL = sessionmaker(bind=engine)
    _SESSION_URL = _ENGINE_URL
    return _SESSION_LOCAL


def init_pgvector_schema() -> bool:
    if Vector is None:
        logger.warning("pgvector package is unavailable.")
        return False

    engine = _get_engine()
    if engine is None:
        logger.warning("PGVector database URL is empty or connection failed.")
        return False

    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(bind=engine)
        return True
    except Exception as exc:
        logger.error("Failed to init pgvector schema: %s", exc)
        return False


def _sanitize_chunk_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {k: v for k, v in chunk.items() if k != "content"}
    for field in _REMOVED_TABLE_METADATA_FIELDS:
        metadata.pop(field, None)
    return metadata


def upsert_chunks_to_pgvector(chunks: List[Dict[str, Any]]) -> int:
    if Vector is None:
        return 0

    session_local = _get_session_local()
    if session_local is None:
        return 0

    from sqlalchemy.dialects.postgresql import insert

    if not chunks:
        return 0

    inserted = 0
    session = session_local()
    try:
        # Auto replace-mode by doc_source:
        # - if doc_source exists in DB, delete old rows of that source first
        # - then insert fresh chunks for this source
        grouped_by_source: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in chunks:
            source = str(chunk.get("doc_source", chunk.get("source", "")) or "").strip()
            grouped_by_source.setdefault(source, []).append(chunk)

        for source, source_chunks in grouped_by_source.items():
            if not source:
                continue
            existing_count = (
                session.query(func.count(PgVectorChunk.id))
                .filter(PgVectorChunk.doc_source == source)
                .scalar()
                or 0
            )
            if existing_count > 0:
                deleted = (
                    session.query(PgVectorChunk)
                    .filter(PgVectorChunk.doc_source == source)
                    .delete(synchronize_session=False)
                )
                logger.info(
                    "PGVector replace mode enabled: doc_source=%s existing=%s deleted=%s incoming=%s",
                    source,
                    existing_count,
                    deleted,
                    len(source_chunks),
                )

        for chunk in chunks:
            content = str(chunk.get("content", "")).strip()
            if not content:
                continue
            vector = embedding(content)
            if vector is None:
                continue

            metadata = _sanitize_chunk_metadata(chunk)
            stmt = insert(PgVectorChunk).values(
                chunk_id=chunk.get("chunk_id"),
                doc_id=chunk.get("doc_id"),
                doc_source=str(chunk.get("doc_source", chunk.get("source", "")) or ""),
                content=content,
                metadata_json=metadata,
                embedding=vector,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[PgVectorChunk.chunk_id],
                set_={
                    "doc_id": stmt.excluded.doc_id,
                    "doc_source": stmt.excluded.doc_source,
                    "content": stmt.excluded.content,
                    "metadata_json": stmt.excluded.metadata_json,
                    "embedding": stmt.excluded.embedding,
                    "created_at": func.now(),
                },
            )
            session.execute(stmt)
            inserted += 1
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("Failed to upsert pgvector chunks: %s", exc)
        inserted = 0
    finally:
        session.close()
    return inserted


def _format_pgvector_row(
    row: "PgVectorChunk",
    similarity: float,
) -> Dict[str, Any]:
    metadata = row.metadata_json or {}
    return {
        "raw_doc": row.content,
        "similarity": max(0.0, min(1.0, float(similarity))),
        "chunk_id": row.chunk_id,
        "doc_id": row.doc_id,
        "doc_source": row.doc_source,
        "chunk_type": metadata.get("chunk_type"),
        "chunk_index": metadata.get("chunk_index"),
        "level1_title": metadata.get("level1_title", ""),
        "level2_title": metadata.get("level2_title", ""),
        "level3_title": metadata.get("level3_title", ""),
        "heading_path": metadata.get("heading_path", "front_matter"),
        "table_id": metadata.get("table_id"),
        "sub_table_id": metadata.get("sub_table_id"),
        "sub_table_index": metadata.get("sub_table_index"),
        "table_id_subtable_count": metadata.get("table_id_subtable_count"),
        "table_context_text": metadata.get("table_context_text"),
        "table_header_text": metadata.get("table_header_text"),
    }


def search_documents_pgvector(query: str, k: int = 5) -> List[Dict[str, Any]]:
    if Vector is None:
        return []

    session_local = _get_session_local()
    if session_local is None:
        return []

    query_embedding = embedding(query)
    if query_embedding is None:
        return []

    session = session_local()
    try:
        distance_expr = PgVectorChunk.embedding.cosine_distance(query_embedding)
        rows = (
            session.query(PgVectorChunk, distance_expr.label("distance"))
            .order_by(distance_expr.asc())
            .limit(k)
            .all()
        )
    except Exception as exc:
        logger.error("Failed to query pgvector: %s", exc)
        return []
    finally:
        session.close()

    result: List[Dict[str, Any]] = []
    for row, distance in rows:
        score = max(0.0, 1.0 - float(distance))
        result.append(_format_pgvector_row(row, similarity=score))
    return result


def list_documents_pgvector(limit: int = 3000) -> List[Dict[str, Any]]:
    if Vector is None:
        return []

    session_local = _get_session_local()
    if session_local is None:
        return []

    session = session_local()
    try:
        rows = session.query(PgVectorChunk).order_by(PgVectorChunk.id.desc()).limit(max(1, int(limit))).all()
    except Exception as exc:
        logger.error("Failed to list pgvector chunks: %s", exc)
        return []
    finally:
        session.close()

    rows = list(reversed(rows))
    return [_format_pgvector_row(row, similarity=0.0) for row in rows]
