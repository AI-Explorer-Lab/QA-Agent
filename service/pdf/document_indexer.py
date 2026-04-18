from __future__ import annotations

from pathlib import Path
import logging
from typing import Any, Dict, List

from exception.business_exception import ValidationException
from middlewares.operation_log import (
    fail_operation_step,
    finish_operation_step,
    log_operation_event,
    start_operation_step,
)
from middlewares.trace_context import get_trace_id
from service.embedding.embedding_service import EmbeddingService, build_embedding_provider_from_config
from service.pdf.mineru_client import MinerUClient
from service.pdf.pdf_loader import collect_pdf_documents
from service.pdf.structured_chunker import ChunkingConfig, StructuredChunker
from service.retrieval.runtime import get_runtime_repository, replace_collection_chunks, upsert_runtime_chunks
from service.session.session_service import get_session_service
from utils.config_loader import get_app_config
from utils.hash_utils import short_hash


def _page_range_to_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ""
        if len(value) == 1:
            return str(value[0])
        return f"{value[0]}-{value[-1]}"
    return str(value or "")


def _normalize_chunk_for_retrieval(chunk: Dict[str, Any], embedding: List[float]) -> Dict[str, Any]:
    content = str(chunk.get("content") or chunk.get("raw_doc") or "")
    normalized = dict(chunk)
    normalized["raw_doc"] = content
    normalized["content"] = content
    normalized["embedding"] = embedding
    normalized["page_range"] = _page_range_to_text(chunk.get("page_range"))
    normalized["heading_path"] = str(chunk.get("heading_path") or "front_matter")
    normalized["chunk_type"] = str(chunk.get("chunk_type") or "text")
    normalized.setdefault("source_channels", [])
    normalized.setdefault("metadata", {})
    return normalized


def _count_mineru_blocks(payload: Dict[str, Any]) -> int:
    return sum(
        len(page.get("para_blocks") or [])
        for page in payload.get("pdf_info") or []
        if isinstance(page, dict)
    )


def _chunk_type_counts(chunks: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for chunk in chunks:
        chunk_type = str(chunk.get("chunk_type") or "text")
        counts[chunk_type] = counts.get(chunk_type, 0) + 1
    return counts


class DocumentIndexingService:
    def __init__(
        self,
        mineru_client: MinerUClient | None = None,
        chunker: StructuredChunker | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        config = get_app_config()
        chunk_cfg = config.get("chunking", {}) if isinstance(config.get("chunking"), dict) else {}
        cache_cfg = config.get("cache", {}) if isinstance(config.get("cache"), dict) else {}
        self.mineru_client = mineru_client or MinerUClient(
            cache_ttl_seconds=int(cache_cfg.get("ttl_seconds", 3600)),
            cache_max_items=int(cache_cfg.get("max_items", 5000)),
        )
        self.chunker = chunker or StructuredChunker(
            ChunkingConfig(
                chunk_size_tokens=int(chunk_cfg.get("chunk_size_tokens", 1024)),
                chunk_overlap_tokens=int(chunk_cfg.get("chunk_overlap_tokens", 200)),
                max_chunk_size_tokens=int(chunk_cfg.get("max_chunk_size", 7000)),
            )
        )
        self.embedding_service = embedding_service or EmbeddingService(
            provider=build_embedding_provider_from_config(config),
            cache_ttl_seconds=int(cache_cfg.get("ttl_seconds", 3600)),
            cache_max_items=int(cache_cfg.get("max_items", 5000)),
        )
        storage_cfg = config.get("storage", {}) if isinstance(config.get("storage"), dict) else {}
        self.configured_storage_backend = str(storage_cfg.get("backend") or "").strip() or "unknown"
        self.session_service = get_session_service()

    async def index_documents(
        self,
        pdf_path: str,
        collection_name: str = "default",
        doc_source: str | None = None,
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        collection = (collection_name or "default").strip() or "default"
        overall_timer = start_operation_step(
            "index.document",
            pdf_path=pdf_path,
            collection_name=collection,
            force_rebuild=force_rebuild,
        )
        try:
            collect_timer = start_operation_step("index.collect_documents", pdf_path=pdf_path)
            try:
                pdf_documents = collect_pdf_documents(pdf_path)
            except Exception as exc:
                fail_operation_step(collect_timer, exc)
                raise
            finish_operation_step(
                collect_timer,
                document_count=len(pdf_documents),
                total_size_bytes=sum(document.size_bytes for document in pdf_documents),
            )

            if doc_source and len(pdf_documents) != 1:
                raise ValidationException(
                    message="doc_source override requires exactly one PDF document.",
                    detail={
                        "pdf_path": pdf_path,
                        "collection_name": collection,
                        "doc_source": doc_source,
                        "document_count": len(pdf_documents),
                    },
                )
            all_chunks: List[Dict[str, Any]] = []
            document_summaries: List[Dict[str, Any]] = []
            skipped_documents: List[Dict[str, Any]] = []

            doc_sources_to_replace: set[str] = set()
            runtime_repository = get_runtime_repository()
            for pdf_doc in pdf_documents:
                doc_source_value = str((doc_source or str(pdf_doc.path)) or "").strip() or str(pdf_doc.path)
                doc_id = "doc_" + short_hash([collection, doc_source_value], length=16)
                doc_timer = start_operation_step(
                    "index.document_one",
                    doc_id=doc_id,
                    doc_source=doc_source_value,
                    file_hash=pdf_doc.file_hash,
                    size_bytes=pdf_doc.size_bytes,
                )
                try:
                    if not force_rebuild:
                        existing = runtime_repository.get_latest_document_by_source(collection, doc_source_value)
                        existing_hash = str((existing or {}).get("doc_hash") or "")
                        if existing and existing_hash and existing_hash == pdf_doc.file_hash:
                            skipped_detail = {
                                "doc_id": existing.get("doc_id") or doc_id,
                                "collection_name": collection,
                                "doc_source": doc_source_value,
                                "doc_hash": pdf_doc.file_hash,
                                "page_count": int(existing.get("page_count") or 0),
                                "reason": "already_indexed_same_hash",
                            }
                            skipped_documents.append(skipped_detail)
                            log_operation_event(
                                "index.document.skipped",
                                status="info",
                                level=logging.INFO,
                                **skipped_detail,
                            )
                            document_summaries.append(
                                {
                                    "doc_id": skipped_detail["doc_id"],
                                    "collection_name": collection,
                                    "doc_source": doc_source_value,
                                    "doc_hash": pdf_doc.file_hash,
                                    "page_count": skipped_detail["page_count"],
                                    "chunk_count": 0,
                                    "parser_source": "skipped",
                                    "title": Path(doc_source_value).stem if doc_source_value else Path(pdf_doc.path).stem,
                                }
                            )
                            finish_operation_step(
                                doc_timer,
                                status="skipped",
                                level=logging.INFO,
                                doc_id=skipped_detail["doc_id"],
                                reason="already_indexed_same_hash",
                            )
                            continue
                        if existing and existing_hash and existing_hash != pdf_doc.file_hash:
                            doc_sources_to_replace.add(doc_source_value)

                    ocr_timer = start_operation_step(
                        "index.ocr",
                        doc_id=doc_id,
                        doc_source=doc_source_value,
                        use_cache=True,
                        force_rebuild=force_rebuild,
                    )
                    try:
                        payload = self.mineru_client.parse_pdf_to_mineru_json(
                            pdf_doc.path,
                            use_cache=True,
                            force_rebuild=force_rebuild,
                        )
                    except Exception as exc:
                        fail_operation_step(ocr_timer, exc, doc_id=doc_id)
                        raise
                    page_count = len(payload.get("pdf_info") or [])
                    finish_operation_step(
                        ocr_timer,
                        doc_id=doc_id,
                        parser_source=payload.get("source", "unknown"),
                        page_count=page_count,
                        block_count=_count_mineru_blocks(payload),
                    )

                    chunk_timer = start_operation_step(
                        "index.chunking",
                        doc_id=doc_id,
                        page_count=page_count,
                    )
                    try:
                        chunks = self.chunker.chunk_mineru_payload(
                            mineru_payload=payload,
                            doc_id=doc_id,
                            collection_name=collection,
                            doc_source=doc_source_value,
                        )
                    except Exception as exc:
                        fail_operation_step(chunk_timer, exc, doc_id=doc_id)
                        raise
                    chunk_counts = _chunk_type_counts(chunks)
                    finish_operation_step(
                        chunk_timer,
                        doc_id=doc_id,
                        chunk_count=len(chunks),
                        text_chunks=chunk_counts.get("text", 0),
                        table_chunks=chunk_counts.get("table", 0),
                    )
                    if not chunks:
                        skipped_detail = {
                            "doc_id": doc_id,
                            "doc_source": str(pdf_doc.path),
                            "doc_hash": pdf_doc.file_hash,
                            "page_count": page_count,
                            "parser_source": payload.get("source", "unknown"),
                            "reason": "no_chunks_after_chunking",
                        }
                        skipped_documents.append(skipped_detail)
                        log_operation_event(
                            "index.document.skipped",
                            status="warning",
                            level=logging.WARNING,
                            **skipped_detail,
                        )
                        finish_operation_step(
                            doc_timer,
                            status="skipped",
                            level=logging.WARNING,
                            doc_id=doc_id,
                            reason="no_chunks_after_chunking",
                        )
                        continue

                    texts = [str(chunk.get("content") or "") for chunk in chunks]
                    embedding_timer = start_operation_step(
                        "index.embedding",
                        doc_id=doc_id,
                        text_count=len(texts),
                        max_concurrency=6,
                    )
                    try:
                        vectors = await self.embedding_service.embed_texts(
                            texts,
                            use_cache=True,
                            chunk_text=True,
                            max_concurrency=6,
                        )
                    except Exception as exc:
                        fail_operation_step(embedding_timer, exc, doc_id=doc_id)
                        raise
                    if len(vectors) != len(chunks):
                        exc = ValidationException(
                            message="Embedding count does not match chunk count.",
                            detail={
                                "doc_id": doc_id,
                                "chunk_count": len(chunks),
                                "embedding_count": len(vectors),
                            },
                        )
                        fail_operation_step(embedding_timer, exc, doc_id=doc_id)
                        raise exc
                    finish_operation_step(
                        embedding_timer,
                        doc_id=doc_id,
                        vector_count=len(vectors),
                        embedding_dim=len(vectors[0]) if vectors else 0,
                        embedding_provider=getattr(self.embedding_service, "provider_name", "unknown"),
                        embedding_model=getattr(self.embedding_service, "provider_model", ""),
                    )

                    normalized_chunks: List[Dict[str, Any]] = []
                    for index, chunk in enumerate(chunks):
                        base = dict(chunk)
                        base["doc_source"] = doc_source_value
                        base["doc_hash"] = pdf_doc.file_hash
                        base["page_count"] = page_count
                        base["title"] = Path(pdf_doc.path).stem
                        normalized = _normalize_chunk_for_retrieval(base, vectors[index])
                        normalized["doc_source"] = doc_source_value
                        normalized["doc_hash"] = pdf_doc.file_hash
                        normalized["page_count"] = page_count
                        normalized["title"] = Path(pdf_doc.path).stem
                        metadata = normalized.get("metadata") or {}
                        if isinstance(metadata, dict):
                            metadata.setdefault("doc_hash", pdf_doc.file_hash)
                            metadata.setdefault("page_count", page_count)
                            metadata.setdefault("doc_source", doc_source_value)
                            normalized["metadata"] = metadata
                        normalized_chunks.append(normalized)

                    all_chunks.extend(normalized_chunks)
                    document_summary = {
                        "doc_id": doc_id,
                        "collection_name": collection,
                        "doc_source": doc_source_value,
                        "doc_hash": pdf_doc.file_hash,
                        "page_count": page_count,
                        "chunk_count": len(normalized_chunks),
                        "parser_source": payload.get("source", "unknown"),
                        "title": Path(pdf_doc.path).stem,
                    }
                    document_summaries.append(document_summary)
                    finish_operation_step(
                        doc_timer,
                        doc_id=doc_id,
                        page_count=page_count,
                        chunk_count=len(normalized_chunks),
                    )
                except Exception as exc:
                    fail_operation_step(doc_timer, exc, doc_id=doc_id)
                    raise

            if not all_chunks:
                if skipped_documents and len(skipped_documents) == len(pdf_documents):
                    return {
                        "success": True,
                        "trace_id": get_trace_id(),
                        "collection_name": collection,
                        "indexed_documents": 0,
                        "indexed_doc_count": 0,
                        "indexed_chunks": 0,
                        "skipped_documents": len(skipped_documents),
                        "skipped_documents_detail": skipped_documents,
                        "documents": document_summaries,
                        "storage": {
                            "configured_backend": self.configured_storage_backend,
                            "effective_vector_backend": str(getattr(runtime_repository, "backend", "unknown") or "unknown"),
                            "target": "pgvector",
                            "persistent_database_write": False,
                            "session_collection_chunks": len(self.session_service.get_collection_chunks(collection)),
                        },
                    }

                raise ValidationException(
                    message="Document indexing produced no chunks. Check OCR/MinerU output or PDF text extraction.",
                    detail={
                        "pdf_path": pdf_path,
                        "collection_name": collection,
                        "document_count": len(pdf_documents),
                        "skipped_documents": skipped_documents,
                    },
                )
            runtime_repository = get_runtime_repository()
            effective_backend = str(getattr(runtime_repository, "backend", "unknown") or "unknown")
            if effective_backend != "pgvector":
                raise ValidationException(
                    message="Document indexing requires pgvector storage.",
                    detail={
                        "configured_storage_backend": self.configured_storage_backend,
                        "effective_vector_backend": effective_backend,
                    },
                )

            database_timer = start_operation_step(
                "index.database",
                collection_name=collection,
                force_rebuild=force_rebuild,
                configured_storage_backend=self.configured_storage_backend,
                effective_vector_backend=effective_backend,
                chunk_count=len(all_chunks),
            )
            try:
                if force_rebuild:
                    indexed_count = replace_collection_chunks(collection, all_chunks)
                else:
                    if doc_sources_to_replace:
                        for source in sorted(doc_sources_to_replace):
                            runtime_repository.delete_documents_by_source(collection, source)
                            try:
                                self.session_service.delete_collection_doc_source(collection, source)
                            except Exception:
                                pass
                    indexed_count = upsert_runtime_chunks(all_chunks)
                session_result = self.session_service.upsert_collection_chunks(
                    collection,
                    all_chunks,
                    force_rebuild=force_rebuild,
                )
            except Exception as exc:
                fail_operation_step(database_timer, exc)
                raise
            finish_operation_step(
                database_timer,
                indexed_chunks=indexed_count,
                session_collection_chunks=session_result.get("chunk_count"),
                persistent_database_write=True,
            )

            result = {
                "success": True,
                "trace_id": get_trace_id(),
                "collection_name": collection,
                "indexed_documents": len(document_summaries),
                "indexed_doc_count": len(document_summaries),
                "indexed_chunks": indexed_count,
                "skipped_documents": len(skipped_documents),
                "skipped_documents_detail": skipped_documents,
                "documents": document_summaries,
                "storage": {
                    "configured_backend": self.configured_storage_backend,
                    "effective_vector_backend": effective_backend,
                    "target": "pgvector",
                    "persistent_database_write": True,
                    "session_collection_chunks": session_result.get("chunk_count"),
                },
            }
            finish_operation_step(
                overall_timer,
                indexed_documents=len(document_summaries),
                indexed_chunks=indexed_count,
                skipped_documents=len(skipped_documents),
            )
            return result
        except Exception as exc:
            fail_operation_step(overall_timer, exc)
            raise


_DEFAULT_INDEXING_SERVICE = DocumentIndexingService()


def get_document_indexing_service() -> DocumentIndexingService:
    return _DEFAULT_INDEXING_SERVICE








