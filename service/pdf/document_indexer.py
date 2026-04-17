from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from service.embedding.embedding_service import EmbeddingService
from service.pdf.mineru_client import MinerUClient
from service.pdf.pdf_loader import collect_pdf_documents
from service.pdf.structured_chunker import ChunkingConfig, StructuredChunker
from service.retrieval.runtime import replace_collection_chunks, upsert_runtime_chunks
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
            cache_ttl_seconds=int(cache_cfg.get("ttl_seconds", 3600)),
            cache_max_items=int(cache_cfg.get("max_items", 5000)),
        )
        self.session_service = get_session_service()

    async def index_documents(
        self,
        pdf_path: str,
        collection_name: str = "default",
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        collection = (collection_name or "default").strip() or "default"
        pdf_documents = collect_pdf_documents(pdf_path)
        all_chunks: List[Dict[str, Any]] = []
        document_summaries: List[Dict[str, Any]] = []
        skipped = 0

        for pdf_doc in pdf_documents:
            doc_id = "doc_" + short_hash([collection, str(pdf_doc.path), pdf_doc.file_hash], length=16)
            payload = self.mineru_client.parse_pdf_to_mineru_json(
                pdf_doc.path,
                use_cache=True,
                force_rebuild=force_rebuild,
            )
            page_count = len(payload.get("pdf_info") or [])
            chunks = self.chunker.chunk_mineru_payload(
                mineru_payload=payload,
                doc_id=doc_id,
                collection_name=collection,
                doc_source=str(pdf_doc.path),
            )
            if not chunks:
                skipped += 1
                continue

            texts = [str(chunk.get("content") or "") for chunk in chunks]
            vectors = await self.embedding_service.embed_texts(
                texts,
                use_cache=True,
                chunk_text=True,
                max_concurrency=6,
            )
            normalized_chunks = [
                _normalize_chunk_for_retrieval(chunk, vectors[index])
                for index, chunk in enumerate(chunks)
            ]
            all_chunks.extend(normalized_chunks)
            document_summaries.append(
                {
                    "doc_id": doc_id,
                    "collection_name": collection,
                    "doc_source": str(pdf_doc.path),
                    "doc_hash": pdf_doc.file_hash,
                    "page_count": page_count,
                    "chunk_count": len(normalized_chunks),
                    "title": Path(pdf_doc.path).stem,
                }
            )

        if force_rebuild:
            indexed_count = replace_collection_chunks(collection, all_chunks)
        else:
            indexed_count = upsert_runtime_chunks(all_chunks)
        self.session_service.upsert_collection_chunks(collection, all_chunks, force_rebuild=force_rebuild)

        return {
            "success": True,
            "collection_name": collection,
            "indexed_documents": len(document_summaries),
            "indexed_doc_count": len(document_summaries),
            "indexed_chunks": indexed_count,
            "skipped_documents": skipped,
            "documents": document_summaries,
        }


_DEFAULT_INDEXING_SERVICE = DocumentIndexingService()


def get_document_indexing_service() -> DocumentIndexingService:
    return _DEFAULT_INDEXING_SERVICE
