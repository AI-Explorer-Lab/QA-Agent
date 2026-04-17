import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = Path(__file__).resolve().parents[1]

from chunking_service.document_processor import process_documents_pdf, process_documents_v2
from core.config_loader import load_runtime_env
from db_service.pgvector_store import init_pgvector_schema, upsert_chunks_to_pgvector
from embedding_service.embedding_processor import embedding

logger = logging.getLogger(__name__)

_cached_vector_store: Optional["FAISSVectorStore"] = None
_REMOVED_TABLE_METADATA_FIELDS = {
    "table_anchor_text",
    "table_anchor_confidence",
    "table_context_chunk_id",
    "table_context_tokens",
    "table_context_absorbed",
}

try:
    import faiss
except Exception:  # pragma: no cover - optional runtime dependency
    faiss = None

load_runtime_env()


def _resolve_input_path(path_value: str) -> Path:
    raw = Path(path_value)
    if raw.is_absolute():
        return raw

    cwd_candidate = (Path.cwd() / raw).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = (PROJECT_ROOT / raw).resolve()
    if project_candidate.exists():
        return project_candidate

    return project_candidate


def _detect_doc_type(input_path: Path, requested_type: str) -> str:
    file_suffix = input_path.suffix.lower().lstrip(".")
    normalized = (requested_type or "").strip().lower()

    # Auto mode: infer from file extension or directory contents.
    if normalized == "auto":
        if input_path.is_file() and file_suffix in {"pdf", "txt"}:
            return file_suffix
        if input_path.is_dir():
            has_pdf = any(input_path.glob("*.pdf"))
            has_txt = any(input_path.glob("*.txt"))
            if has_pdf and not has_txt:
                return "pdf"
            if has_txt and not has_pdf:
                return "txt"
            if has_pdf and has_txt:
                logger.warning("Both PDF and TXT found in %s; auto mode chooses pdf.", input_path)
                return "pdf"
        return "pdf"

    # If caller passed explicit type but path is a file with different extension, align to extension.
    if input_path.is_file() and file_suffix in {"pdf", "txt"} and normalized in {"pdf", "txt"}:
        if file_suffix != normalized:
            logger.warning(
                "file_type=%s mismatches file extension .%s; using %s.",
                normalized,
                file_suffix,
                file_suffix,
            )
            return file_suffix

    return normalized or "pdf"


class FAISSVectorStore:
    """Simple FAISS vector store using cosine similarity with normalized vectors."""

    def __init__(self, dimension: int = 1024):
        if faiss is None:
            raise RuntimeError("faiss is not installed")
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: List[Dict[str, Any]] = []
        self.doc_id_to_index: Dict[str, int] = {}

    def add_embeddings(self, embeddings: List[np.ndarray], documents: List[Dict[str, Any]]) -> None:
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")

        normalized_embeddings = []
        for emb in embeddings:
            if isinstance(emb, list):
                emb = np.array(emb, dtype="float32")
            norm = np.linalg.norm(emb)
            normalized = emb if norm == 0 else emb / norm
            normalized_embeddings.append(normalized.astype("float32"))

        self.index.add(np.vstack(normalized_embeddings))
        start_index = len(self.documents)

        for offset, document in enumerate(documents):
            index = start_index + offset
            self.documents.append(document)
            chunk_id = document.get("chunk_id", f"doc_{index}")
            self.doc_id_to_index[chunk_id] = index

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding, dtype="float32")
        norm = np.linalg.norm(query_embedding)
        normalized_query = query_embedding if norm == 0 else query_embedding / norm
        scores, indices = self.index.search(normalized_query.reshape(1, -1).astype("float32"), k)

        results: List[Dict[str, Any]] = []
        for score, index in zip(scores[0], indices[0]):
            if index == -1 or index >= len(self.documents):
                continue
            results.append(
                {
                    "document": self.documents[index],
                    "similarity_score": float(score),
                    "index": int(index),
                }
            )
        return results

    def save(self, index_path: str, metadata_path: str) -> None:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
        metadata = {
            "documents": self.documents,
            "doc_id_to_index": self.doc_id_to_index,
            "dimension": self.dimension,
        }
        with open(metadata_path, "wb") as handle:
            pickle.dump(metadata, handle)

    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> "FAISSVectorStore":
        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as handle:
            metadata = pickle.load(handle)

        store = cls(dimension=metadata["dimension"])
        store.index = index
        store.documents = metadata["documents"]
        store.doc_id_to_index = metadata["doc_id_to_index"]
        return store

    def get_document_count(self) -> int:
        return len(self.documents)


def get_or_load_faiss(
    index_path: str = "./vector_stores/faiss_index.bin",
    metadata_path: str = "./vector_stores/faiss_metadata.pkl",
) -> Optional[FAISSVectorStore]:
    global _cached_vector_store

    if _cached_vector_store is not None:
        return _cached_vector_store

    try:
        if faiss is None:
            logger.warning("faiss is not installed")
            return None
        loaded = load_faiss(index_path, metadata_path)
        if loaded:
            _cached_vector_store = loaded
            logger.info("Loaded FAISS vector store with %s docs", loaded.get_document_count())
        return loaded
    except Exception as exc:
        logger.error("Failed to load FAISS index: %s", exc)
        return None


def clear_faiss_cache() -> None:
    global _cached_vector_store
    _cached_vector_store = None


def _sanitize_chunk_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(chunk)
    for field in _REMOVED_TABLE_METADATA_FIELDS:
        sanitized.pop(field, None)
    return sanitized


def process_and_save_to_faiss(
    document_path: str,
    index_path: str = "./vector_stores/faiss_index.bin",
    metadata_path: str = "./vector_stores/faiss_metadata.pkl",
    type: str = "txt",
) -> bool:
    try:
        if faiss is None:
            logger.error("faiss is not installed, cannot build index.")
            return False

        input_path = _resolve_input_path(document_path)
        if not input_path.exists():
            logger.error("Input path does not exist: %s", input_path)
            return False

        effective_type = _detect_doc_type(input_path, type)
        if effective_type not in {"pdf", "txt"}:
            logger.error("Unsupported file type: %s", effective_type)
            return False

        chunks: List[Dict[str, Any]] = []
        if effective_type == "txt":
            chunks = process_documents_v2(str(input_path))
        elif effective_type == "pdf":
            chunks = process_documents_pdf(str(input_path))

        if not chunks:
            logger.warning("No chunks to index.")
            return False

        embeddings: List[List[float]] = []
        records: List[Dict[str, Any]] = []
        failed_count = 0

        for index, chunk in enumerate(chunks, start=1):
            if index % 10 == 0:
                logger.info("Embedding progress %s/%s", index, len(chunks))
            sanitized_chunk = _sanitize_chunk_metadata(chunk)
            text_content = str(sanitized_chunk.get("content", "")).strip()
            if not text_content:
                failed_count += 1
                continue
            vector = embedding(text_content)
            if vector is None:
                failed_count += 1
                continue
            embeddings.append(vector)
            records.append(sanitized_chunk)

        if not embeddings:
            logger.error("No valid embeddings were generated.")
            return False

        store = FAISSVectorStore(dimension=len(embeddings[0]))
        store.add_embeddings(embeddings, records)
        store.save(index_path, metadata_path)
        clear_faiss_cache()

        vector_backend = os.getenv("VECTOR_STORE_BACKEND", "faiss").strip().lower()
        if vector_backend in {"pgvector", "hybrid", "both"}:
            if init_pgvector_schema():
                upserted = upsert_chunks_to_pgvector(records)
                logger.info("PGVector upsert completed. rows=%s", upserted)
            else:
                logger.warning("PGVector schema init skipped/failed; only FAISS index is saved.")

        logger.info(
            "FAISS saved: indexed=%s failed=%s index_path=%s metadata_path=%s",
            len(records),
            failed_count,
            index_path,
            metadata_path,
        )
        return True
    except Exception as exc:
        logger.error("Failed to process/save FAISS: %s", exc)
        return False


def save_embeddings_to_faiss(
    embeddings: List[List[float]],
    documents: List[Dict[str, Any]],
    index_path: str = "./vector_stores/faiss_index.bin",
    metadata_path: str = "./vector_stores/faiss_metadata.pkl",
) -> bool:
    try:
        if faiss is None:
            logger.error("faiss is not installed, cannot save embeddings.")
            return False
        if not embeddings:
            logger.error("Embeddings list is empty.")
            return False
        store = FAISSVectorStore(dimension=len(embeddings[0]))
        sanitized_documents = [_sanitize_chunk_metadata(document) for document in documents]
        store.add_embeddings(embeddings, sanitized_documents)
        store.save(index_path, metadata_path)
        clear_faiss_cache()
        return True
    except Exception as exc:
        logger.error("Failed to save FAISS embeddings: %s", exc)
        return False


def load_faiss(index_path: str, metadata_path: str) -> Optional[FAISSVectorStore]:
    try:
        if faiss is None:
            return None
        return FAISSVectorStore.load(index_path, metadata_path)
    except Exception as exc:
        logger.error("Failed to load FAISS files: %s", exc)
        return None


def search_documents(
    query: str,
    index_path: str = "./vector_stores/faiss_index.bin",
    metadata_path: str = "./vector_stores/faiss_metadata.pkl",
    k: int = 5,
) -> List[Dict[str, Any]]:
    try:
        if faiss is None:
            return []
        store = load_faiss(index_path, metadata_path)
        if not store:
            return []
        query_embedding = embedding(query)
        if query_embedding is None:
            return []
        return store.search(query_embedding, k=k)
    except Exception as exc:
        logger.error("FAISS search failed: %s", exc)
        return []


def _format_search_result(result: Dict[str, Any]) -> Dict[str, Any]:
    document = result.get("document", {})
    return _format_document(document, similarity=float(result.get("similarity_score", 0.0)))


def _format_document(document: Dict[str, Any], similarity: float = 0.0) -> Dict[str, Any]:
    raw_doc = str(document.get("content", ""))
    return {
        "raw_doc": raw_doc,
        "similarity": float(similarity),
        "chunk_id": document.get("chunk_id"),
        "doc_id": document.get("doc_id"),
        "doc_source": document.get("doc_source", document.get("source")),
        "chunk_type": document.get("chunk_type"),
        "chunk_index": document.get("chunk_index"),
        "level1_title": document.get("level1_title", ""),
        "level2_title": document.get("level2_title", ""),
        "level3_title": document.get("level3_title", ""),
        "heading_path": document.get("heading_path", "front_matter"),
        "table_id": document.get("table_id"),
        "sub_table_id": document.get("sub_table_id"),
        "sub_table_index": document.get("sub_table_index"),
        "table_id_subtable_count": document.get("table_id_subtable_count"),
        "table_context_text": document.get("table_context_text"),
        "table_header_text": document.get("table_header_text"),
    }


def list_faiss_documents(
    index_path: str = "./vector_stores/faiss_index.bin",
    metadata_path: str = "./vector_stores/faiss_metadata.pkl",
    limit: int = 0,
) -> List[Dict[str, Any]]:
    if faiss is None:
        return []

    store = get_or_load_faiss(index_path=index_path, metadata_path=metadata_path)
    if not store:
        return []

    documents = list(store.documents)
    if limit and limit > 0:
        documents = documents[: int(limit)]

    return [_format_document(document, similarity=0.0) for document in documents]


def search_documents_v2(query: str, k: int) -> List[Dict[str, Any]]:
    index_path = "./vector_stores/faiss_index.bin"
    metadata_path = "./vector_stores/faiss_metadata.pkl"
    try:
        if faiss is None:
            return [{"raw_doc": "FAISS is not installed; local vector search is unavailable.", "similarity": 0.0}]

        store = get_or_load_faiss(index_path=index_path, metadata_path=metadata_path)
        if not store:
            return [
                {
                    "raw_doc": "Failed to load FAISS index. Please build the index first.",
                    "similarity": 0.0,
                }
            ]

        query_embedding = embedding(query)
        if query_embedding is None:
            return [
                {
                    "raw_doc": "Failed to generate query embedding.",
                    "similarity": 0.0,
                }
            ]

        results = store.search(query_embedding, k=k)
        return [_format_search_result(result) for result in results]
    except Exception as exc:
        logger.error("search_documents_v2 failed: %s", exc)
        return [{"raw_doc": f"Search failed: {exc}", "similarity": 0.0}]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk -> Embedding -> FAISS (and optional PGVector sync).")
    parser.add_argument(
        "--document-path",
        type=str,
        default="./docs/pdf_docs",
        help="Input directory path. Example: ./docs/pdf_docs or ./docs/test_docs",
    )
    parser.add_argument(
        "--type",
        choices=["pdf", "txt", "auto"],
        default="auto",
        help="Input document type.",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="./vector_stores/faiss_index.bin",
        help="Output FAISS index path.",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="./vector_stores/faiss_metadata.pkl",
        help="Output FAISS metadata path.",
    )
    parser.add_argument(
        "--backend",
        choices=["faiss", "pgvector", "hybrid", "both"],
        default=None,
        help="Optional override for VECTOR_STORE_BACKEND at runtime.",
    )
    return parser


def _main() -> int:
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args()

    if args.backend:
        os.environ["VECTOR_STORE_BACKEND"] = args.backend
        logger.info("VECTOR_STORE_BACKEND override=%s", args.backend)

    ok = process_and_save_to_faiss(
        document_path=args.document_path,
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        type=args.type,
    )
    if not ok:
        logger.error("Index build failed.")
        return 1

    logger.info("Index build succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
