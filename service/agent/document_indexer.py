from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List

from service.session.session_service import get_session_service
from utils.content_normalizer import normalize_whitespace
from utils.hash_utils import short_hash

try:
    from service.pdf.pdf_loader import collect_pdf_documents
except Exception:  # pragma: no cover
    collect_pdf_documents = None  # type: ignore[assignment]


def _build_chunk(
    *,
    doc_id: str,
    doc_source: str,
    page_idx: int,
    chunk_index: int,
    content: str,
    chunk_type: str = "text",
    heading_path: str = "front_matter",
) -> Dict[str, Any]:
    normalized = normalize_whitespace(content, preserve_newlines=True)
    chunk_id = f"{doc_id}_{page_idx}_{chunk_index}_{short_hash(normalized, 8)}"
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_source": doc_source,
        "page_idx": page_idx,
        "heading_path": heading_path,
        "level1_title": "",
        "level2_title": "",
        "level3_title": "",
        "chunk_type": chunk_type,
        "chunk_index": chunk_index,
        "table_id": "" if chunk_type != "table" else f"table_{chunk_index}",
        "sub_table_id": "",
        "table_header_text": "",
        "table_context_text": "",
        "content": normalized,
    }


def _is_table_like(text: str) -> bool:
    lowered = text.lower()
    if "|" in text or "\t" in text:
        return True
    return any(token in lowered for token in ["表", "指标", "%", "同比", "环比", "收入", "利润"])


async def _try_external_chunker(pdf_path: str, collection_name: str) -> List[Dict[str, Any]] | None:
    candidates = [
        ("service.pdf.structured_chunker", ("chunk_pdf_document", "build_structured_chunks", "chunk_document")),
        ("service.pdf.pdf_loader", ("load_and_chunk_pdf",)),
    ]

    for module_name, function_names in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for function_name in function_names:
            fn = getattr(module, function_name, None)
            if not callable(fn):
                continue

            try:
                signature = inspect.signature(fn)
                kwargs = {
                    "pdf_path": pdf_path,
                    "collection_name": collection_name,
                }
                if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
                    result = fn(**kwargs)
                else:
                    accepted = set(signature.parameters.keys())
                    filtered = {key: value for key, value in kwargs.items() if key in accepted}
                    result = fn(**filtered)
                if inspect.isawaitable(result):
                    result = await result
            except Exception:
                continue

            if isinstance(result, list):
                return [item for item in result if isinstance(item, dict)]
    return None


def _fallback_collect_documents(pdf_path: str | Path) -> List[Dict[str, Any]]:
    path = Path(pdf_path)
    if path.is_file() and path.suffix.lower() == ".pdf":
        files = [path.resolve()]
    else:
        files = [item.resolve() for item in path.rglob("*.pdf") if item.is_file()]

    docs: List[Dict[str, Any]] = []
    for file_path in files:
        docs.append(
            {
                "path": file_path,
                "file_hash": short_hash(str(file_path), 16),
                "size_bytes": file_path.stat().st_size,
            }
        )
    return docs


def _extract_pdf_chunks_with_pymupdf(doc_id: str, pdf_file: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []

    try:
        import fitz  # type: ignore

        with fitz.open(str(pdf_file)) as pdf_doc:
            for page_number, page in enumerate(pdf_doc, start=1):
                text = normalize_whitespace(page.get_text("text"), preserve_newlines=True)
                if not text:
                    continue

                paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
                if not paragraphs:
                    paragraphs = [text]

                for idx, paragraph in enumerate(paragraphs, start=1):
                    chunk_type = "table" if _is_table_like(paragraph) else "text"
                    chunks.append(
                        _build_chunk(
                            doc_id=doc_id,
                            doc_source=pdf_file.name,
                            page_idx=page_number,
                            chunk_index=idx,
                            content=paragraph,
                            chunk_type=chunk_type,
                        )
                    )
    except Exception:
        pass

    if chunks:
        return chunks

    placeholder = (
        f"文档 {pdf_file.name} 已索引（local_dev fallback）。"
        "若需高精度解析，请集成 MinerU 解析链路。"
    )
    return [
        _build_chunk(
            doc_id=doc_id,
            doc_source=pdf_file.name,
            page_idx=1,
            chunk_index=1,
            content=placeholder,
            chunk_type="text",
        )
    ]


class DocumentIndexService:
    def __init__(self) -> None:
        self.session_service = get_session_service()

    async def index_documents(
        self,
        *,
        pdf_path: str,
        collection_name: str,
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        c_name = str(collection_name or "default").strip() or "default"
        source_path = Path(pdf_path).expanduser()
        if not source_path.is_absolute():
            source_path = (Path.cwd() / source_path).resolve()

        if not source_path.exists():
            raise FileNotFoundError(f"PDF path not found: {source_path}")

        if collect_pdf_documents is not None:
            documents = collect_pdf_documents(source_path)
            normalized_docs = [
                {
                    "path": item.path,
                    "file_hash": item.file_hash,
                    "size_bytes": item.size_bytes,
                }
                for item in documents
            ]
        else:
            normalized_docs = _fallback_collect_documents(source_path)

        if not normalized_docs:
            raise ValueError(f"No PDF files found under: {source_path}")

        all_chunks: List[Dict[str, Any]] = []
        indexed_docs: List[Dict[str, Any]] = []

        for index, doc in enumerate(normalized_docs):
            file_path = Path(doc["path"])
            file_hash = str(doc.get("file_hash") or short_hash(str(file_path), 16))
            doc_id = f"doc_{file_hash[:12]}"

            external_chunks = await _try_external_chunker(str(file_path), c_name)
            if external_chunks:
                chunks = []
                for idx, item in enumerate(external_chunks, start=1):
                    normalized = normalize_whitespace(item.get("content") or item.get("raw_doc") or "", preserve_newlines=True)
                    if not normalized:
                        continue
                    chunk = _build_chunk(
                        doc_id=doc_id,
                        doc_source=file_path.name,
                        page_idx=int(item.get("page_idx", 1) or 1),
                        chunk_index=int(item.get("chunk_index", idx) or idx),
                        content=normalized,
                        chunk_type=str(item.get("chunk_type") or "text"),
                        heading_path=str(item.get("heading_path") or "front_matter"),
                    )
                    chunk["table_header_text"] = item.get("table_header_text") or ""
                    chunk["table_context_text"] = item.get("table_context_text") or ""
                    chunks.append(chunk)
            else:
                chunks = _extract_pdf_chunks_with_pymupdf(doc_id=doc_id, pdf_file=file_path)

            all_chunks.extend(chunks)
            indexed_docs.append(
                {
                    "doc_id": doc_id,
                    "doc_source": file_path.name,
                    "file_hash": file_hash,
                    "size_bytes": int(doc.get("size_bytes") or 0),
                    "chunk_count": len(chunks),
                }
            )

            if index == 0:
                await self.session_service.upsert_collection_chunks(c_name, chunks, force_rebuild=force_rebuild)
            else:
                await self.session_service.upsert_collection_chunks(c_name, chunks, force_rebuild=False)

        return {
            "collection_name": c_name,
            "indexed_doc_count": len(indexed_docs),
            "indexed_chunk_count": len(all_chunks),
            "indexed_docs": indexed_docs,
        }


_DOCUMENT_INDEX_SERVICE = DocumentIndexService()


def get_document_index_service() -> DocumentIndexService:
    return _DOCUMENT_INDEX_SERVICE
