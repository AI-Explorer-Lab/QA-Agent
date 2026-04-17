from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from chunking_service.heading_recovery import recover_high_conf_headings_from_mineru_payload  # noqa: E402
from chunking_service.structured_chunking import (  # noqa: E402
    CHUNK_TOKEN_OVERLAP,
    CHUNK_TOKEN_SIZE,
    MAX_CHUNK_SIZE,
    chunk_mineru_json_payload,
    chunk_plain_text,
    render_mineru_json_elements_as_text,
)
from core.config_loader import load_runtime_env  # noqa: E402
from ingest_service.documents_ingestion import DocumentLoader  # noqa: E402

load_dotenv(project_root / ".env")
load_runtime_env()

logger = logging.getLogger(__name__)

try:
    import fitz
except Exception:  # pragma: no cover - optional runtime dependency
    fitz = None


_INVALID_FILE_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def split_into_chunks(text: str, max_chars: int = 500, overlap_ratio: float = 0.3) -> List[str]:
    """
    Legacy splitter kept for compatibility.
    New logic uses token-based recursive chunking.
    """
    del max_chars, overlap_ratio
    return [item["content"] for item in chunk_plain_text(text, "legacy", "legacy")]


def split_into_chunks_v2(text: str) -> List[str]:
    """
    Token-based recursive splitter using tiktoken-compatible behavior.
    chunk_size=1024, chunk_overlap=200.
    """
    return [item["content"] for item in chunk_plain_text(text, "legacy_v2", "legacy_v2")]


def split_into_chunks_pdf(text: str, max_chars: int = 500, overlap_ratio: float = 0.3) -> List[str]:
    del max_chars, overlap_ratio
    return split_into_chunks_v2(text)


def _build_text_chunk_records(doc: Dict[str, Any], chunk_start_index: int = 0) -> List[Dict[str, Any]]:
    records = chunk_plain_text(
        text=doc.get("content", ""),
        doc_id=doc["id"],
        doc_source=doc["source"],
        chunk_start_index=chunk_start_index,
    )
    for record in records:
        record["original_id"] = doc["id"]
    return records


def process_documents(directory: str) -> List[Dict[str, Any]]:
    """
    Legacy API retained. Internally uses token-based recursive chunking.
    """
    return process_documents_v2(directory)


def process_documents_v2(directory: str) -> List[Dict[str, Any]]:
    documents = DocumentLoader(directory, file_type="txt").load_documents()
    processed: List[Dict[str, Any]] = []

    for doc in documents:
        processed.extend(_build_text_chunk_records(doc=doc))

    _save_generated_chunks(processed)
    logger.info("Processed %s chunks from %s text documents", len(processed), len(documents))
    return processed


def _extract_mineru_payload_from_zip(zip_file: zipfile.ZipFile) -> Optional[Dict[str, Any]]:
    json_files = [name for name in zip_file.namelist() if name.lower().endswith(".json")]
    for file_name in json_files:
        try:
            payload = json.loads(zip_file.read(file_name).decode("utf-8"))
        except Exception as exc:
            logger.debug("Skip invalid MinerU json file %s: %s", file_name, exc)
            continue

        if not isinstance(payload, dict) or "pdf_info" not in payload:
            continue
        return payload
    return None


def _save_extracted_text(doc_source: str, text_content: str) -> None:
    if not text_content.strip():
        return

    generated_docs_dir = project_root / "docs" / "generated_docs"
    generated_docs_dir.mkdir(parents=True, exist_ok=True)

    source_name = Path(doc_source).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = generated_docs_dir / f"{source_name}_{timestamp}.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text_content)

    logger.info("Saved extracted doc text to %s", output_path)


def _safe_file_name(value: str, fallback: str = "chunk") -> str:
    text = str(value or "").strip().replace("\r", " ").replace("\n", " ")
    text = _INVALID_FILE_CHARS_RE.sub("_", text).strip(" .")
    if not text:
        text = fallback
    if len(text) > 180:
        text = text[:180].rstrip(" .")
    return text or fallback


def _save_generated_chunks(chunks: List[Dict[str, Any]]) -> None:
    if not chunks:
        return

    generated_chunks_dir = project_root / "docs" / "generated_chunks"
    generated_chunks_dir.mkdir(parents=True, exist_ok=True)
    for legacy_file in generated_chunks_dir.glob("*.txt"):
        try:
            legacy_file.unlink()
        except Exception:
            pass

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in chunks:
        source = str(chunk.get("doc_source") or chunk.get("source") or "unknown")
        grouped.setdefault(source, []).append(chunk)

    total_saved = 0
    for source, items in grouped.items():
        source_name = _safe_file_name(Path(source).stem or "unknown", fallback="unknown")
        source_dir = generated_chunks_dir / source_name
        source_dir.mkdir(parents=True, exist_ok=True)

        for old_file in source_dir.glob("*.txt"):
            try:
                old_file.unlink()
            except Exception:
                pass

        def _sort_key(item: Dict[str, Any]) -> tuple:
            index_value = item.get("chunk_index")
            try:
                index_number = int(index_value)
            except Exception:
                index_number = 10**9
            return index_number, str(item.get("chunk_id", ""))

        used_file_names = set()
        for idx, item in enumerate(sorted(items, key=_sort_key), start=1):
            chunk_id = str(item.get("chunk_id") or f"chunk_{idx}")
            file_base = _safe_file_name(chunk_id, fallback=f"chunk_{idx}")
            file_name = f"{file_base}.txt"
            suffix = 2
            while file_name.lower() in used_file_names:
                file_name = f"{file_base}_{suffix}.txt"
                suffix += 1
            used_file_names.add(file_name.lower())

            output_path = source_dir / file_name
            output_path.write_text(str(item.get("content", "")), encoding="utf-8")
            total_saved += 1

    logger.info("Saved %s chunk files to %s", total_saved, generated_chunks_dir)


def _request_mineru_upload_url(
    headers: Dict[str, str],
    split_filename: str,
) -> Optional[Dict[str, Any]]:
    request_body = {
        "files": [{"name": split_filename}],
        "model_version": "vlm",
        "is_ocr": True,
    }
    upload_url_response = requests.post(
        "https://mineru.net/api/v4/file-urls/batch",
        headers=headers,
        json=request_body,
        timeout=30,
    )
    if upload_url_response.status_code != 200:
        logger.warning(
            "MinerU upload URL request failed (%s): %s",
            upload_url_response.status_code,
            upload_url_response.text,
        )
        return None

    upload_url_payload = upload_url_response.json()
    if upload_url_payload.get("code") != 0:
        logger.warning(
            "MinerU upload URL API error: %s",
            upload_url_payload.get("msg"),
        )
        return None

    return upload_url_payload


def process_documents_pdf(directory: str) -> List[Dict[str, Any]]:
    """
    Process PDF documents via MinerU and chunk results with structured strategy.

    Strategy:
    - Parse MinerU JSON in original page/block order (paragraph + table)
    - Recover heading path from MinerU JSON title blocks (default split by L1/L2; configurable)
    - Paragraphs: segmented by heading path, then token recursive split (1024/200)
    - Tables: keep row-level structure, split by row when token > max_chunk_size(7000),
      and inject header into each sub-table
    - Table chunks are independent (not merged into text chunks)
    - Attach heading/table metadata required by business rules
    """
    token = os.getenv("MinerU_API_KEY")
    if not token:
        logger.error("MinerU_API_KEY not found in .env file")
        return []
    if fitz is None:
        logger.error("pymupdf is not installed. Please install dependencies first.")
        return []

    pdf_docs = DocumentLoader(directory, "pdf").load_documents()
    if not pdf_docs:
        return []

    processed: List[Dict[str, Any]] = []
    temp_dir = project_root / "docs" / "temp_pdf"
    temp_dir.mkdir(exist_ok=True)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    for doc in pdf_docs:
        pdf_path = Path(doc["source"])
        logger.info("Processing PDF: %s", pdf_path.name)
        chunk_start_index = 0
        extracted_text_parts: List[str] = []

        try:
            pdf_doc = fitz.open(str(pdf_path))
            page_count = pdf_doc.page_count
            pdf_doc.close()
        except Exception as exc:
            logger.error("Failed to read PDF page count for %s: %s", pdf_path.name, exc)
            continue

        num_splits = math.ceil(page_count / 200)
        logger.info(
            "PDF pages=%s, split into %s task(s) for MinerU (max 200 pages/task)",
            page_count,
            num_splits,
        )
        split_tasks: List[Dict[str, Any]] = []

        for split_idx in range(num_splits):
            split_filename = f"split_{split_idx}_{int(time.time())}.pdf"
            split_path = temp_dir / split_filename
            logger.info("Preparing split %s/%s: %s", split_idx + 1, num_splits, split_filename)

            try:
                if num_splits > 1:
                    start_page = split_idx * 200
                    end_page = min((split_idx + 1) * 200, page_count)
                    source_doc = fitz.open(str(pdf_path))
                    sub_doc = fitz.open()
                    for page_number in range(start_page, end_page):
                        sub_doc.insert_pdf(source_doc, from_page=page_number, to_page=page_number)
                    sub_doc.save(str(split_path))
                    sub_doc.close()
                    source_doc.close()
                else:
                    shutil.copy(str(pdf_path), str(split_path))
            except Exception as exc:
                logger.error("Failed to create split PDF %s: %s", split_filename, exc)
                continue

            try:
                upload_url_payload = _request_mineru_upload_url(
                    headers=headers,
                    split_filename=split_filename,
                )
                if not upload_url_payload:
                    logger.error("MinerU upload URL API error: request failed")
                    continue

                batch_id = upload_url_payload["data"]["batch_id"]
                file_url = upload_url_payload["data"]["file_urls"][0]
            except Exception as exc:
                logger.error("Failed to request MinerU upload URL for %s: %s", split_filename, exc)
                continue

            try:
                with open(split_path, "rb") as file:
                    upload_result = requests.put(file_url, data=file, timeout=120)
                if upload_result.status_code != 200:
                    logger.error("MinerU upload failed (%s)", upload_result.status_code)
                    continue
                logger.info("Uploaded split %s/%s, batch_id=%s", split_idx + 1, num_splits, batch_id)
                split_tasks.append(
                    {
                        "batch_id": batch_id,
                        "split_idx": split_idx,
                    }
                )
            except Exception as exc:
                logger.error("Failed to upload split PDF %s: %s", split_filename, exc)
                continue

        for task in split_tasks:
            batch_id = task["batch_id"]
            split_idx = task["split_idx"]
            max_attempts = 60

            for attempt in range(1, max_attempts + 1):
                try:
                    result_response = requests.get(
                        f"https://mineru.net/api/v4/extract-results/batch/{batch_id}",
                        headers=headers,
                        timeout=30,
                    )
                    if result_response.status_code != 200:
                        logger.info(
                            "Polling split %s/%s attempt %s/%s: http=%s",
                            split_idx + 1,
                            num_splits,
                            attempt,
                            max_attempts,
                            result_response.status_code,
                        )
                        time.sleep(15)
                        continue

                    result_payload = result_response.json()
                    if result_payload.get("code") != 0:
                        logger.info(
                            "Polling split %s/%s attempt %s/%s: api_code=%s",
                            split_idx + 1,
                            num_splits,
                            attempt,
                            max_attempts,
                            result_payload.get("code"),
                        )
                        time.sleep(15)
                        continue

                    extract_results = result_payload["data"].get("extract_result", [])
                    if not extract_results:
                        logger.info(
                            "Polling split %s/%s attempt %s/%s: no extract_result yet",
                            split_idx + 1,
                            num_splits,
                            attempt,
                            max_attempts,
                        )
                        break

                    extract_result = extract_results[0]
                    state = extract_result.get("state")
                    logger.info(
                        "Polling split %s/%s attempt %s/%s: state=%s",
                        split_idx + 1,
                        num_splits,
                        attempt,
                        max_attempts,
                        state,
                    )

                    if state == "done":
                        zip_url = extract_result.get("full_zip_url")
                        if not zip_url:
                            break

                        zip_response = requests.get(zip_url, timeout=120)
                        if zip_response.status_code != 200:
                            break

                        with zipfile.ZipFile(io.BytesIO(zip_response.content)) as archive:
                            payload = _extract_mineru_payload_from_zip(archive)
                            markdown_files = [
                                name for name in archive.namelist() if name.lower().endswith(".md")
                            ]

                            if payload:
                                recovered_headings = recover_high_conf_headings_from_mineru_payload(payload)
                                if recovered_headings:
                                    logger.info(
                                        "Recovered %s high-confidence L1/L2/L3 headings from MinerU JSON",
                                        len(recovered_headings),
                                    )
                                text_dump = render_mineru_json_elements_as_text(payload)
                                if text_dump:
                                    extracted_text_parts.append(text_dump)

                                json_chunks = chunk_mineru_json_payload(
                                    payload=payload,
                                    doc_id=f"{doc['id']}_part_{split_idx + 1}",
                                    doc_source=doc["source"],
                                    chunk_start_index=chunk_start_index,
                                    max_chunk_size=MAX_CHUNK_SIZE,
                                    recovered_headings=recovered_headings,
                                )
                                processed.extend(json_chunks)
                                chunk_start_index += len(json_chunks)
                            elif markdown_files:
                                md_name = markdown_files[0]
                                md_content = archive.read(md_name).decode("utf-8")
                                extracted_text_parts.append(md_content)
                                md_chunks = chunk_plain_text(
                                    text=md_content,
                                    doc_id=f"{doc['id']}_part_{split_idx + 1}",
                                    doc_source=doc["source"],
                                    chunk_start_index=chunk_start_index,
                                )
                                processed.extend(md_chunks)
                                chunk_start_index += len(md_chunks)
                        break

                    if state in {"failed"}:
                        logger.error("MinerU extraction failed for split %s", split_idx + 1)
                        break

                    time.sleep(15)
                except Exception as exc:
                    logger.error("Error polling MinerU results for split %s: %s", split_idx + 1, exc)
                    time.sleep(15)

        if extracted_text_parts:
            _save_extracted_text(doc["source"], "\n\n".join(extracted_text_parts))

    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass

    logger.info(
        "Processed %s chunks from %s PDF documents (chunk_size=%s overlap=%s max_chunk_size=%s)",
        len(processed),
        len(pdf_docs),
        CHUNK_TOKEN_SIZE,
        CHUNK_TOKEN_OVERLAP,
        MAX_CHUNK_SIZE,
    )
    _save_generated_chunks(processed)
    return processed


if __name__ == "__main__":
    docs = process_documents_pdf("./docs/pdf_docs")
    print(f"Total chunks created: {len(docs)}")
