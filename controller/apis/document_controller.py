from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
import re
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError

from exception.business_exception import ValidationException
from service.pdf.document_indexer import get_document_indexing_service
from service.pdf.index_progress import EVENT_TO_STEP, get_index_progress_tracker

router = APIRouter()
_PATH_FIELD_PATTERN = re.compile(
    r'(?P<prefix>"(?P<key>pdf_path|doc_source)"\s*:\s*")(?P<value>[^"]*)(?P<suffix>")'
)


class DocumentIndexRequest(BaseModel):
    doc_source: str = ""
    pdf_path: str
    force_rebuild: bool = False
    collection_name: str = "default"


def _step_label(key: str) -> str:
    return {
        "upload": "Upload file",
        "collect": "Collect documents",
        "dedupe": "Duplicate check",
        "ocr": "OCR / PDF parse",
        "chunking": "Chunking",
        "embedding": "Embedding",
        "database": "Database write",
    }.get(key, key)


def _build_pipeline_steps(result: dict[str, Any], *, uploaded: bool) -> list[dict[str, Any]]:
    indexed_chunks = int(result.get("indexed_chunks") or 0)
    skipped_documents = int(result.get("skipped_documents") or 0)
    indexed_documents = int(result.get("indexed_doc_count") or result.get("indexed_documents") or 0)
    all_skipped = skipped_documents > 0 and indexed_chunks <= 0

    first_key = "upload" if uploaded else "collect"
    first_detail = "File received by backend" if uploaded else "Local PDF path collected"
    if all_skipped:
        steps = [
            {"key": first_key, "label": _step_label(first_key), "status": "completed", "progress": 100, "detail": first_detail},
        ]
        if uploaded:
            steps.append({"key": "collect", "label": _step_label("collect"), "status": "completed", "progress": 100, "detail": "Collected uploaded PDF"})
        steps.extend([
            {
                "key": "dedupe",
                "label": _step_label("dedupe"),
                "status": "completed",
                "progress": 100,
                "detail": "Same document hash already indexed",
            },
            {"key": "ocr", "label": _step_label("ocr"), "status": "skipped", "progress": 100, "detail": "Skipped by duplicate check"},
            {"key": "chunking", "label": _step_label("chunking"), "status": "skipped", "progress": 100, "detail": "Skipped by duplicate check"},
            {"key": "embedding", "label": _step_label("embedding"), "status": "skipped", "progress": 100, "detail": "Skipped by duplicate check"},
            {"key": "database", "label": _step_label("database"), "status": "completed", "progress": 100, "detail": "Existing records are available"},
        ])
        return steps

    steps = [
        {"key": first_key, "label": _step_label(first_key), "status": "completed", "progress": 100, "detail": first_detail},
    ]
    if uploaded:
        steps.append({"key": "collect", "label": _step_label("collect"), "status": "completed", "progress": 100, "detail": "Collected uploaded PDF"})
    steps.extend([
        {"key": "ocr", "label": _step_label("ocr"), "status": "completed", "progress": 100, "detail": f"Parsed {indexed_documents} document(s)"},
        {"key": "chunking", "label": _step_label("chunking"), "status": "completed", "progress": 100, "detail": f"Generated {indexed_chunks} chunk(s)"},
        {"key": "embedding", "label": _step_label("embedding"), "status": "completed", "progress": 100, "detail": "Generated chunk embeddings"},
        {"key": "database", "label": _step_label("database"), "status": "completed", "progress": 100, "detail": f"Wrote {indexed_chunks} vector row(s)"},
    ])
    return steps


def _augment_index_result(
    result: dict[str, Any],
    *,
    uploaded: bool,
    upload_file_name: str = "",
    upload_size_bytes: int = 0,
) -> dict[str, Any]:
    augmented = dict(result)
    augmented["pipeline_steps"] = _build_pipeline_steps(augmented, uploaded=uploaded)
    if uploaded:
        augmented["upload"] = {
            "file_name": upload_file_name,
            "size_bytes": upload_size_bytes,
        }
    return augmented


def _escape_invalid_json_backslashes(raw_body: str) -> str:
    repaired: list[str] = []
    in_string = False
    escaped = False
    index = 0

    while index < len(raw_body):
        char = raw_body[index]

        if not in_string:
            repaired.append(char)
            if char == '"':
                in_string = True
            index += 1
            continue

        if escaped:
            repaired.append(char)
            escaped = False
            index += 1
            continue

        if char == "\\":
            next_char = raw_body[index + 1] if index + 1 < len(raw_body) else ""
            is_unicode_escape = (
                next_char == "u"
                and index + 5 < len(raw_body)
                and all(ch in "0123456789abcdefABCDEF" for ch in raw_body[index + 2 : index + 6])
            )
            if next_char in {'"', "\\", "/", "b", "f", "n", "r", "t"} or is_unicode_escape:
                repaired.append(char)
                escaped = True
            else:
                repaired.append("\\\\")
            index += 1
            continue

        repaired.append(char)
        if char == '"':
            in_string = False
        index += 1

    return "".join(repaired)


def _normalize_windows_path_value(raw_value: str) -> str:
    normalized: list[str] = []
    index = 0

    while index < len(raw_value):
        char = raw_value[index]
        if char != "\\":
            normalized.append(char)
            index += 1
            continue

        normalized.append("\\\\")
        if index + 1 < len(raw_value) and raw_value[index + 1] == "\\":
            index += 2
        else:
            index += 1

    return "".join(normalized)


def _escape_windows_path_fields(raw_body: str) -> str:
    def replace(match: re.Match[str]) -> str:
        value = _normalize_windows_path_value(match.group("value"))
        return f'{match.group("prefix")}{value}{match.group("suffix")}'

    return _PATH_FIELD_PATTERN.sub(replace, raw_body)


async def _parse_document_index_request(request: Request) -> DocumentIndexRequest:
    raw_body = (await request.body()).decode("utf-8")
    repaired_body = _escape_invalid_json_backslashes(_escape_windows_path_fields(raw_body))

    try:
        payload = json.loads(repaired_body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="Invalid JSON request body") from exc

    try:
        model_validate = getattr(DocumentIndexRequest, "model_validate", None)
        if callable(model_validate):
            return model_validate(payload)
        return DocumentIndexRequest.parse_obj(payload)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


def _progress_detail(event: str, status: str, fields: dict[str, Any]) -> str:
    if status == "started":
        return "Backend step started"
    if status == "failed":
        return str(fields.get("error") or "Backend step failed")
    if event == "index.collect_documents":
        return f"Collected {fields.get('document_count', 0)} document(s)"
    if event == "index.ocr":
        return f"Parsed {fields.get('page_count', 0)} page(s), {fields.get('block_count', 0)} block(s)"
    if event == "index.chunking":
        return f"Generated {fields.get('chunk_count', 0)} chunk(s)"
    if event == "index.embedding":
        return f"Generated {fields.get('vector_count', 0)} embedding vector(s)"
    if event == "index.database":
        return f"Wrote {fields.get('indexed_chunks', 0)} chunk row(s)"
    return "Backend step completed"


def _progress_callback_for(task_id: str):
    tracker = get_index_progress_tracker()

    def callback(event: str, status: str, fields: dict[str, Any]) -> None:
        step_key = EVENT_TO_STEP.get(event)
        if not step_key:
            return
        progress_status = "running" if status == "started" else status
        tracker.update_step(
            task_id,
            step_key,
            status=progress_status,
            progress=None,
            detail=_progress_detail(event, status, fields),
            fields=fields,
        )

    return callback


async def _save_upload_to_temp(file: UploadFile) -> tuple[Path, str, int]:
    original_name = Path(file.filename or "uploaded.pdf").name
    if Path(original_name).suffix.lower() != ".pdf":
        raise ValidationException(
            "Only PDF upload is supported.",
            detail={"file_name": original_name, "expected_suffix": ".pdf"},
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="trusted_qa_upload_"))
    target_path = temp_dir / original_name
    size_bytes = 0
    with target_path.open("wb") as output:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size_bytes += len(chunk)
            output.write(chunk)

    if size_bytes <= 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValidationException("Uploaded file is empty.", detail={"file_name": original_name})
    return target_path, original_name, size_bytes


async def _run_upload_index_task(
    task_id: str,
    pdf_path: str,
    file_name: str,
    size_bytes: int,
    collection_name: str,
    force_rebuild: bool,
    doc_source: str,
) -> None:
    tracker = get_index_progress_tracker()
    try:
        tracker.update_step(
            task_id,
            "upload",
            status="completed",
            progress=100,
            detail="File saved and ready for indexing",
            fields={"file_name": file_name, "size_bytes": size_bytes},
        )
        source = (doc_source or "").strip() or file_name
        result = await get_document_indexing_service().index_documents(
            pdf_path=pdf_path,
            force_rebuild=force_rebuild,
            collection_name=collection_name,
            doc_source=source,
            progress_callback=_progress_callback_for(task_id),
        )
        tracker.complete(
            task_id,
            _augment_index_result(
                result,
                uploaded=True,
                upload_file_name=file_name,
                upload_size_bytes=size_bytes,
            ),
        )
    except Exception as exc:
        tracker.fail(task_id, str(exc))
    finally:
        shutil.rmtree(str(Path(pdf_path).parent), ignore_errors=True)


@router.post("/documents/index")
async def index_documents(request: Request):
    payload = await _parse_document_index_request(request)
    result = await get_document_indexing_service().index_documents(
        pdf_path=payload.pdf_path,
        force_rebuild=payload.force_rebuild,
        collection_name=payload.collection_name,
        doc_source=payload.doc_source or None,
    )
    return _augment_index_result(result, uploaded=False)


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(..., description="PDF file to index"),
    collection_name: str = Form("default"),
    force_rebuild: bool = Form(False),
    doc_source: str = Form(""),
):
    target_path, original_name, size_bytes = await _save_upload_to_temp(file)
    collection = (collection_name or "default").strip() or "default"
    try:
        source = (doc_source or "").strip() or original_name
        result = await get_document_indexing_service().index_documents(
            pdf_path=str(target_path),
            force_rebuild=force_rebuild,
            collection_name=collection,
            doc_source=source,
        )
        return _augment_index_result(
            result,
            uploaded=True,
            upload_file_name=original_name,
            upload_size_bytes=size_bytes,
        )
    finally:
        shutil.rmtree(str(target_path.parent), ignore_errors=True)


@router.post("/documents/upload/start")
async def start_upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to index"),
    collection_name: str = Form("default"),
    force_rebuild: bool = Form(False),
    doc_source: str = Form(""),
):
    target_path, original_name, size_bytes = await _save_upload_to_temp(file)
    collection = (collection_name or "default").strip() or "default"
    tracker = get_index_progress_tracker()
    task = tracker.create(collection_name=collection, file_name=original_name)
    task_id = str(task["task_id"])
    tracker.update_step(
        task_id,
        "upload",
        status="completed",
        progress=100,
        detail="File saved and ready for indexing",
        fields={"file_name": original_name, "size_bytes": size_bytes},
    )
    background_tasks.add_task(
        _run_upload_index_task,
        task_id,
        str(target_path),
        original_name,
        size_bytes,
        collection,
        force_rebuild,
        doc_source,
    )
    return tracker.get(task_id)


@router.get("/documents/tasks/{task_id}")
async def get_document_task(task_id: str):
    task = get_index_progress_tracker().get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail={"task_id": task_id, "message": "Document task not found"})
    return task
