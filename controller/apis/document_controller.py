from __future__ import annotations

import json
import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError

from service.pdf.document_indexer import get_document_indexing_service

router = APIRouter()
_PATH_FIELD_PATTERN = re.compile(
    r'(?P<prefix>"(?P<key>pdf_path|doc_source)"\s*:\s*")(?P<value>[^"]*)(?P<suffix>")'
)


class DocumentIndexRequest(BaseModel):
    doc_source: str = ""
    pdf_path: str
    force_rebuild: bool = False
    collection_name: str = "default"


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


@router.post("/documents/index")
async def index_documents(request: Request):
    payload = await _parse_document_index_request(request)
    return await get_document_indexing_service().index_documents(
        pdf_path=payload.pdf_path,
        force_rebuild=payload.force_rebuild,
        collection_name=payload.collection_name,
        doc_source=payload.doc_source or None,
    )
