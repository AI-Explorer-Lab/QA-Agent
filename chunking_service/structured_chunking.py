from __future__ import annotations

import logging
import os
import re
from html import unescape
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunking_service.heading_recovery import (
    RecoveredHeading,
    normalize_heading_match_key,
    normalize_text,
    recover_high_conf_headings_from_mineru_payload,
    recover_headings_from_paragraph_texts,
)
from core.config_loader import load_runtime_env

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency at runtime
    tiktoken = None

load_runtime_env()


CHUNK_TOKEN_SIZE = int(os.getenv("CHUNK_SIZE_TOKENS", "1024"))
CHUNK_TOKEN_OVERLAP = int(os.getenv("CHUNK_OVERLAP_TOKENS", "200"))
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "7000"))
TABLE_CONTEXT_SHORT_TEXT_TOKENS = int(os.getenv("TABLE_CONTEXT_SHORT_TEXT_TOKENS", "40"))
CHUNK_HEADING_MAX_LEVEL = max(1, min(3, int(os.getenv("CHUNK_HEADING_MAX_LEVEL", "2"))))

_ENCODING_NAME = "cl100k_base"
_HEADING_PATH_SEPARATOR = " > "


def count_tokens(text: str) -> int:
    """Count tokens with tiktoken, fallback to rough character estimate."""
    if not text:
        return 0
    if tiktoken is None:
        return max(1, len(text) // 4)

    try:
        encoding = tiktoken.get_encoding(_ENCODING_NAME)
        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def build_recursive_splitter() -> RecursiveCharacterTextSplitter:
    separators = [
        "\n\n",
        "\n",
        "\u3002",
        "\uff1b",
        "\uff01",
        "\uff1f",
        "\uff0c",
        "\uff1a",
        "\u3001",
        ".",
        "!",
        "?",
        ",",
        ":",
        " ",
        "",
    ]
    try:
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=_ENCODING_NAME,
            chunk_size=CHUNK_TOKEN_SIZE,
            chunk_overlap=CHUNK_TOKEN_OVERLAP,
            separators=separators,
            is_separator_regex=False,
        )
    except Exception:
        return RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=CHUNK_TOKEN_SIZE,
            chunk_overlap=CHUNK_TOKEN_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )


def _empty_heading_metadata() -> Dict[str, str]:
    return _build_heading_metadata()


def _build_heading_metadata(
    level1_title: str = "",
    level2_title: str = "",
    level3_title: str = "",
) -> Dict[str, str]:
    level1_title = normalize_text(level1_title)
    level2_title = normalize_text(level2_title)
    level3_title = normalize_text(level3_title)
    path_parts = [item for item in [level1_title, level2_title, level3_title] if item]
    return {
        "level1_title": level1_title,
        "level2_title": level2_title,
        "level3_title": level3_title,
        "heading_path": _HEADING_PATH_SEPARATOR.join(path_parts) if path_parts else "front_matter",
    }


def chunk_plain_text(
    text: str,
    doc_id: str,
    doc_source: str,
    chunk_start_index: int = 0,
) -> List[Dict[str, Any]]:
    splitter = build_recursive_splitter()
    text_chunks = splitter.split_text(text or "")
    chunks: List[Dict[str, Any]] = []

    chunk_index = chunk_start_index
    heading_meta = _empty_heading_metadata()
    for item in text_chunks:
        item = normalize_text(item)
        if not item:
            continue
        chunk_id = f"{doc_id}_chunk_{chunk_index}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "doc_source": doc_source,
                "source": doc_source,
                "content": item,
                "chunk_type": "text",
                "chunk_index": chunk_index,
                **heading_meta,
            }
        )
        chunk_index += 1

    return chunks


def _rows_to_markdown(rows: List[List[str]]) -> str:
    if not rows:
        return ""

    width = len(rows[0])
    normalized_rows: List[List[str]] = []
    for row in rows:
        row = [str(cell).strip() for cell in row]
        if len(row) < width:
            row = row + [""] * (width - len(row))
        elif len(row) > width:
            row = row[:width]
        normalized_rows.append(row)

    header = normalized_rows[0]
    body = normalized_rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def wrap_table_markdown(rows: List[List[str]]) -> str:
    markdown = _rows_to_markdown(rows)
    return f"<TABLE_START>\n{markdown}\n</TABLE_END>"


class _TableHTMLParser(HTMLParser):
    """Lightweight HTML table parser for MinerU span['html'] payload."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self._in_row = False
        self._in_cell = False
        self._cell_parts: List[str] = []
        self._cell_rowspan = 1
        self._cell_colspan = 1
        self._current_row: List[str] = []
        self._current_col = 0
        self._pending_rowspans: Dict[int, Tuple[int, str]] = {}

    def _apply_pending_cells(self) -> None:
        while self._current_col in self._pending_rowspans:
            remain, text = self._pending_rowspans[self._current_col]
            self._current_row.append(text)
            self._current_col += 1
            if remain <= 1:
                del self._pending_rowspans[self._current_col - 1]
            else:
                self._pending_rowspans[self._current_col - 1] = (remain - 1, text)

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        name = tag.lower()
        if name == "tr":
            self._in_row = True
            self._current_row = []
            self._current_col = 0
            self._apply_pending_cells()
            return

        if name not in {"td", "th"} or not self._in_row:
            return

        attr_map = {str(key).lower(): str(value or "") for key, value in attrs}
        try:
            self._cell_rowspan = max(1, int(attr_map.get("rowspan", "1")))
        except Exception:
            self._cell_rowspan = 1
        try:
            self._cell_colspan = max(1, int(attr_map.get("colspan", "1")))
        except Exception:
            self._cell_colspan = 1
        self._in_cell = True
        self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_cell and data:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name in {"td", "th"} and self._in_row and self._in_cell:
            self._apply_pending_cells()
            text = normalize_text(unescape("".join(self._cell_parts)))
            for _ in range(self._cell_colspan):
                self._current_row.append(text)
                if self._cell_rowspan > 1:
                    self._pending_rowspans[self._current_col] = (self._cell_rowspan - 1, text)
                self._current_col += 1
            self._in_cell = False
            self._cell_parts = []
            self._cell_rowspan = 1
            self._cell_colspan = 1
            return

        if name == "tr" and self._in_row:
            self._apply_pending_cells()
            if any(normalize_text(item) for item in self._current_row):
                self.rows.append([normalize_text(item) for item in self._current_row])
            self._in_row = False
            self._current_row = []
            self._current_col = 0


def _parse_table_html_rows(html_content: str) -> List[List[str]]:
    text = str(html_content or "").strip()
    if not text:
        return []
    parser = _TableHTMLParser()
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        return []
    rows = parser.rows
    return [row for row in rows if any(normalize_text(cell) for cell in row)]


def _shrink_row_to_fit(
    header: List[str],
    row: List[str],
    max_chunk_size: int,
) -> List[str]:
    candidate = [str(cell) for cell in row]
    attempts = 0
    while count_tokens(wrap_table_markdown([header, candidate])) > max_chunk_size and attempts < 200:
        longest_idx = max(range(len(candidate)), key=lambda idx: len(candidate[idx]))
        text = candidate[longest_idx]
        if len(text) <= 16:
            break
        candidate[longest_idx] = text[: max(8, int(len(text) * 0.8))] + "..."
        attempts += 1

    if count_tokens(wrap_table_markdown([header, candidate])) > max_chunk_size:
        candidate = ["[ROW_EXCEEDS_MAX_CHUNK_SIZE]"] * len(header)
    return candidate


def split_table_rows_by_token(
    rows: List[List[str]],
    max_chunk_size: int = MAX_CHUNK_SIZE,
) -> List[List[List[str]]]:
    if not rows:
        return []

    header = [str(cell).strip() for cell in rows[0]]
    if not header:
        return []

    data_rows = [[str(cell).strip() for cell in row] for row in rows[1:]]
    if not data_rows:
        single = [header]
        if count_tokens(wrap_table_markdown(single)) > max_chunk_size:
            header = _shrink_row_to_fit(header, header, max_chunk_size)
            single = [header]
        return [single]

    subtables: List[List[List[str]]] = []
    current: List[List[str]] = [header]

    for row in data_rows:
        candidate = current + [row]
        if count_tokens(wrap_table_markdown(candidate)) <= max_chunk_size:
            current.append(row)
            continue

        if len(current) > 1:
            subtables.append(current)

        adjusted_row = row
        if count_tokens(wrap_table_markdown([header, adjusted_row])) > max_chunk_size:
            adjusted_row = _shrink_row_to_fit(header, adjusted_row, max_chunk_size)

        current = [header, adjusted_row]

    if current:
        subtables.append(current)

    return subtables


def _build_table_context_meta(text_chunk: Dict[str, Any]) -> Dict[str, Any]:
    text = normalize_text(text_chunk.get("content", ""))
    if not text:
        return {}
    return {
        "table_context_text": text[:800],
    }


def _build_table_header_text(rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return ""
    header_row = rows[0] if rows else []
    header_cells = [normalize_text(str(cell)) for cell in header_row]
    header_cells = [cell for cell in header_cells if cell]
    if not header_cells:
        return ""
    return " | ".join(header_cells)[:1000]


def _extract_docx_elements(doc) -> List[Dict[str, Any]]:
    elements: List[Dict[str, Any]] = []
    table_index = 0

    paragraph_map = {id(paragraph._element): paragraph for paragraph in doc.paragraphs}
    table_map = {id(table._element): table for table in doc.tables}

    for element in doc.element.body:
        if element.tag.endswith("p"):
            paragraph = paragraph_map.get(id(element))
            if paragraph is None:
                continue
            text = normalize_text(paragraph.text)
            if text:
                style_name = getattr(getattr(paragraph, "style", None), "name", "")
                elements.append(
                    {
                        "type": "paragraph",
                        "text": text,
                        "style_name": style_name,
                    }
                )
            continue

        if element.tag.endswith("tbl"):
            table = table_map.get(id(element))
            if table is None:
                continue
            rows = [[normalize_text(cell.text) for cell in row.cells] for row in table.rows]
            rows = [row for row in rows if any(row)]
            if rows:
                table_index += 1
                elements.append(
                    {
                        "type": "table",
                        "rows": rows,
                        "table_id": f"table_{table_index}",
                    }
                )

    return elements


def _extract_mineru_block_text(block: Dict[str, Any]) -> str:
    fragments: List[str] = []
    for line in block.get("lines", []) or []:
        if not isinstance(line, dict):
            continue
        for span in line.get("spans", []) or []:
            if not isinstance(span, dict):
                continue
            content = normalize_text(span.get("content"))
            if content:
                fragments.append(content)
    return normalize_text("".join(fragments))


def _extract_table_rows_from_mineru_block(block: Dict[str, Any]) -> List[List[str]]:
    for inner_block in block.get("blocks", []) or []:
        if not isinstance(inner_block, dict):
            continue
        for line in inner_block.get("lines", []) or []:
            if not isinstance(line, dict):
                continue
            for span in line.get("spans", []) or []:
                if not isinstance(span, dict):
                    continue
                if span.get("type") != "table":
                    continue
                rows = _parse_table_html_rows(str(span.get("html", "")))
                if rows:
                    return rows

    text_fallback = _extract_mineru_block_text(block)
    if text_fallback:
        return [["table_text"], [text_fallback]]
    return []


def _extract_mineru_json_elements(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    elements: List[Dict[str, Any]] = []
    pages = payload.get("pdf_info", []) or []
    if not isinstance(pages, list):
        return elements

    table_index = 0
    sorted_pages = sorted(
        [item for item in pages if isinstance(item, dict)],
        key=lambda page: int(page.get("page_idx", -1)),
    )
    for page in sorted_pages:
        blocks = page.get("para_blocks", []) or []
        sorted_blocks = sorted(
            [item for item in blocks if isinstance(item, dict)],
            key=lambda block: int(block.get("index", -1)),
        )
        for block in sorted_blocks:
            block_type = str(block.get("type", "")).strip().lower()
            if block_type in {"text", "title", "list"}:
                text = _extract_mineru_block_text(block)
                if text:
                    elements.append({"type": "paragraph", "text": text, "style_name": block_type})
                continue

            if block_type == "table":
                rows = _extract_table_rows_from_mineru_block(block)
                rows = [row for row in rows if any(normalize_text(cell) for cell in row)]
                if rows:
                    table_index += 1
                    elements.append(
                        {
                            "type": "table",
                            "rows": rows,
                            "table_id": f"table_{table_index}",
                        }
                    )

    return elements


def render_docx_elements_as_text(doc) -> str:
    elements = _extract_docx_elements(doc)
    parts: List[str] = []
    for item in elements:
        if item["type"] == "paragraph":
            parts.append(item["text"])
        elif item["type"] == "table":
            parts.append(wrap_table_markdown(item["rows"]))
    return "\n\n".join(parts)


def render_mineru_json_elements_as_text(payload: Dict[str, Any]) -> str:
    elements = _extract_mineru_json_elements(payload)
    parts: List[str] = []
    for item in elements:
        if item["type"] == "paragraph":
            parts.append(item["text"])
        elif item["type"] == "table":
            parts.append(wrap_table_markdown(item["rows"]))
    return "\n\n".join(parts)


def _build_heading_fallback(parsed_elements: Sequence[Dict[str, Any]]) -> List[RecoveredHeading]:
    paragraph_texts = [
        item["text"]
        for item in parsed_elements
        if item.get("type") == "paragraph" and normalize_text(item.get("text"))
    ]
    return recover_headings_from_paragraph_texts(paragraph_texts)


def _heading_keys_match(paragraph_text: str, heading: RecoveredHeading) -> bool:
    paragraph_key = normalize_heading_match_key(paragraph_text)
    heading_key = heading.match_key or normalize_heading_match_key(heading.text)
    return (
        paragraph_key == heading_key
        or paragraph_key.startswith(heading_key)
        or heading_key.startswith(paragraph_key)
    )


def _match_next_heading(
    paragraph_text: str,
    recovered_headings: Sequence[RecoveredHeading],
    start_index: int,
    lookahead: int = 6,
) -> Tuple[Optional[RecoveredHeading], int]:
    if start_index >= len(recovered_headings):
        return None, start_index

    upper_bound = min(len(recovered_headings), start_index + lookahead)
    for idx in range(start_index, upper_bound):
        heading = recovered_headings[idx]
        if _heading_keys_match(paragraph_text, heading):
            return heading, idx + 1
    return None, start_index


def _filter_headings_by_level(
    headings: Sequence[RecoveredHeading],
    max_level: int,
) -> List[RecoveredHeading]:
    limit = max(1, min(3, int(max_level)))
    return [item for item in headings if int(item.level) <= limit]


def _apply_heading_state(
    current_state: Dict[str, str],
    heading: RecoveredHeading,
) -> Dict[str, str]:
    level1_title = current_state.get("level1_title", "")
    level2_title = current_state.get("level2_title", "")
    level3_title = current_state.get("level3_title", "")

    if heading.level == 1:
        level1_title = heading.text
        level2_title = ""
        level3_title = ""
    elif heading.level == 2:
        level2_title = heading.text
        level3_title = ""
    elif heading.level == 3:
        level3_title = heading.text

    return _build_heading_metadata(
        level1_title=level1_title,
        level2_title=level2_title,
        level3_title=level3_title,
    )


def _chunk_structured_elements(
    parsed_elements: Sequence[Dict[str, Any]],
    doc_id: str,
    doc_source: str,
    chunk_start_index: int = 0,
    max_chunk_size: int = MAX_CHUNK_SIZE,
    recovered_headings: Optional[Sequence[RecoveredHeading]] = None,
) -> List[Dict[str, Any]]:
    splitter = build_recursive_splitter()
    heading_entries = list(recovered_headings or [])
    if not heading_entries:
        heading_entries = _build_heading_fallback(parsed_elements)
    heading_entries = _filter_headings_by_level(heading_entries, CHUNK_HEADING_MAX_LEVEL)

    chunks: List[Dict[str, Any]] = []
    chunk_index = chunk_start_index
    heading_cursor = 0
    current_heading_meta = _build_heading_metadata()
    paragraph_buffer: List[str] = []
    buffer_has_body = False
    recent_paragraphs: List[str] = []
    last_element_type = ""

    def flush_paragraph_buffer(force: bool = False) -> List[Dict[str, Any]]:
        nonlocal chunk_index, buffer_has_body
        emitted_chunks: List[Dict[str, Any]] = []
        if not paragraph_buffer:
            return emitted_chunks

        merged_text = "\n\n".join(normalize_text(item) for item in paragraph_buffer if normalize_text(item))
        paragraph_buffer.clear()
        if not merged_text:
            buffer_has_body = False
            return emitted_chunks

        if not force and not buffer_has_body:
            buffer_has_body = False
            return emitted_chunks

        for text_chunk in splitter.split_text(merged_text):
            text_chunk = normalize_text(text_chunk)
            if not text_chunk:
                continue

            chunk_id = f"{doc_id}_chunk_{chunk_index}"
            chunk_record: Dict[str, Any] = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "doc_source": doc_source,
                "source": doc_source,
                "content": text_chunk,
                "chunk_type": "text",
                "chunk_index": chunk_index,
                **current_heading_meta,
            }
            chunks.append(chunk_record)
            emitted_chunks.append(chunk_record)
            chunk_index += 1

        buffer_has_body = False
        return emitted_chunks

    for item in parsed_elements:
        if item["type"] == "paragraph":
            paragraph_text = normalize_text(item["text"])
            if not paragraph_text:
                continue

            matched_heading, new_cursor = _match_next_heading(
                paragraph_text=paragraph_text,
                recovered_headings=heading_entries,
                start_index=heading_cursor,
            )
            if matched_heading is not None:
                flush_paragraph_buffer()
                current_heading_meta = _apply_heading_state(current_heading_meta, matched_heading)
                heading_cursor = new_cursor
                paragraph_buffer.append(paragraph_text)
                recent_paragraphs.append(paragraph_text)
                recent_paragraphs = recent_paragraphs[-20:]
                last_element_type = "paragraph"
                continue

            paragraph_buffer.append(paragraph_text)
            recent_paragraphs.append(paragraph_text)
            recent_paragraphs = recent_paragraphs[-20:]
            buffer_has_body = True
            last_element_type = "paragraph"
            continue

        emitted_text_chunks = flush_paragraph_buffer()

        table_context_meta: Dict[str, Any] = {}
        if last_element_type == "paragraph" and emitted_text_chunks:
            candidate_text_chunk = emitted_text_chunks[-1]
            candidate_tokens = count_tokens(candidate_text_chunk.get("content", ""))
            same_heading = (
                candidate_text_chunk.get("heading_path", "") == current_heading_meta.get("heading_path", "")
            )
            if candidate_tokens <= TABLE_CONTEXT_SHORT_TEXT_TOKENS and same_heading:
                if chunks and chunks[-1].get("chunk_id") == candidate_text_chunk.get("chunk_id"):
                    chunks.pop()
                    chunk_index = max(chunk_start_index, chunk_index - 1)
                table_context_meta = _build_table_context_meta(candidate_text_chunk)

        rows = item["rows"]
        table_id = f"{doc_id}_{item['table_id']}"
        sub_tables = split_table_rows_by_token(rows, max_chunk_size=max_chunk_size)
        sub_count = max(1, len(sub_tables))

        for sub_idx, sub_table_rows in enumerate(sub_tables, start=1):
            table_content = wrap_table_markdown(sub_table_rows)
            table_tokens = count_tokens(table_content)
            if table_tokens > max_chunk_size:
                logging.warning(
                    "Table sub-table still exceeds max token size after split. "
                    "table_id=%s sub_table_id=%s tokens=%s limit=%s",
                    table_id,
                    sub_idx,
                    table_tokens,
                    max_chunk_size,
                )

            chunk_id = f"{doc_id}_chunk_{chunk_index}"
            table_chunk: Dict[str, Any] = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "doc_source": doc_source,
                "source": doc_source,
                "content": table_content,
                "chunk_type": "table",
                "chunk_index": chunk_index,
                "table_id": table_id,
                "sub_table_id": f"{table_id}_{sub_idx}",
                "sub_table_index": sub_idx,
                "table_id_subtable_count": sub_count,
                "table_header_text": _build_table_header_text(sub_table_rows),
                **current_heading_meta,
            }
            table_chunk.update(table_context_meta)
            chunks.append(table_chunk)
            chunk_index += 1

        last_element_type = "table"

    flush_paragraph_buffer()
    return chunks


def chunk_docx_document(
    doc,
    doc_id: str,
    doc_source: str,
    chunk_start_index: int = 0,
    max_chunk_size: int = MAX_CHUNK_SIZE,
    recovered_headings: Optional[Sequence[RecoveredHeading]] = None,
) -> List[Dict[str, Any]]:
    parsed_elements = _extract_docx_elements(doc)
    return _chunk_structured_elements(
        parsed_elements=parsed_elements,
        doc_id=doc_id,
        doc_source=doc_source,
        chunk_start_index=chunk_start_index,
        max_chunk_size=max_chunk_size,
        recovered_headings=recovered_headings,
    )


def chunk_mineru_json_payload(
    payload: Dict[str, Any],
    doc_id: str,
    doc_source: str,
    chunk_start_index: int = 0,
    max_chunk_size: int = MAX_CHUNK_SIZE,
    recovered_headings: Optional[Sequence[RecoveredHeading]] = None,
) -> List[Dict[str, Any]]:
    parsed_elements = _extract_mineru_json_elements(payload)
    heading_entries = list(recovered_headings or [])
    if not heading_entries:
        heading_entries = recover_high_conf_headings_from_mineru_payload(payload)
    return _chunk_structured_elements(
        parsed_elements=parsed_elements,
        doc_id=doc_id,
        doc_source=doc_source,
        chunk_start_index=chunk_start_index,
        max_chunk_size=max_chunk_size,
        recovered_headings=heading_entries,
    )
