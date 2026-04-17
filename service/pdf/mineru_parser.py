from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
from typing import Any, Dict, List

from utils.content_normalizer import normalize_whitespace


class _SimpleTableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self._in_row = False
        self._in_cell = False
        self._current_row: List[str] = []
        self._cell_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs):
        name = tag.lower()
        if name == "tr":
            self._in_row = True
            self._current_row = []
            return
        if self._in_row and name in {"td", "th"}:
            self._in_cell = True
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_cell and data:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str):
        name = tag.lower()
        if name in {"td", "th"} and self._in_row and self._in_cell:
            text = normalize_whitespace(unescape("".join(self._cell_parts)), preserve_newlines=False)
            self._current_row.append(text)
            self._in_cell = False
            self._cell_parts = []
            return

        if name == "tr" and self._in_row:
            if any(cell for cell in self._current_row):
                self.rows.append(self._current_row)
            self._in_row = False
            self._current_row = []


def _parse_table_html_rows(html_text: str) -> List[List[str]]:
    raw = str(html_text or "").strip()
    if not raw:
        return []

    parser = _SimpleTableHTMLParser()
    try:
        parser.feed(raw)
        parser.close()
    except Exception:
        return []
    return parser.rows


def _extract_block_text(block: Dict[str, Any]) -> str:
    fragments: List[str] = []
    for line in block.get("lines", []) or []:
        if not isinstance(line, dict):
            continue
        for span in line.get("spans", []) or []:
            if not isinstance(span, dict):
                continue
            content = normalize_whitespace(span.get("content"), preserve_newlines=False)
            if content:
                fragments.append(content)
    return normalize_whitespace("".join(fragments), preserve_newlines=False)


def _extract_table_rows(block: Dict[str, Any]) -> List[List[str]]:
    rows = block.get("table_rows")
    if isinstance(rows, list):
        normalized_rows = [
            [normalize_whitespace(cell, preserve_newlines=False) for cell in row]
            for row in rows
            if isinstance(row, list)
        ]
        normalized_rows = [row for row in normalized_rows if any(cell for cell in row)]
        if normalized_rows:
            return normalized_rows

    for inner_block in block.get("blocks", []) or []:
        if not isinstance(inner_block, dict):
            continue
        for line in inner_block.get("lines", []) or []:
            if not isinstance(line, dict):
                continue
            for span in line.get("spans", []) or []:
                if not isinstance(span, dict):
                    continue
                if str(span.get("type", "")).strip().lower() != "table":
                    continue
                parsed_rows = _parse_table_html_rows(str(span.get("html", "")))
                if parsed_rows:
                    return parsed_rows

    text_fallback = _extract_block_text(block)
    if text_fallback:
        return [["table_text"], [text_fallback]]
    return []


def parse_mineru_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    pdf_info = payload.get("pdf_info", []) or []
    if not isinstance(pdf_info, list):
        return []

    ordered_blocks: List[Dict[str, Any]] = []
    table_counter = 0

    pages = sorted(
        [page for page in pdf_info if isinstance(page, dict)],
        key=lambda page: int(page.get("page_idx", -1)),
    )

    for page in pages:
        page_idx = int(page.get("page_idx", -1))
        para_blocks = page.get("para_blocks", []) or []
        sorted_para_blocks = sorted(
            [block for block in para_blocks if isinstance(block, dict)],
            key=lambda block: int(block.get("index", -1)),
        )

        for block in sorted_para_blocks:
            raw_type = str(block.get("type", "text")).strip().lower()
            block_type = raw_type if raw_type in {"title", "text", "list", "table"} else "text"
            block_index = int(block.get("index", -1))

            if block_type == "table":
                rows = _extract_table_rows(block)
                if not rows:
                    continue
                table_counter += 1
                ordered_blocks.append(
                    {
                        "type": "table",
                        "page_idx": page_idx,
                        "block_index": block_index,
                        "rows": rows,
                        "table_id": f"table_{table_counter}",
                    }
                )
                continue

            text = _extract_block_text(block)
            if not text:
                continue
            ordered_blocks.append(
                {
                    "type": block_type,
                    "page_idx": page_idx,
                    "block_index": block_index,
                    "text": text,
                }
            )

    ordered_blocks.sort(key=lambda item: (int(item.get("page_idx", -1)), int(item.get("block_index", -1))))
    return ordered_blocks
