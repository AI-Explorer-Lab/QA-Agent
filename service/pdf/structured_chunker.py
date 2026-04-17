from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence

from service.pdf.heading_recovery import HeadingState, apply_heading, build_heading_metadata, detect_heading_level
from service.pdf.mineru_parser import parse_mineru_payload
from utils.content_normalizer import normalize_whitespace
from utils.token_counter import count_tokens

try:  # pragma: no cover - optional runtime dependency
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    RecursiveCharacterTextSplitter = None

if TYPE_CHECKING:  # pragma: no cover - integration hook for upcoming domain package
    try:
        from domain.chunk import PdfChunk  # type: ignore
    except Exception:
        PdfChunk = Dict[str, Any]  # type: ignore

EMBEDDING_DIMENSION = 1024


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size_tokens: int = 1024
    chunk_overlap_tokens: int = 200
    max_chunk_size_tokens: int = 7000
    embedding_dim: int = EMBEDDING_DIMENSION


class StructuredChunker:
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        self._splitter = self._build_recursive_splitter()

    def _build_recursive_splitter(self):
        if RecursiveCharacterTextSplitter is None:
            return None

        separators = [
            "\n\n",
            "\n",
            "。",
            "；",
            "！",
            "？",
            "，",
            "、",
            ".",
            ";",
            "!",
            "?",
            ",",
            " ",
            "",
        ]

        try:
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=self.config.chunk_size_tokens,
                chunk_overlap=self.config.chunk_overlap_tokens,
                separators=separators,
                is_separator_regex=False,
            )
        except Exception:
            return RecursiveCharacterTextSplitter(
                separators=separators,
                chunk_size=self.config.chunk_size_tokens,
                chunk_overlap=self.config.chunk_overlap_tokens,
                length_function=count_tokens,
                is_separator_regex=False,
            )

    def _split_text(self, text: str) -> List[str]:
        normalized = normalize_whitespace(text, preserve_newlines=True)
        if not normalized:
            return []

        if self._splitter is None:
            return self._fallback_split_text(normalized)

        pieces = self._splitter.split_text(normalized)
        return [normalize_whitespace(piece, preserve_newlines=True) for piece in pieces if piece and piece.strip()]

    def _fallback_split_text(self, text: str) -> List[str]:
        max_tokens = max(1, int(self.config.chunk_size_tokens))
        overlap = max(0, int(self.config.chunk_overlap_tokens))
        words = text.split()
        if not words:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = start
            segment_words: List[str] = []
            while end < len(words):
                candidate = " ".join(segment_words + [words[end]])
                if count_tokens(candidate) > max_tokens and segment_words:
                    break
                segment_words.append(words[end])
                end += 1

            chunk_text = normalize_whitespace(" ".join(segment_words), preserve_newlines=True)
            if chunk_text:
                chunks.append(chunk_text)

            if end >= len(words):
                break
            if overlap <= 0:
                start = end
                continue
            overlap_words = max(1, overlap)
            start = max(start + 1, end - overlap_words)
        return chunks

    @staticmethod
    def _normalize_table_rows(rows: Sequence[Sequence[Any]]) -> List[List[str]]:
        normalized = [
            [normalize_whitespace(cell, preserve_newlines=False) for cell in row]
            for row in rows
            if isinstance(row, (list, tuple))
        ]
        normalized = [row for row in normalized if any(cell for cell in row)]
        if not normalized:
            return []

        width = max(len(row) for row in normalized)
        return [row + [""] * (width - len(row)) for row in normalized]

    @staticmethod
    def _rows_to_markdown(rows: Sequence[Sequence[str]]) -> str:
        normalized_rows = [list(row) for row in rows if row]
        if not normalized_rows:
            return ""

        header = normalized_rows[0]
        body = normalized_rows[1:]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]
        for row in body:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def _table_to_chunk_text(self, rows: Sequence[Sequence[str]]) -> str:
        markdown = self._rows_to_markdown(rows)
        if not markdown:
            return ""
        return f"<TABLE_START>\n{markdown}\n</TABLE_END>"

    def _truncate_row_to_limit(self, header: List[str], row: List[str]) -> List[str]:
        candidate = list(row)
        for _ in range(256):
            if count_tokens(self._table_to_chunk_text([header, candidate])) <= self.config.max_chunk_size_tokens:
                return candidate
            idx = max(range(len(candidate)), key=lambda i: len(candidate[i]))
            if len(candidate[idx]) <= 16:
                break
            candidate[idx] = candidate[idx][: max(8, int(len(candidate[idx]) * 0.8))] + "..."

        if count_tokens(self._table_to_chunk_text([header, candidate])) <= self.config.max_chunk_size_tokens:
            return candidate
        return ["[ROW_EXCEEDS_MAX_CHUNK_SIZE]"] * len(header)

    def split_table_rows_by_token(self, rows: Sequence[Sequence[Any]]) -> List[List[List[str]]]:
        normalized = self._normalize_table_rows(rows)
        if not normalized:
            return []

        header = normalized[0]
        data_rows = normalized[1:]
        if not data_rows:
            return [[header]]

        subtables: List[List[List[str]]] = []
        current: List[List[str]] = [header]

        for row in data_rows:
            candidate = current + [row]
            if count_tokens(self._table_to_chunk_text(candidate)) <= self.config.max_chunk_size_tokens:
                current.append(row)
                continue

            if len(current) > 1:
                subtables.append(current)

            adjusted_row = row
            if count_tokens(self._table_to_chunk_text([header, adjusted_row])) > self.config.max_chunk_size_tokens:
                adjusted_row = self._truncate_row_to_limit(header, adjusted_row)

            current = [header, adjusted_row]

        if current:
            subtables.append(current)
        return subtables

    @staticmethod
    def _page_range_from_pages(pages: Sequence[int]) -> List[int]:
        valid_pages = [int(page) for page in pages if isinstance(page, int) and page >= 0]
        if not valid_pages:
            return [-1, -1]
        return [min(valid_pages), max(valid_pages)]

    def chunk_parsed_blocks(
        self,
        parsed_blocks: Sequence[Dict[str, Any]],
        doc_id: str,
        collection_name: str,
        doc_source: str,
        chunk_start_index: int = 0,
    ) -> List[Dict[str, Any]]:
        ordered_blocks = sorted(
            [dict(block) for block in parsed_blocks if isinstance(block, dict)],
            key=lambda block: (int(block.get("page_idx", -1)), int(block.get("block_index", -1))),
        )

        chunks: List[Dict[str, Any]] = []
        chunk_index = int(chunk_start_index)
        heading_state = HeadingState()

        text_buffer: List[str] = []
        text_pages: List[int] = []
        last_text_chunk_content = ""
        auto_table_counter = 0

        def flush_text_buffer() -> str:
            nonlocal chunk_index, text_buffer, text_pages, last_text_chunk_content
            merged = normalize_whitespace("\n\n".join(text_buffer), preserve_newlines=True)
            if not merged:
                text_buffer = []
                text_pages = []
                return ""

            heading_meta = build_heading_metadata(heading_state)
            page_range = self._page_range_from_pages(text_pages)
            page_idx = page_range[0]
            emitted_last_text = ""

            for part in self._split_text(merged):
                chunk_id = f"{doc_id}_chunk_{chunk_index}"
                chunk_record = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "collection_name": collection_name,
                    "doc_source": doc_source,
                    "content": part,
                    "page_idx": page_idx,
                    "page_range": list(page_range),
                    "chunk_type": "text",
                    "chunk_index": chunk_index,
                    "table_id": "",
                    "sub_table_id": "",
                    "table_header_text": "",
                    "table_context_text": "",
                    "embedding_dim": EMBEDDING_DIMENSION,
                    **heading_meta,
                }
                chunks.append(chunk_record)
                emitted_last_text = part
                chunk_index += 1

            text_buffer = []
            text_pages = []
            if emitted_last_text:
                last_text_chunk_content = emitted_last_text
            return emitted_last_text

        for block in ordered_blocks:
            block_type = str(block.get("type", "text")).strip().lower()
            page_idx = int(block.get("page_idx", -1))

            if block_type in {"title", "text", "list"}:
                text = normalize_whitespace(block.get("text", ""), preserve_newlines=False)
                if not text:
                    continue

                if block_type == "title":
                    heading_level = detect_heading_level(text)
                    if heading_level > 0:
                        flush_text_buffer()
                        heading_state = apply_heading(heading_state, text, level=heading_level)

                text_buffer.append(text)
                text_pages.append(page_idx)
                continue

            if block_type != "table":
                continue

            flush_text_buffer()
            table_rows = self._normalize_table_rows(block.get("rows", []))
            if not table_rows:
                continue

            auto_table_counter += 1
            table_id = str(block.get("table_id") or f"table_{auto_table_counter}")
            heading_meta = build_heading_metadata(heading_state)
            table_context_text = normalize_whitespace(last_text_chunk_content, preserve_newlines=False)[:800]
            sub_tables = self.split_table_rows_by_token(table_rows)
            total_sub_tables = max(1, len(sub_tables))

            for sub_idx, sub_rows in enumerate(sub_tables, start=1):
                table_content = self._table_to_chunk_text(sub_rows)
                if not table_content:
                    continue

                chunk_id = f"{doc_id}_chunk_{chunk_index}"
                table_chunk = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "collection_name": collection_name,
                    "doc_source": doc_source,
                    "content": table_content,
                    "page_idx": page_idx,
                    "page_range": [page_idx, page_idx],
                    "chunk_type": "table",
                    "chunk_index": chunk_index,
                    "table_id": table_id,
                    "sub_table_id": f"{table_id}_{sub_idx}",
                    "table_header_text": " | ".join(sub_rows[0]) if sub_rows else "",
                    "table_context_text": table_context_text,
                    "embedding_dim": EMBEDDING_DIMENSION,
                    "table_id_subtable_count": total_sub_tables,
                    **heading_meta,
                }
                chunks.append(table_chunk)
                chunk_index += 1

        flush_text_buffer()
        return chunks

    def chunk_mineru_payload(
        self,
        mineru_payload: Dict[str, Any],
        doc_id: str,
        collection_name: str,
        doc_source: str,
        chunk_start_index: int = 0,
    ) -> List[Dict[str, Any]]:
        parsed_blocks = parse_mineru_payload(mineru_payload)
        return self.chunk_parsed_blocks(
            parsed_blocks=parsed_blocks,
            doc_id=doc_id,
            collection_name=collection_name,
            doc_source=doc_source,
            chunk_start_index=chunk_start_index,
        )


_default_chunker = StructuredChunker()


def chunk_parsed_blocks(
    parsed_blocks: Sequence[Dict[str, Any]],
    doc_id: str,
    collection_name: str,
    doc_source: str,
    chunk_start_index: int = 0,
) -> List[Dict[str, Any]]:
    return _default_chunker.chunk_parsed_blocks(
        parsed_blocks=parsed_blocks,
        doc_id=doc_id,
        collection_name=collection_name,
        doc_source=doc_source,
        chunk_start_index=chunk_start_index,
    )


def chunk_mineru_payload(
    mineru_payload: Dict[str, Any],
    doc_id: str,
    collection_name: str,
    doc_source: str,
    chunk_start_index: int = 0,
) -> List[Dict[str, Any]]:
    return _default_chunker.chunk_mineru_payload(
        mineru_payload=mineru_payload,
        doc_id=doc_id,
        collection_name=collection_name,
        doc_source=doc_source,
        chunk_start_index=chunk_start_index,
    )
