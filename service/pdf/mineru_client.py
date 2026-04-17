from __future__ import annotations

import copy
import json
import os
import re
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from service.pdf.document_parse_cache import DocumentParseCache
from service.pdf.heading_recovery import detect_heading_level
from service.pdf.pdf_loader import PdfLoaderError, collect_pdf_paths
from utils.content_normalizer import normalize_whitespace
from utils.hash_utils import build_cache_key, file_sha256

try:  # pragma: no cover - optional runtime dependency
    import fitz
except Exception:  # pragma: no cover
    fitz = None

try:  # pragma: no cover - optional integration point
    from exception.business_exception import BusinessException as _BusinessException
except Exception:  # pragma: no cover
    class _BusinessException(RuntimeError):
        pass


class MinerUClientError(_BusinessException):
    pass


_LIST_RE = re.compile(r"^(?:[-*•●]|\d+[、.．])\s+.+$")


class MinerUClient:
    def __init__(
        self,
        cache_ttl_seconds: int = 3600,
        cache_max_items: int = 5000,
        document_parse_cache: DocumentParseCache | None = None,
        mineru_json_root: str | Path | None = None,
    ) -> None:
        self.document_parse_cache = document_parse_cache or DocumentParseCache(
            ttl_seconds=cache_ttl_seconds,
            max_items=cache_max_items,
        )
        self.mineru_json_root = Path(mineru_json_root).resolve() if mineru_json_root else None

    def _cache_key(self, pdf_file: Path) -> str:
        digest = file_sha256(pdf_file)
        mtime = int(pdf_file.stat().st_mtime_ns)
        return build_cache_key(str(pdf_file), digest, mtime, prefix="document_parse")

    def _resolve_mineru_json_path(self, pdf_file: Path, mineru_json_path: str | Path | None) -> Optional[Path]:
        if mineru_json_path:
            candidate = Path(mineru_json_path).expanduser()
            if not candidate.is_absolute():
                candidate = (Path.cwd() / candidate).resolve()
            else:
                candidate = candidate.resolve()

            if candidate.is_file():
                return candidate
            if candidate.is_dir():
                path_in_dir = candidate / f"{pdf_file.stem}.json"
                if path_in_dir.exists() and path_in_dir.is_file():
                    return path_in_dir
                raise MinerUClientError(f"MinerU json file not found for PDF: {path_in_dir}")
            raise MinerUClientError(f"Invalid mineru_json_path: {candidate}")

        if self.mineru_json_root and self.mineru_json_root.exists():
            candidate = self.mineru_json_root / f"{pdf_file.stem}.json"
            if candidate.exists() and candidate.is_file():
                return candidate

        env_dir = os.getenv("MINERU_JSON_DIR", "").strip()
        if env_dir:
            env_root = Path(env_dir).expanduser()
            if not env_root.is_absolute():
                env_root = (Path.cwd() / env_root).resolve()
            else:
                env_root = env_root.resolve()
            candidate = env_root / f"{pdf_file.stem}.json"
            if candidate.exists() and candidate.is_file():
                return candidate

        return None

    @staticmethod
    def _split_text_into_paragraphs(block_text: str) -> List[str]:
        normalized = str(block_text or "").replace("\r\n", "\n").replace("\r", "\n")
        parts = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part and part.strip()]
        if not parts:
            return []

        paragraphs: List[str] = []
        for part in parts:
            lines = [normalize_whitespace(line, preserve_newlines=False) for line in part.split("\n") if line.strip()]
            if len(lines) > 1:
                paragraphs.extend(line for line in lines if line)
            else:
                merged = normalize_whitespace(part, preserve_newlines=False)
                if merged:
                    paragraphs.append(merged)
        return paragraphs

    @staticmethod
    def _rows_to_html(rows: List[List[str]]) -> str:
        rendered_rows: List[str] = []
        for row in rows:
            cells = "".join(f"<td>{escape(str(cell))}</td>" for cell in row)
            rendered_rows.append(f"<tr>{cells}</tr>")
        return "<table>" + "".join(rendered_rows) + "</table>"

    @staticmethod
    def _normalize_table_rows(rows: List[List[Any]]) -> List[List[str]]:
        normalized_rows = [
            [normalize_whitespace(cell, preserve_newlines=False) for cell in row]
            for row in rows
            if isinstance(row, list)
        ]
        normalized_rows = [row for row in normalized_rows if any(cell for cell in row)]
        if not normalized_rows:
            return []

        width = max(len(row) for row in normalized_rows)
        return [row + [""] * (width - len(row)) for row in normalized_rows]

    @staticmethod
    def _classify_text_block_type(paragraph_text: str) -> str:
        text = normalize_whitespace(paragraph_text, preserve_newlines=False)
        if not text:
            return "text"
        if detect_heading_level(text) > 0:
            return "title"
        if _LIST_RE.match(text):
            return "list"
        return "text"

    def _build_text_para_block(self, text: str, block_index: int, bbox: List[float]) -> Dict[str, Any]:
        block_type = self._classify_text_block_type(text)
        return {
            "index": block_index,
            "type": block_type,
            "bbox": bbox,
            "lines": [{"spans": [{"content": text}]}],
        }

    def _build_table_para_block(self, rows: List[List[str]], block_index: int, bbox: List[float]) -> Dict[str, Any]:
        html_text = self._rows_to_html(rows)
        return {
            "index": block_index,
            "type": "table",
            "bbox": bbox,
            "table_rows": rows,
            "blocks": [
                {
                    "lines": [
                        {
                            "spans": [
                                {
                                    "type": "table",
                                    "html": html_text,
                                }
                            ]
                        }
                    ]
                }
            ],
        }

    def _extract_tables_from_page(self, page) -> List[Tuple[List[List[str]], List[float]]]:
        if not hasattr(page, "find_tables"):
            return []

        try:
            found = page.find_tables()
        except Exception:
            return []

        tables = getattr(found, "tables", []) or []
        extracted: List[Tuple[List[List[str]], List[float]]] = []
        for table in tables:
            try:
                rows = self._normalize_table_rows(table.extract() or [])
            except Exception:
                continue
            if not rows:
                continue
            bbox_obj = getattr(table, "bbox", None)
            if bbox_obj and len(bbox_obj) == 4:
                bbox = [float(bbox_obj[0]), float(bbox_obj[1]), float(bbox_obj[2]), float(bbox_obj[3])]
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]
            extracted.append((rows, bbox))
        return extracted

    def _minimal_fallback_payload(self, pdf_file: Path) -> Dict[str, Any]:
        summary_text = normalize_whitespace(f"{pdf_file.stem} document parsed by minimal MinerU fallback", preserve_newlines=False)
        block_type = "title" if detect_heading_level(summary_text) > 0 else "text"
        return {
            "source": "mineru_fallback_minimal",
            "pdf_path": str(pdf_file),
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [0.0, 0.0],
                    "para_blocks": [
                        {
                            "index": 0,
                            "type": block_type,
                            "bbox": [0.0, 0.0, 0.0, 0.0],
                            "lines": [{"spans": [{"content": summary_text}]}],
                        }
                    ],
                }
            ],
        }

    def _fallback_parse_with_pymupdf(self, pdf_file: Path) -> Dict[str, Any]:
        if fitz is None:
            return self._minimal_fallback_payload(pdf_file)

        try:
            document = fitz.open(str(pdf_file))
        except Exception:
            return self._minimal_fallback_payload(pdf_file)

        pdf_info: List[Dict[str, Any]] = []

        try:
            for page_idx in range(document.page_count):
                page = document.load_page(page_idx)
                items: List[Dict[str, Any]] = []

                for block in page.get_text("blocks") or []:
                    if len(block) < 5:
                        continue
                    if len(block) >= 7:
                        try:
                            if int(block[6]) != 0:
                                continue
                        except Exception:
                            pass

                    raw_text = str(block[4] or "")
                    if not raw_text.strip():
                        continue

                    items.append(
                        {
                            "kind": "text",
                            "x": float(block[0]),
                            "y": float(block[1]),
                            "bbox": [float(block[0]), float(block[1]), float(block[2]), float(block[3])],
                            "text": raw_text,
                        }
                    )

                for rows, bbox in self._extract_tables_from_page(page):
                    items.append(
                        {
                            "kind": "table",
                            "x": float(bbox[0]) if len(bbox) == 4 else 0.0,
                            "y": float(bbox[1]) if len(bbox) == 4 else 0.0,
                            "bbox": bbox,
                            "rows": rows,
                        }
                    )

                items.sort(key=lambda item: (float(item.get("y", 0.0)), float(item.get("x", 0.0))))

                para_blocks: List[Dict[str, Any]] = []
                block_index = 0
                for item in items:
                    if item["kind"] == "table":
                        para_blocks.append(
                            self._build_table_para_block(
                                rows=item.get("rows", []),
                                block_index=block_index,
                                bbox=item.get("bbox", [0.0, 0.0, 0.0, 0.0]),
                            )
                        )
                        block_index += 1
                        continue

                    for paragraph in self._split_text_into_paragraphs(str(item.get("text", ""))):
                        para_blocks.append(
                            self._build_text_para_block(
                                text=paragraph,
                                block_index=block_index,
                                bbox=item.get("bbox", [0.0, 0.0, 0.0, 0.0]),
                            )
                        )
                        block_index += 1

                if not para_blocks:
                    para_blocks = [
                        {
                            "index": 0,
                            "type": "text",
                            "bbox": [0.0, 0.0, 0.0, 0.0],
                            "lines": [{"spans": [{"content": ""}]}],
                        }
                    ]

                pdf_info.append(
                    {
                        "page_idx": page_idx,
                        "page_size": [float(page.rect.width), float(page.rect.height)],
                        "para_blocks": para_blocks,
                    }
                )
        finally:
            document.close()

        if not pdf_info:
            return self._minimal_fallback_payload(pdf_file)

        return {
            "source": "mineru_fallback_pymupdf",
            "pdf_path": str(pdf_file),
            "pdf_info": pdf_info,
        }

    @staticmethod
    def _load_json_payload(json_path: Path) -> Dict[str, Any]:
        try:
            with json_path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except Exception as exc:
            raise MinerUClientError(f"Failed to load MinerU JSON: {json_path}") from exc

        if not isinstance(payload, dict):
            raise MinerUClientError(f"Invalid MinerU JSON payload format: {json_path}")
        return payload

    def _try_remote_mineru(self, pdf_file: Path) -> Optional[Dict[str, Any]]:
        _ = pdf_file
        return None

    def parse_pdf_to_mineru_json(
        self,
        pdf_path: str | Path,
        mineru_json_path: str | Path | None = None,
        use_cache: bool = True,
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        pdf_files = collect_pdf_paths(pdf_path)
        if len(pdf_files) != 1:
            raise PdfLoaderError("MinerU client expects a single PDF file path")

        pdf_file = pdf_files[0]
        cache_key = self._cache_key(pdf_file)

        if use_cache and not force_rebuild:
            cached_payload = self.document_parse_cache.get(cache_key)
            if cached_payload is not None:
                return copy.deepcopy(cached_payload)

        resolved_json_path = self._resolve_mineru_json_path(pdf_file, mineru_json_path)
        if resolved_json_path:
            payload = self._load_json_payload(resolved_json_path)
        else:
            payload = self._try_remote_mineru(pdf_file)
            if payload is None:
                payload = self._fallback_parse_with_pymupdf(pdf_file)

        if not isinstance(payload, dict) or "pdf_info" not in payload:
            raise MinerUClientError("Invalid MinerU payload: missing pdf_info")

        if use_cache:
            self.document_parse_cache.set(cache_key, copy.deepcopy(payload))
        return payload


_default_mineru_client = MinerUClient()


def parse_pdf_to_mineru_json(
    pdf_path: str | Path,
    mineru_json_path: str | Path | None = None,
    use_cache: bool = True,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    return _default_mineru_client.parse_pdf_to_mineru_json(
        pdf_path=pdf_path,
        mineru_json_path=mineru_json_path,
        use_cache=use_cache,
        force_rebuild=force_rebuild,
    )
