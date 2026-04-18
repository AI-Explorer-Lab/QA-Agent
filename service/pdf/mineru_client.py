from __future__ import annotations

import copy
import io
import json
import logging
import math
import os
import re
import shutil
import time
import zipfile
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from core.config_loader import load_runtime_env
from middlewares.operation_log import log_operation_event
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
        remote_enabled: bool | None = None,
        max_pages_per_task: int | None = None,
        poll_interval_seconds: float = 15.0,
        max_poll_attempts: int = 60,
    ) -> None:
        self.document_parse_cache = document_parse_cache or DocumentParseCache(
            ttl_seconds=cache_ttl_seconds,
            max_items=cache_max_items,
        )
        self.mineru_json_root = Path(mineru_json_root).resolve() if mineru_json_root else None
        self.remote_enabled = self._resolve_remote_enabled(remote_enabled)
        self.max_pages_per_task = self._resolve_max_pages_per_task(max_pages_per_task)
        self.poll_interval_seconds = max(0.1, float(poll_interval_seconds))
        self.max_poll_attempts = max(1, int(max_poll_attempts))

    @staticmethod
    def _resolve_remote_enabled(value: bool | None) -> bool:
        if value is not None:
            return bool(value)
        config_value = None
        try:
            from utils.config_loader import get_app_config

            pdf_cfg = get_app_config().get("pdf", {})
            if isinstance(pdf_cfg, dict):
                config_value = pdf_cfg.get("mineru_remote_enabled")
        except Exception:
            config_value = None

        if config_value is None:
            config_value = os.getenv("MINERU_REMOTE_ENABLED", "true")
        return str(config_value).strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _resolve_max_pages_per_task(value: int | None) -> int:
        if value is not None:
            return max(1, int(value))
        try:
            from utils.config_loader import get_app_config

            pdf_cfg = get_app_config().get("pdf", {})
            if isinstance(pdf_cfg, dict):
                configured = pdf_cfg.get("max_pages_per_task")
                if configured:
                    return max(1, int(configured))
        except Exception:
            pass
        return 200

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

    @staticmethod
    def _safe_response_text(response: requests.Response, limit: int = 700) -> str:
        try:
            text = response.text
        except Exception:
            text = ""
        return str(text or "")[:limit]

    @staticmethod
    def _resolve_token_from_env() -> str:
        load_runtime_env()
        configured_env_name = os.getenv("MINERU_API_KEY_ENV", "MinerU_API_KEY").strip() or "MinerU_API_KEY"
        candidates = [configured_env_name, "MinerU_API_KEY", "MINERU_API_KEY"]
        for name in candidates:
            value = os.getenv(name, "").strip()
            if value:
                return value
        return ""

    @staticmethod
    def _read_page_count(pdf_file: Path) -> int:
        if fitz is None:
            return 1
        try:
            document = fitz.open(str(pdf_file))
            try:
                return max(1, int(document.page_count))
            finally:
                document.close()
        except Exception:
            return 1

    @staticmethod
    def _extract_mineru_payload_from_zip(zip_file: zipfile.ZipFile) -> Optional[Dict[str, Any]]:
        json_files = [name for name in zip_file.namelist() if name.lower().endswith(".json")]
        for file_name in json_files:
            try:
                payload = json.loads(zip_file.read(file_name).decode("utf-8"))
            except Exception as exc:
                log_operation_event(
                    "index.ocr.remote.zip_json_skipped",
                    status="warning",
                    level=logging.WARNING,
                    zip_member=file_name,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                continue

            if isinstance(payload, dict) and "pdf_info" in payload:
                return payload
        return None

    def _build_markdown_payload(self, pdf_file: Path, markdown_text: str) -> Dict[str, Any]:
        paragraphs = self._split_text_into_paragraphs(markdown_text)
        para_blocks = [
            self._build_text_para_block(text=paragraph, block_index=index, bbox=[0.0, 0.0, 0.0, 0.0])
            for index, paragraph in enumerate(paragraphs)
            if paragraph
        ]
        if not para_blocks:
            return self._minimal_fallback_payload(pdf_file)
        return {
            "source": "mineru_remote_markdown",
            "pdf_path": str(pdf_file),
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [0.0, 0.0],
                    "para_blocks": para_blocks,
                }
            ],
        }

    def _extract_remote_payload_from_zip_bytes(self, pdf_file: Path, content: bytes) -> Dict[str, Any]:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            payload = self._extract_mineru_payload_from_zip(archive)
            if payload is not None:
                payload = copy.deepcopy(payload)
                payload["source"] = "mineru_remote"
                return payload

            markdown_files = [name for name in archive.namelist() if name.lower().endswith(".md")]
            if markdown_files:
                markdown_parts: List[str] = []
                for name in markdown_files:
                    try:
                        markdown_parts.append(archive.read(name).decode("utf-8"))
                    except Exception:
                        continue
                if markdown_parts:
                    return self._build_markdown_payload(pdf_file, "\n\n".join(markdown_parts))

        raise MinerUClientError("MinerU result zip does not contain a usable JSON or Markdown payload.")

    def _request_remote_upload_url(self, headers: Dict[str, str], file_name: str) -> Tuple[str, str]:
        request_body = {
            "files": [{"name": file_name}],
            "model_version": "vlm",
            "is_ocr": True,
        }
        log_operation_event(
            "index.ocr.remote.upload_url",
            status="started",
            file_name=file_name,
            model_version=request_body["model_version"],
            is_ocr=request_body["is_ocr"],
        )
        response = requests.post(
            "https://mineru.net/api/v4/file-urls/batch",
            headers=headers,
            json=request_body,
            timeout=30,
        )
        if response.status_code != 200:
            raise MinerUClientError(
                f"MinerU upload URL request failed: http={response.status_code}, body={self._safe_response_text(response)}"
            )

        payload = response.json()
        if payload.get("code") != 0:
            raise MinerUClientError(f"MinerU upload URL API error: {payload.get('msg') or payload.get('code')}")

        data = payload.get("data") or {}
        batch_id = str(data.get("batch_id") or "").strip()
        file_urls = data.get("file_urls") or []
        file_url = str(file_urls[0] if file_urls else "").strip()
        if not batch_id or not file_url:
            raise MinerUClientError("MinerU upload URL API returned empty batch_id or file_url.")

        log_operation_event(
            "index.ocr.remote.upload_url",
            status="completed",
            file_name=file_name,
            batch_id=batch_id,
        )
        return batch_id, file_url

    def _upload_remote_pdf(self, file_path: Path, file_url: str, batch_id: str, split_index: int, split_count: int) -> None:
        log_operation_event(
            "index.ocr.remote.upload",
            status="started",
            split_index=split_index + 1,
            split_count=split_count,
            file_name=file_path.name,
            size_bytes=file_path.stat().st_size,
            batch_id=batch_id,
        )
        with file_path.open("rb") as file:
            response = requests.put(file_url, data=file, timeout=120)
        if response.status_code != 200:
            raise MinerUClientError(
                f"MinerU upload failed: http={response.status_code}, body={self._safe_response_text(response)}"
            )
        log_operation_event(
            "index.ocr.remote.upload",
            status="completed",
            split_index=split_index + 1,
            split_count=split_count,
            file_name=file_path.name,
            batch_id=batch_id,
        )

    def _poll_remote_payload(
        self,
        headers: Dict[str, str],
        batch_id: str,
        pdf_file: Path,
        split_index: int,
        split_count: int,
    ) -> Dict[str, Any]:
        result_url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
        for attempt in range(1, self.max_poll_attempts + 1):
            response = requests.get(result_url, headers=headers, timeout=30)
            if response.status_code != 200:
                log_operation_event(
                    "index.ocr.remote.poll",
                    status="waiting",
                    level=logging.WARNING,
                    split_index=split_index + 1,
                    split_count=split_count,
                    attempt=attempt,
                    max_attempts=self.max_poll_attempts,
                    batch_id=batch_id,
                    http_status=response.status_code,
                    response_text=self._safe_response_text(response, limit=300),
                )
                time.sleep(self.poll_interval_seconds)
                continue

            result_payload = response.json()
            if result_payload.get("code") != 0:
                log_operation_event(
                    "index.ocr.remote.poll",
                    status="waiting",
                    level=logging.WARNING,
                    split_index=split_index + 1,
                    split_count=split_count,
                    attempt=attempt,
                    max_attempts=self.max_poll_attempts,
                    batch_id=batch_id,
                    api_code=result_payload.get("code"),
                    api_message=result_payload.get("msg"),
                )
                time.sleep(self.poll_interval_seconds)
                continue

            extract_results = (result_payload.get("data") or {}).get("extract_result") or []
            if not extract_results:
                log_operation_event(
                    "index.ocr.remote.poll",
                    status="waiting",
                    split_index=split_index + 1,
                    split_count=split_count,
                    attempt=attempt,
                    max_attempts=self.max_poll_attempts,
                    batch_id=batch_id,
                    state="no_extract_result",
                )
                time.sleep(self.poll_interval_seconds)
                continue

            extract_result = extract_results[0] or {}
            state = str(extract_result.get("state") or "").strip().lower()
            log_operation_event(
                "index.ocr.remote.poll",
                status="waiting" if state != "done" else "completed",
                split_index=split_index + 1,
                split_count=split_count,
                attempt=attempt,
                max_attempts=self.max_poll_attempts,
                batch_id=batch_id,
                state=state or "unknown",
            )

            if state == "done":
                zip_url = str(extract_result.get("full_zip_url") or "").strip()
                if not zip_url:
                    raise MinerUClientError("MinerU extraction completed without full_zip_url.")
                zip_response = requests.get(zip_url, timeout=120)
                if zip_response.status_code != 200:
                    raise MinerUClientError(
                        f"MinerU result zip download failed: http={zip_response.status_code}, body={self._safe_response_text(zip_response)}"
                    )
                log_operation_event(
                    "index.ocr.remote.download",
                    status="completed",
                    split_index=split_index + 1,
                    split_count=split_count,
                    batch_id=batch_id,
                    zip_size_bytes=len(zip_response.content or b""),
                )
                return self._extract_remote_payload_from_zip_bytes(pdf_file, zip_response.content)

            if state == "failed":
                raise MinerUClientError(f"MinerU extraction failed for batch_id={batch_id}.")

            time.sleep(self.poll_interval_seconds)

        raise MinerUClientError(
            f"MinerU extraction timed out after {self.max_poll_attempts} polls for batch_id={batch_id}."
        )

    def _create_split_tasks(self, pdf_file: Path, page_count: int) -> Tuple[List[Tuple[int, Path]], Optional[Path]]:
        if fitz is None or page_count <= self.max_pages_per_task:
            return [(0, pdf_file)], None

        temp_dir = Path.cwd() / "docs" / "temp_pdf"
        temp_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        split_count = math.ceil(page_count / self.max_pages_per_task)
        split_tasks: List[Tuple[int, Path]] = []

        source_doc = fitz.open(str(pdf_file))
        try:
            for split_index in range(split_count):
                start_page = split_index * self.max_pages_per_task
                end_page = min((split_index + 1) * self.max_pages_per_task, page_count)
                split_path = temp_dir / f"{pdf_file.stem}_split_{split_index + 1}_{timestamp}.pdf"
                sub_doc = fitz.open()
                try:
                    for page_number in range(start_page, end_page):
                        sub_doc.insert_pdf(source_doc, from_page=page_number, to_page=page_number)
                    sub_doc.save(str(split_path))
                finally:
                    sub_doc.close()
                split_tasks.append((split_index, split_path))
        finally:
            source_doc.close()

        return split_tasks, temp_dir

    @staticmethod
    def _merge_remote_payloads(pdf_file: Path, payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_pdf_info: List[Dict[str, Any]] = []
        for payload in payloads:
            for page in payload.get("pdf_info") or []:
                if not isinstance(page, dict):
                    continue
                page_copy = copy.deepcopy(page)
                page_copy["page_idx"] = len(merged_pdf_info)
                merged_pdf_info.append(page_copy)

        if not merged_pdf_info:
            raise MinerUClientError("MinerU remote payload contains no pdf_info pages.")

        return {
            "source": "mineru_remote",
            "pdf_path": str(pdf_file),
            "remote_split_count": len(payloads),
            "pdf_info": merged_pdf_info,
        }

    def _try_remote_mineru(self, pdf_file: Path) -> Optional[Dict[str, Any]]:
        if not self.remote_enabled:
            log_operation_event(
                "index.ocr.remote",
                status="disabled",
                pdf_path=str(pdf_file),
                message="MinerU remote OCR is disabled by config or environment.",
            )
            return None

        token = self._resolve_token_from_env()
        if not token:
            log_operation_event(
                "index.ocr.remote",
                status="skipped",
                level=logging.WARNING,
                pdf_path=str(pdf_file),
                reason="missing_mineru_api_key",
            )
            return None

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        page_count = self._read_page_count(pdf_file)
        log_operation_event(
            "index.ocr.source",
            status="selected",
            pdf_path=str(pdf_file),
            source="mineru_remote",
            page_count=page_count,
            max_pages_per_task=self.max_pages_per_task,
        )

        temp_dir: Optional[Path] = None
        try:
            split_tasks, temp_dir = self._create_split_tasks(pdf_file, page_count)
            split_count = len(split_tasks)
            payloads: List[Dict[str, Any]] = []
            log_operation_event(
                "index.ocr.remote",
                status="started",
                pdf_path=str(pdf_file),
                page_count=page_count,
                split_count=split_count,
            )

            for split_index, split_path in split_tasks:
                batch_id, file_url = self._request_remote_upload_url(headers, split_path.name)
                self._upload_remote_pdf(split_path, file_url, batch_id, split_index, split_count)
                payload = self._poll_remote_payload(headers, batch_id, pdf_file, split_index, split_count)
                payloads.append(payload)

            merged = self._merge_remote_payloads(pdf_file, payloads)
            log_operation_event(
                "index.ocr.remote",
                status="completed",
                pdf_path=str(pdf_file),
                page_count=len(merged.get("pdf_info") or []),
                split_count=split_count,
            )
            return merged
        except Exception as exc:
            log_operation_event(
                "index.ocr.remote",
                status="failed",
                level=logging.ERROR,
                pdf_path=str(pdf_file),
                error_type=type(exc).__name__,
                error=str(exc),
                message="MinerU remote OCR failed; local fallback will be used if available.",
            )
            return None
        finally:
            if temp_dir is not None:
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass

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
                log_operation_event(
                    "index.ocr.cache",
                    status="hit",
                    pdf_path=str(pdf_file),
                    cache_key=cache_key,
                )
                return copy.deepcopy(cached_payload)
            log_operation_event(
                "index.ocr.cache",
                status="miss",
                pdf_path=str(pdf_file),
                cache_key=cache_key,
            )

        resolved_json_path = self._resolve_mineru_json_path(pdf_file, mineru_json_path)
        if resolved_json_path:
            log_operation_event(
                "index.ocr.source",
                status="selected",
                pdf_path=str(pdf_file),
                source="mineru_json_file",
                mineru_json_path=str(resolved_json_path),
            )
            payload = self._load_json_payload(resolved_json_path)
        else:
            payload = self._try_remote_mineru(pdf_file)
            if payload is None:
                log_operation_event(
                    "index.ocr.source",
                    status="warning",
                    level=logging.WARNING,
                    pdf_path=str(pdf_file),
                    source="pymupdf_fallback",
                    message="Remote MinerU OCR is unavailable; falling back to local PyMuPDF/minimal parsing.",
                )
                payload = self._fallback_parse_with_pymupdf(pdf_file)

        if not isinstance(payload, dict) or "pdf_info" not in payload:
            raise MinerUClientError("Invalid MinerU payload: missing pdf_info")

        if use_cache:
            self.document_parse_cache.set(cache_key, copy.deepcopy(payload))
            log_operation_event(
                "index.ocr.cache",
                status="stored",
                pdf_path=str(pdf_file),
                cache_key=cache_key,
                parser_source=payload.get("source", "unknown"),
            )
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
