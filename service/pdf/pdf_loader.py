from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from utils.hash_utils import file_sha256

try:  # pragma: no cover - optional integration point
    from exceptions.base_exception import AppBaseException as _BusinessException
except Exception:  # pragma: no cover - fallback for this repository stage
    class _BusinessException(RuntimeError):
        pass


class PdfLoaderError(_BusinessException):
    """Raised when the input path is not a valid PDF source."""


@dataclass(frozen=True)
class PdfDocument:
    path: Path
    file_hash: str
    size_bytes: int



def _resolve_path(pdf_path: str | Path) -> Path:
    path = Path(pdf_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return path


def _validate_pdf_file(path: Path) -> Path:
    if path.suffix.lower() != ".pdf":
        raise PdfLoaderError(f"Only PDF files are supported, got: {path}")
    return path


def collect_pdf_paths(pdf_path: str | Path) -> List[Path]:
    path = _resolve_path(pdf_path)
    if not path.exists():
        raise PdfLoaderError(f"PDF path not found: {path}")

    if path.is_file():
        return [_validate_pdf_file(path)]

    if not path.is_dir():
        raise PdfLoaderError(f"Input must be a PDF file or directory: {path}")

    candidates = sorted(
        [item.resolve() for item in path.rglob("*.pdf") if item.is_file()],
        key=lambda item: str(item).lower(),
    )
    if not candidates:
        raise PdfLoaderError(f"No PDF files found in directory: {path}")
    return candidates


def collect_pdf_documents(pdf_path: str | Path) -> List[PdfDocument]:
    documents: List[PdfDocument] = []
    for path in collect_pdf_paths(pdf_path):
        documents.append(
            PdfDocument(
                path=path,
                file_hash=file_sha256(path),
                size_bytes=path.stat().st_size,
            )
        )
    return documents


