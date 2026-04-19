from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

import pytest

from service.embedding.embedding_service import EmbeddingService
from service.pdf.document_indexer import DocumentIndexingService
from service.pdf.structured_chunker import StructuredChunker


@pytest.fixture
def workspace_tmp_dir() -> Path:
    root = Path.cwd() / ".tmp_pytest_workspace"
    root.mkdir(parents=True, exist_ok=True)
    case_dir = Path(tempfile.mkdtemp(prefix="index_logging_", dir=str(root)))
    try:
        yield case_dir
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


class EmptyMinerUClient:
    def parse_pdf_to_mineru_json(self, pdf_path, use_cache=True, force_rebuild=False):
        return {
            "source": "test_empty_ocr",
            "pdf_info": [
                {
                    "page_idx": 0,
                    "para_blocks": [
                        {
                            "index": 0,
                            "type": "text",
                            "lines": [{"spans": [{"content": ""}]}],
                        }
                    ],
                }
            ],
        }


def test_index_documents_skips_when_ocr_produces_no_chunks(workspace_tmp_dir, caplog):
    pdf_path = workspace_tmp_dir / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n% empty test pdf\n%%EOF\n")
    service = DocumentIndexingService(
        mineru_client=EmptyMinerUClient(),
        chunker=StructuredChunker(),
        embedding_service=EmbeddingService(),
    )

    with caplog.at_level(logging.INFO, logger="trusted_qa.operation"):
        result = asyncio.run(
            service.index_documents(
                pdf_path=str(pdf_path),
                collection_name="empty_ocr",
                force_rebuild=True,
            )
        )

    assert result["success"] is True
    assert result["collection_name"] == "empty_ocr"
    assert result["indexed_doc_count"] == 0
    assert result["indexed_chunks"] == 0
    assert result["skipped_documents"] == 1
    assert result["skipped_documents_detail"][0]["reason"] == "no_chunks_after_chunking"

    log_output = "\n".join(record.getMessage() for record in caplog.records)
    assert "index.collect_documents" in log_output
    assert "index.ocr" in log_output
    assert "index.chunking" in log_output
    assert "index.document.skipped" in log_output


def test_app_config_is_single_merged_config():
    from utils.config_loader import get_app_config

    config = get_app_config(reload=True)
    assert config["agent"]["orchestration"] == "trusted_qa_workflow"
    assert config["storage"]["backend"] == "pgvector"
    assert config["pdf"]["parser"] == "mineru"