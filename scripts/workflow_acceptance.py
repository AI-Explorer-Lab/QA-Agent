from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TRUSTED_QA_ENABLE_REAL_LLM"] = "0"

from service.agent.trusted_qa_workflow import get_trusted_qa_workflow
from service.retrieval.runtime import reset_runtime_repository, upsert_runtime_chunks


async def main() -> None:
    reset_runtime_repository()
    upsert_runtime_chunks(
        [
            {
                "chunk_id": "acceptance-1",
                "doc_id": "doc-acceptance",
                "collection_name": "acceptance",
                "doc_source": "policy.pdf",
                "raw_doc": "The employee handbook states that the probation period is three months and conversion requires manager approval.",
                "content": "The employee handbook states that the probation period is three months and conversion requires manager approval.",
                "chunk_type": "text",
                "page_idx": 1,
                "page_range": "1-1",
                "heading_path": "Section 1 > Probation",
            },
            {
                "chunk_id": "acceptance-2",
                "doc_id": "doc-acceptance",
                "collection_name": "acceptance",
                "doc_source": "policy.pdf",
                "raw_doc": "After the probation period, the employee must submit a conversion request for manager approval.",
                "content": "After the probation period, the employee must submit a conversion request for manager approval.",
                "chunk_type": "text",
                "page_idx": 1,
                "page_range": "1-1",
                "heading_path": "Section 1 > Probation",
            }
        ]
    )
    result = await get_trusted_qa_workflow().ask(
        question="How long is the probation period?",
        collection_name="acceptance",
        top_k=3,
    )
    print(json.dumps({"decision": result["decision"], "query_type": result["query_type"], "citations": len(result["citations"]), "answer": result["answer"]}, ensure_ascii=False, indent=2))
    assert result["decision"] == "answer"
    assert result["citations"]


if __name__ == "__main__":
    asyncio.run(main())
