from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TRUSTED_QA_ENABLE_REAL_LLM"] = "0"

from fastapi.testclient import TestClient

from main import app


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "acceptance.pdf"
        pdf.write_bytes(b"%PDF-1.4\n% minimal acceptance pdf\n")
        client = TestClient(app)
        index_resp = client.post("/documents/index", json={"pdf_path": str(pdf), "collection_name": "e2e", "force_rebuild": True})
        assert index_resp.status_code == 200, index_resp.text
        ask_resp = client.post("/qa/ask", json={"question": "acceptance document parsed", "collection_name": "e2e"})
        assert ask_resp.status_code == 200, ask_resp.text
        payload = {"index": index_resp.json(), "ask": ask_resp.json()}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        assert payload["ask"]["decision"] in {"answer", "refuse", "clarify"}
        assert "retrieval_trace" in payload["ask"]


if __name__ == "__main__":
    main()
