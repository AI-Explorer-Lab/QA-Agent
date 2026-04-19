import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TRUSTED_QA_ENABLE_REAL_LLM"] = "0"

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def _create_sample_pdf(path: Path) -> None:
    try:
        import fitz  # type: ignore

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "2025 revenue was 123 billion yuan, gross margin was 34 percent.")
        page.insert_text((72, 120), "Table 4 Metric | Value | Unit")
        doc.save(str(path))
        doc.close()
    except Exception:
        path.write_bytes(b"%PDF-1.4\n% test fallback\n")


class QAApiTestCase(unittest.TestCase):
    def test_only_required_endpoints_exposed(self):
        self.assertEqual(client.post("/askLLM").status_code, 404)
        self.assertEqual(client.post("/team-leader-task").status_code, 404)
        self.assertEqual(client.post("/react-ask").status_code, 404)

    def test_health_endpoint(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")

    def test_qa_api_flow_smoke(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "qa_api_sample.pdf"
            _create_sample_pdf(pdf_path)

            index_resp = client.post(
                "/documents/index",
                json={
                    "pdf_path": str(pdf_path),
                    "collection_name": "qa_api_test",
                    "force_rebuild": True,
                },
            )
            self.assertEqual(index_resp.status_code, 200, index_resp.text)
            index_payload = index_resp.json()
            self.assertTrue(index_payload["success"])
            self.assertGreaterEqual(index_payload["indexed_doc_count"], 1)

            ask_resp = client.post(
                "/qa/ask",
                json={
                    "question": "Based on the document, what was 2025 revenue? Provide citations.",
                    "collection_name": "qa_api_test",
                    "top_k": 5,
                    "expand_query_num": 3,
                    "enable_cache": True,
                },
            )
            self.assertEqual(ask_resp.status_code, 200, ask_resp.text)
            ask_payload = ask_resp.json()

            required = {
                "answer",
                "decision",
                "query_type",
                "confidence",
                "citations",
                "evidence",
                "retrieval_trace",
                "rerank_trace",
                "session_id",
                "skill_trace",
                "react_observations",
            }
            self.assertTrue(required.issubset(set(ask_payload.keys())))

            session_id = ask_payload["session_id"]
            session_resp = client.get(f"/qa/sessions/{session_id}")
            self.assertEqual(session_resp.status_code, 200, session_resp.text)

            session_payload = session_resp.json()
            self.assertIn("messages", session_payload)
            self.assertIn("retrieval_traces", session_payload)
            self.assertGreaterEqual(len(session_payload["messages"]), 2)


if __name__ == "__main__":
    unittest.main()