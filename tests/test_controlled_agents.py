from __future__ import annotations

import asyncio
import unittest

from service.agent.controlled_agents import (
    EvidenceAuditAgent,
    IntentUnderstandingAgent,
    SlotFillingAgent,
    merge_audit_and_rule_gate,
)


class ControlledAgentsTestCase(unittest.TestCase):
    def test_intent_agent_maps_to_fixed_query_type_without_confidence(self):
        result = asyncio.run(IntentUnderstandingAgent().classify("Where is this sentence cited? Provide the page source."))

        self.assertEqual(result["query_type"], "citation_locate")
        self.assertEqual(result["matched_keyword_group"], "_CITATION_KEYWORDS")
        self.assertEqual(result["intent"], "locate_source")
        self.assertNotIn("confidence", result)

    def test_slot_agent_returns_fixed_schema_and_full_year(self):
        result = asyncio.run(SlotFillingAgent().fill("What was 2025 revenue?", "table_qa"))

        self.assertEqual(set(result.keys()), {"years", "metric", "period", "target_statement", "compare_targets", "scope"})
        self.assertEqual(result["years"], ["2025"])
        self.assertEqual(result["period"], "2025")
        self.assertEqual(result["metric"], "revenue")

    def test_evidence_audit_can_request_retry_for_missing_table_aspect(self):
        audit = asyncio.run(
            EvidenceAuditAgent().audit(
                question="What was 2025 revenue?",
                query_type="table_qa",
                slots={"years": ["2025"], "metric": "revenue", "period": "2025"},
                selected_skill="TableQASkill",
                evidence=[{"content": "2025 net profit was 10.", "chunk_type": "text", "final_score": 0.9}],
                rerank_trace={},
            )
        )

        self.assertEqual(audit["semantic_decision"], "retry")
        self.assertIn("table_evidence", audit["missing_aspects"])
        self.assertEqual(audit["evidence_coverage"], "partial")

    def test_merge_audit_and_rule_gate_is_conservative(self):
        merged = merge_audit_and_rule_gate(
            {"decision": "answer", "reason": "evidence_passed", "confidence": 0.8},
            {
                "semantic_decision": "retry",
                "missing_aspects": ["metric"],
                "evidence_coverage": "partial",
                "conflict_detected": False,
                "suggested_retry_query": "2025 revenue table",
                "reason": "metric missing",
            },
        )

        self.assertEqual(merged["decision"], "retry")
        self.assertEqual(merged["rule_decision"], "answer")
        self.assertEqual(merged["semantic_decision"], "retry")
        self.assertEqual(merged["suggested_retry_query"], "2025 revenue table")


if __name__ == "__main__":
    unittest.main()