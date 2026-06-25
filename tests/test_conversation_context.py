from __future__ import annotations

import asyncio
import unittest

from service.agent.conversation_context import ConversationContextService


class RouteLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def structured_json(self, system_prompt, user_payload, schema, max_tokens=512):
        del system_prompt, schema, max_tokens
        self.calls.append(user_payload)
        return self.payload


def _assistant(content, decision="answer", slots=None, citations=None, evidence=None, effective_question=""):
    return {
        "message_id": "assistant-1",
        "role": "assistant",
        "content": content,
        "metadata": {
            "decision": decision,
            "query_type": "table_qa",
            "effective_question": effective_question,
            "citations": citations or [],
            "evidence": evidence or [],
            "skill_trace": {"slots": slots or {}},
            "retrieval_trace_id": "trace-1",
        },
    }


class ConversationContextTestCase(unittest.TestCase):
    def test_new_query_does_not_inherit_existing_focus(self):
        llm = RouteLLM(
            {
                "turn_type": "new_rag_query",
                "should_use_history": False,
                "context_source": "none",
                "effective_question": "合同里的违约责任有哪些？",
                "history_refs": [],
                "confidence": 0.9,
            }
        )
        service = ConversationContextService(llm)
        session = {
            "metadata": {
                "conversation_focus": {
                    "active_topic": "上海芯导 2025 年财报",
                    "company": "上海芯导",
                    "period": "2025",
                }
            },
            "messages": [],
        }

        result = asyncio.run(service.prepare_context(session, "合同里的违约责任有哪些？", "default"))

        self.assertEqual(result["turn_route"]["turn_type"], "new_rag_query")
        self.assertEqual(result["turn_route"]["context_source"], "none")
        self.assertFalse(result["turn_route"]["should_use_history"])

    def test_follow_up_uses_conversation_focus(self):
        llm = RouteLLM(
            {
                "turn_type": "follow_up",
                "should_use_history": True,
                "context_source": "conversation_focus",
                "effective_question": "上海芯导 2025 年净利润是多少？",
                "history_refs": ["focus"],
                "confidence": 0.92,
            }
        )
        service = ConversationContextService(llm)
        session = {
            "metadata": {
                "conversation_focus": {
                    "active_topic": "上海芯导 2025 年营业收入是多少？",
                    "company": "上海芯导",
                    "period": "2025",
                    "metric": "营业收入",
                }
            },
            "messages": [],
        }

        result = asyncio.run(service.prepare_context(session, "那净利润呢？", "default"))

        self.assertEqual(result["effective_question"], "上海芯导 2025 年净利润是多少？")
        self.assertEqual(result["turn_route"]["context_source"], "conversation_focus")
        self.assertEqual(result["turn_route"]["history_refs"], ["focus"])

    def test_clarification_reply_uses_latest_pending_turn(self):
        llm = RouteLLM(
            {
                "turn_type": "clarification_reply",
                "should_use_history": True,
                "context_source": "clarification_pending",
                "effective_question": "",
                "history_refs": [],
                "confidence": 0.95,
            }
        )
        service = ConversationContextService(llm)
        session = {
            "metadata": {},
            "messages": [
                {"message_id": "user-1", "role": "user", "content": "营业收入是多少？", "metadata": {}},
                _assistant(
                    "为了给出可信答案，请补充：时间范围。",
                    decision="clarify",
                    slots={"metric": "营业收入", "__missing_required__": ["period"]},
                    effective_question="营业收入是多少？",
                ),
            ],
        }

        result = asyncio.run(service.prepare_context(session, "2025 年", "default"))

        self.assertEqual(result["turn_route"]["turn_type"], "clarification_reply")
        self.assertEqual(result["turn_route"]["context_source"], "clarification_pending")
        self.assertIn("营业收入是多少？", result["effective_question"])
        self.assertIn("2025 年", result["effective_question"])

    def test_citation_followup_uses_last_evidence_when_available(self):
        llm = RouteLLM(
            {
                "turn_type": "citation_followup",
                "should_use_history": True,
                "context_source": "last_evidence",
                "effective_question": "这个数据来自哪一页？",
                "history_refs": [],
                "confidence": 0.91,
            }
        )
        service = ConversationContextService(llm)
        session = {
            "metadata": {},
            "messages": [
                {"message_id": "user-1", "role": "user", "content": "上海芯导 2025 年营业收入是多少？", "metadata": {}},
                _assistant(
                    "营业收入为 10 亿元 [C1]",
                    citations=[{"citation_id": "C1", "page_idx": 12}],
                    evidence=[{"chunk_id": "chunk-1", "doc_source": "report.pdf", "content": "营业收入为 10 亿元"}],
                    effective_question="上海芯导 2025 年营业收入是多少？",
                ),
            ],
        }

        result = asyncio.run(service.prepare_context(session, "这个数据来自哪一页？", "default"))

        self.assertEqual(result["turn_route"]["context_source"], "last_evidence")
        self.assertEqual(result["turn_route"]["history_refs"], ["last_evidence"])

    def test_focus_updates_after_answer_and_keeps_refuse_from_overwriting(self):
        service = ConversationContextService()
        previous = {"active_topic": "旧主题", "period": "2024"}
        answered = service.build_focus_after_response(
            previous_focus=previous,
            effective_question="上海芯导 2025 年净利润是多少？",
            query_type="table_qa",
            slots={"period": "2025", "metric": "净利润"},
            response={"decision": "answer", "evidence": [{"doc_source": "report.pdf"}]},
            turn_route={"turn_type": "follow_up"},
        )

        self.assertEqual(answered["period"], "2025")
        self.assertEqual(answered["metric"], "净利润")
        self.assertEqual(answered["doc_scope"], ["report.pdf"])

        refused = service.build_focus_after_response(
            previous_focus=answered,
            effective_question="另一个问题",
            query_type="fact_lookup",
            slots={},
            response={"decision": "refuse", "evidence": []},
            turn_route={"turn_type": "new_rag_query"},
        )

        self.assertEqual(refused["active_topic"], answered["active_topic"])


if __name__ == "__main__":
    unittest.main()
