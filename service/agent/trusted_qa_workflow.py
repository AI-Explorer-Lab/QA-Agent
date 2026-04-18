from __future__ import annotations

from typing import Any, Dict, List
from uuid import uuid4

from service.agent.answer_generator import AnswerGenerator
from service.agent.clarify_gate import run_clarify_gate
from service.agent.evidence_gate import EvidenceGate
from service.agent.query_classifier import classify_query_type
from service.agent.query_expander import expand_queries
from service.agent.skill_registry import DEFAULT_SKILL_REGISTRY
from service.embedding.embedding_service import EmbeddingService, build_embedding_provider_from_config
from service.evaluation.ragas_evaluator import evaluate_qa_result
from service.llm import get_llm_service
from service.retrieval.hybrid_retriever import HybridRetriever
from service.retrieval.parallel_query_executor import ParallelQueryExecutor
from service.retrieval.retrieval_cache import RetrievalResultCache
from service.retrieval.runtime import get_runtime_repository
from service.session.session_service import get_session_service
from utils.config_loader import get_app_config


def _query_expander_for_executor(question: str, expand_query_num: int) -> List[str]:
    return expand_queries(question, "fact_lookup", expand_query_num)[1:]


class TrustedQAWorkflow:
    def __init__(self) -> None:
        self.config = get_app_config()
        retrieval_cfg = self.config.get("retrieval", {}) if isinstance(self.config.get("retrieval"), dict) else {}
        cache_cfg = self.config.get("cache", {}) if isinstance(self.config.get("cache"), dict) else {}
        guard_cfg = self.config.get("guardrails", {}) if isinstance(self.config.get("guardrails"), dict) else {}
        self.session_service = get_session_service()
        self.skill_registry = DEFAULT_SKILL_REGISTRY
        self.llm_service = get_llm_service()
        self.embedding_service = EmbeddingService(provider=build_embedding_provider_from_config(self.config))
        self.retriever = HybridRetriever(
            ParallelQueryExecutor(
                repository=get_runtime_repository(),
                retrieval_cache=RetrievalResultCache(
                    ttl_seconds=int(cache_cfg.get("ttl_seconds", 3600)),
                    max_items=int(cache_cfg.get("max_items", 5000)),
                ),
                query_expander=_query_expander_for_executor,
                async_embedding_builder=lambda text_value: self.embedding_service.embed_text(text_value, use_cache=True, chunk_text=False),
                max_concurrency=int(retrieval_cfg.get("max_concurrency", 6)),
                query_timeout_seconds=float(retrieval_cfg.get("query_timeout_seconds", 20)),
            ),
            table_evidence_quota=int(retrieval_cfg.get("table_evidence_quota", 2)),
        )
        self.evidence_gate = EvidenceGate(
            evidence_min_docs=int(guard_cfg.get("evidence_min_docs", 1)),
            evidence_min_top_score=float(guard_cfg.get("evidence_min_top_score", 0.20)),
            evidence_min_avg_score=float(guard_cfg.get("evidence_min_avg_score", 0.10)),
            retry_limit=int(guard_cfg.get("retry_limit", 2)),
            refuse_on_low_evidence=bool(guard_cfg.get("refuse_on_low_evidence", True)),
        )
        self.answer_generator = AnswerGenerator()
        self.table_evidence_quota = int(retrieval_cfg.get("table_evidence_quota", 2))

    async def ask(
        self,
        question: str,
        collection_name: str = "default",
        session_id: str | None = None,
        top_k: int = 5,
        expand_query_num: int = 3,
        enable_cache: bool = True,
    ) -> Dict[str, Any]:
        session = self.session_service.load_session(session_id, collection_name=collection_name)
        sid = session["session_id"]
        query_type = classify_query_type(question)
        selected_skill = self.skill_registry.select_skill(query_type)
        llm_expanded = await self.llm_service.expand_queries(question, query_type, expand_query_num)
        expanded = llm_expanded or expand_queries(question, query_type, expand_query_num)
        llm_query_expansion_used = bool(llm_expanded)
        observations: List[Dict[str, Any]] = [
            {"phase": "load_session", "session_id": sid},
            {"phase": "classify_query_type", "query_type": query_type},
            {"phase": "select_skill_from_registry", "selected_skill": selected_skill.skill_name},
        ]

        clarify = run_clarify_gate(question, query_type, collection_name)
        if clarify["decision"] == "clarify":
            answer_payload = self.answer_generator.generate(
                question=question,
                query_type=query_type,
                evidence=[],
                decision="clarify",
                gate_reason="missing_slots",
            )
            response = self._build_response(
                sid,
                query_type,
                "clarify",
                answer_payload,
                {"expanded_queries": expanded, "observations": observations, "clarify": clarify},
                {},
                selected_skill.skill_name,
            )
            response["answer"] = clarify.get("clarify_question") or response["answer"]
            response["skill_trace"]["tool_chain"] = list(getattr(selected_skill, "tool_chain", []))
            response["retrieval_trace"]["tool_chain"] = response["skill_trace"]["tool_chain"]
            llm_trace = self.llm_service.trace_metadata()
            llm_trace.update(
                {
                    "query_expansion_used": llm_query_expansion_used,
                    "answer_generation_used": False,
                }
            )
            response["retrieval_trace"]["llm"] = llm_trace
            self._save(sid, question, response)
            return response

        retrieval_result = await self.retriever.retrieve(
            question=question,
            collection_name=collection_name,
            top_k=max(1, int(top_k)),
            query_type=query_type,
            expand_query_num=max(1, int(expand_query_num)),
            enable_cache=enable_cache,
            expanded_queries=expanded,
        )
        evidence = list(retrieval_result.get("evidence") or [])
        observations.append({"phase": "parallel_hybrid_retrieval", "evidence_count": len(evidence)})
        slots = clarify.get("slots", {})
        gate = self.evidence_gate.evaluate(
            evidence,
            query_type=query_type,
            retry_count=0,
            table_evidence_quota=self.table_evidence_quota,
            slots=slots,
        )
        observations.append({"phase": "evidence_gate", "gate": gate})

        retry_count = 0
        while gate.get("decision") == "retry" and retry_count < self.evidence_gate.retry_limit:
            retry_count += 1
            retry_result = await self.retriever.retrieve(
                question=question + " " + gate.get("reason", ""),
                collection_name=collection_name,
                top_k=max(1, int(top_k)),
                query_type=query_type,
                expand_query_num=max(1, int(expand_query_num)),
                enable_cache=False,
            )
            evidence = list(retry_result.get("evidence") or evidence)
            retrieval_result = retry_result
            gate = self.evidence_gate.evaluate(
                evidence,
                query_type=query_type,
                retry_count=retry_count,
                table_evidence_quota=self.table_evidence_quota,
                slots=slots,
            )
            observations.append({"phase": "retry_retrieval", "retry_count": retry_count, "gate": gate, "evidence_count": len(evidence)})

        decision = gate.get("decision", "refuse")
        if decision == "retry":
            decision = "refuse"
            gate = dict(gate)
            gate["decision"] = "refuse"
            gate["reason"] = gate.get("reason", "retry_limit_reached")
        if decision not in {"answer", "clarify", "refuse"}:
            decision = "refuse"

        answer_payload = self.answer_generator.generate(
            question=question,
            query_type=query_type,
            evidence=evidence,
            decision=decision,
            gate_reason=gate.get("reason", ""),
        )
        llm_answer_used = False
        if decision == "answer":
            llm_answer = await self.llm_service.generate_grounded_answer(
                question=question,
                query_type=query_type,
                evidence=answer_payload.get("evidence", []),
                citations=answer_payload.get("citations", []),
            )
            if llm_answer:
                answer_payload["answer"] = llm_answer
                llm_answer_used = True
        response = self._build_response(
            sid,
            query_type,
            decision,
            answer_payload,
            retrieval_result.get("retrieval_trace") or {},
            retrieval_result.get("rerank_trace") or {},
            selected_skill.skill_name,
            observations=observations,
            expanded_queries=expanded,
            gate=gate,
        )
        llm_trace = self.llm_service.trace_metadata()
        llm_trace.update(
            {
                "query_expansion_used": llm_query_expansion_used,
                "answer_generation_used": llm_answer_used,
            }
        )
        response["retrieval_trace"]["llm"] = llm_trace
        response["skill_trace"]["tool_chain"] = list(getattr(selected_skill, "tool_chain", []))
        response["retrieval_trace"]["tool_chain"] = response["skill_trace"]["tool_chain"]
        evaluation = evaluate_qa_result(
            question=question,
            answer=response["answer"],
            decision=response["decision"],
            citations=response["citations"],
            evidence=response["evidence"],
        )
        response["confidence"] = max(response["confidence"], float(evaluation.get("confidence", 0.0)))
        response["retrieval_trace"].setdefault("evaluation", evaluation)
        self._save(sid, question, response)
        return response

    def _build_response(
        self,
        session_id: str,
        query_type: str,
        decision: str,
        answer_payload: Dict[str, Any],
        retrieval_trace: Dict[str, Any],
        rerank_trace: Dict[str, Any],
        selected_skill: str,
        observations: List[Dict[str, Any]] | None = None,
        expanded_queries: List[str] | None = None,
        gate: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        trace = dict(retrieval_trace or {})
        trace.setdefault("trace_id", str(uuid4()))
        trace.setdefault("selected_skill", selected_skill)
        trace.setdefault("expanded_queries", expanded_queries or trace.get("query_variants", []))
        trace.setdefault("react_observations", observations or [])
        if gate:
            trace.setdefault("gate_decision", gate)
        react_observations = observations or []
        skill_trace = {
            "selected_skill": selected_skill,
            "tool_chain": trace.get("tool_chain", []),
            "observations": react_observations,
            "gate_decision": gate or trace.get("gate_decision", {}),
        }
        return {
            "answer": answer_payload.get("answer", ""),
            "decision": decision,
            "query_type": query_type,
            "confidence": float(answer_payload.get("confidence") or 0.0),
            "citations": answer_payload.get("citations", []),
            "evidence": answer_payload.get("evidence", []),
            "retrieval_trace": trace,
            "rerank_trace": dict(rerank_trace or {}),
            "skill_trace": skill_trace,
            "react_observations": react_observations,
            "session_id": session_id,
        }

    def _save(self, session_id: str, question: str, response: Dict[str, Any]) -> None:
        self.session_service.save_session(
            session_id=session_id,
            user_question=question,
            assistant_payload=response,
            retrieval_trace=response.get("retrieval_trace") or {},
        )


_DEFAULT_WORKFLOW = TrustedQAWorkflow()


def get_trusted_qa_workflow() -> TrustedQAWorkflow:
    return _DEFAULT_WORKFLOW


