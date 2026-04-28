from __future__ import annotations

import os
from contextvars import ContextVar

from typing import Any, Dict, List, Optional
from uuid import uuid4

from service.agent.answer_generator import AnswerGenerator
from service.agent.clarify_gate import build_clarify_question
from service.agent.controlled_agents import IntentUnderstandingAgent, SlotFillingAgent
from service.agent.evidence_gate import EvidenceDecisionEngine
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

try:
    from langgraph.graph import StateGraph
except Exception:
    StateGraph = None


_LANGGRAPH_BYPASS: ContextVar[bool] = ContextVar("trusted_qa_langgraph_bypass", default=False)


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


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
        self.intent_agent = IntentUnderstandingAgent(self.llm_service)
        self.slot_agent = SlotFillingAgent(self.llm_service)
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
        self.evidence_decision = EvidenceDecisionEngine(
            llm_service=self.llm_service,
            evidence_min_docs=int(guard_cfg.get("evidence_min_docs", 1)),
            evidence_min_top_score=float(guard_cfg.get("evidence_min_top_score", 0.20)),
            evidence_min_avg_score=float(guard_cfg.get("evidence_min_avg_score", 0.10)),
            retry_limit=int(guard_cfg.get("retry_limit", 2)),
            refuse_on_low_evidence=bool(guard_cfg.get("refuse_on_low_evidence", True)),
        )
        self.answer_generator = AnswerGenerator()
        self.table_evidence_quota = int(retrieval_cfg.get("table_evidence_quota", 2))
        workflow_cfg = self.config.get("workflow", {}) if isinstance(self.config.get("workflow"), dict) else {}
        self.langgraph_enabled = _env_truthy(
            "TRUSTED_QA_ENABLE_LANGGRAPH",
            bool(workflow_cfg.get("enable_langgraph", False)),
        )
        self.langgraph_available = StateGraph is not None
        self._langgraph_app: Optional[Any] = None

    def _langgraph_app_instance(self) -> Optional[Any]:
        if not self.langgraph_available:
            return None
        if self._langgraph_app is not None:
            return self._langgraph_app

        graph = StateGraph(dict)

        async def run_linear(state: Dict[str, Any]) -> Dict[str, Any]:
            token = _LANGGRAPH_BYPASS.set(True)
            try:
                response = await self.ask(
                    question=str(state.get("question") or ""),
                    collection_name=str(state.get("collection_name") or "default"),
                    session_id=state.get("session_id"),
                    top_k=int(state.get("top_k") or 5),
                    expand_query_num=int(state.get("expand_query_num") or 3),
                    enable_cache=bool(state.get("enable_cache", True)),
                )
            finally:
                _LANGGRAPH_BYPASS.reset(token)
            trace = response.get("retrieval_trace") if isinstance(response, dict) else None
            if isinstance(trace, dict):
                trace["workflow_runner"] = "langgraph"
            return {"response": response}

        graph.add_node("run_linear", run_linear)
        graph.set_entry_point("run_linear")
        graph.set_finish_point("run_linear")
        self._langgraph_app = graph.compile()
        return self._langgraph_app

    async def _ask_with_langgraph(
        self,
        question: str,
        collection_name: str,
        session_id: str | None,
        top_k: int,
        expand_query_num: int,
        enable_cache: bool,
    ) -> Optional[Dict[str, Any]]:
        app = self._langgraph_app_instance()
        if app is None:
            return None
        result = await app.ainvoke(
            {
                "question": question,
                "collection_name": collection_name,
                "session_id": session_id,
                "top_k": top_k,
                "expand_query_num": expand_query_num,
                "enable_cache": enable_cache,
            }
        )
        if isinstance(result, dict) and isinstance(result.get("response"), dict):
            return result.get("response")
        return None

    async def _maybe_run_langgraph(
        self,
        question: str,
        collection_name: str,
        session_id: str | None,
        top_k: int,
        expand_query_num: int,
        enable_cache: bool,
    ) -> Optional[Dict[str, Any]]:
        if not self.langgraph_enabled or not self.langgraph_available or _LANGGRAPH_BYPASS.get():
            return None
        try:
            return await self._ask_with_langgraph(
                question=question,
                collection_name=collection_name,
                session_id=session_id,
                top_k=top_k,
                expand_query_num=expand_query_num,
                enable_cache=enable_cache,
            )
        except Exception:
            return None

    async def ask(
        self,
        question: str,
        collection_name: str = "default",
        session_id: str | None = None,
        top_k: int = 5,
        expand_query_num: int = 3,
        enable_cache: bool = True,
    ) -> Dict[str, Any]:
        langgraph_response = await self._maybe_run_langgraph(
            question=question,
            collection_name=collection_name,
            session_id=session_id,
            top_k=top_k,
            expand_query_num=expand_query_num,
            enable_cache=enable_cache,
        )
        if isinstance(langgraph_response, dict):
            return langgraph_response

        session = self.session_service.load_session(session_id, collection_name=collection_name)
        sid = session["session_id"]
        intent_trace = await self.intent_agent.classify(question)
        query_type = str(intent_trace.get("query_type") or "fact_lookup")
        selected_skill = self.skill_registry.select_skill(query_type)
        slots = await self.slot_agent.fill(question, query_type, selected_skill)
        missing_slots = selected_skill.get_missing_slots(slots)
        skill_package = selected_skill.package_metadata()
        if not str(collection_name or "").strip():
            missing_slots.append("collection_name")
        clarify = {
            "decision": "clarify" if missing_slots else "answer",
            "missing_slots": missing_slots,
            "clarify_question": build_clarify_question(query_type, missing_slots) if missing_slots else "",
            "slots": slots,
        }
        observations: List[Dict[str, Any]] = [
            {"phase": "load_session", "session_id": sid},
            {"phase": "intent_understanding_agent", "intent": intent_trace},
            {"phase": "select_skill_from_registry", "selected_skill": selected_skill.skill_name, "skill_package": skill_package},
            {"phase": "slot_filling_agent", "slots": slots, "missing_slots": missing_slots},
        ]

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
                {
                    "expanded_queries": [],
                    "observations": observations,
                    "clarify": clarify,
                    "intent_trace": intent_trace,
                    "slots": slots,
                },
                {},
                selected_skill.skill_name,
            )
            response["answer"] = clarify.get("clarify_question") or response["answer"]
            response["skill_trace"]["tool_chain"] = list(getattr(selected_skill, "tool_chain", []))
            response["skill_trace"]["skill_package"] = skill_package
            response["skill_trace"]["intent_trace"] = intent_trace
            response["skill_trace"]["slots"] = slots
            response["retrieval_trace"]["tool_chain"] = response["skill_trace"]["tool_chain"]
            response["retrieval_trace"]["skill_package"] = skill_package
            llm_trace = self.llm_service.trace_metadata()
            llm_trace.update(
                {
                    "query_expansion_used": False,
                    "answer_generation_used": False,
                }
            )
            response["retrieval_trace"]["llm"] = llm_trace
            response["retrieval_trace"].setdefault("workflow_runner", "python")
            self._save(sid, question, response)
            return response

        llm_expanded = await self.llm_service.expand_queries(question, query_type, expand_query_num)
        expanded = llm_expanded or expand_queries(question, query_type, expand_query_num)
        llm_expansion_used = bool(llm_expanded)

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
        gate = await self.evidence_decision.evaluate(
            question=question,
            query_type=query_type,
            slots=slots,
            selected_skill=selected_skill.skill_name,
            evidence=evidence,
            rerank_trace=retrieval_result.get("rerank_trace") or {},
            retry_count=0,
            table_evidence_quota=self.table_evidence_quota,
        )
        observations.append(
            {
                "phase": "evidence_decision",
                "rule_gate": gate.get("rule_gate") or {},
                "audit": gate.get("evidence_audit") or {},
                "gate": gate,
            }
        )

        retry_count = 0
        while gate.get("decision") == "retry" and retry_count < self.evidence_decision.retry_limit:
            retry_count += 1
            retry_question = str(gate.get("suggested_retry_query") or "").strip() or question
            retry_result = await self.retriever.retrieve(
                question=retry_question,
                collection_name=collection_name,
                top_k=max(1, int(top_k)),
                query_type=query_type,
                expand_query_num=max(1, int(expand_query_num)),
                enable_cache=False,
            )
            evidence = list(retry_result.get("evidence") or evidence)
            retrieval_result = retry_result
            gate = await self.evidence_decision.evaluate(
                question=question,
                query_type=query_type,
                slots=slots,
                selected_skill=selected_skill.skill_name,
                evidence=evidence,
                rerank_trace=retrieval_result.get("rerank_trace") or {},
                retry_count=retry_count,
                table_evidence_quota=self.table_evidence_quota,
            )
            observations.append(
                {
                    "phase": "retry_retrieval",
                    "retry_count": retry_count,
                    "retry_question": retry_question,
                    "audit": gate.get("evidence_audit") or {},
                    "rule_gate": gate.get("rule_gate") or {},
                    "gate": gate,
                    "evidence_count": len(evidence),
                }
            )

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
                "query_expansion_used": llm_expansion_used,
                "answer_generation_used": llm_answer_used,
            }
        )
        response["retrieval_trace"]["llm"] = llm_trace
        response["retrieval_trace"].setdefault("workflow_runner", "python")
        response["skill_trace"]["tool_chain"] = list(getattr(selected_skill, "tool_chain", []))
        response["skill_trace"]["skill_package"] = skill_package
        response["skill_trace"]["intent_trace"] = intent_trace
        response["skill_trace"]["slots"] = slots
        response["skill_trace"]["evidence_audit"] = gate.get("evidence_audit", {})
        response["retrieval_trace"]["tool_chain"] = response["skill_trace"]["tool_chain"]
        response["retrieval_trace"]["skill_package"] = skill_package
        response["retrieval_trace"]["intent_trace"] = intent_trace
        response["retrieval_trace"]["slots"] = slots
        response["retrieval_trace"]["evidence_audit"] = gate.get("evidence_audit", {})
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

