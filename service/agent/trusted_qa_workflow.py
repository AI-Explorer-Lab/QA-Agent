from __future__ import annotations

import hashlib
import json
import os
import time
import copy
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Dict, List, Optional
from uuid import uuid4

from service.agent.answer_generator import AnswerGenerator
from service.agent.clarify_gate import build_clarify_question
from service.agent.conversation_context import ConversationContextService
from service.agent.controlled_agents import IntentUnderstandingAgent, QuestionUnderstandingAgent, SlotFillingAgent
from service.agent.evidence_gate import EvidenceDecisionEngine
from service.agent.query_expander import FIXED_QUERY_VARIANT_TOTAL, expand_queries
from service.agent.skill_registry import DEFAULT_SKILL_REGISTRY
from service.embedding.embedding_service import EmbeddingService, build_embedding_provider_from_config
from service.evaluation.ragas_evaluator import evaluate_qa_result
from service.llm import get_llm_service
from service.retrieval.hybrid_retriever import HybridRetriever
from service.retrieval.parallel_query_executor import ParallelQueryExecutor
from service.retrieval.retrieval_cache import RetrievalResultCache
from service.retrieval.runtime import get_runtime_repository
from service.retrieval.two_stage_hybrid_reranker import TwoStageHybridReranker
from service.session.session_service import get_session_service
from utils.config_loader import get_app_config

try:
    from langgraph.graph import StateGraph
except Exception:
    StateGraph = None


_LANGGRAPH_BYPASS: ContextVar[bool] = ContextVar("trusted_qa_langgraph_bypass", default=False)
ProgressCallback = Callable[[Dict[str, Any]], Awaitable[None]]


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _query_expander_for_executor(question: str, expand_query_num: int) -> List[str]:
    del expand_query_num
    return expand_queries(question, "fact_lookup", FIXED_QUERY_VARIANT_TOTAL)[1:]


def _fixed_query_variants(question: str, query_type: str, candidates: List[str] | None) -> List[str]:
    fallback = expand_queries(question, query_type, FIXED_QUERY_VARIANT_TOTAL)
    original = str(question or "").strip()
    scene_variant = fallback[-1] if fallback else original
    rewrite_pool = list(candidates or []) + fallback[1:-1]

    merged: List[str] = []
    for item in [original] + rewrite_pool:
        value = str(item or "").strip()
        if value and value not in merged:
            merged.append(value)
        if len(merged) >= FIXED_QUERY_VARIANT_TOTAL - 1:
            break
    scene_value = str(scene_variant or "").strip()
    if scene_value and scene_value not in merged:
        merged.append(scene_value)
    for item in fallback:
        if len(merged) >= FIXED_QUERY_VARIANT_TOTAL:
            break
        value = str(item or "").strip()
        if value and value not in merged:
            merged.append(value)
    return merged


def _usable_llm_answer(answer: Any) -> bool:
    text = str(answer or "").strip()
    if len(text) < 12:
        return False
    incomplete_endings = ("根据所提供的", "根据提供的", "根据现有证据", "根据证据")
    return text not in incomplete_endings


def _duration_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def _progress_stage_from_observation(item: Dict[str, Any]) -> Dict[str, Any] | None:
    phase = str(item.get("phase") or "").strip()
    if not phase:
        return None
    return {
        "phase": phase,
        "stage": phase,
        "status": "completed",
        "duration_ms": int(item.get("duration_ms") or 0),
        "timed": "duration_ms" in item,
        "cache_hit": bool(item.get("cache_hit", False)),
        "cache_precheck_hit": bool(item.get("cache_precheck_hit", False)),
        "query_expansion_cache_hit": bool(item.get("query_expansion_cache_hit", False)),
        "query_expansion_skipped": str(item.get("query_expansion_skipped") or ""),
        "llm_answer_cache_hit": bool(item.get("llm_answer_cache_hit", False)),
        "llm_query_expansion_used": bool(item.get("llm_query_expansion_used", False)),
        "evidence_count": int(item.get("evidence_count") or 0),
    }


async def _emit_progress_stage(progress_callback: ProgressCallback | None, item: Dict[str, Any]) -> None:
    if progress_callback is None:
        return
    stage = _progress_stage_from_observation(item)
    if stage is not None:
        await progress_callback(stage)


async def _emit_progress_marker(progress_callback: ProgressCallback | None, phase: str, status: str = "running", **extra: Any) -> None:
    if progress_callback is None:
        return
    stage = {
        "phase": phase,
        "stage": phase,
        "status": status,
        "duration_ms": int(extra.pop("duration_ms", 0) or 0),
        "timed": bool(extra.pop("timed", False)),
        "cache_hit": bool(extra.pop("cache_hit", False)),
        "cache_precheck_hit": bool(extra.pop("cache_precheck_hit", False)),
        "query_expansion_cache_hit": bool(extra.pop("query_expansion_cache_hit", False)),
        "query_expansion_skipped": str(extra.pop("query_expansion_skipped", "") or ""),
        "llm_answer_cache_hit": bool(extra.pop("llm_answer_cache_hit", False)),
        "llm_query_expansion_used": bool(extra.pop("llm_query_expansion_used", False)),
        "evidence_count": int(extra.pop("evidence_count", 0) or 0),
    }
    stage.update(extra)
    await progress_callback(stage)


async def _emit_response_finalize_stage(progress_callback: ProgressCallback | None, response: Dict[str, Any]) -> None:
    if progress_callback is None:
        return
    stages = response.get("retrieval_trace", {}).get("progress_stages", [])
    if not isinstance(stages, list) or not stages:
        return
    final_stage = stages[-1]
    if isinstance(final_stage, dict) and final_stage.get("phase") == "finalize_response":
        await _emit_progress_stage(progress_callback, final_stage)


class TrustedQAWorkflow:
    def __init__(self) -> None:
        self.config = get_app_config()
        retrieval_cfg = self.config.get("retrieval", {}) if isinstance(self.config.get("retrieval"), dict) else {}
        reranker_cfg = self.config.get("reranker", {}) if isinstance(self.config.get("reranker"), dict) else {}
        cache_cfg = self.config.get("cache", {}) if isinstance(self.config.get("cache"), dict) else {}
        guard_cfg = self.config.get("guardrails", {}) if isinstance(self.config.get("guardrails"), dict) else {}
        self.session_service = get_session_service()
        self.skill_registry = DEFAULT_SKILL_REGISTRY
        self.llm_service = get_llm_service()
        self.conversation_context = ConversationContextService(self.llm_service)
        self.intent_agent = IntentUnderstandingAgent(self.llm_service)
        self.slot_agent = SlotFillingAgent(self.llm_service)
        self.understanding_agent = QuestionUnderstandingAgent(self.llm_service)
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
            reranker=TwoStageHybridReranker(
                dense_weight=float(reranker_cfg.get("dense_weight", 0.50)),
                bm25_weight=float(reranker_cfg.get("bm25_weight", 0.35)),
                metadata_boost_weight=float(reranker_cfg.get("metadata_boost_weight", 0.10)),
                table_boost_weight=float(reranker_cfg.get("table_boost_weight", 0.05)),
                near_duplicate_threshold=float(reranker_cfg.get("near_duplicate_threshold", 0.90)),
                table_evidence_quota=int(retrieval_cfg.get("table_evidence_quota", 2)),
                cross_encoder_enabled=bool(reranker_cfg.get("cross_encoder_enabled", True)),
                cross_encoder_model=str(reranker_cfg.get("cross_encoder_model", "BAAI/bge-reranker-base")),
                cross_encoder_candidate_pool=int(reranker_cfg.get("cross_encoder_candidate_pool", 30)),
                cross_encoder_batch_size=int(reranker_cfg.get("cross_encoder_batch_size", 8)),
                cross_encoder_max_length=int(reranker_cfg.get("cross_encoder_max_length", 512)),
                cross_encoder_local_files_only=bool(reranker_cfg.get("cross_encoder_local_files_only", False)),
                cross_encoder_load_on_request=bool(reranker_cfg.get("cross_encoder_load_on_request", False)),
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
        self.query_expansion_cache_enabled = bool(cache_cfg.get("query_expansion_cache_enabled", cache_cfg.get("enabled", True)))
        self.query_expansion_cache_ttl_seconds = max(1, int(cache_cfg.get("query_expansion_cache_ttl_seconds", cache_cfg.get("ttl_seconds", 3600))))
        self.query_expansion_cache_max_items = max(1, int(cache_cfg.get("query_expansion_cache_max_items", min(int(cache_cfg.get("max_items", 5000)), 1024))))
        self._query_expansion_cache: Dict[tuple[str, str, int], tuple[float, List[str], List[str]]] = {}
        self.answer_cache_enabled = bool(cache_cfg.get("answer_cache_enabled", cache_cfg.get("enabled", True)))
        self.answer_cache_ttl_seconds = max(1, int(cache_cfg.get("answer_cache_ttl_seconds", cache_cfg.get("ttl_seconds", 3600))))
        self.answer_cache_max_items = max(1, int(cache_cfg.get("answer_cache_max_items", min(int(cache_cfg.get("max_items", 5000)), 1024))))
        self._answer_cache: Dict[str, tuple[float, str]] = {}
        self.response_cache_enabled = bool(cache_cfg.get("response_cache_enabled", cache_cfg.get("enabled", True)))
        self.response_cache_ttl_seconds = max(1, int(cache_cfg.get("response_cache_ttl_seconds", cache_cfg.get("ttl_seconds", 3600))))
        self.response_cache_max_items = max(1, int(cache_cfg.get("response_cache_max_items", min(int(cache_cfg.get("max_items", 5000)), 1024))))
        self._response_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
        workflow_cfg = self.config.get("workflow", {}) if isinstance(self.config.get("workflow"), dict) else {}
        self.langgraph_enabled = _env_truthy(
            "TRUSTED_QA_ENABLE_LANGGRAPH",
            bool(workflow_cfg.get("enable_langgraph", True)),
        )
        self.langgraph_available = StateGraph is not None
        self._langgraph_app: Optional[Any] = None

    @staticmethod
    def _query_expansion_cache_key(question: str, query_type: str, expand_query_num: int) -> tuple[str, str, int]:
        return (str(question or "").strip(), str(query_type or "fact_lookup").strip(), max(1, int(expand_query_num)))

    def _get_cached_query_expansion(self, question: str, query_type: str, expand_query_num: int) -> tuple[List[str], List[str]] | None:
        if not self.query_expansion_cache_enabled:
            return None
        key = self._query_expansion_cache_key(question, query_type, expand_query_num)
        cached = self._query_expansion_cache.get(key)
        if cached is None:
            return None
        cached_at, expanded, llm_expanded = cached
        if time.time() - cached_at > self.query_expansion_cache_ttl_seconds:
            self._query_expansion_cache.pop(key, None)
            return None
        return list(expanded), list(llm_expanded)

    def _store_query_expansion_cache(
        self,
        question: str,
        query_type: str,
        expand_query_num: int,
        expanded: List[str],
        llm_expanded: List[str] | None,
    ) -> None:
        if not self.query_expansion_cache_enabled or not llm_expanded:
            return
        if len(self._query_expansion_cache) >= self.query_expansion_cache_max_items:
            oldest_key = min(self._query_expansion_cache.items(), key=lambda item: item[1][0])[0]
            self._query_expansion_cache.pop(oldest_key, None)
        key = self._query_expansion_cache_key(question, query_type, expand_query_num)
        self._query_expansion_cache[key] = (time.time(), list(expanded), list(llm_expanded))

    def _answer_cache_key(
        self,
        question: str,
        query_type: str,
        evidence: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
    ) -> str:
        evidence_fingerprint = []
        for item in evidence:
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            evidence_fingerprint.append(
                {
                    "chunk_id": item.get("chunk_id", ""),
                    "doc_id": item.get("doc_id", ""),
                    "doc_source": item.get("doc_source", ""),
                    "content": item.get("content", ""),
                    "page_idx": metadata.get("page_idx"),
                    "heading_path": metadata.get("heading_path", ""),
                }
            )
        citation_fingerprint = [
            {
                "chunk_id": item.get("chunk_id", ""),
                "doc_id": item.get("doc_id", ""),
                "quote": item.get("quote", ""),
            }
            for item in citations
        ]
        payload = {
            "version": "grounded-answer-v1",
            "question": str(question or "").strip(),
            "query_type": str(query_type or "fact_lookup"),
            "provider": str(getattr(self.llm_service, "provider_name", "")),
            "model": str(getattr(self.llm_service, "model", "")),
            "evidence": evidence_fingerprint,
            "citations": citation_fingerprint,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_cached_answer(self, cache_key: str) -> str | None:
        if not self.answer_cache_enabled:
            return None
        cached = self._answer_cache.get(cache_key)
        if cached is None:
            return None
        cached_at, answer = cached
        if time.time() - cached_at > self.answer_cache_ttl_seconds:
            self._answer_cache.pop(cache_key, None)
            return None
        return str(answer or "").strip() or None

    def _store_answer_cache(self, cache_key: str, answer: str | None) -> None:
        text = str(answer or "").strip()
        if not self.answer_cache_enabled or not text:
            return
        if len(self._answer_cache) >= self.answer_cache_max_items:
            oldest_key = min(self._answer_cache.items(), key=lambda item: item[1][0])[0]
            self._answer_cache.pop(oldest_key, None)
        self._answer_cache[cache_key] = (time.time(), text)

    async def _response_cache_key(
        self,
        question: str,
        collection_name: str,
        query_type: str,
        top_k: int,
        expand_query_num: int,
    ) -> str:
        repository = getattr(getattr(self.retriever, "parallel_executor", None), "repository", None)
        repository_revision = int(getattr(repository, "revision", 0) or 0)
        try:
            repository_collection_count = int(await repository.count_collection_chunks(collection_name)) if repository is not None else -1
        except Exception:
            repository_collection_count = -1
        payload = {
            "version": "qa-response-v1",
            "question": str(question or "").strip(),
            "collection_name": str(collection_name or "default"),
            "query_type": str(query_type or "fact_lookup"),
            "top_k": max(1, int(top_k)),
            "expand_query_num": max(1, int(expand_query_num)),
            "provider": str(getattr(self.llm_service, "provider_name", "")),
            "model": str(getattr(self.llm_service, "model", "")),
            "repository_revision": repository_revision,
            "repository_collection_count": repository_collection_count,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Dict[str, Any] | None:
        if not self.response_cache_enabled:
            return None
        cached = self._response_cache.get(cache_key)
        if cached is None:
            return None
        cached_at, response = cached
        if time.time() - cached_at > self.response_cache_ttl_seconds:
            self._response_cache.pop(cache_key, None)
            return None
        return copy.deepcopy(response)

    def _store_response_cache(self, cache_key: str | None, response: Dict[str, Any]) -> None:
        if not self.response_cache_enabled or not cache_key:
            return
        if len(self._response_cache) >= self.response_cache_max_items:
            oldest_key = min(self._response_cache.items(), key=lambda item: item[1][0])[0]
            self._response_cache.pop(oldest_key, None)
        self._response_cache[cache_key] = (time.time(), copy.deepcopy(response))

    async def _expand_queries_for_retrieval(
        self,
        question: str,
        query_type: str,
        expand_query_num: int,
    ) -> tuple[List[str], List[str] | None, bool]:
        del expand_query_num
        llm_expanded = await self.llm_service.expand_queries(question, query_type, FIXED_QUERY_VARIANT_TOTAL)
        expanded = _fixed_query_variants(question, query_type, llm_expanded)
        return expanded, llm_expanded, bool(llm_expanded)

    async def _retrieve_with_cache_aware_expansion(
        self,
        question: str,
        collection_name: str,
        top_k: int,
        query_type: str,
        expand_query_num: int,
        enable_cache: bool,
    ) -> tuple[Dict[str, Any], List[str], List[str] | None, bool, Dict[str, Any]]:
        started = time.perf_counter()
        effective_top_k = max(1, int(top_k))
        effective_expand_num = max(1, int(expand_query_num))
        cache_precheck_hit = False
        query_expansion_cache_hit = False
        expanded: List[str]
        llm_expanded: List[str] | None
        llm_expansion_used: bool

        cached_stage1 = None
        if enable_cache:
            cached_stage1 = await self.retriever.get_cached_stage1(
                question=question,
                collection_name=collection_name,
                top_k=effective_top_k,
                query_type=query_type,
            )

        if cached_stage1 is not None:
            cache_precheck_hit = True
            trace = dict(cached_stage1.get("retrieval_trace") or {})
            cached_variants = trace.get("query_variants") or trace.get("expanded_queries")
            expanded = [str(item or "").strip() for item in list(cached_variants or []) if str(item or "").strip()]
            if not expanded:
                expanded = _fixed_query_variants(question, query_type, None)
            llm_expanded = None
            llm_expansion_used = False
        else:
            cached_expansion = self._get_cached_query_expansion(question, query_type, effective_expand_num) if enable_cache else None
            if cached_expansion is not None:
                query_expansion_cache_hit = True
                expanded, _cached_llm_expanded = cached_expansion
                llm_expanded = None
                llm_expansion_used = False
            else:
                expanded, llm_expanded, llm_expansion_used = await self._expand_queries_for_retrieval(
                    question,
                    query_type,
                    effective_expand_num,
                )
                if enable_cache:
                    self._store_query_expansion_cache(question, query_type, effective_expand_num, expanded, llm_expanded)

        retrieval_result = await self.retriever.retrieve(
            question=question,
            collection_name=collection_name,
            top_k=effective_top_k,
            query_type=query_type,
            expand_query_num=effective_expand_num,
            enable_cache=enable_cache,
            expanded_queries=expanded,
        )
        trace = retrieval_result.setdefault("retrieval_trace", {})
        trace["query_expansion_skipped"] = "retrieval_cache_hit" if cache_precheck_hit else ("query_expansion_cache_hit" if query_expansion_cache_hit else "")
        trace["query_expansion_cache_hit"] = bool(query_expansion_cache_hit)
        trace["llm_query_expansion_used"] = bool(llm_expansion_used)
        query_expansion_skipped = str(trace.get("query_expansion_skipped") or "")
        stage = {
            "phase": "parallel_hybrid_retrieval",
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "cache_precheck_hit": cache_precheck_hit,
            "query_expansion_cache_hit": query_expansion_cache_hit,
            "query_expansion_skipped": query_expansion_skipped,
            "cache_hit": bool(trace.get("cache_hit", False)),
            "llm_query_expansion_used": bool(llm_expansion_used),
            "query_variant_count": len(expanded),
        }
        return retrieval_result, expanded, llm_expanded, llm_expansion_used, stage

    async def _understand_question(self, question: str, conversation_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return await self.understanding_agent.understand(
            question,
            skill_registry=self.skill_registry,
            conversation_context=conversation_context or {},
        )

    @staticmethod
    def _build_clarify_payload(
        query_type: str,
        collection_name: str,
        slots: Dict[str, Any],
        selected_skill: Any,
    ) -> Dict[str, Any]:
        missing_slots = selected_skill.get_missing_slots(slots) if selected_skill is not None else []
        if not str(collection_name or "").strip():
            missing_slots.append("collection_name")
        return {
            "decision": "clarify" if missing_slots else "answer",
            "missing_slots": missing_slots,
            "clarify_question": build_clarify_question(query_type, missing_slots) if missing_slots else "",
            "slots": slots,
        }

    def _langgraph_initial_state(
        self,
        question: str,
        collection_name: str,
        session_id: str | None,
        top_k: int,
        expand_query_num: int,
        enable_cache: bool,
    ) -> Dict[str, Any]:
        workflow_started_at = time.perf_counter()
        return {
            "question": question,
            "collection_name": collection_name,
            "session_id": session_id,
            "top_k": max(1, int(top_k)),
            "expand_query_num": max(1, int(expand_query_num)),
            "enable_cache": bool(enable_cache),
            "retry_count": 0,
            "observations": [],
            "workflow_started_at": workflow_started_at,
        }

    def _langgraph_app_instance(self) -> Optional[Any]:
        if not self.langgraph_available:
            return None
        if self._langgraph_app is not None:
            return self._langgraph_app

        graph = StateGraph(dict)
        graph.add_node("load_session", self._graph_load_session)
        graph.add_node("build_conversation_context", self._graph_build_conversation_context)
        graph.add_node("understand_intent_and_slots", self._graph_understand_intent_and_slots)
        graph.add_node("run_clarify_gate", self._graph_run_clarify_gate)
        graph.add_node("build_clarify_response", self._graph_build_clarify_response)
        graph.add_node("retrieve_evidence", self._graph_retrieve_evidence)
        graph.add_node("evaluate_evidence", self._graph_evaluate_evidence)
        graph.add_node("retry_retrieval", self._graph_retry_retrieval)
        graph.add_node("build_answer_response", self._graph_build_answer_response)
        graph.add_node("persist_response", self._graph_persist_response)

        graph.set_entry_point("load_session")
        graph.add_edge("load_session", "build_conversation_context")
        graph.add_edge("build_conversation_context", "understand_intent_and_slots")
        graph.add_edge("understand_intent_and_slots", "run_clarify_gate")
        graph.add_conditional_edges(
            "run_clarify_gate",
            self._graph_route_after_clarify_gate,
            {"clarify": "build_clarify_response", "retrieve": "retrieve_evidence"},
        )
        graph.add_edge("retrieve_evidence", "evaluate_evidence")
        graph.add_conditional_edges(
            "evaluate_evidence",
            self._graph_route_after_gate,
            {"retry": "retry_retrieval", "final": "build_answer_response"},
        )
        graph.add_conditional_edges(
            "retry_retrieval",
            self._graph_route_after_retry,
            {"retry": "retry_retrieval", "final": "build_answer_response"},
        )
        graph.add_edge("build_clarify_response", "persist_response")
        graph.add_edge("build_answer_response", "persist_response")
        graph.set_finish_point("persist_response")
        self._langgraph_app = graph.compile()
        return self._langgraph_app

    async def _graph_load_session(self, state: Dict[str, Any]) -> Dict[str, Any]:
        started_at = time.perf_counter()
        session = await self.session_service.load_session(state.get("session_id"), collection_name=str(state.get("collection_name") or "default"))
        sid = session["session_id"]
        observations = list(state.get("observations") or [])
        observations.append({"phase": "load_session", "session_id": sid, "duration_ms": _duration_ms(started_at)})
        next_state = dict(state)
        next_state.update({"session": session, "sid": sid, "observations": observations})
        return next_state

    async def _graph_build_conversation_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        started_at = time.perf_counter()
        context = await self.conversation_context.prepare_context(
            state.get("session") or {},
            str(state.get("question") or ""),
            str(state.get("collection_name") or "default"),
        )
        observations = list(state.get("observations") or [])
        turn_route = dict(context.get("turn_route") or {})
        observations.append(
            {
                "phase": "conversation_context",
                "turn_type": turn_route.get("turn_type", "new_rag_query"),
                "context_source": turn_route.get("context_source", "none"),
                "effective_question": context.get("effective_question", state.get("question") or ""),
                "history_refs": turn_route.get("history_refs", []),
                "duration_ms": _duration_ms(started_at),
            }
        )
        return {
            **state,
            "conversation_state": context.get("conversation_state") or {},
            "turn_route": turn_route,
            "original_question": context.get("original_question") or str(state.get("question") or ""),
            "effective_question": context.get("effective_question") or str(state.get("question") or ""),
            "observations": observations,
        }

    async def _graph_understand_intent_and_slots(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = str(state.get("effective_question") or state.get("question") or "")
        started_at = time.perf_counter()
        understanding = await self._understand_question(question, conversation_context=state.get("turn_route") or {})
        intent_trace = dict(understanding.get("intent_trace") or {})
        query_type = str(understanding.get("query_type") or intent_trace.get("query_type") or "fact_lookup")
        intent_duration_ms = _duration_ms(started_at)
        skill_started_at = time.perf_counter()
        selected_skill = understanding.get("selected_skill") or self.skill_registry.select_skill(query_type)
        slots = dict(understanding.get("slots") or {})
        skill_package = selected_skill.package_metadata()
        observations = list(state.get("observations") or [])
        observations.append({"phase": "intent_slot_understanding_agent", "intent": intent_trace, "slots": slots, "duration_ms": intent_duration_ms})
        observations.append({"phase": "select_skill_from_registry", "selected_skill": selected_skill.skill_name, "skill_package": skill_package, "duration_ms": _duration_ms(skill_started_at)})
        next_state = dict(state)
        next_state.update(
            {
                "intent_trace": intent_trace,
                "query_type": query_type,
                "selected_skill": selected_skill,
                "skill_package": skill_package,
                "slots": slots,
                "observations": observations,
            }
        )
        return next_state

    async def _graph_run_clarify_gate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        started_at = time.perf_counter()
        query_type = str(state.get("query_type") or "fact_lookup")
        collection_name = str(state.get("collection_name") or "default")
        slots = dict(state.get("slots") or {})
        selected_skill = state.get("selected_skill")
        clarify = self._build_clarify_payload(query_type, collection_name, slots, selected_skill)
        observations = list(state.get("observations") or [])
        observations.append(
            {
                "phase": "clarify_gate",
                "slots": slots,
                "missing_slots": clarify.get("missing_slots") or [],
                "decision": clarify.get("decision") or "answer",
                "duration_ms": _duration_ms(started_at),
            }
        )
        next_state = dict(state)
        next_state.update(
            {
                "missing_slots": list(clarify.get("missing_slots") or []),
                "clarify": clarify,
                "observations": observations,
            }
        )
        return next_state

    def _graph_route_after_clarify_gate(self, state: Dict[str, Any]) -> str:
        clarify = state.get("clarify") or {}
        return "clarify" if clarify.get("decision") == "clarify" else "retrieve"

    async def _graph_build_clarify_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        started_at = time.perf_counter()
        question = str(state.get("effective_question") or state.get("question") or "")
        query_type = str(state.get("query_type") or "fact_lookup")
        clarify = dict(state.get("clarify") or {})
        answer_payload = self.answer_generator.generate(question=question, query_type=query_type, evidence=[], decision="clarify", gate_reason="missing_slots")
        response = self._build_response(
            str(state.get("sid") or ""),
            query_type,
            "clarify",
            answer_payload,
            {"expanded_queries": [], "observations": state.get("observations") or [], "clarify": clarify, "intent_trace": state.get("intent_trace") or {}, "slots": state.get("slots") or {}},
            {},
            str(getattr(state.get("selected_skill"), "skill_name", "")),
        )
        response["answer"] = clarify.get("clarify_question") or response.get("answer", "")
        observations = list(state.get("observations") or [])
        observations.append({"phase": "answer_generation", "duration_ms": _duration_ms(started_at), "llm_answer_used": False})
        response["retrieval_trace"]["react_observations"] = observations
        response["react_observations"] = observations
        next_state = dict(state)
        next_state.update({"response": response, "decision": "clarify", "llm_expansion_used": False, "llm_answer_used": False, "observations": observations})
        return next_state
    async def _graph_retrieve_evidence(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = str(state.get("effective_question") or state.get("question") or "")
        query_type = str(state.get("query_type") or "fact_lookup")
        expand_query_num = int(state.get("expand_query_num") or 3)
        retrieval_result, expanded, llm_expanded, llm_expansion_used, retrieval_stage = await self._retrieve_with_cache_aware_expansion(
            question=question,
            collection_name=str(state.get("collection_name") or "default"),
            top_k=max(1, int(state.get("top_k") or 5)),
            query_type=query_type,
            expand_query_num=expand_query_num,
            enable_cache=bool(state.get("enable_cache", True)),
        )
        evidence = list(retrieval_result.get("evidence") or [])
        observations = list(state.get("observations") or [])
        observations.append({**retrieval_stage, "evidence_count": len(evidence)})
        next_state = dict(state)
        next_state.update(
            {
                "llm_expanded": llm_expanded,
                "expanded": expanded,
                "llm_expansion_used": llm_expansion_used,
                "retrieval_result": retrieval_result,
                "evidence": evidence,
                "observations": observations,
            }
        )
        return next_state

    async def _graph_evaluate_evidence(self, state: Dict[str, Any]) -> Dict[str, Any]:
        started_at = time.perf_counter()
        gate = await self.evidence_decision.evaluate(
            question=str(state.get("effective_question") or state.get("question") or ""),
            query_type=str(state.get("query_type") or "fact_lookup"),
            slots=state.get("slots") or {},
            selected_skill=str(getattr(state.get("selected_skill"), "skill_name", "")),
            evidence=list(state.get("evidence") or []),
            rerank_trace=(state.get("retrieval_result") or {}).get("rerank_trace") or {},
            retry_count=int(state.get("retry_count") or 0),
            table_evidence_quota=self.table_evidence_quota,
        )
        observations = list(state.get("observations") or [])
        observations.append({"phase": "evidence_decision", "rule_gate": gate.get("rule_gate") or {}, "audit": gate.get("evidence_audit") or {}, "gate": gate, "duration_ms": _duration_ms(started_at)})
        next_state = dict(state)
        next_state.update({"gate": gate, "observations": observations})
        return next_state

    def _graph_route_after_gate(self, state: Dict[str, Any]) -> str:
        gate = state.get("gate") or {}
        retry_count = int(state.get("retry_count") or 0)
        if gate.get("decision") == "retry" and retry_count < self.evidence_decision.retry_limit:
            return "retry"
        return "final"

    async def _graph_retry_retrieval(self, state: Dict[str, Any]) -> Dict[str, Any]:
        started_at = time.perf_counter()
        retry_count = int(state.get("retry_count") or 0) + 1
        gate = dict(state.get("gate") or {})
        retry_question = str(gate.get("suggested_retry_query") or "").strip() or str(state.get("effective_question") or state.get("question") or "")
        query_type = str(state.get("query_type") or "fact_lookup")
        expand_query_num = max(1, int(state.get("expand_query_num") or 3))
        retry_expanded, retry_llm_expanded, retry_llm_expansion_used = await self._expand_queries_for_retrieval(
            retry_question,
            query_type,
            expand_query_num,
        )
        retry_result = await self.retriever.retrieve(
            question=retry_question,
            collection_name=str(state.get("collection_name") or "default"),
            top_k=max(1, int(state.get("top_k") or 5)),
            query_type=query_type,
            expand_query_num=expand_query_num,
            enable_cache=False,
            expanded_queries=retry_expanded,
        )
        evidence = list(retry_result.get("evidence") or state.get("evidence") or [])
        gate = await self.evidence_decision.evaluate(
            question=str(state.get("effective_question") or state.get("question") or ""),
            query_type=query_type,
            slots=state.get("slots") or {},
            selected_skill=str(getattr(state.get("selected_skill"), "skill_name", "")),
            evidence=evidence,
            rerank_trace=retry_result.get("rerank_trace") or {},
            retry_count=retry_count,
            table_evidence_quota=self.table_evidence_quota,
        )
        observations = list(state.get("observations") or [])
        observations.append({"phase": "retry_retrieval", "retry_count": retry_count, "retry_question": retry_question, "expanded_queries": retry_expanded, "audit": gate.get("evidence_audit") or {}, "rule_gate": gate.get("rule_gate") or {}, "gate": gate, "evidence_count": len(evidence), "duration_ms": _duration_ms(started_at)})
        next_state = dict(state)
        next_state.update(
            {
                "retry_count": retry_count,
                "retry_question": retry_question,
                "llm_expanded": retry_llm_expanded,
                "expanded": retry_expanded,
                "llm_expansion_used": bool(state.get("llm_expansion_used")) or retry_llm_expansion_used,
                "retrieval_result": retry_result,
                "evidence": evidence,
                "gate": gate,
                "observations": observations,
            }
        )
        return next_state

    def _graph_route_after_retry(self, state: Dict[str, Any]) -> str:
        gate = state.get("gate") or {}
        retry_count = int(state.get("retry_count") or 0)
        if gate.get("decision") == "retry" and retry_count < self.evidence_decision.retry_limit:
            return "retry"
        return "final"

    async def _graph_build_answer_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        started_at = time.perf_counter()
        question = str(state.get("effective_question") or state.get("question") or "")
        query_type = str(state.get("query_type") or "fact_lookup")
        gate = dict(state.get("gate") or {})
        evidence = list(state.get("evidence") or [])
        decision = gate.get("decision", "refuse")
        if decision == "retry":
            decision = "refuse"
            gate["decision"] = "refuse"
            gate["reason"] = gate.get("reason", "retry_limit_reached")
        if decision not in {"answer", "clarify", "refuse"}:
            decision = "refuse"
        answer_payload = self.answer_generator.generate(question=question, query_type=query_type, evidence=evidence, decision=decision, gate_reason=gate.get("reason", ""))
        llm_answer_used = False
        llm_answer_cache_hit = False
        if decision == "answer":
            answer_cache_key = self._answer_cache_key(
                question,
                query_type,
                list(answer_payload.get("evidence") or []),
                list(answer_payload.get("citations") or []),
            )
            llm_answer = self._get_cached_answer(answer_cache_key) if bool(state.get("enable_cache", True)) else None
            if llm_answer:
                llm_answer_cache_hit = True
            else:
                llm_answer = await self.llm_service.generate_grounded_answer(
                    question=question,
                    query_type=query_type,
                    evidence=answer_payload.get("evidence", []),
                    citations=answer_payload.get("citations", []),
                )
            if _usable_llm_answer(llm_answer):
                answer_payload["answer"] = llm_answer
                llm_answer_used = not llm_answer_cache_hit
                if not llm_answer_cache_hit and bool(state.get("enable_cache", True)):
                    self._store_answer_cache(answer_cache_key, llm_answer)
        observations = list(state.get("observations") or [])
        observations.append(
            {
                "phase": "answer_generation",
                "duration_ms": _duration_ms(started_at),
                "llm_answer_used": llm_answer_used,
                "llm_answer_cache_hit": llm_answer_cache_hit,
            }
        )
        response = self._build_response(
            str(state.get("sid") or ""),
            query_type,
            decision,
            answer_payload,
            (state.get("retrieval_result") or {}).get("retrieval_trace") or {},
            (state.get("retrieval_result") or {}).get("rerank_trace") or {},
            str(getattr(state.get("selected_skill"), "skill_name", "")),
            observations=observations,
            expanded_queries=state.get("expanded") or [],
            gate=gate,
        )
        next_state = dict(state)
        next_state.update(
            {
                "response": response,
                "decision": decision,
                "gate": gate,
                "llm_answer_used": llm_answer_used,
                "llm_answer_cache_hit": llm_answer_cache_hit,
                "observations": observations,
            }
        )
        return next_state

    async def _graph_persist_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        workflow_started_at = state.get("workflow_started_at")
        response = dict(state.get("response") or {})
        if not response:
            next_state = dict(state)
            next_state.update({"response": {}})
            return next_state
        selected_skill = state.get("selected_skill")
        response = self._apply_response_traces(
            response=response,
            selected_skill=selected_skill,
            skill_package=state.get("skill_package") or {},
            intent_trace=state.get("intent_trace") or {},
            slots=state.get("slots") or {},
            gate=state.get("gate") or {},
            llm_expansion_used=bool(state.get("llm_expansion_used")),
            llm_answer_used=bool(state.get("llm_answer_used")),
            llm_answer_cache_hit=bool(state.get("llm_answer_cache_hit")),
            workflow_runner="langgraph",
            evaluate=response.get("decision") != "clarify",
            question=str(state.get("effective_question") or state.get("question") or ""),
            original_question=str(state.get("original_question") or state.get("question") or ""),
            effective_question=str(state.get("effective_question") or state.get("question") or ""),
            conversation_state=state.get("conversation_state") or {},
            turn_route=state.get("turn_route") or {},
            workflow_duration_ms=_duration_ms(float(workflow_started_at)) if isinstance(workflow_started_at, (int, float)) else None,
        )
        await self._save(str(state.get("sid") or ""), str(state.get("original_question") or state.get("question") or ""), response)
        await self._update_conversation_focus(
            session_id=str(state.get("sid") or ""),
            effective_question=str(state.get("effective_question") or state.get("question") or ""),
            query_type=str(response.get("query_type") or state.get("query_type") or "fact_lookup"),
            slots=state.get("slots") or {},
            response=response,
            conversation_state=state.get("conversation_state") or {},
            turn_route=state.get("turn_route") or {},
        )
        if isinstance(workflow_started_at, (int, float)):
            self._set_workflow_duration(response, _duration_ms(float(workflow_started_at)))
        next_state = dict(state)
        next_state.update({"response": response})
        return next_state

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
        result = await app.ainvoke(self._langgraph_initial_state(question, collection_name, session_id, top_k, expand_query_num, enable_cache))
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
            return await self._ask_with_langgraph(question=question, collection_name=collection_name, session_id=session_id, top_k=top_k, expand_query_num=expand_query_num, enable_cache=enable_cache)
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
        progress_callback: ProgressCallback | None = None,
    ) -> Dict[str, Any]:
        if progress_callback is not None:
            return await self._ask_legacy(
                question=question,
                collection_name=collection_name,
                session_id=session_id,
                top_k=top_k,
                expand_query_num=expand_query_num,
                enable_cache=enable_cache,
                progress_callback=progress_callback,
            )
        langgraph_response = await self._maybe_run_langgraph(question=question, collection_name=collection_name, session_id=session_id, top_k=top_k, expand_query_num=expand_query_num, enable_cache=enable_cache)
        if isinstance(langgraph_response, dict):
            return langgraph_response
        return await self._ask_legacy(question=question, collection_name=collection_name, session_id=session_id, top_k=top_k, expand_query_num=expand_query_num, enable_cache=enable_cache)
    async def _ask_legacy(
        self,
        question: str,
        collection_name: str = "default",
        session_id: str | None = None,
        top_k: int = 5,
        expand_query_num: int = 3,
        enable_cache: bool = True,
        progress_callback: ProgressCallback | None = None,
    ) -> Dict[str, Any]:
        workflow_started_at = time.perf_counter()
        load_session_started_at = time.perf_counter()
        await _emit_progress_marker(progress_callback, "load_session")
        session = await self.session_service.load_session(session_id, collection_name=collection_name)
        sid = session["session_id"]
        load_session_duration_ms = _duration_ms(load_session_started_at)
        observations: List[Dict[str, Any]] = [{"phase": "load_session", "session_id": sid, "duration_ms": load_session_duration_ms}]
        await _emit_progress_stage(progress_callback, observations[-1])

        context_started_at = time.perf_counter()
        await _emit_progress_marker(progress_callback, "conversation_context")
        context = await self.conversation_context.prepare_context(session, question, collection_name)
        context_duration_ms = _duration_ms(context_started_at)
        original_question = str(context.get("original_question") or question)
        effective_question = str(context.get("effective_question") or question)
        conversation_state = context.get("conversation_state") or {}
        turn_route = context.get("turn_route") or {}
        observations.append(
            {
                "phase": "conversation_context",
                "turn_type": turn_route.get("turn_type", "new_rag_query"),
                "context_source": turn_route.get("context_source", "none"),
                "effective_question": effective_question,
                "history_refs": turn_route.get("history_refs", []),
                "duration_ms": context_duration_ms,
            }
        )
        await _emit_progress_stage(progress_callback, observations[-1])

        intent_started_at = time.perf_counter()
        await _emit_progress_marker(progress_callback, "intent_slot_understanding_agent")
        understanding = await self._understand_question(effective_question, conversation_context=turn_route)
        intent_duration_ms = _duration_ms(intent_started_at)
        intent_trace = dict(understanding.get("intent_trace") or {})
        query_type = str(understanding.get("query_type") or intent_trace.get("query_type") or "fact_lookup")
        slots = dict(understanding.get("slots") or {})
        observations.append({"phase": "intent_slot_understanding_agent", "intent": intent_trace, "slots": slots, "duration_ms": intent_duration_ms})
        await _emit_progress_stage(progress_callback, observations[-1])

        skill_started_at = time.perf_counter()
        await _emit_progress_marker(progress_callback, "select_skill_from_registry")
        selected_skill = understanding.get("selected_skill") or self.skill_registry.select_skill(query_type)
        skill_package = selected_skill.package_metadata()
        skill_duration_ms = _duration_ms(skill_started_at)
        observations.append({"phase": "select_skill_from_registry", "selected_skill": selected_skill.skill_name, "skill_package": skill_package, "duration_ms": skill_duration_ms})
        await _emit_progress_stage(progress_callback, observations[-1])

        clarify_started_at = time.perf_counter()
        await _emit_progress_marker(progress_callback, "clarify_gate")
        clarify = self._build_clarify_payload(query_type, collection_name, slots, selected_skill)
        clarify_duration_ms = _duration_ms(clarify_started_at)
        observations.append(
            {
                "phase": "clarify_gate",
                "slots": slots,
                "missing_slots": clarify.get("missing_slots") or [],
                "decision": clarify.get("decision") or "answer",
                "duration_ms": clarify_duration_ms,
            }
        )
        await _emit_progress_stage(progress_callback, observations[-1])

        if clarify["decision"] == "clarify":
            answer_started_at = time.perf_counter()
            await _emit_progress_marker(progress_callback, "answer_generation")
            answer_payload = self.answer_generator.generate(question=effective_question, query_type=query_type, evidence=[], decision="clarify", gate_reason="missing_slots")
            answer_observation = {"phase": "answer_generation", "duration_ms": _duration_ms(answer_started_at), "llm_answer_used": False}
            observations.append(answer_observation)
            await _emit_progress_stage(progress_callback, answer_observation)
            response = self._build_response(
                sid,
                query_type,
                "clarify",
                answer_payload,
                {"expanded_queries": [], "observations": observations, "clarify": clarify, "intent_trace": intent_trace, "slots": slots},
                {},
                selected_skill.skill_name,
            )
            response["answer"] = clarify.get("clarify_question") or response["answer"]
            response = self._apply_response_traces(
                response=response,
                selected_skill=selected_skill,
                skill_package=skill_package,
                intent_trace=intent_trace,
                slots=slots,
                gate={},
                llm_expansion_used=False,
                llm_answer_used=False,
                llm_answer_cache_hit=False,
                workflow_runner="python",
                evaluate=False,
                question=effective_question,
                original_question=original_question,
                effective_question=effective_question,
                conversation_state=conversation_state,
                turn_route=turn_route,
                workflow_duration_ms=_duration_ms(workflow_started_at),
            )
            await _emit_progress_marker(progress_callback, "finalize_response")
            await self._save(sid, original_question, response)
            await self._update_conversation_focus(
                session_id=sid,
                effective_question=effective_question,
                query_type=query_type,
                slots=slots,
                response=response,
                conversation_state=conversation_state,
                turn_route=turn_route,
            )
            self._set_workflow_duration(response, _duration_ms(workflow_started_at))
            await _emit_response_finalize_stage(progress_callback, response)
            return response

        response_cache_key = await self._response_cache_key(
            effective_question,
            collection_name,
            query_type,
            max(1, int(top_k)),
            max(1, int(expand_query_num)),
        )
        cached_response = self._get_cached_response(response_cache_key) if enable_cache else None
        if cached_response is not None:
            evidence_count = len(cached_response.get("evidence") or [])
            cached_observations = list(observations)
            cached_observations.extend(
                [
                    {
                        "phase": "parallel_hybrid_retrieval",
                        "duration_ms": 0,
                        "cache_hit": True,
                        "cache_precheck_hit": True,
                        "query_expansion_skipped": "final_response_cache_hit",
                        "evidence_count": evidence_count,
                    },
                    {"phase": "evidence_decision", "duration_ms": 0, "evidence_count": evidence_count},
                    {
                        "phase": "answer_generation",
                        "duration_ms": 0,
                        "llm_answer_used": False,
                        "llm_answer_cache_hit": True,
                    },
                ]
            )
            for cached_observation in cached_observations[len(observations) :]:
                await _emit_progress_stage(progress_callback, cached_observation)
            response = copy.deepcopy(cached_response)
            response["session_id"] = sid
            response["original_question"] = original_question
            response["effective_question"] = effective_question
            response.setdefault("retrieval_trace", {})
            response["retrieval_trace"]["react_observations"] = cached_observations
            response["react_observations"] = cached_observations
            response["retrieval_trace"]["final_response_cache_hit"] = True
            response["retrieval_trace"]["cache_hit"] = True
            response["retrieval_trace"]["query_expansion_skipped"] = "final_response_cache_hit"
            response = self._apply_response_traces(
                response=response,
                selected_skill=selected_skill,
                skill_package=skill_package,
                intent_trace=intent_trace,
                slots=slots,
                gate=response.get("retrieval_trace", {}).get("gate_decision") or {},
                llm_expansion_used=False,
                llm_answer_used=False,
                llm_answer_cache_hit=True,
                workflow_runner="python",
                evaluate=True,
                question=effective_question,
                original_question=original_question,
                effective_question=effective_question,
                conversation_state=conversation_state,
                turn_route=turn_route,
                workflow_duration_ms=_duration_ms(workflow_started_at),
            )
            await _emit_progress_marker(progress_callback, "finalize_response")
            await self._save(sid, original_question, response)
            await self._update_conversation_focus(
                session_id=sid,
                effective_question=effective_question,
                query_type=query_type,
                slots=slots,
                response=response,
                conversation_state=conversation_state,
                turn_route=turn_route,
            )
            self._set_workflow_duration(response, _duration_ms(workflow_started_at))
            await _emit_response_finalize_stage(progress_callback, response)
            return response

        await _emit_progress_marker(progress_callback, "parallel_hybrid_retrieval")
        retrieval_result, expanded, llm_expanded, llm_expansion_used, retrieval_stage = await self._retrieve_with_cache_aware_expansion(
            question=effective_question,
            collection_name=collection_name,
            top_k=max(1, int(top_k)),
            query_type=query_type,
            expand_query_num=max(1, int(expand_query_num)),
            enable_cache=enable_cache,
        )
        evidence = list(retrieval_result.get("evidence") or [])
        retrieval_observation = {**retrieval_stage, "evidence_count": len(evidence)}
        observations.append(retrieval_observation)
        await _emit_progress_stage(progress_callback, retrieval_observation)
        evidence_started_at = time.perf_counter()
        await _emit_progress_marker(progress_callback, "evidence_decision")
        gate = await self.evidence_decision.evaluate(
            question=effective_question,
            query_type=query_type,
            slots=slots,
            selected_skill=selected_skill.skill_name,
            evidence=evidence,
            rerank_trace=retrieval_result.get("rerank_trace") or {},
            retry_count=0,
            table_evidence_quota=self.table_evidence_quota,
        )
        evidence_observation = {"phase": "evidence_decision", "rule_gate": gate.get("rule_gate") or {}, "audit": gate.get("evidence_audit") or {}, "gate": gate, "duration_ms": _duration_ms(evidence_started_at)}
        observations.append(evidence_observation)
        await _emit_progress_stage(progress_callback, evidence_observation)

        retry_count = 0
        while gate.get("decision") == "retry" and retry_count < self.evidence_decision.retry_limit:
            retry_started_at = time.perf_counter()
            await _emit_progress_marker(progress_callback, "retry_retrieval")
            retry_count += 1
            retry_question = str(gate.get("suggested_retry_query") or "").strip() or effective_question
            retry_expanded, retry_llm_expanded, retry_llm_expansion_used = await self._expand_queries_for_retrieval(
                retry_question,
                query_type,
                expand_query_num,
            )
            llm_expansion_used = llm_expansion_used or retry_llm_expansion_used
            expanded = retry_expanded
            retry_result = await self.retriever.retrieve(
                question=retry_question,
                collection_name=collection_name,
                top_k=max(1, int(top_k)),
                query_type=query_type,
                expand_query_num=max(1, int(expand_query_num)),
                enable_cache=False,
                expanded_queries=retry_expanded,
            )
            evidence = list(retry_result.get("evidence") or evidence)
            retrieval_result = retry_result
            gate = await self.evidence_decision.evaluate(
                question=effective_question,
                query_type=query_type,
                slots=slots,
                selected_skill=selected_skill.skill_name,
                evidence=evidence,
                rerank_trace=retrieval_result.get("rerank_trace") or {},
                retry_count=retry_count,
                table_evidence_quota=self.table_evidence_quota,
            )
            retry_observation = {"phase": "retry_retrieval", "retry_count": retry_count, "retry_question": retry_question, "expanded_queries": retry_expanded, "llm_expanded": retry_llm_expanded, "audit": gate.get("evidence_audit") or {}, "rule_gate": gate.get("rule_gate") or {}, "gate": gate, "evidence_count": len(evidence), "duration_ms": _duration_ms(retry_started_at)}
            observations.append(retry_observation)
            await _emit_progress_stage(progress_callback, retry_observation)

        decision = gate.get("decision", "refuse")
        if decision == "retry":
            decision = "refuse"
            gate = dict(gate)
            gate["decision"] = "refuse"
            gate["reason"] = gate.get("reason", "retry_limit_reached")
        if decision not in {"answer", "clarify", "refuse"}:
            decision = "refuse"

        answer_started_at = time.perf_counter()
        await _emit_progress_marker(progress_callback, "answer_generation")
        answer_payload = self.answer_generator.generate(question=effective_question, query_type=query_type, evidence=evidence, decision=decision, gate_reason=gate.get("reason", ""))
        llm_answer_used = False
        llm_answer_cache_hit = False
        if decision == "answer":
            answer_cache_key = self._answer_cache_key(
                effective_question,
                query_type,
                list(answer_payload.get("evidence") or []),
                list(answer_payload.get("citations") or []),
            )
            llm_answer = self._get_cached_answer(answer_cache_key) if enable_cache else None
            if llm_answer:
                llm_answer_cache_hit = True
            else:
                llm_answer = await self.llm_service.generate_grounded_answer(
                    question=effective_question,
                    query_type=query_type,
                    evidence=answer_payload.get("evidence", []),
                    citations=answer_payload.get("citations", []),
                )
            if _usable_llm_answer(llm_answer):
                answer_payload["answer"] = llm_answer
                llm_answer_used = not llm_answer_cache_hit
                if not llm_answer_cache_hit and enable_cache:
                    self._store_answer_cache(answer_cache_key, llm_answer)
        answer_observation = {
            "phase": "answer_generation",
            "duration_ms": _duration_ms(answer_started_at),
            "llm_answer_used": llm_answer_used,
            "llm_answer_cache_hit": llm_answer_cache_hit,
        }
        observations.append(answer_observation)
        await _emit_progress_stage(progress_callback, answer_observation)
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
        response = self._apply_response_traces(
            response=response,
            selected_skill=selected_skill,
            skill_package=skill_package,
            intent_trace=intent_trace,
            slots=slots,
            gate=gate,
            llm_expansion_used=llm_expansion_used,
            llm_answer_used=llm_answer_used,
            llm_answer_cache_hit=llm_answer_cache_hit,
            workflow_runner="python",
            evaluate=True,
            question=effective_question,
            original_question=original_question,
            effective_question=effective_question,
            conversation_state=conversation_state,
            turn_route=turn_route,
            workflow_duration_ms=_duration_ms(workflow_started_at),
        )
        if enable_cache and decision == "answer":
            self._store_response_cache(response_cache_key, response)
        await _emit_progress_marker(progress_callback, "finalize_response")
        await self._save(sid, original_question, response)
        await self._update_conversation_focus(
            session_id=sid,
            effective_question=effective_question,
            query_type=query_type,
            slots=slots,
            response=response,
            conversation_state=conversation_state,
            turn_route=turn_route,
        )
        self._set_workflow_duration(response, _duration_ms(workflow_started_at))
        await _emit_response_finalize_stage(progress_callback, response)
        return response

    @staticmethod
    def _set_workflow_duration(response: Dict[str, Any], duration_ms: int) -> None:
        total = max(0, int(duration_ms))
        trace = response.setdefault("retrieval_trace", {})
        trace["workflow_duration_ms"] = total
        stages = trace.get("progress_stages")
        if not isinstance(stages, list):
            return
        kept_stages = [
            stage
            for stage in stages
            if not (isinstance(stage, dict) and stage.get("phase") == "finalize_response")
        ]
        timed_sum = 0
        for stage in kept_stages:
            if isinstance(stage, dict) and stage.get("timed", True):
                timed_sum += max(0, int(stage.get("duration_ms") or 0))
        residual = total - timed_sum
        if residual > 0:
            kept_stages.append(
                {
                    "phase": "finalize_response",
                    "status": "completed",
                    "duration_ms": residual,
                    "timed": True,
                    "cache_hit": False,
                    "cache_precheck_hit": False,
                    "query_expansion_cache_hit": False,
                    "query_expansion_skipped": "",
                    "llm_answer_cache_hit": False,
                    "llm_query_expansion_used": False,
                    "evidence_count": 0,
                }
            )
        trace["progress_stages"] = kept_stages

    def _apply_response_traces(
        self,
        response: Dict[str, Any],
        selected_skill: Any,
        skill_package: Dict[str, Any],
        intent_trace: Dict[str, Any],
        slots: Dict[str, Any],
        gate: Dict[str, Any],
        llm_expansion_used: bool,
        llm_answer_used: bool,
        llm_answer_cache_hit: bool,
        workflow_runner: str,
        evaluate: bool,
        question: str,
        original_question: str | None = None,
        effective_question: str | None = None,
        conversation_state: Dict[str, Any] | None = None,
        turn_route: Dict[str, Any] | None = None,
        workflow_duration_ms: int | None = None,
    ) -> Dict[str, Any]:
        trace_metadata = getattr(self.llm_service, "trace_metadata", None)
        llm_trace = trace_metadata() if callable(trace_metadata) else {}
        llm_trace.update(
            {
                "query_expansion_used": bool(llm_expansion_used),
                "answer_generation_used": bool(llm_answer_used),
                "answer_cache_hit": bool(llm_answer_cache_hit),
            }
        )
        original = str(original_question or question or "")
        effective = str(effective_question or question or "")
        route = dict(turn_route or {})
        state_snapshot = dict(conversation_state or {})
        response.setdefault("retrieval_trace", {})
        response.setdefault("skill_trace", {})
        if workflow_duration_ms is not None:
            response["retrieval_trace"]["workflow_duration_ms"] = int(workflow_duration_ms)
        response["original_question"] = original
        response["effective_question"] = effective
        response["turn_type"] = route.get("turn_type", "new_rag_query")
        response["retrieval_trace"]["llm"] = llm_trace
        response["retrieval_trace"]["workflow_runner"] = workflow_runner
        response["retrieval_trace"]["turn_routing"] = {
            "turn_type": route.get("turn_type", "new_rag_query"),
            "context_source": route.get("context_source", "none"),
            "original_question": original,
            "effective_question": effective,
            "history_refs": route.get("history_refs", []),
            "missing_info": route.get("missing_info", []),
            "confidence": route.get("confidence", 0.0),
            "reason": route.get("reason", ""),
            "requires_clarification": bool(route.get("requires_clarification", False)),
        }
        response["retrieval_trace"]["conversation_state_snapshot"] = {
            "latest_clarification_pending": state_snapshot.get("latest_clarification_pending"),
            "conversation_focus": state_snapshot.get("conversation_focus"),
            "recent_history_count": len(state_snapshot.get("recent_history") or []),
            "last_citations_count": len(state_snapshot.get("last_citations") or []),
        }
        response["skill_trace"]["tool_chain"] = list(getattr(selected_skill, "tool_chain", []))
        response["skill_trace"]["skill_package"] = skill_package
        response["skill_trace"]["intent_trace"] = intent_trace
        response["skill_trace"]["slots"] = slots
        response["skill_trace"]["turn_routing"] = response["retrieval_trace"]["turn_routing"]
        response["retrieval_trace"]["tool_chain"] = response["skill_trace"]["tool_chain"]
        response["retrieval_trace"]["skill_package"] = skill_package
        response["retrieval_trace"]["intent_trace"] = intent_trace
        response["retrieval_trace"]["slots"] = slots
        if gate:
            response["skill_trace"]["evidence_audit"] = gate.get("evidence_audit", {})
            response["retrieval_trace"]["evidence_audit"] = gate.get("evidence_audit", {})
        if evaluate:
            evaluation = evaluate_qa_result(
                question=question,
                answer=response.get("answer", ""),
                decision=response.get("decision", "refuse"),
                citations=response.get("citations", []),
                evidence=response.get("evidence", []),
            )
            response["confidence"] = max(float(response.get("confidence") or 0.0), float(evaluation.get("confidence", 0.0)))
            response["retrieval_trace"].setdefault("evaluation", evaluation)
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
        progress_stages = []
        for item in react_observations:
            if not isinstance(item, dict):
                continue
            phase = str(item.get("phase") or "").strip()
            if not phase:
                continue
            progress_stages.append(
                {
                    "phase": phase,
                    "status": "completed",
                    "duration_ms": int(item.get("duration_ms") or 0),
                    "timed": "duration_ms" in item,
                    "cache_hit": bool(item.get("cache_hit", False)),
                    "cache_precheck_hit": bool(item.get("cache_precheck_hit", False)),
                    "query_expansion_cache_hit": bool(item.get("query_expansion_cache_hit", False)),
                    "query_expansion_skipped": str(item.get("query_expansion_skipped") or ""),
                    "llm_answer_cache_hit": bool(item.get("llm_answer_cache_hit", False)),
                    "llm_query_expansion_used": bool(item.get("llm_query_expansion_used", False)),
                    "evidence_count": int(item.get("evidence_count") or 0),
                }
            )
        trace.setdefault("progress_stages", progress_stages)
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

    async def _update_conversation_focus(
        self,
        session_id: str,
        effective_question: str,
        query_type: str,
        slots: Dict[str, Any],
        response: Dict[str, Any],
        conversation_state: Dict[str, Any],
        turn_route: Dict[str, Any],
    ) -> None:
        updater = getattr(self.session_service, "update_session_metadata", None)
        if not callable(updater):
            return
        previous_focus = (conversation_state or {}).get("conversation_focus")
        next_focus = self.conversation_context.build_focus_after_response(
            previous_focus=previous_focus if isinstance(previous_focus, dict) else None,
            effective_question=effective_question,
            query_type=query_type,
            slots=slots,
            response=response,
            turn_route=turn_route,
        )
        if next_focus is None:
            return
        result = updater(session_id, {"conversation_focus": next_focus})
        if hasattr(result, "__await__"):
            await result

    async def _save(self, session_id: str, question: str, response: Dict[str, Any]) -> None:
        await self.session_service.save_session(session_id=session_id, user_question=question, assistant_payload=response, retrieval_trace=response.get("retrieval_trace") or {})


_DEFAULT_WORKFLOW = TrustedQAWorkflow()


def get_trusted_qa_workflow() -> TrustedQAWorkflow:
    return _DEFAULT_WORKFLOW
