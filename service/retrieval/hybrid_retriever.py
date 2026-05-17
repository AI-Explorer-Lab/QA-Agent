from __future__ import annotations

import asyncio
from typing import Any, Dict, Sequence

from .parallel_query_executor import ParallelQueryExecutor
from .two_stage_hybrid_reranker import TwoStageHybridReranker


class HybridRetriever:
    """
    Fixed hybrid retrieval strategy:
    stage-1 parallel dense + BM25 + table-prioritized retrieval
    stage-2 two-stage rerank
    """

    def __init__(
        self,
        parallel_executor: ParallelQueryExecutor,
        reranker: TwoStageHybridReranker | None = None,
        table_evidence_quota: int = 2,
    ) -> None:
        self.parallel_executor = parallel_executor
        self.reranker = reranker or TwoStageHybridReranker(table_evidence_quota=table_evidence_quota)
        self.table_evidence_quota = max(0, int(table_evidence_quota))

    async def retrieve(
        self,
        question: str,
        collection_name: str,
        top_k: int,
        query_type: str = "fact_lookup",
        expand_query_num: int = 3,
        enable_cache: bool = True,
        expanded_queries: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        candidate_pool_size = max(1, int(getattr(self.reranker, "cross_encoder_candidate_pool", 30) or 30))
        stage_top_k = max(max(1, int(top_k)) * 4, candidate_pool_size)
        stage1 = await self.parallel_executor.execute(
            question=question,
            collection_name=collection_name,
            top_k=stage_top_k,
            query_type=query_type,
            expand_query_num=expand_query_num,
            enable_cache=enable_cache,
            expanded_queries=expanded_queries,
        )

        candidates = list(stage1.get("candidates") or [])
        reranked, rerank_trace = self.reranker.rerank(
            query=question,
            candidates=candidates,
            top_k=top_k,
            query_type=query_type,
            table_evidence_quota=self.table_evidence_quota,
        )

        retrieval_trace = dict(stage1.get("retrieval_trace") or {})
        retrieval_trace["candidate_pool_size"] = len(candidates)

        return {
            "query_type": query_type,
            "evidence": reranked,
            "candidates": reranked,
            "retrieval_trace": retrieval_trace,
            "rerank_trace": rerank_trace,
        }

    def retrieve_sync(
        self,
        question: str,
        collection_name: str,
        top_k: int,
        query_type: str = "fact_lookup",
        expand_query_num: int = 3,
        enable_cache: bool = True,
        expanded_queries: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        coroutine = self.retrieve(
            question=question,
            collection_name=collection_name,
            top_k=top_k,
            query_type=query_type,
            expand_query_num=expand_query_num,
            enable_cache=enable_cache,
            expanded_queries=expanded_queries,
        )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        if loop.is_running():
            return loop.create_task(coroutine)  # type: ignore[return-value]
        return loop.run_until_complete(coroutine)
