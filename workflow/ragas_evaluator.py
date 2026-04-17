import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)


def _normalize(text: str) -> List[str]:
    text = (text or "").lower()
    parts = re.split(r"[^\w\u4e00-\u9fff]+", text)
    return [part for part in parts if part]


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa = set(a)
    sb = set(b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _fallback_evaluation(question: str, answer: str, contexts: List[str], similarities: List[float]) -> Dict:
    q_tokens = _normalize(question)
    a_tokens = _normalize(answer)
    c_tokens = _normalize("\n".join(contexts))

    answer_relevancy = _jaccard(q_tokens, a_tokens)
    faithfulness_proxy = _jaccard(a_tokens, c_tokens)
    if similarities:
        context_precision = sum(max(0.0, min(1.0, score)) for score in similarities) / len(similarities)
    else:
        context_precision = 0.0

    overall = round((answer_relevancy + faithfulness_proxy + context_precision) / 3, 4)
    return {
        "framework": "ragas-fallback",
        "answer_relevancy": round(answer_relevancy, 4),
        "faithfulness": round(faithfulness_proxy, 4),
        "context_precision": round(context_precision, 4),
        "overall_score": overall,
    }


def evaluate_rag_response(question: str, answer: str, retrieved_docs: List[Dict]) -> Dict:
    """
    RAGAS evaluation entry.
    - If ragas dependency is available, it can be plugged in here in future iteration.
    - Current implementation provides deterministic fallback metrics to evaluate RAG quality.
    """
    contexts = [str(doc.get("raw_doc", "")) for doc in retrieved_docs if doc.get("raw_doc")]
    similarities = [float(doc.get("similarity", 0.0)) for doc in retrieved_docs if "similarity" in doc]

    try:
        # Keep interface ready for native ragas integration without breaking runtime today.
        import ragas  # noqa: F401

        result = _fallback_evaluation(question, answer, contexts, similarities)
        result["framework"] = "ragas-compatible"
        result["note"] = "Native ragas dataset pipeline can be enabled when evaluation dataset is configured."
        return result
    except Exception as exc:
        logger.debug("RAGAS package unavailable, fallback evaluator applied: %s", exc)
        return _fallback_evaluation(question, answer, contexts, similarities)

