import logging
import sys
from pathlib import Path
from typing import List
from typing_extensions import NotRequired, TypedDict

from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from db_service.vector_store_router import search_documents  # noqa: E402
from workflow.ragas_evaluator import evaluate_rag_response  # noqa: E402
from core.config_loader import get_llm_runtime_config, load_runtime_env  # noqa: E402
from core.content_normalizer import normalize_content  # noqa: E402
from core.retrieval_deduper import dedupe_ranked_documents  # noqa: E402

load_runtime_env()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class State(TypedDict):
    conversation_history: List[str]
    input: str
    messages: list
    output: str
    task_completed: bool
    expanded_queries: NotRequired[list]
    expand_query_num: NotRequired[int]
    retrieved_answers: NotRequired[int]
    retrieved_docs: NotRequired[list]
    original_input: NotRequired[str]
    ragas_evaluation: NotRequired[dict]


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        runtime = get_llm_runtime_config()
        _agent = ChatOpenAI(
            api_key=runtime["api_key"],
            base_url=runtime["base_url"],
            model=runtime["model"],
            use_responses_api=runtime["use_responses_api"],
        )
    return _agent


async def rag_query_expand_node(state: State) -> State:
    original_input = state.get("original_input", state["input"])
    expand_query_num = state.get("expand_query_num", 3)
    prompt = f"""You are a helpful assistant that expands a user's question to include relevant context.
Question: {original_input}
The number of expanded queries should be {expand_query_num}.

Requirements:
1. DO NOT repeat the original question in each expanded query.
2. Each query should explore a different aspect.
3. Output each query on a separate line without numbering.
4. Generate exactly {expand_query_num} queries.
"""
    response = await get_agent().ainvoke([HumanMessage(content=prompt)])
    expanded_query = normalize_content(response.content)
    context = state.get("conversation_history", "")

    new_state = state.copy()
    new_state["original_input"] = original_input
    new_state["expanded_queries"] = expanded_query
    new_state["input"] = f"{original_input}\n{context}".strip()
    return new_state


async def rag_retrieve_node(state: State) -> State:
    k = state.get("retrieved_answers", 5)
    original_query = state.get("original_input", state["input"])
    expanded_queries_str = normalize_content(state.get("expanded_queries", ""))
    expanded_queries = [item.strip() for item in expanded_queries_str.split("\n") if item.strip()]
    all_queries = [original_query] + expanded_queries

    all_retrieved_docs = []
    for query in all_queries:
        all_retrieved_docs.extend(search_documents(query, k))

    limited_docs = dedupe_ranked_documents(all_retrieved_docs, k=max(1, int(k)))

    context_parts = [f"[DOC{i}] {doc.get('raw_doc', '')}" for i, doc in enumerate(limited_docs, start=1)]
    context = "\n\n".join(context_parts)

    new_state = state.copy()
    new_state["expanded_queries"] = expanded_queries
    new_state["retrieved_docs"] = limited_docs
    new_state["conversation_history"] = context
    new_state["output"] = context
    return new_state


async def rag_generate_node(state: State) -> State:
    query = state.get("original_input", state["input"])
    retrieved_docs = state.get("retrieved_docs", [])
    expanded_queries = state.get("expanded_queries", [])

    prompt = f"""You are a professional information summarization assistant.
Please answer the user based strictly on retrieved documents.

User question: {query}
Expanded queries: {expanded_queries}
Retrieved relevant documents:
{retrieved_docs}

Requirements:
1. Answer directly without conversational opening.
2. Do not add facts not present in retrieved documents.
3. Use objective and professional language.
4. If documents differ, present differences objectively.
"""
    response = await get_agent().ainvoke([HumanMessage(content=prompt)])
    answer_text = normalize_content(response.content)

    ragas_eval = evaluate_rag_response(
        question=query,
        answer=answer_text,
        retrieved_docs=retrieved_docs,
    )

    state_copy = state.copy()
    state_copy["output"] = answer_text
    state_copy["ragas_evaluation"] = ragas_eval
    logger.info("RAGAS evaluation: %s", ragas_eval)
    return state_copy


workflow = StateGraph(State)
workflow.add_node("rag_query_expand_node", rag_query_expand_node)
workflow.add_node("rag_retrieve_node", rag_retrieve_node)
workflow.add_node("rag_generate_node", rag_generate_node)

workflow.add_edge(START, "rag_query_expand_node")
workflow.add_edge("rag_query_expand_node", "rag_retrieve_node")
workflow.add_edge("rag_retrieve_node", "rag_generate_node")
workflow.add_edge("rag_generate_node", END)

rag_graph = workflow.compile()
