import logging
import os
from fastapi import APIRouter
# api.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage
from starlette.concurrency import run_in_threadpool

from db_service.history_conversations import load_history_conversation
from db_service.save_conversations import *
from db_service.db import save_conversation_sql
from db_service.faiss_store import process_and_save_to_faiss
from core.config_loader import get_llm_runtime_config
from core.content_normalizer import normalize_content
from chat_langchain import app as langgraph_app
import uuid
from workflow.team_leader_workflow import graph
from workflow.react_workflow import run_react

app = FastAPI(title="RAG Q&A Backend (dev)")

router = APIRouter()


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _format_retrieved_doc(doc: dict) -> dict:
    return {
        "raw_doc": doc.get("raw_doc", ""),
        "similarity": _safe_float(doc.get("similarity", 0.0)),
        "chunk_id": doc.get("chunk_id"),
        "doc_id": doc.get("doc_id"),
        "doc_source": doc.get("doc_source"),
        "chunk_type": doc.get("chunk_type"),
        "chunk_index": doc.get("chunk_index"),
        "level1_title": doc.get("level1_title", ""),
        "level2_title": doc.get("level2_title", ""),
        "level3_title": doc.get("level3_title", ""),
        "heading_path": doc.get("heading_path", "front_matter"),
        "table_id": doc.get("table_id"),
        "sub_table_id": doc.get("sub_table_id"),
        "sub_table_index": doc.get("sub_table_index"),
        "table_id_subtable_count": doc.get("table_id_subtable_count"),
        "table_context_text": doc.get("table_context_text"),
        "table_header_text": doc.get("table_header_text"),
    }


@router.get("/health")
def read_root():
    return {"message": "Hello, world! Backend is running."}


@router.post("/vector/build-index")
async def build_vector_index(
    document_path: str = "./docs/pdf_docs/上海芯导电子科技股份有限公司财报_2025.pdf",
    file_type: str = "pdf",
    index_path: str = "./vector_stores/faiss_index.bin",
    metadata_path: str = "./vector_stores/faiss_metadata.pkl",
    backend: str = "",
):
    """
    Trigger document ingestion pipeline:
    chunking -> embedding -> FAISS (and optional pgvector sync).
    """
    file_type = (file_type or "").strip().lower()
    if file_type not in {"pdf", "txt"}:
        return JSONResponse(
            status_code=400,
            content={"error": "file_type must be one of: pdf, txt"},
        )

    if backend:
        backend_value = backend.strip().lower()
        if backend_value not in {"faiss", "pgvector", "hybrid", "both"}:
            return JSONResponse(
                status_code=400,
                content={"error": "backend must be one of: faiss, pgvector, hybrid, both"},
            )
        os.environ["VECTOR_STORE_BACKEND"] = backend_value

    success = await run_in_threadpool(
        process_and_save_to_faiss,
        document_path=document_path,
        index_path=index_path,
        metadata_path=metadata_path,
        type=file_type,
    )
    if not success:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Index build failed. Please check server logs and input path.",
                "document_path": document_path,
                "file_type": file_type,
                "backend": os.getenv("VECTOR_STORE_BACKEND", "faiss"),
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": "Index build completed.",
            "document_path": document_path,
            "file_type": file_type,
            "backend": os.getenv("VECTOR_STORE_BACKEND", "faiss"),
            "index_path": index_path,
            "metadata_path": metadata_path,
        },
    )


@router.post("/askLLM")
async def ask_llm(model_name: str = "", question: str = "hello", session_id: str = None):
    if not model_name:
        model_name = get_llm_runtime_config().get("model", "")
    # 如果没有 session_id，说明是新会话
    is_new_session = not session_id
    if is_new_session:
        session_id = str(uuid.uuid4())

    # 加载历史对话，返回完整的消息列表
    messages = load_history_conversation(question, session_id)

    # 使用 LangGraph 应用处理请求
    # 构造输入消息
    input_messages = {
        "messages": messages
    }
    logging.debug(f"input_messages: {input_messages}")

    # 配置线程 ID 用于会话记忆
    config = {"configurable": {"thread_id": session_id}}

    # 异步调用 LangGraph 应用
    result = await langgraph_app.ainvoke(input_messages, config=config)

    # 提取 AI 回复
    answer_text = ""
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage):
            answer_text = normalize_content(message.content)
            break

    # 保存对话历史到数据库 (在后台异步执行)
    save_conversation_json(session_id, question, answer_text, model_name)
    save_conversation_sql(session_id, question, answer_text, model_name)

    # 返回结果包含 session_id
    return JSONResponse(status_code=200, content={
        "session_id": session_id,
        "answer": answer_text
    })


# @router.post("/embedding")
# def embedding_text(text: str = "你好"):
#     return JSONResponse(status_code=200, content={
#         "embedding_result": embedding_processor(text)
#     })


@router.post("/team-leader-task")  # will rename to chat-task
async def team_leader_task(question: str, retrieved_answers:int=5):
    """
    API endpoint that invokes the team leader workflow to handle user tasks
    """
    try:
        # Initialize the state for the workflow
        initial_state = {
            "input": question,
            "output": "",
            "conversation_history": [],
            "messages": [],
            "retrieved_answers": retrieved_answers,
        }
        logging.debug(f"Initial state: {initial_state}")

        # Run the workflow asynchronously
        final_state = await graph.ainvoke(initial_state)

        # Extract tool output and retrieved docs from messages
        output = ""
        retrieved_docs = []
        ragas_evaluation = {}
        for msg in final_state.get("messages", []):
            if hasattr(msg, 'content') and msg.content:
                # Parse llm_rag JSON output
                try:
                    import json
                    content = normalize_content(msg.content)
                    if isinstance(content, str) and content.startswith('{'):
                        parsed = json.loads(content)
                        if "summary" in parsed:
                            output = parsed.get("summary", content)
                            retrieved_docs = parsed.get("retrieved_docs", [])
                            ragas_evaluation = parsed.get("ragas_evaluation", {})
                        else:
                            output = content
                    else:
                        output = content
                except (json.JSONDecodeError, TypeError):
                    output = normalize_content(content)
                break

        final_answer = output or "未能生成有效回答"
        logging.info("Retrieved docs count: %s", len(retrieved_docs))

        # Build messages_summary with structured document data
        messages_summary = []
        if retrieved_docs:
            # Format retrieved documents as structured data
            for doc in retrieved_docs:
                messages_summary.append({
                    "content": _format_retrieved_doc(doc)
                })
        else:
            # If no retrieved docs, include a simple message summary
            for msg in final_state.get("messages", []):
                if hasattr(msg, 'content') and msg.content:
                    messages_summary.append({
                        "content": normalize_content(msg.content)[:500]
                    })

        return JSONResponse(
            status_code=200,
            content={
                "question": question,
                "answer": final_answer,
                "retrieved_answers": final_state.get("retrieved_answers"),
                "messages_summary": messages_summary,
                "ragas_evaluation": ragas_evaluation,
            }
        )
    except Exception as e:
        logging.error(f"Error in team_leader_task: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": f"处理请求时发生错误: {str(e)}"
            }
        )



@router.post("/react-ask")
async def react_ask(question: str, max_iterations: int = 10, expand_query_num: int=3, retrieved_answers: int = 5, session_id: str = None):
    """
    ReAct Agent API endpoint that uses reasoning-acting loop to handle user queries.

    The ReAct agent will:
    1. THOUGHT: Analyze the user's request
    2. ACTION: Call appropriate tools (llm_chat, llm_query, llm_rag)
    3. OBSERVATION: Review tool results
    4. ITERATE: Continue until satisfied or max_iterations reached

    Args:
        question: User's query or request
        max_iterations: Maximum number of ReAct iterations (default: 10)
        expand_query_num: Number of query that expand based on question (default: 3)
        reretrieved_answers: Number of documents to retrieve when using RAG (default: 5)
        session_id: Optional session ID for conversation tracking

    Returns:
        JSON response with:
            - question: Original question
            - answer: Final answer from the agent
            - iteration_count: Number of iterations performed
            - tool_calls_summary: Summary of tools used
            - messages_count: Total messages in conversation
            - session_id: Session identifier
    """
    try:
        # Generate session_id if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        logging.info(f"ReAct request - Session: {session_id}, Question: {question[:50]}...")

        # Run the ReAct workflow
        result = await run_react(question, max_iterations=max_iterations, expand_query_num=expand_query_num, retrieved_answers=retrieved_answers, session_id=session_id)

        # Check if successful
        if not result.get("success"):
            return JSONResponse(
                status_code=500,
                content={
                    "error": result.get("error", "Unknown error"),
                    "question": question,
                    "session_id": session_id
                }
            )

        # Extract messages and find the final answer
        messages = result.get("messages", [])
        answer = ""

        # Find the last AI message as the final answer
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                if not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                    # This is the final answer (no tool calls)
                    answer = normalize_content(msg.content)
                break

        # If no answer found, try to get from result output
        if not answer:
            answer = normalize_content(result.get("output", ""))

        # Build messages_summary with retrieved documents (same format as team-leader-task)
        retrieved_docs = result.get("retrieved_docs", [])
        messages_summary = []

        if retrieved_docs:
            # Format retrieved documents as structured data
            for doc in retrieved_docs:
                messages_summary.append({
                    "content": _format_retrieved_doc(doc)
                })
        else:
            # If no retrieved docs, include conversation summary
            for msg in messages:
                if isinstance(msg, AIMessage) and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                    messages_summary.append({
                        "content": normalize_content(msg.content)[:500]
                    })
                    break

        # Get retrieved_answers count and iteration_count
        retrieved_count = result.get("retrieved_answers", retrieved_answers)
        iteration_count = result.get("iteration_count", 0)
        ragas_evaluation = result.get("ragas_evaluation", {})
        final_decision = result.get("final_decision", "")
        gate_reason = result.get("gate_reason", "")
        evidence_summary = result.get("evidence_summary", {})
        retrieval_retry_count = result.get("retrieval_retry_count", 0)

        # Save conversation to database
        save_conversation_json(session_id, question, answer, "react-agent")
        save_conversation_sql(session_id, question, answer, "react-agent")

        logging.info(f"ReAct completed - Session: {session_id}, Iterations: {iteration_count}")

        return JSONResponse(
            status_code=200,
            content={
                "question": question,
                "answer": answer,
                "retrieved_answers": retrieved_count,
                "messages_summary": messages_summary,
                "iteration_count": iteration_count,
                "session_id": session_id,
                "ragas_evaluation": ragas_evaluation,
                "final_decision": final_decision,
                "gate_reason": gate_reason,
                "evidence_summary": evidence_summary,
                "retrieval_retry_count": retrieval_retry_count,
            }
        )

    except Exception as e:
        logging.error(f"Error in react_ask: {str(e)}")
        import traceback
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": f"处理请求时发生错误: {str(e)}",
                "question": question,
                "session_id": session_id
            }
        )


if __name__ == "__main__":
    pass
