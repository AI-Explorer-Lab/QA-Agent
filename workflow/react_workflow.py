"""
ReAct Workflow Implementation

This module implements a ReAct (Reasoning + Acting) agent pattern that:
1. THOUGHT: Analyzes the current situation and decides what to do
2. ACTION: Executes a tool or provides a final answer
3. OBSERVATION: Observes the result of the action
4. Iterates until the task is complete

The RAG subgraph (rag_workflow.py) remains unchanged and is called as a tool.
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
# This allows the file to be run directly from the workflow directory
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
from typing_extensions import TypedDict, NotRequired, Literal
from typing import List, Any, Dict
from langgraph.graph import StateGraph, START, END
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from core.config_loader import get_llm_runtime_config, load_runtime_env
from core.content_normalizer import normalize_content

# Import existing tools
from workflow.team_leader_workflow import llm_chat, llm_query, llm_rag
from workflow.react_gates import (
    load_react_guardrails_config,
    should_clarify_query,
    build_clarify_question,
    summarize_evidence,
    decide_evidence_action,
    build_retry_system_prompt,
    build_refusal_answer,
)

# Import history loading
from db_service.history_conversations import load_history_conversation

load_dotenv()
load_runtime_env()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReActState(TypedDict):
    """
    State for the ReAct workflow.

    Attributes:
        messages: Conversation history including user messages, AI responses, and tool results
        input: Original user input
        output: NotRequired - Final answer when the task is complete
        next: NotRequired - Next step in the workflow ('agent', 'tools', 'end')
        iteration_count: NotRequired - Track iterations to prevent infinite loops
        max_iterations: NotRequired - Maximum number of ReAct iterations allowed
        retrieved_docs: NotRequired - Retrieved documents with similarity scores from RAG
        retrieved_answers: NotRequired - Count of retrieved answers from RAG
    """
    messages: List[Any]
    input: str
    output: NotRequired[str]
    next: NotRequired[str]
    iteration_count: NotRequired[int]
    max_iterations: NotRequired[int]
    expand_query_num: NotRequired[int]
    retrieved_docs: NotRequired[List[dict]]
    retrieved_answers: NotRequired[int]
    ragas_evaluation: NotRequired[dict]
    last_tool_name: NotRequired[str]
    retrieval_retry_count: NotRequired[int]
    gate_action: NotRequired[str]
    gate_reason: NotRequired[str]
    clarify_needed: NotRequired[bool]
    clarify_question: NotRequired[str]
    final_decision: NotRequired[str]
    evidence_summary: NotRequired[Dict[str, Any]]


# Global LLM instance for the ReAct agent
_react_agent = None
_react_agent_signature = None

# # GLM 4.7 model
# def get_react_agent():
#     """Get or create global ReAct agent instance."""
#     global _react_agent
#     if _react_agent is None:
#         _react_agent = ChatOpenAI(
#             openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
#             openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
#             model="glm-4.7",
#             temperature=0  # Lower temperature for more deterministic reasoning
#         )
#     return _react_agent

def get_react_agent():
    """Get or create global ReAct agent instance."""
    global _react_agent, _react_agent_signature
    runtime = get_llm_runtime_config()
    runtime_signature = (
        runtime.get("model", ""),
        runtime.get("base_url", ""),
        runtime.get("api_key", ""),
        bool(runtime.get("use_responses_api", False)),
    )

    if _react_agent is None or _react_agent_signature != runtime_signature:
        _react_agent = ChatOpenAI(
            api_key=runtime["api_key"],
            base_url=runtime["base_url"],
            model=runtime["model"],
            use_responses_api=runtime["use_responses_api"],
            temperature=0  # Lower temperature for more deterministic reasoning
        )
        _react_agent_signature = runtime_signature
    return _react_agent


def create_react_system_prompt(expand_query_num: int = 3, retrieved_answers: int = 5) -> str:
    """
    Create the system prompt for the ReAct agent.

    Args:
        retrieved_answers: Number of documents to retrieve when using RAG
        expanded_query: Number of query that expand based on question

    Returns:
        System prompt that instructs the LLM to follow ReAct pattern
    """
    return f"""You are an intelligent AI assistant that helps users with their questions.

    ## Your Internal Thinking Process (do not include in your response):

    1. Analyze the user's request and the current situation
    2. Decide what to do:
    - If you need more information, call a tool
    - If you have enough information, provide a final answer
    3. After a tool is executed, review the result
    4. If needed, perform another action; otherwise, answer

    ## Available Tools:

    - **llm_chat(query: str)**: Use for general conversations, greetings, casual chat
    - **llm_query(query: str)**: Use for math calculations, weather information, or general factual queries
    - **llm_rag(query: str, retrieved_answers: int)**: If and only if user ask Psychology, camera , and financial report related question, Use it for document retrieval, research. If you use llm_rag, you are not allowed to use other tools.

    ## Guidelines:

    1. Think step by step internally before taking action
    2. Be specific when calling tools - provide clear and specific queries
    3. Use tools efficiently - don't call tools if you can answer from your knowledge
    4. Iterate if needed - if the first tool call doesn't give you enough information, try another app
    5. Provide clear, natural, and conversational answers to users

    ## CRITICAL - Response Format:

    - DO NOT include "THOUGHT:", "ANSWER:", "ACTION:", or "OBSERVATION:" labels in your responses
    - DO NOT show your internal reasoning process to users
    - When providing your final answer, respond naturally as if you're having a conversation
    - Your responses should be clean, user-friendly text without structured tags or labels
    - Users should see only your final answer, not your thinking process

    ## Important:

    - You MUST call exactly ONE tool at a time
    - If you call llm_rag, you MUST use expand_query_num={expand_query_num}, retrieved_answers={retrieved_answers}
    - After each tool execution, evaluate if you need more actions
    - When you're ready to answer, provide the final response naturally WITHOUT calling another tool

    Remember: Keep your responses conversational and professional. Users should not see any internal reasoning labels."""


async def clarify_gate_node(state: ReActState) -> ReActState:
    config = load_react_guardrails_config()
    new_state = state.copy()

    if not config["clarify_enabled"]:
        new_state["gate_action"] = "agent"
        new_state["clarify_needed"] = False
        return new_state

    need_clarify, reason, missing_slots = should_clarify_query(
        query=state.get("input", ""),
        rag_only=bool(config["clarify_rag_only"]),
    )

    clarify_needed = need_clarify
    if clarify_needed:
        max_turns = int(config["clarify_max_turns"])
        clarifies_so_far = int(state.get("retrieval_retry_count", 0))
        if clarifies_so_far >= max_turns:
            clarify_needed = False

    if clarify_needed:
        new_state["gate_action"] = "clarify_response"
        new_state["gate_reason"] = reason or "clarify_needed"
        new_state["clarify_needed"] = True
        new_state["clarify_question"] = build_clarify_question(
            state.get("input", ""),
            missing_slots,
        )
        new_state["final_decision"] = "clarify"
    else:
        new_state["gate_action"] = "agent"
        new_state["clarify_needed"] = False
        new_state["clarify_question"] = ""

    return new_state


def route_after_clarify_gate(state: ReActState) -> Literal["clarify_response", "agent"]:
    if state.get("gate_action") == "clarify_response":
        return "clarify_response"
    return "agent"


async def clarify_response_node(state: ReActState) -> ReActState:
    question = state.get("clarify_question") or "请补充更多关键信息后我再继续检索。"
    ai_message = AIMessage(content=question)

    new_state = state.copy()
    new_state["messages"] = state.get("messages", []) + [ai_message]
    new_state["output"] = question
    new_state["final_decision"] = "clarify"
    new_state["gate_action"] = "clarify_response"
    new_state["gate_reason"] = state.get("gate_reason", "clarify_needed")
    return new_state


def _build_docs_context(retrieved_docs: List[Dict[str, Any]], max_docs: int = 5, max_chars: int = 1200) -> str:
    lines: List[str] = []
    for index, doc in enumerate(retrieved_docs[:max_docs], start=1):
        raw_doc = normalize_content(str(doc.get("raw_doc", "")))[:max_chars]
        similarity = doc.get("similarity", 0.0)
        heading_path = doc.get("heading_path", "front_matter")
        lines.append(
            f"[DOC{index}] similarity={similarity} heading_path={heading_path}\n{raw_doc}"
        )
    return "\n\n".join(lines)


async def evidence_gate_node(state: ReActState) -> ReActState:
    new_state = state.copy()
    last_tool_name = state.get("last_tool_name", "")

    if last_tool_name != "llm_rag":
        new_state["gate_action"] = "agent"
        new_state["gate_reason"] = "non_rag_tool_skip_evidence_gate"
        new_state["evidence_summary"] = {}
        return new_state

    config = load_react_guardrails_config()
    evidence_summary = summarize_evidence(
        retrieved_docs=state.get("retrieved_docs", []),
        ragas_evaluation=state.get("ragas_evaluation", {}),
    )
    retry_count = int(state.get("retrieval_retry_count", 0))
    action, reason = decide_evidence_action(
        evidence_summary=evidence_summary,
        retry_count=retry_count,
        config=config,
    )

    new_state["evidence_summary"] = evidence_summary
    new_state["gate_action"] = action
    new_state["gate_reason"] = reason

    if action == "agent_retry":
        retry_prompt = build_retry_system_prompt(
            question=state.get("input", ""),
            evidence_summary=evidence_summary,
            reason=reason,
        )
        new_state["messages"] = state.get("messages", []) + [SystemMessage(content=retry_prompt)]
        new_state["retrieval_retry_count"] = retry_count + 1
    else:
        new_state["retrieval_retry_count"] = retry_count

    return new_state


def route_after_evidence_gate(state: ReActState) -> Literal["agent", "final_answer", "refuse_answer"]:
    action = state.get("gate_action", "agent")
    if action == "refuse_answer":
        return "refuse_answer"
    if action == "final_answer":
        return "final_answer"
    return "agent"


async def final_answer_node(state: ReActState) -> ReActState:
    messages = state.get("messages", [])
    question = state.get("input", "")
    retrieved_docs = state.get("retrieved_docs", [])
    evidence_summary = state.get("evidence_summary", {})

    prompt = (
        "You are in final-answer stage.\n"
        "Provide a concise and reliable answer strictly based on retrieved evidence.\n"
        "If evidence has uncertainty, explicitly state the uncertainty.\n"
        "Do not call tools.\n\n"
        f"Original question: {question}\n"
        f"Evidence summary: {evidence_summary}\n"
        f"Retrieved docs:\n{_build_docs_context(retrieved_docs)}"
    )

    response = await get_react_agent().ainvoke(messages + [HumanMessage(content=prompt)])
    final_text = normalize_content(response.content)

    new_state = state.copy()
    new_state["messages"] = messages + [response]
    new_state["output"] = final_text
    new_state["final_decision"] = "final_answer"
    return new_state


async def refuse_answer_node(state: ReActState) -> ReActState:
    answer = build_refusal_answer(
        question=state.get("input", ""),
        evidence_summary=state.get("evidence_summary", {}),
    )
    ai_message = AIMessage(content=answer)

    new_state = state.copy()
    new_state["messages"] = state.get("messages", []) + [ai_message]
    new_state["output"] = answer
    new_state["final_decision"] = "refuse_answer"
    return new_state


async def react_agent_node(state: ReActState) -> ReActState:
    """
    ReAct agent node that performs reasoning and decides actions.

    This node:
    1. Analyzes the conversation history and user input
    2. Decides whether to call a tool or provide a final answer
    3. Returns the AI's response (which may include tool_calls)

    Args:
        state: Current ReAct state containing messages and iteration info

    Returns:
        Updated state with the AI's response added to messages
    """
    messages = state["messages"]
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)
    expand_query_num = state.get("expand_query_num", 3)
    retrieved_answers = state.get("retrieved_answers", 5)
    gate_action = state.get("gate_action", "")
    gate_reason = state.get("gate_reason", "")

    # Check iteration limit to prevent infinite loops
    if iteration_count >= max_iterations:
        logger.warning(f"Max iterations ({max_iterations}) reached, forcing completion\n")

        # Generate final answer based on all previous tool results
        final_prompt = """Based on all the information gathered from tool calls in this conversation,
        please provide a comprehensive final answer to the user's original question.
        Summarize all findings and give a clear, structured response."""

        final_messages = messages + [HumanMessage(content=final_prompt)]
        response = await get_react_agent().ainvoke(final_messages)
        final_text = normalize_content(response.content)

        new_state = state.copy()
        new_state["messages"] = messages + [response]
        new_state["output"] = final_text
        new_state["next"] = "end"
        new_state["final_decision"] = "max_iterations_fallback"

        return new_state

    # Prepare messages for LLM
    # Only add system prompt on first iteration
    if iteration_count == 0:
        # Check if system prompt already exists
        has_system = any(isinstance(msg, SystemMessage) for msg in messages)
        if not has_system:
            system_msg = SystemMessage(content=create_react_system_prompt(expand_query_num=expand_query_num, retrieved_answers=retrieved_answers))
            messages_with_system = [system_msg] + messages
        else:
            messages_with_system = messages
    else:
        # On subsequent iterations, use messages as-is (they already have the full history)
        messages_with_system = messages

    if gate_action == "agent_retry":
        retry_instruction = (
            "Guardrail triggered evidence retry. "
            "Before giving final answer, call llm_rag once with a refined query. "
            f"Reason: {gate_reason}"
        )
        messages_with_system = messages_with_system + [SystemMessage(content=retry_instruction)]

    # Call the LLM to get reasoning and action decisionroach
    try:
        # Bind tools to the agent so it can call them
        agent_with_tools = get_react_agent().bind_tools([llm_chat, llm_query, llm_rag])

        response = await agent_with_tools.ainvoke(messages_with_system)

        new_state = state.copy()
        new_state["messages"] = messages_with_system + [response]
        new_state["iteration_count"] = iteration_count + 1
        if gate_action == "agent_retry":
            new_state["gate_action"] = ""

        # ============ THOUGHT ============
        logger.info(f"\n{'='*60}")
        logger.info(f"[REACT ITERATION] {iteration_count + 1}")
        logger.info(f"{'='*60}")

        # Log the reasoning (if present in response content)
        if hasattr(response, 'content') and response.content:
            logger.info("\n[AGENT REASONING]")
            logger.info(f"{'-'*60}")
            logger.info(f"{response.content}")
            logger.info(f"{'-'*60}")

        # Log tool calls if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for call in response.tool_calls:
                tool_name = call['name']
                logger.info("\n[DECISION REASON] Based on above reasoning content.")
                logger.info(f"[DECISION RESULT] Call tool [{tool_name}]")
                logger.info(f"  args: {call['args']}\n")
        else:
            logger.info("\n[DECISION REASON] Based on above reasoning content.")
            logger.info("[DECISION RESULT] Answer user directly, no tool call needed\n")

        return new_state

    except Exception as e:
        logger.error(f"Error in react_agent_node: {e}\n")

        # Return error state
        error_state = state.copy()
        error_state["messages"] = messages + [
            AIMessage(content=f"I encountered an error: {str(e)}. Please try again.")
        ]
        error_state["next"] = "end"
        return error_state


def should_continue(state: ReActState) -> Literal["tools", "end"]:
    """
    Determine whether to continue the ReAct loop or end.

    This function checks the last message in the conversation:
    - If it has tool_calls, continue to tools node
    - Otherwise, end the conversation

    Args:
        state: Current ReAct state

    Returns:
        "tools" if the agent wants to call tools, "end" otherwise
    """
    messages = state["messages"]

    if not messages:
        return "end"

    last_message = messages[-1]

    # Check if the last message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info(f"\n{'='*60}")
        logger.info("[ACTION] Continue tool execution")
        logger.info(f"  tools: {[call['name'] for call in last_message.tool_calls]}")
        logger.info(f"{'='*60}\n")
        return "tools"

    # No tool calls, meaning the agent provided a final answer
    if isinstance(last_message, AIMessage):
        state["output"] = normalize_content(last_message.content)
        if not state.get("final_decision"):
            state["final_decision"] = "agent_direct_answer"
        logger.info(f"\n{'='*60}")
        logger.info("[ACTION] Final answer generated, ending ReAct loop")
        logger.info(f"{'='*60}\n")

    return "end"


async def custom_tool_node(state: ReActState) -> ReActState:
    """
    Custom tool execution node that properly maintains message history.

    This node:
    1. Extracts tool calls from the last AI message
    2. Executes each tool
    3. Appends tool results as ToolMessages to the messages list
    4. Extracts retrieved_docs from RAG tool results

    Args:
        state: Current ReAct state with messages containing tool calls

    Returns:
        Updated state with tool results added to messages and retrieved_docs
    """
    messages = state["messages"]

    # Find the last AI message with tool calls
    last_ai_message = None
    tool_calls = []

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            last_ai_message = msg
            tool_calls = msg.tool_calls
            break

    if not tool_calls:
        logger.warning("No tool calls found in messages\n")
        return state

    # Execute each tool call
    tool_results = []
    tool_map = {
        "llm_chat": llm_chat,
        "llm_query": llm_query,
        "llm_rag": llm_rag
    }

    # Track retrieved documents from RAG
    retrieved_docs = []
    retrieved_answers = state.get("retrieved_answers", 5)
    ragas_evaluation = {}
    rag_invoked = False
    last_tool_name = ""

    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id", "")
        last_tool_name = str(tool_name or "")

        logger.info(f"\n{'-'*60}")
        logger.info(f"[ACTION EXEC] {tool_name}")
        logger.info(f"  args: {tool_args}")
        logger.info(f"{'-'*60}")

        try:
            # Get the tool
            tool = tool_map.get(tool_name)

            if not tool:
                raise ValueError(f"Unknown tool: {tool_name}")
            # Execute the tool using .ainvoke() method for LangChain tools
            result = await tool.ainvoke(tool_args)

            # ============ OBSERVATION ============
            logger.info("\n[OBSERVATION] Tool execution result:")
            logger.info(f"{'-'*60}")
            # Truncate long outputs in logs.
            result_str = str(result)
            if len(result_str) > 500:
                logger.info(f"{result_str[:500]}... (truncated, total {len(result_str)} chars)")
            else:
                logger.info(f"{result_str}")
            logger.info(f"{'-'*60}\n")

            # Special handling for llm_rag to extract retrieved_docs
            if tool_name == "llm_rag":
                rag_invoked = True
                try:
                    # Parse JSON response from RAG
                    import json
                    if isinstance(result, str):
                        parsed_result = json.loads(result)
                    else:
                        parsed_result = result

                    # Extract retrieved_docs and summary
                    if isinstance(parsed_result, dict):
                        retrieved_docs = parsed_result.get("retrieved_docs", [])
                        ragas_evaluation = parsed_result.get("ragas_evaluation", {})
                        # Update retrieved_answers from args
                        retrieved_answers = tool_args.get("retrieved_answers", 5)
                        logger.info(f"Extracted {len(retrieved_docs)} retrieved documents from RAG\n")

                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse RAG result as JSON: {e}\n")
                    retrieved_docs = []
                    ragas_evaluation = {}

            # Create ToolMessage with the result
            tool_result = ToolMessage(
                content=str(result),
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_results.append(tool_result)

            logger.info(f"Tool {tool_name} executed successfully\n")

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}\n")

            # Create error ToolMessage
            tool_result = ToolMessage(
                content=f"Error: {str(e)}",
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_results.append(tool_result)

    # Return state with tool results appended to messages
    new_state = state.copy()
    new_state["messages"] = messages + tool_results
    new_state["last_tool_name"] = last_tool_name

    # Update RAG retrieval fields whenever llm_rag was invoked.
    # This avoids stale evidence from previous rounds.
    if rag_invoked:
        new_state["retrieved_docs"] = retrieved_docs
        new_state["retrieved_answers"] = retrieved_answers
        new_state["ragas_evaluation"] = ragas_evaluation
        logger.info(f"Added {len(retrieved_docs)} retrieved documents to state\n")

    return new_state


def create_react_graph(max_iterations: int = 10):
    """
    Create and compile the ReAct workflow graph.

    The graph structure:
        START -> clarify_gate -> (clarify_response | agent)
        agent -> (tools | END)
        tools -> evidence_gate -> (agent | final_answer | refuse_answer)

    Args:
        max_iterations: Maximum number of ReAct iterations to prevent infinite loops

    Returns:
        Compiled LangGraph StateGraph
    """
    # Create the workflow graph
    workflow = StateGraph(ReActState)
    logging.debug(workflow.state_schema)

    # Add nodes
    workflow.add_node("clarify_gate", clarify_gate_node)
    workflow.add_node("clarify_response", clarify_response_node)
    workflow.add_node("agent", react_agent_node)
    workflow.add_node("tools", custom_tool_node)  # Use custom tool node instead of ToolNode
    workflow.add_node("evidence_gate", evidence_gate_node)
    workflow.add_node("final_answer", final_answer_node)
    workflow.add_node("refuse_answer", refuse_answer_node)

    # Set entry point
    workflow.add_edge(START, "clarify_gate")

    workflow.add_conditional_edges(
        "clarify_gate",
        route_after_clarify_gate,
        {
            "clarify_response": "clarify_response",
            "agent": "agent",
        },
    )
    workflow.add_edge("clarify_response", END)

    # Add conditional edge from agent to either tools or end
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )

    # After tool execution, enter evidence gate first.
    workflow.add_edge("tools", "evidence_gate")

    workflow.add_conditional_edges(
        "evidence_gate",
        route_after_evidence_gate,
        {
            "agent": "agent",
            "final_answer": "final_answer",
            "refuse_answer": "refuse_answer",
        },
    )
    workflow.add_edge("final_answer", END)
    workflow.add_edge("refuse_answer", END)

    # Compile the graph
    react_graph = workflow.compile()

    logger.info("ReAct workflow graph created successfully\n")
    return react_graph


# Create the default ReAct graph instance
react_graph = create_react_graph(max_iterations=10)


# Helper function to run ReAct workflow
async def run_react(input_message: str, max_iterations: int = 10, expand_query_num: int = 3, retrieved_answers: int = 5, session_id: str = None) -> dict:
    """
    Run the ReAct workflow with a user input.

    Args:
        input_message: User's query or request
        max_iterations: Maximum number of ReAct iterations
        expanded_query: Number of query that expand based on question
        retrieved_answers: Number of documents to retrieve when using RAG
        session_id: Optional session ID for loading conversation history

    Returns:
        Dictionary containing:
            - messages: Full conversation history
            - output: Final answer
            - iteration_count: Number of iterations performed
            - retrieved_docs: Retrieved documents with similarity scores
            - retrieved_answers: Count of retrieved answers
    """
    # Load conversation history if session_id is provided
    if session_id:
        messages = load_history_conversation(input_message, session_id)
        logger.info(f"Message {messages} loaded\n")
        logger.info(f"Loaded {len(messages) - 1} historical messages for session {session_id}\n")
    else:
        messages = [HumanMessage(content=input_message)]

    initial_state: ReActState = {
        "messages": messages,
        "input": input_message,
        "max_iterations": max_iterations,
        "expand_query_num": expand_query_num,
        "iteration_count": 0,
        "retrieved_answers": retrieved_answers,
        "retrieval_retry_count": 0,
        "gate_action": "",
        "gate_reason": "",
        "final_decision": "",
    }

    try:
        result = await react_graph.ainvoke(initial_state)

        return {
            "messages": result.get("messages", []),
            "output": result.get("output", ""),
            "iteration_count": result.get("iteration_count", 0),
            "retrieved_docs": result.get("retrieved_docs", []),
            "retrieved_answers": result.get("retrieved_answers", 5),
            "ragas_evaluation": result.get("ragas_evaluation", {}),
            "final_decision": result.get("final_decision", ""),
            "gate_reason": result.get("gate_reason", ""),
            "evidence_summary": result.get("evidence_summary", {}),
            "retrieval_retry_count": result.get("retrieval_retry_count", 0),
            "success": True
        }

    except Exception as e:
        logger.error(f"Error running ReAct workflow: {e}\n")

        return {
            "messages": initial_state["messages"],
            "output": f"An error occurred: {str(e)}",
            "iteration_count": 0,
            "retrieved_docs": [],
            "retrieved_answers": 5,
            "ragas_evaluation": {},
            "final_decision": "",
            "gate_reason": "",
            "evidence_summary": {},
            "retrieval_retry_count": 0,
            "success": False,
            "error": str(e)
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def test_react():
        """Test the ReAct workflow with various queries."""

        test_queries = [
            "hello",  # Simple greeting - should use llm_chat
            "calculate (125 + 37) * 5",  # Calculation - should use llm_query
            "what causes herd mentality?",  # Knowledge retrieval - should use llm_rag
            "what is weather in beijing today?",  # Weather - should use llm_query
        ]

        print("\n" + "="*70)
        print(" "*20 + "ReAct Workflow Test Suite")
        print("="*70)

        for i, query in enumerate(test_queries, 1):
            print(f"\n{'-'*70}")
            print(f"Test {i}/4: {query}")
            print(f"{'-'*70}")

            result = await run_react(query, max_iterations=10)

            # Show iterations
            iterations = result['iteration_count']
            print(f"Completed in {iterations} iteration{'s' if iterations > 1 else ''}")

            # Show final answer
            
            if result.get('output'):
                print(f"\nFinal Answer:\n{result['output'][:300]}")
                if len(result['output']) > 300:
                    print("...")

            # Show status
            status = "Success" if result['success'] else "Failed"
            print(f"\nStatus: {status}")

            # Show conversation summary
            if result.get("success"):
                messages = result['messages']
                tool_calls_count = sum(1 for m in messages if isinstance(m, AIMessage) and hasattr(m, 'tool_calls') and m.tool_calls)
                tool_results_count = sum(1 for m in messages if isinstance(m, ToolMessage))
                print(f"Conversation: {len(messages)} messages ({tool_calls_count} tool calls, {tool_results_count} tool results)")

        print(f"\n{'='*70}")
        print(" "*25 + "All Tests Completed")
        print("="*70 + "\n")

    # Run tests
    asyncio.run(test_react())
