from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from workflow.rag_workflow import rag_graph
from core.config_loader import get_llm_runtime_config, load_runtime_env
from core.content_normalizer import normalize_content
import logging

load_dotenv()
load_runtime_env()

class State(TypedDict):
    input: str  # user query input
    output: NotRequired[str]  # model output answer (deprecated, extracted from messages)
    conversation_history: NotRequired[list]  # store all conversation history
    messages: list  # LangChain message history for tool calling
    retrieved_answers: NotRequired[int]  # count of retrieved answers, defaults to 5
    retrieved_docs: NotRequired[list]  # raw retrieved documents with similarity scores
    ragas_evaluation: NotRequired[dict]

# Global agent instance
_runtime = get_llm_runtime_config()
agent = ChatOpenAI(
    api_key=_runtime["api_key"],
    base_url=_runtime["base_url"],
    model=_runtime["model"],
    use_responses_api=_runtime["use_responses_api"],
)


def _build_runtime_chat_model() -> ChatOpenAI:
    runtime = get_llm_runtime_config()
    return ChatOpenAI(
        api_key=runtime["api_key"],
        base_url=runtime["base_url"],
        model=runtime["model"],
        use_responses_api=runtime["use_responses_api"],
        temperature=0,
    )

@tool
async def llm_chat(query: str) -> str:
    """
    MCP-based tool for common conversations
    """
    result = await agent.ainvoke([HumanMessage(content=query)])
    # ainvoke returns AIMessage directly
    if isinstance(result, AIMessage):
        return normalize_content(result.content)
    return "No response"

@tool
async def llm_query(query: str) -> str:
    """
    MCP-based tool for calculator / weather / general query
    """
    client = MultiServerMCPClient(
        {
            "calculator_service": {
                "transport": "stdio",
                "command": "python",
                "args": ["mcp_tool/calculator_server.py"],
            },
            "weather_service": {
                "transport": "stdio",
                "command": "python",
                "args": ["mcp_tool/weather_server.py"],
            }
        }
    )

    try:
        mcp_tools = await client.get_tools()

        query_agent = create_agent(
            _build_runtime_chat_model(),
            mcp_tools,
            system_prompt=SystemMessage(
                content=f"You are a helpful AI assistant. Answer the user's query: {query}"
            )
        )

        result = await query_agent.ainvoke({
            "messages": [HumanMessage(content=query)]
        })

        ai_message = None
        if "messages" in result:
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    ai_message = msg
                    break

        return normalize_content(ai_message.content) if ai_message else "No response"

    except Exception as e:
        logging.error(f"llm_query error: {e}")
        return f"Error: {e}"

@tool
async def llm_rag(
    query: str,
    expand_query: int = 3,
    retrieved_answers: int = 5
) -> str:
    """
    RAG tool: Retrieve documents and generate summary based on them.
    Returns JSON string with summary and retrieved documents.
    """
    try:
        result = await rag_graph.ainvoke({
            "input": query,
            "retrieved_answers": retrieved_answers,
            "expand_query_num": expand_query,
            "messages": [],
            "conversation_history": [],
            "output": "",
            "task_completed": False,
        })

        # Extract the summary output and retrieved documents
        summary = normalize_content(result.get("output", "No summary generated"))
        retrieved_docs = result.get("retrieved_docs", [])
        ragas_evaluation = result.get("ragas_evaluation", {})

        # Return as JSON string
        import json
        return json.dumps({
            "summary": summary,
            "retrieved_docs": retrieved_docs,
            "ragas_evaluation": ragas_evaluation,
        }, ensure_ascii=False)

    except Exception as e:
        logging.error(f"llm_rag error: {e}")
        import json
        return json.dumps({
            "summary": f"RAG error: {e}",
            "retrieved_docs": [],
            "ragas_evaluation": {},
        }, ensure_ascii=False)

# def team_leader(state: State) -> State:
#     """
#     Use agent to understand user intent and decide workflow path by calling appropriate tool
#     """

#     prompt = f"""
#     You are a helpful AI assistant. Based on the user's query: {state['input']},
#     decide whether to use llm_chat for common conversations, llm_query for calculations, weather info,
#     or llm_rag for document retrieval or research.

#     If the query is about common chat (for example, common greeting ,.etc), call the llm_chat tool.
#     If the query is related to calculations, math operations, weather information or common conversations, call the llm_query tool.
#     If it requires document retrieval, research, or detailed information retrieval, call the llm_rag tool.
    
    
#     Additionally, if you choose weather realated tool in llm_query, you should translate the query city name into English with lowercase before calling llm_query.
#     If you call llm_rag, you MUST pass:
#     - query = the user query
#     - retrieved_answers = {state.get("retrieved_answers")}
#     """
#     logging.debug(f"team_leader state: {state}")
#     # Bind the tools to the client
#     model_with_tool = agent.bind_tools([llm_chat, llm_query, llm_rag])

#     response = model_with_tool.invoke([
#         HumanMessage(content=prompt)
#     ])

#     new_state = state.copy()
#     new_state["messages"] = [
#         HumanMessage(content=state["input"]),
#         response
#     ]

async def team_leader(state: State) -> State:
    """Route user query to appropriate tool using LLM."""
    # Get existing messages or initialize with user input
    messages = state.get("messages", [])

    # If this is the first call, add the user's message
    if not messages or len(messages) == 0:
        messages = [HumanMessage(content=state["input"])]

    # Build the prompt for the router
    prompt = f"""
You are a router agent.

User query:
{state['input']}

Decide which tool to call:

- llm_chat(query: str) - for common conversations
- llm_query(query: str) - for calculations, weather info, general queries
- llm_rag(query: str, retrieved_answers: int) - for document retrieval or research

IMPORTANT:
You MUST call exactly ONE tool.
If you choose llm_rag, you MUST pass retrieved_answers={state.get("retrieved_answers", 5)}.
"""

    # Add the prompt as a human message
    messages.append(HumanMessage(content=prompt))

    model_with_tools = agent.bind_tools(
        [llm_chat, llm_query, llm_rag]
    )

    ai_message = await model_with_tools.ainvoke(messages)

    # Update state with the AI response (which should contain tool_calls)
    new_state = state.copy()
    new_state["messages"] = messages + [ai_message]

    return new_state


# Create ToolNode with all tools
tool_node = ToolNode([llm_chat, llm_query, llm_rag])


# Create workflow graph
workflow = StateGraph(State)

# Define nodes
workflow.add_node("team_leader", team_leader)
workflow.add_node("tools", tool_node)

# Add entry point
workflow.add_edge(START, "team_leader")

# team_leader always calls a tool (enforced by prompt), so go directly to tools
workflow.add_edge("team_leader", "tools")

# After tools execute, go directly to END
workflow.add_edge("tools", END)

# Compile the graph (async-compatible)
graph = workflow.compile()
