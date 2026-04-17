import asyncio

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from core.config_loader import get_llm_runtime_config, load_runtime_env

load_dotenv()
load_runtime_env()


async def llm_agent_demo():
    runtime = get_llm_runtime_config()
    model_name = runtime.get("model", "")
    if not model_name:
        raise RuntimeError("LLM model is empty. Please configure llm.current_model in config/app.yaml.")

    # Create an MCP client that connects to both calculator and weather servers
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
            },
        }
    )

    # Get the tools from the server
    tools = await client.get_tools()
    print("Available tools:", [tool.name for tool in tools])

    # Create an agent using the configured model
    agent = create_agent(
        ChatOpenAI(
            api_key=runtime["api_key"],
            base_url=runtime["base_url"],
            model=model_name,
            use_responses_api=runtime["use_responses_api"],
            temperature=0,
        ),
        tools,
        system_prompt=SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": "You are an AI assistant tasked with math calculations and short factual answers. Be accurate and concise.",
                }
            ]
        ),
    )

    queries = [
        "Calculate 15 plus 27 for me",
        "Can you multiply 123 and 3?",
        "How about 6 divide by 2?",
        "What's the current weather in Beijing?",
    ]

    for query_num, query in enumerate(queries, start=1):
        print(f"\n--- Query {query_num}: {query} ---")
        try:
            result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})

            ai_message = None
            if "messages" in result:
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage):
                        ai_message = msg
                        break

            if ai_message:
                print(f"AI Response: {ai_message.content}")
            else:
                print("Could not find AI response in the result")

        except Exception as exc:  # noqa: BLE001
            print(f"Error processing query '{query}': {exc}")


if __name__ == "__main__":
    asyncio.run(llm_agent_demo())
