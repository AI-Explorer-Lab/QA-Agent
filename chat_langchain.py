import getpass
import logging
from dotenv import load_dotenv

## sys.path.insert(0, r'C:\Users\dell\AppData\Roaming\Python\Python314\site-packages')

from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from core.config_loader import get_llm_runtime_config, load_runtime_env

load_dotenv()
load_runtime_env()

workflow = StateGraph(state_schema=MessagesState)

def build_client():
    runtime = get_llm_runtime_config()
    base_url = runtime["base_url"]
    model_name = runtime["model"]
    api_key = runtime["api_key"]

    if not (base_url and api_key and model_name):
        raise RuntimeError(
            "LLM config is incomplete. Please set llm.current_model/provider config in config/app.yaml"
        )

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        use_responses_api=runtime["use_responses_api"],
        temperature=0,
        timeout=120,
        max_retries=2,
    )


client = build_client()


def chat_with_llm(state: MessagesState):
    model = client
    system_prompt = (
        "You are a helpful assistant. "
        "Answer all questions to the best of your ability. "
        "The provided chat history includes a summary of the earlier conversation."
        "If \"AI最近一条回复\" appeared, that means this is a latest message stored in SQL that AI answered. You should read and understand it before you answer."
    )
    system_message = SystemMessage(content=system_prompt)
    message_history = state["messages"][:-1]  # exclude the most recent user input
    if len(message_history) >= 2:
        last_human_message = state["messages"][-1]
        # Invoke the model to generate conversation summary
        summary_prompt = (
            "Distill the above chat messages into a single summary message. "
            "Include as many specific details as you can."
            "Language of summary must in Chinese."
        )
        summary_message = model.invoke(
            input=message_history + [HumanMessage(content=summary_prompt)]
        )
        logging.debug("summary_message: ", summary_message.content)

        # Delete messages that we no longer want to show up
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
        # Re-add user message
        human_message = HumanMessage(content=last_human_message.content)
        # Call the model with summary & response
        response = model.invoke([system_message, summary_message, human_message])
        message_updates = [summary_message, human_message, response] + delete_messages
    else:
        message_updates = model.invoke([system_message] + state["messages"])

    return {"messages": message_updates}


# Define the node and edge
workflow.add_node("model", chat_with_llm)
workflow.add_edge(START, "model")

# Add simple in-memory checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    demo_ephemeral_chat_history = [
        HumanMessage(content="Hey there! I'm Nemo."),
        AIMessage(content="Hello!"),
        HumanMessage(content="How are you today?"),
        AIMessage(content="Fine thanks!"),
    ]
    result = app.invoke(
        {
            "messages": demo_ephemeral_chat_history
                        + [HumanMessage("What did I say my name was?")]
        },
        config={"configurable": {"thread_id": "4"}},
    )

