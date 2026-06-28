import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from core.config_loader import get_llm_runtime_config, load_runtime_env

load_dotenv(Path(__file__).resolve().parent / ".env", override=False, encoding="utf-8-sig")
load_runtime_env()


def _build_client_and_model(model_override: str = "") -> tuple[OpenAI, str]:
    runtime = get_llm_runtime_config()
    resolved_model = (model_override or "").strip() or runtime["model"]
    if not runtime["api_key"] or not runtime["base_url"] or not resolved_model:
        raise RuntimeError(
            "LLM runtime config incomplete. Please check config/app.yaml -> llm.current_model and providers."
        )
    client = OpenAI(api_key=runtime["api_key"], base_url=runtime["base_url"])
    return client, resolved_model


# Connect to configured LLM model (Responses API).
def openai_chat(user_content: str = "Write a one-sentence bedtime story about a unicorn."):
    client, resolved_model = _build_client_and_model()
    response = client.responses.create(
        model=resolved_model,
        input=user_content,
    )
    logging.debug("response: %s", response)
    return response


# Connect to configured LLM model (Chat Completions API).
def deepseek_chat(model: str, user_content: str):
    client, resolved_model = _build_client_and_model(model_override=model)
    resp = client.chat.completions.create(
        model=resolved_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": user_content},
        ],
        stream=False,
    )
    return resp.choices[0].message.content


def model_choose(model_name: str, user_content: str):
    model_name = (model_name or "").strip().lower()
    if model_name in {"openai", "responses"}:
        return openai_chat(user_content)

    target_model = model_name or get_llm_runtime_config()["model"]
    return deepseek_chat(target_model, user_content)


if __name__ == "__main__":
    runtime = get_llm_runtime_config()
    model_choose(runtime.get("model", ""), "你好")
