from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import httpx

from utils.config_loader import get_app_config

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    import yaml
except Exception:
    yaml = None

try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None

try:
    from pydantic import BaseModel
except Exception:
    BaseModel = object  # type: ignore[assignment]

try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    ChatPromptTemplate = None

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_APP_CONFIG = PROJECT_ROOT / "config" / "app.yaml"


def _load_env_once() -> None:
    if load_dotenv is None:
        return
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _env_truthy(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _extract_json_array(text: str) -> Optional[List[str]]:
    raw = (text or "").strip()
    if not raw:
        return None

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        if isinstance(parsed, dict):
            for key in ("queries", "expanded_queries", "items"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [str(item).strip() for item in value if str(item).strip()]
    except Exception:
        pass

    match = re.search(r"\[[\s\S]*\]", raw)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass

    object_match = re.search(r"\{[\s\S]*\}", raw)
    if object_match:
        try:
            parsed = json.loads(object_match.group(0))
            if isinstance(parsed, dict):
                for key in ("queries", "expanded_queries", "items"):
                    value = parsed.get(key)
                    if isinstance(value, list):
                        return [str(item).strip() for item in value if str(item).strip()]
        except Exception:
            pass

    line_queries: List[str] = []
    for line in raw.splitlines():
        value = re.sub(r"^\s*(?:[-*•]|\d+[.)、:]|[（(]?\d+[）)])\s*", "", line).strip()
        value = value.strip("\"'“”‘’`，,。;；")
        if not value:
            continue
        if value.startswith(("[", "]", "{", "}")):
            continue
        lowered = value.lower()
        if lowered.startswith(("here are", "queries", "expanded queries", "json")):
            continue
        if value not in line_queries:
            line_queries.append(value)
        if len(line_queries) >= 4:
            break
    return line_queries or None


def _extract_json_object(text: str) -> Dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _load_legacy_llm_config() -> Dict[str, Any]:
    if yaml is None or not LEGACY_APP_CONFIG.exists():
        return {}
    try:
        loaded = yaml.safe_load(LEGACY_APP_CONFIG.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}
    llm = loaded.get("llm")
    return llm if isinstance(llm, dict) else {}


def _deep_merge_nonempty(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        if value in (None, ""):
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_nonempty(merged[key], value)
        else:
            merged[key] = value
    return merged


def _env_or_literal(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    env_value = os.getenv(raw)
    if env_value:
        return env_value.strip()
    if raw.startswith(("sk-", "sk_")) or not re.fullmatch(r"[A-Z0-9_]+", raw):
        return raw
    return ""


def _safe_error(error: Exception, secret: str = "") -> str:
    text = f"{type(error).__name__}: {error}"
    if secret:
        text = text.replace(secret, "***")
    return text[:700]


def _extract_response_text(response: Any) -> str:
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    chunks: List[str] = []
    output = getattr(response, "output", None) or []
    for item in output:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        if not content:
            continue
        for part in content:
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
    return "\n".join(chunks).strip()


class LLMService:
    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        _load_env_once()
        self.config = config or get_app_config(reload=True)
        loaded_llm = self.config.get("llm", {}) if isinstance(self.config.get("llm"), dict) else {}
        legacy_llm = _load_legacy_llm_config() if config is None else {}
        llm_cfg = _deep_merge_nonempty(legacy_llm, loaded_llm)

        providers = llm_cfg.get("providers", {}) if isinstance(llm_cfg.get("providers"), dict) else {}
        selector = str(llm_cfg.get("current_model") or llm_cfg.get("model") or "gpt-4o-mini").strip()
        provider_name = str(llm_cfg.get("provider") or "").strip()
        provider_cfg = providers.get(provider_name) if provider_name in providers else None
        model_key = selector

        if provider_cfg is not None and provider_name and selector.startswith(provider_name + "-"):
            model_key = selector[len(provider_name) + 1 :]
        elif provider_cfg is None and "-" in selector:
            prefix, remainder = selector.split("-", 1)
            if prefix in providers:
                provider_name = prefix
                provider_cfg = providers.get(prefix)
                model_key = remainder

        provider_cfg = provider_cfg if isinstance(provider_cfg, dict) else {}
        models = provider_cfg.get("models", {}) if isinstance(provider_cfg.get("models"), dict) else {}
        model_cfg = models.get(model_key, {}) if isinstance(models.get(model_key), dict) else {}

        self.provider_name = provider_name or "openai_compatible"
        self.model = (
            os.getenv("TRUSTED_QA_LLM_MODEL")
            or str(model_cfg.get("model") or llm_cfg.get("model") or selector or "gpt-4o-mini")
        )
        self.temperature = float(os.getenv("TRUSTED_QA_LLM_TEMPERATURE") or llm_cfg.get("temperature", 0.2) or 0.2)
        default_enabled = bool(llm_cfg.get("enable_real_generation", True))
        self.enabled = _env_truthy("TRUSTED_QA_ENABLE_REAL_LLM", default_enabled)
        self.timeout_seconds = float(
            os.getenv("TRUSTED_QA_LLM_TIMEOUT_SECONDS")
            or model_cfg.get("timeout_seconds")
            or provider_cfg.get("timeout_seconds")
            or llm_cfg.get("timeout_seconds", 30)
            or 30
        )
        configured_max_retries = (
            os.getenv("TRUSTED_QA_LLM_MAX_RETRIES")
            or model_cfg.get("max_retries")
            or provider_cfg.get("max_retries")
            or llm_cfg.get("max_retries")
            or 2
        )
        self.max_retries = max(0, _safe_int(configured_max_retries, 2))
        self.client_mode = str(
            os.getenv("TRUSTED_QA_LLM_CLIENT_MODE")
            or model_cfg.get("client_mode")
            or provider_cfg.get("client_mode")
            or llm_cfg.get("client_mode")
            or ("direct" if provider_name.lower() == "rightcode" else "sdk")
        ).strip().lower()
        self.use_responses_api = bool(llm_cfg.get("use_responses_api", provider_cfg.get("use_responses_api", False)))

        yaml_api_key = str(llm_cfg.get("api_key") or "").strip()
        provider_api_key = str(provider_cfg.get("api_key") or "").strip()
        api_key_env = str(llm_cfg.get("api_key_env") or "").strip()
        provider_api_key_env = str(provider_cfg.get("api_key_env") or "").strip()
        self.api_key = (
            yaml_api_key
            or provider_api_key
            or _env_or_literal(api_key_env)
            or _env_or_literal(provider_api_key_env)
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("ANYROUTER_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or ""
        )

        base_url_env = str(llm_cfg.get("base_url_env") or "").strip()
        provider_base_url_env = str(provider_cfg.get("base_url_env") or "").strip()
        self.base_url = (
            str(llm_cfg.get("base_url") or "").strip()
            or str(provider_cfg.get("base_url") or "").strip()
            or (os.getenv(base_url_env) if base_url_env else "")
            or (os.getenv(provider_base_url_env) if provider_base_url_env else "")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("ANYROUTER_BASE_URL")
            or os.getenv("DASHSCOPE_BASE_URL")
            or None
        )

        self._install_proxy(provider_cfg)
        self._client = None
        self._langchain_client = None
        self.last_error = ""
        self.last_call_mode = ""
        self.call_attempt_count = 0

    def _install_proxy(self, provider_cfg: Dict[str, Any]) -> None:
        proxy_pairs = {
            "HTTP_PROXY": provider_cfg.get("http_proxy"),
            "HTTPS_PROXY": provider_cfg.get("https_proxy"),
            "NO_PROXY": provider_cfg.get("no_proxy"),
        }
        for env_name, value in proxy_pairs.items():
            if value and not os.getenv(env_name):
                os.environ[env_name] = str(value)

    @property
    def is_available(self) -> bool:
        if getattr(self, "client_mode", "") == "direct":
            return bool(self.enabled and self.api_key and self.base_url)
        return bool(self.enabled and self.api_key and AsyncOpenAI is not None)

    @property
    def langchain_available(self) -> bool:
        if getattr(self, "client_mode", "") == "direct":
            return False
        return bool(self.enabled and self.api_key and ChatPromptTemplate is not None and ChatOpenAI is not None)

    def trace_metadata(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "available": self.is_available,
            "provider": self.provider_name,
            "model": self.model,
            "base_url_set": bool(self.base_url),
            "client_mode": self.client_mode,
            "use_responses_api": self.use_responses_api,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "langchain_available": self.langchain_available,
            "call_attempt_count": self.call_attempt_count,
            "last_call_mode": self.last_call_mode,
            "last_error": self.last_error,
        }

    def _client_instance(self):
        if not self.is_available:
            if not self.enabled:
                self.last_error = "real LLM disabled by TRUSTED_QA_ENABLE_REAL_LLM or YAML"
            elif not self.api_key:
                self.last_error = "missing LLM API key"
            elif AsyncOpenAI is None:
                self.last_error = "openai package is not installed"
            return None
        if self._client is None:
            kwargs = {"api_key": self.api_key, "timeout": self.timeout_seconds, "max_retries": self.max_retries}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    def _langchain_client_instance(self):
        if not self.langchain_available:
            return None
        if self._langchain_client is None:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "temperature": self.temperature,
                "api_key": self.api_key,
                "timeout": self.timeout_seconds,
                "max_retries": self.max_retries,
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._langchain_client = ChatOpenAI(**kwargs)
        return self._langchain_client

    async def _complete_direct(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        model_override: str | None = None,
    ) -> Optional[str]:
        if not self.enabled:
            self.last_error = "real LLM disabled by TRUSTED_QA_ENABLE_REAL_LLM or YAML"
            return None
        if not self.api_key:
            self.last_error = "missing LLM API key"
            return None
        if not self.base_url:
            self.last_error = "missing LLM base_url"
            return None

        self.call_attempt_count += 1
        self.last_call_mode = "direct.chat.completions"
        self.last_error = ""

        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": str(model_override or self.model),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") if isinstance(data, dict) else None
            if not choices:
                self.last_error = "direct chat.completions returned no choices"
                return None
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            content = message.get("content") if isinstance(message, dict) else ""
            if isinstance(content, list):
                content = "\n".join(str(part.get("text") if isinstance(part, dict) else part) for part in content if str(part).strip())
            text = str(content or "").strip()
            if not text:
                self.last_error = "direct chat.completions returned empty content"
                return None
            return text
        except Exception as error:
            self.last_error = "direct.chat.completions " + _safe_error(error, self.api_key)
            return None

    async def structured_json(
        self,
        system_prompt: str,
        user_payload: Any,
        schema: Type[Any],
        max_tokens: int = 512,
    ) -> Optional[Dict[str, Any]]:
        user_prompt = user_payload if isinstance(user_payload, str) else json.dumps(user_payload, ensure_ascii=False)
        if self.langchain_available:
            model = self._langchain_client_instance()
            if model is not None:
                self.call_attempt_count += 1
                try:
                    self.last_call_mode = "langchain.structured_output"
                    prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", "{system_prompt}"),
                            ("user", "{user_prompt}"),
                        ]
                    )
                    messages = prompt.format_messages(system_prompt=system_prompt, user_prompt=user_prompt)
                    runnable = model.with_structured_output(schema)
                    result = await runnable.ainvoke(messages)
                    self.last_error = ""
                    if hasattr(result, "model_dump"):
                        return result.model_dump()
                    if isinstance(result, dict):
                        return result
                    if hasattr(result, "dict"):
                        return result.dict()
                except Exception as error:
                    self.last_error = "langchain.structured_output " + _safe_error(error, self.api_key)

        content = await self.complete(system_prompt, user_prompt, max_tokens=max_tokens)
        return _extract_json_object(content or "")

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        model_override: str | None = None,
    ) -> Optional[str]:
        if self.client_mode == "direct":
            return await self._complete_direct(system_prompt, user_prompt, max_tokens=max_tokens, model_override=model_override)

        if self.langchain_available:
            model = self._langchain_client_instance()
            if model is not None:
                self.call_attempt_count += 1
                try:
                    self.last_call_mode = "langchain.chat"
                    prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", "{system_prompt}"),
                            ("user", "{user_prompt}"),
                        ]
                    )
                    chain = prompt | model
                    response = await chain.ainvoke({"system_prompt": system_prompt, "user_prompt": user_prompt})
                    content = getattr(response, "content", response)
                    if isinstance(content, list):
                        content = "\n".join(str(part) for part in content if str(part).strip())
                    if isinstance(content, str) and content.strip():
                        self.last_error = ""
                        return content.strip()
                except Exception as error:
                    self.last_error = "langchain.chat " + _safe_error(error, self.api_key)

        client = self._client_instance()
        if client is None:
            return None

        self.call_attempt_count += 1
        self.last_error = ""
        errors: List[str] = []

        if self.use_responses_api and hasattr(client, "responses"):
            try:
                self.last_call_mode = "responses"
                response = await client.responses.create(
                    model=str(model_override or self.model),
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_output_tokens=max_tokens,
                )
                text = _extract_response_text(response)
                if text:
                    self.last_error = ""
                    return text
            except Exception as error:
                errors.append("responses " + _safe_error(error, self.api_key))

        try:
            self.last_call_mode = "chat.completions"
            response = await client.chat.completions.create(
                model=str(model_override or self.model),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
            if response.choices:
                text = response.choices[0].message.content or ""
                if text.strip():
                    self.last_error = ""
                    return text.strip()
        except Exception as error:
            errors.append("chat.completions " + _safe_error(error, self.api_key))

        self.last_error = "; ".join(errors)[:1000] if errors else "LLM returned empty response"
        return None

    def _query_expansion_model(self) -> str:
        configured = str(os.getenv("TRUSTED_QA_QUERY_EXPANSION_MODEL") or "").strip()
        if configured:
            return configured
        if self.provider_name.lower() == "deepseek" and self.model != "deepseek-chat":
            return "deepseek-chat"
        return self.model

    async def expand_queries(self, question: str, query_type: str, expand_query_num: int) -> Optional[List[str]]:
        if not (self.is_available or self.langchain_available):
            self._client_instance()
            return None
        del expand_query_num
        total = 4
        system_prompt = (
            "You rewrite enterprise PDF QA questions for hybrid retrieval. "
            "Return valid JSON only: an array of exactly four retrieval query strings. "
            "Do not answer the question and do not extract values."
        )
        user_prompt = (
            f"Question: {question}\n"
            f"Query type: {query_type}\n"
            "Return exactly 4 Chinese-friendly search queries: the original question first, "
            "then two concise noise-reduced rewrite variants, then one query_type scene-enhanced variant. "
            "Keep named entities, years, metrics, table headers and document targets. "
            "Do not add page numbers unless the original question already contains them. "
            "Each item must be a complete search query, not a field name, value, unit, or answer. "
            "Output format example: [\"2025 operating revenue\", \"2025 revenue table\", \"operating revenue value unit 2025\", \"table_qa operating revenue metric 2025\"]"
        )
        content = await self.complete(
            system_prompt,
            user_prompt,
            max_tokens=320,
            model_override=self._query_expansion_model(),
        )
        queries = _extract_json_array(content or "")
        if not queries:
            return None
        deduped: List[str] = []
        for item in [question] + queries:
            value = str(item or "").strip()
            if value and value not in deduped:
                deduped.append(value)
            if len(deduped) >= total:
                break
        return deduped

    async def generate_grounded_answer(
        self,
        question: str,
        query_type: str,
        evidence: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
    ) -> Optional[str]:
        if not (self.is_available or self.langchain_available) or not evidence:
            self._client_instance()
            return None
        evidence_lines = []
        for index, item in enumerate(evidence, start=1):
            citation_id = citations[index - 1].get("citation_id", f"C{index}") if index - 1 < len(citations) else f"C{index}"
            evidence_lines.append(
                f"[{citation_id}] doc={item.get('doc_source')} page={item.get('metadata', {}).get('page_idx')} "
                f"heading={item.get('metadata', {}).get('heading_path')} content={item.get('content')}"
            )
        system_prompt = (
            "You are a trusted enterprise PDF QA agent. Answer only from the provided evidence. "
            "Every key claim must cite citation ids like [C1]. If evidence is insufficient, say so. "
            "For table QA, include metric, value, unit, period and source when present. "
            "Do not use ellipses or omit cited facts by writing ...; cite the relevant evidence ids explicitly. "
            "The evidence is pre-ordered by requested entity and document section. Preserve that logical order. "
            "Do not put history before profile. Do not include unrelated financial statement sections unless the user asks for them. "
            "Do not output raw TABLE_START or TABLE_END markers; render useful tables as normal Markdown tables."
        )
        user_prompt = (
            f"Question: {question}\n"
            f"Query type: {query_type}\n"
            "Evidence:\n" + "\n".join(evidence_lines) + "\n"
            "Write the final answer in Chinese. Do not simply list evidence snippets; synthesize them into the requested structure."
        )
        answer = await self.complete(system_prompt, user_prompt, max_tokens=1600)
        if not answer or not answer.strip():
            return None
        return answer.strip()


_DEFAULT_LLM_SERVICE: LLMService | None = None


def get_llm_service() -> LLMService:
    global _DEFAULT_LLM_SERVICE
    if _DEFAULT_LLM_SERVICE is None:
        _DEFAULT_LLM_SERVICE = LLMService()
    return _DEFAULT_LLM_SERVICE
