from __future__ import annotations

from fastapi import FastAPI
import inspect


def _patch_httpx_testclient_compat() -> None:
    try:
        import httpx
    except Exception:
        return
    signature = inspect.signature(httpx.Client.__init__)
    if "app" in signature.parameters:
        return
    original_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.pop("app", None)
        return original_init(self, *args, **kwargs)

    httpx.Client.__init__ = patched_init


_patch_httpx_testclient_compat()


from core.config_loader import load_runtime_env

# Load .env/YAML-backed runtime settings before importing modules that
# initialize database-backed services at import time.
load_runtime_env()

from controller.apis import router
from middlewares.exception_handler import app_exception_handler
from middlewares.operation_log import log_operation_event
from middlewares.request_log import RequestLogMiddleware
from utils.config_loader import get_app_config

config = get_app_config()
app = FastAPI(title=config.get("app", {}).get("name", "trusted-pdf-qa"))
app.add_middleware(RequestLogMiddleware)
app.add_exception_handler(Exception, app_exception_handler)
app.include_router(router)


@app.on_event("startup")
async def preload_runtime_models() -> None:
    reranker_cfg = config.get("reranker", {}) if isinstance(config.get("reranker"), dict) else {}
    if not bool(reranker_cfg.get("cross_encoder_preload_on_startup", False)):
        return
    try:
        import asyncio

        from service.agent.trusted_qa_workflow import get_trusted_qa_workflow

        workflow = get_trusted_qa_workflow()
        reranker = getattr(getattr(workflow, "retriever", None), "reranker", None)
        warmup = getattr(reranker, "warmup_cross_encoder", None)
        if not callable(warmup):
            log_operation_event("runtime.cross_encoder_preload", status="skipped", reason="warmup_unavailable")
            return
        log_operation_event("runtime.cross_encoder_preload", status="started")
        result = await asyncio.to_thread(warmup)
        log_operation_event("runtime.cross_encoder_preload", **result)
    except Exception as exc:
        log_operation_event(
            "runtime.cross_encoder_preload",
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc)[:500],
        )



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
