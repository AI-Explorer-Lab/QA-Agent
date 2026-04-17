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


from controller.apis import router
from middlewares.exception_handler import app_exception_handler
from middlewares.request_log import RequestLogMiddleware
from utils.config_loader import get_app_config

config = get_app_config()
app = FastAPI(title=config.get("app", {}).get("name", "trusted-pdf-qa"))
app.add_middleware(RequestLogMiddleware)
app.add_exception_handler(Exception, app_exception_handler)
app.include_router(router)

# Backward-compatible variable name for older run commands, but only new routes are included.
app_chat_llm = app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
