import hashlib
import logging
import os
from http import HTTPStatus
from typing import List, Optional

from dotenv import load_dotenv
from core.config_loader import load_runtime_env

load_dotenv()
load_runtime_env()

logger = logging.getLogger(__name__)

try:
    import dashscope
except Exception:  # pragma: no cover - optional runtime dependency
    dashscope = None


def _local_embedding(text: str, dimension: int = 1024) -> List[float]:
    """
    Deterministic fallback embedding for offline/sandbox testing.
    This keeps the system runnable when remote embedding service is unavailable.
    """
    text = text or ""
    values = [0.0] * dimension
    for idx, token in enumerate(text.split()):
        digest = hashlib.sha256(f"{token}-{idx}".encode("utf-8")).digest()
        for i in range(0, min(dimension, len(digest))):
            values[(idx + i) % dimension] += digest[i] / 255.0

    norm = sum(v * v for v in values) ** 0.5
    if norm == 0:
        return values
    return [v / norm for v in values]


def embedding(input_text: str = "测试文本") -> Optional[List[float]]:
    qwen_key = os.getenv("QWEN_API_KEY", "").strip()
    if dashscope is not None and qwen_key:
        try:
            response = dashscope.TextEmbedding.call(
                model="text-embedding-v4",
                input=input_text,
                api_key=qwen_key,
                timeout=10,
            )
            if response.status_code == HTTPStatus.OK:
                return response.output["embeddings"][0]["embedding"]
            logger.warning("Dashscope embedding failed: %s - %s", response.status_code, response.message)
        except Exception as exc:
            logger.warning("Dashscope embedding error: %s", exc)

    # Offline-safe fallback
    fallback_dim = 1536
    try:
        fallback_dim = int(os.getenv("PGVECTOR_EMBEDDING_DIM", str(fallback_dim)))
    except Exception:
        fallback_dim = 1536
    return _local_embedding(input_text, dimension=fallback_dim)


if __name__ == "__main__":
    vector = embedding()
    print(f"embedding_dim={len(vector) if vector else 0}")
