from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass
class _CacheEntry(Generic[T]):
    value: T
    expires_at: float


class TTLCache(Generic[T]):
    def __init__(self, ttl_seconds: int = 3600, max_items: int = 5000) -> None:
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.max_items = max(1, int(max_items))
        self._lock = threading.Lock()
        self._entries: "OrderedDict[str, _CacheEntry[T]]" = OrderedDict()

    def _now(self) -> float:
        return time.time()

    def _purge_expired_unlocked(self, now: float) -> None:
        expired_keys = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired_keys:
            self._entries.pop(key, None)

    def _evict_overflow_unlocked(self) -> None:
        while len(self._entries) > self.max_items:
            self._entries.popitem(last=False)

    def get(self, key: str) -> Optional[T]:
        now = self._now()
        with self._lock:
            self._purge_expired_unlocked(now)
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry.value

    def set(self, key: str, value: T) -> None:
        now = self._now()
        expires_at = now + self.ttl_seconds
        with self._lock:
            self._purge_expired_unlocked(now)
            self._entries[key] = _CacheEntry(value=value, expires_at=expires_at)
            self._entries.move_to_end(key)
            self._evict_overflow_unlocked()

    def pop(self, key: str, default: Optional[T] = None) -> Optional[T]:
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                return default
            return entry.value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        now = self._now()
        with self._lock:
            self._purge_expired_unlocked(now)
            return len(self._entries)


class EmbeddingCache(TTLCache[list[float]]):
    pass


class ChunkEmbeddingCache(TTLCache[list[float]]):
    pass
