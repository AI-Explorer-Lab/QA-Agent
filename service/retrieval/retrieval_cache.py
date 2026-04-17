from __future__ import annotations

import copy
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalCacheKey:
    collection_name: str
    question_hash: str
    query_type: str
    top_k: int

    def as_storage_key(self) -> str:
        return "retrieval:{collection}:{question}:{query_type}:{top_k}".format(
            collection=self.collection_name,
            question=self.question_hash,
            query_type=self.query_type,
            top_k=self.top_k,
        )


class RetrievalResultCache:
    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_items: int = 5000,
        time_fn: Any | None = None,
    ) -> None:
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.max_items = max(1, int(max_items))
        self._time_fn = time_fn or time.time
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def build_key(
        self,
        collection_name: str,
        question_hash: str,
        query_type: str,
        top_k: int,
    ) -> RetrievalCacheKey:
        return RetrievalCacheKey(
            collection_name=str(collection_name or "default"),
            question_hash=str(question_hash or ""),
            query_type=str(query_type or "fact_lookup"),
            top_k=max(1, int(top_k)),
        )

    def get(self, key: RetrievalCacheKey) -> Any | None:
        self._evict_expired()
        storage_key = key.as_storage_key()
        entry = self._store.get(storage_key)
        if entry is None:
            return None

        expires_at, value = entry
        now = self._time_fn()
        if expires_at <= now:
            self._store.pop(storage_key, None)
            return None

        self._store.move_to_end(storage_key)
        return copy.deepcopy(value)

    def set(self, key: RetrievalCacheKey, value: Any) -> None:
        self._evict_expired()
        storage_key = key.as_storage_key()
        expires_at = self._time_fn() + self.ttl_seconds
        self._store[storage_key] = (expires_at, copy.deepcopy(value))
        self._store.move_to_end(storage_key)
        self._trim_overflow()

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        self._evict_expired()
        return len(self._store)

    def _evict_expired(self) -> None:
        if not self._store:
            return

        now = self._time_fn()
        expired_keys = [key for key, (expires_at, _) in self._store.items() if expires_at <= now]
        for key in expired_keys:
            self._store.pop(key, None)

    def _trim_overflow(self) -> None:
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)
