from __future__ import annotations

import asyncio
from typing import Awaitable, Iterable, List, Sequence, TypeVar

T = TypeVar("T")


async def bounded_gather(
    awaitables: Sequence[Awaitable[T]] | Iterable[Awaitable[T]],
    limit: int = 5,
    timeout_seconds: float | None = None,
    return_exceptions: bool = False,
) -> List[T]:
    semaphore = asyncio.Semaphore(max(1, int(limit)))
    items = list(awaitables)

    async def _runner(awaitable: Awaitable[T]) -> T:
        async with semaphore:
            if timeout_seconds is None:
                return await awaitable
            return await asyncio.wait_for(awaitable, timeout=timeout_seconds)

    tasks = [asyncio.create_task(_runner(item)) for item in items]
    try:
        results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        return list(results)
    finally:
        for task in tasks:
            if task.done():
                continue
            task.cancel()
