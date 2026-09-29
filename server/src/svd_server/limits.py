"""Rate limiting and the single-worker transcription queue."""

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager


class SlidingWindowLimiter:
    """At most `limit` events per key within the last `window_seconds`."""

    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._events: dict[str, deque[float]] = {}

    def allow(self, key: str) -> bool:
        """Record an event for `key` and return True, unless the limit is reached."""
        if self.blocked(key):
            return False
        self.record(key)
        return True

    def blocked(self, key: str) -> bool:
        return len(self._recent(key)) >= self._limit

    def record(self, key: str) -> None:
        self._events.setdefault(key, deque()).append(self._clock())

    def _recent(self, key: str) -> deque[float]:
        events = self._events.get(key)
        if events is None:
            return deque()
        threshold = self._clock() - self._window
        while events and events[0] <= threshold:
            events.popleft()
        if not events:
            del self._events[key]  # keeps memory bounded to active keys
        return events


class QueueFull(Exception):
    """Raised when the transcription queue cannot accept another request."""


class TranscriptionQueue:
    """One transcription at a time (CPU-bound), plus at most `queue_size` waiting."""

    def __init__(self, queue_size: int) -> None:
        self._capacity = 1 + queue_size
        self._inside = 0
        self._running = asyncio.Semaphore(1)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        if self._inside >= self._capacity:
            raise QueueFull
        self._inside += 1
        try:
            async with self._running:
                yield
        finally:
            self._inside -= 1
