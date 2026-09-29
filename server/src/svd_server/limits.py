"""Rate limiting and the single-worker transcription queue."""

import asyncio
import time
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager


class SlidingWindowLimiter:
    """At most `limit` events per key within the last `window_seconds`.

    Keys may be attacker-chosen (client IPs of failed authentications), so memory
    is bounded two ways: keys whose events have all expired are swept at most once
    per window, and at most `max_keys` keys are tracked (the least recently active
    one is evicted to make room for a new key).
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 100_000,
    ) -> None:
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._max_keys = max_keys
        # Ordered by most recent event, oldest first (the clock is monotonic).
        self._events: OrderedDict[str, deque[float]] = OrderedDict()
        self._last_sweep = clock()

    def __len__(self) -> int:
        return len(self._events)

    @property
    def tracked_keys(self) -> set[str]:
        return set(self._events)

    def allow(self, key: str) -> bool:
        """Record an event for `key` and return True, unless the limit is reached."""
        if self.blocked(key):
            return False
        self.record(key)
        return True

    def blocked(self, key: str) -> bool:
        self._maybe_sweep()
        return len(self._recent(key)) >= self._limit

    def record(self, key: str) -> None:
        self._maybe_sweep()
        events = self._events.get(key)
        if events is None:
            while len(self._events) >= self._max_keys:
                self._events.popitem(last=False)  # least recently active key
            events = self._events[key] = deque()
        else:
            self._events.move_to_end(key)
        events.append(self._clock())

    def _maybe_sweep(self) -> None:
        now = self._clock()
        if now - self._last_sweep < self._window:
            return
        self._last_sweep = now
        threshold = now - self._window
        while self._events:
            key, events = next(iter(self._events.items()))
            if events[-1] > threshold:
                break  # every later key had an event even more recently
            del self._events[key]

    def _recent(self, key: str) -> deque[float]:
        events = self._events.get(key)
        if events is None:
            return deque()
        threshold = self._clock() - self._window
        while events and events[0] <= threshold:
            events.popleft()
        if not events:
            del self._events[key]
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
