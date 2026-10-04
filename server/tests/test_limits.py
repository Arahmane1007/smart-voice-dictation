import asyncio

import pytest
from svd_server.limits import QueueFull, SlidingWindowLimiter, TranscriptionQueue


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_allow_up_to_limit_then_refuse() -> None:
    limiter = SlidingWindowLimiter(limit=2, clock=FakeClock())
    assert limiter.allow("k")
    assert limiter.allow("k")
    assert not limiter.allow("k")


def test_keys_are_independent() -> None:
    limiter = SlidingWindowLimiter(limit=1, clock=FakeClock())
    assert limiter.allow("a")
    assert limiter.allow("b")
    assert not limiter.allow("a")


def test_window_slides() -> None:
    clock = FakeClock()
    limiter = SlidingWindowLimiter(limit=1, window_seconds=60, clock=clock)
    assert limiter.allow("k")
    clock.now += 59
    assert not limiter.allow("k")
    clock.now += 2
    assert limiter.allow("k")


def test_blocked_and_record() -> None:
    clock = FakeClock()
    limiter = SlidingWindowLimiter(limit=2, window_seconds=60, clock=clock)
    assert not limiter.blocked("ip")
    limiter.record("ip")
    assert not limiter.blocked("ip")
    limiter.record("ip")
    assert limiter.blocked("ip")
    clock.now += 61
    assert not limiter.blocked("ip")


async def test_queue_allows_one_running_plus_queue_size_waiting() -> None:
    queue = TranscriptionQueue(queue_size=2)
    release = asyncio.Event()
    entered = 0

    async def worker() -> None:
        nonlocal entered
        async with queue.slot():
            entered += 1
            await release.wait()

    tasks = [asyncio.create_task(worker()) for _ in range(3)]
    for _ in range(5):
        await asyncio.sleep(0)
    assert entered == 1  # only one transcription runs at a time

    with pytest.raises(QueueFull):
        async with queue.slot():
            pass

    release.set()
    await asyncio.gather(*tasks)
    assert entered == 3

    async with queue.slot():  # free again once everything is done
        pass


async def test_queue_size_zero_rejects_any_concurrent_request() -> None:
    queue = TranscriptionQueue(queue_size=0)
    release = asyncio.Event()

    async def worker() -> None:
        async with queue.slot():
            await release.wait()

    task = asyncio.create_task(worker())
    await asyncio.sleep(0)
    with pytest.raises(QueueFull):
        async with queue.slot():
            pass
    release.set()
    await task


async def test_queue_frees_slot_when_body_raises() -> None:
    queue = TranscriptionQueue(queue_size=0)
    with pytest.raises(RuntimeError):
        async with queue.slot():
            raise RuntimeError("boom")
    async with queue.slot():
        pass


def test_one_shot_keys_are_purged_after_the_window() -> None:
    clock = FakeClock()
    limiter = SlidingWindowLimiter(limit=5, window_seconds=60, clock=clock)
    for i in range(100):
        limiter.record(f"ip-{i}")
    assert len(limiter) == 100
    clock.now += 61
    limiter.record("another")  # any activity after a full window sweeps expired keys
    assert len(limiter) == 1


def test_sweep_runs_at_most_once_per_window() -> None:
    clock = FakeClock()  # t=1000, the limiter's first sweep is due at t=1060
    limiter = SlidingWindowLimiter(limit=5, window_seconds=60, clock=clock)
    limiter.record("a")  # t=1000
    clock.now += 30
    limiter.record("b")  # t=1030
    clock.now += 31
    limiter.record("c")  # t=1061: sweep, "a" is gone
    assert limiter.tracked_keys == {"b", "c"}
    clock.now += 39
    limiter.record("d")  # t=1100: "b" expired, but the last sweep was 39 s ago
    assert limiter.tracked_keys == {"b", "c", "d"}
    clock.now += 21
    limiter.record("e")  # t=1121: next sweep, "b" and "c" are gone
    assert limiter.tracked_keys == {"d", "e"}


def test_key_cap_evicts_the_least_recently_active_key() -> None:
    clock = FakeClock()
    limiter = SlidingWindowLimiter(limit=5, window_seconds=60, clock=clock, max_keys=3)
    limiter.record("old")
    clock.now += 1
    limiter.record("mid")
    clock.now += 1
    limiter.record("new")
    clock.now += 1
    limiter.record("old")  # "old" is active again, so "mid" is now the stalest key
    clock.now += 1
    limiter.record("extra")
    assert len(limiter) == 3
    assert limiter.tracked_keys == {"old", "new", "extra"}


def test_known_key_is_recorded_even_when_full() -> None:
    limiter = SlidingWindowLimiter(limit=2, window_seconds=60, clock=FakeClock(), max_keys=1)
    limiter.record("a")
    limiter.record("a")
    assert limiter.blocked("a")
