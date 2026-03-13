"""
state_bus.py — thread-safe pub/sub bridge between robot pipeline and dashboard.

Hot path  (~1 Hz per audio chunk):
    Audio thread calls publish_audio_chunk() → SimpleQueue → async drainer
    → WebSocket subscribers.

Cold path (polled by browser):
    pipeline() coroutine writes to pipeline_state dict directly (async-only).
    set_current_emotion() is thread-safe via lock (may be called from Strands
    executor thread).
"""

import asyncio
import collections
import queue
import threading
import time
from typing import Callable, Set

# ── Hot path: audio thread → async WebSocket ─────────────────────────────────
# stdlib SimpleQueue is genuinely thread-safe; asyncio.Queue is NOT safe to
# put() from a thread without loop.call_soon_threadsafe().
_audio_q: queue.SimpleQueue = queue.SimpleQueue()
_audio_subscribers: Set[Callable] = set()


def publish_audio_chunk(sim: float, state: str, peak: float) -> None:
    """Call from audio THREAD. Non-blocking; drops oldest if queue is full."""
    try:
        if _audio_q.qsize() < 20:
            _audio_q.put_nowait({"sim": sim, "state": state, "peak": peak})
    except Exception:
        pass


def subscribe_audio(cb: Callable) -> None:
    _audio_subscribers.add(cb)


def unsubscribe_audio(cb: Callable) -> None:
    _audio_subscribers.discard(cb)


async def _audio_drainer() -> None:
    """Async task: drain queue, append to history, fan out to WS clients."""
    while True:
        await asyncio.sleep(0.05)
        while True:
            try:
                payload = _audio_q.get_nowait()
            except queue.Empty:
                break
            sim_history.append(payload)
            if _audio_subscribers:
                await asyncio.gather(
                    *[cb(payload) for cb in list(_audio_subscribers)],
                    return_exceptions=True,
                )


# ── Similarity history ring buffer (written by drainer, read by HTTP) ────────
sim_history: collections.deque = collections.deque(maxlen=300)

# ── Pipeline state (async-only writes from pipeline coroutine) ────────────────
pipeline_state: dict = {
    "phase": "idle",
    "heard": "",
    "intent": "",
    "response": "",
    "ts": 0.0,
}

# ── Manual wake trigger (set from async HTTP handler, read from audio thread) ──
# threading.Event is safe to set() from any thread or coroutine.
manual_trigger: threading.Event = threading.Event()

# ── Emotion (thread-safe: may be set from Strands executor thread) ────────────
_emotion_lock = threading.Lock()
_current_emotion: str = "idle"


def set_current_emotion(name: str) -> None:
    with _emotion_lock:
        global _current_emotion
        _current_emotion = name


def get_current_emotion() -> str:
    with _emotion_lock:
        return _current_emotion
