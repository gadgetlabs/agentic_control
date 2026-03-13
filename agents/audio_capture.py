"""
Audio Capture Agent
===================
Single blocking loop, two-state FSM, one PyAudio stream.

  IDLE
    Read one chunk from the mic (1 s).
    Maintain a 2-chunk (2 s) sliding window — scores the last two chunks
    concatenated so multi-syllable wake words are never split at a boundary.
    Log every few chunks so similarity is visible.
    On detection → switch to LISTENING.

  LISTENING
    Read chunks and accumulate into a speech buffer.
    When the buffer is full → return the tensor and go back to IDLE.

Using PyAudio (not sounddevice) keeps the audio format identical to the
format used by SimpleWakeWords during enrolment/detection, eliminating
any conversion artefacts that were causing near-zero similarity.

Usage:
    audio = await audio_capture.listen()   # blocks through wake word + speech
    text  = await stt.transcribe(audio)
"""

import asyncio
import collections
import os
import struct
import sys
import time

import state_bus

import pyaudio
import torch
import torch.nn.functional as F

_SW_DIR = os.path.join(os.path.dirname(__file__), '..', 'simple-wake-word')
sys.path.insert(0, _SW_DIR)

from SimpleWakeWords import (
    CHUNK_SAMPLES,
    SAMPLE_RATE,
    _audio_to_embedding,
    enroll_wake_word,
    set_threshold,
)

_PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EMBEDDING_FILE = os.path.join(_PROJECT_ROOT, "wake_word_embedding.pt")


class AudioCaptureAgent:
    """Wake word detection + speech capture in a single read loop."""

    RATE  = SAMPLE_RATE    # 16 000 Hz
    CHUNK = CHUNK_SAMPLES  # samples per read (matches SimpleWakeWords exactly)

    def __init__(self, wake_word: str, speech_seconds: int = 4):
        _v = os.getenv("MIC_DEVICE_INDEX", "").strip()
        device_index = int(_v) if _v else None

        self._pa = pyaudio.PyAudio()
        dev = (self._pa.get_device_info_by_index(device_index)
               if device_index is not None
               else self._pa.get_default_input_device_info())

        print(f"[audio] mic [{dev['index']}] '{dev['name']}' "
              f"@ {self.RATE} Hz  chunk={self.CHUNK}")

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.CHUNK,
        )

        if os.path.exists(_EMBEDDING_FILE):
            self._target = torch.load(
                _EMBEDDING_FILE, weights_only=True, map_location="cpu"
            )
            print(f"[audio] loaded wake word embedding from {_EMBEDDING_FILE}")
        else:
            print(f"[audio] no embedding found – enrolling '{wake_word}' ...")
            self._target = enroll_wake_word(wake_word)

        self._speech_chunks = speech_seconds  # CHUNK = 1 s, so chunks == seconds

        # Threshold can be tuned via env without touching code.
        # Default 0.2 matches the SimpleWakeWords reference — the model's peak
        # similarity is ~0.2-0.22 when the wake word is spoken correctly.
        self._threshold = float(os.getenv("WAKE_WORD_THRESHOLD", "0.2"))
        set_threshold(self._threshold)
        print(f"[audio] threshold={self._threshold:.2f}  "
              f"speech_window={speech_seconds}s")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _read(self) -> torch.Tensor:
        """Read one chunk from PyAudio, return float32 tensor [-1, 1]."""
        raw     = self._stream.read(self.CHUNK, exception_on_overflow=False)
        samples = struct.unpack(f"{self.CHUNK}h", raw)
        return torch.tensor(samples, dtype=torch.float32) / 32768.0

    def _sim(self, chunk: torch.Tensor) -> float:
        t0  = time.perf_counter()
        emb = _audio_to_embedding(chunk)
        sim = F.cosine_similarity(
            emb.unsqueeze(0), self._target.unsqueeze(0)
        ).item()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > 200:   # only log when it's suspiciously slow
            print(f"[perf] embed       {elapsed_ms:.0f}ms  (>{200}ms threshold)")
        return sim

    # ── FSM loop (runs in a thread via asyncio.to_thread) ─────────────────────

    def _loop(self) -> torch.Tensor:
        state  = "IDLE"
        speech = []
        n      = 0
        # 2-chunk (2 s) sliding window so "didgeridoo" is never cut across
        # a block boundary.  PyAudio still reads in 1 s chunks; we just cat
        # the last two before scoring.
        win_buf: collections.deque = collections.deque(maxlen=2)

        while True:
            chunk = self._read()
            n    += 1

            if state == "IDLE":
                win_buf.append(chunk)
                window = torch.cat(list(win_buf))   # 1 s until buf fills, then 2 s
                sim = self._sim(window)
                state_bus.publish_audio_chunk(
                    sim=sim, state="IDLE", peak=float(chunk.abs().max())
                )
                if n % 3 == 0:                          # log every ~3 s
                    print(f"[audio] IDLE  sim={sim:.3f}  threshold={self._threshold:.2f}")

                if sim > self._threshold:
                    print(f"[audio] *** WAKE WORD (sim={sim:.3f}) ***")
                    state  = "LISTENING"
                    speech = []

            elif state == "LISTENING":
                speech.append(chunk)
                state_bus.publish_audio_chunk(
                    sim=1.0, state="LISTENING", peak=float(chunk.abs().max())
                )
                remaining = self._speech_chunks - len(speech)
                print(f"[audio] LISTENING {len(speech)}/{self._speech_chunks}  "
                      f"({remaining}s left)")

                if len(speech) >= self._speech_chunks:
                    result = torch.cat(speech)
                    print(f"[audio] captured {result.shape[0]} samples  "
                          f"peak={result.abs().max():.4f}")
                    state_bus.publish_audio_chunk(
                        sim=1.0, state="PROCESSING", peak=float(result.abs().max())
                    )
                    return result

    # ── Public API ────────────────────────────────────────────────────────────

    async def listen(self) -> torch.Tensor:
        """Block until wake word fires, then return the speech audio tensor."""
        return await asyncio.to_thread(self._loop)
