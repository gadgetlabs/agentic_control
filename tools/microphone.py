"""
tools/microphone.py  -  Shared microphone resource manager.

Opens the PyAudio input stream exactly once and exposes two interfaces:

  mic.next_chunk()      → torch.Tensor   blocking; used by WakeWordAgent
  await mic.record(s)   → torch.Tensor   async; used by SpeechCaptureAgent

State machine
─────────────
  WAKE_WORD  (default)
    Reader thread pushes chunks onto a small queue.
    next_chunk() pops from that queue.

  CAPTURE
    reader thread accumulates chunks until `seconds` of audio is collected,
    then fires a threading.Event and switches back to WAKE_WORD.
    record() awaits that event and returns the concatenated tensor.

Because only one state is active at a time the mic stream is never shared
between two consumers simultaneously.
"""

import os
import queue
import struct
import sys
import threading

import torch

# ── SimpleWakeWords lives as a cloned repo, not a pip package ────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simple-wake-word'))
from SimpleWakeWords import CHUNK_SAMPLES, SAMPLE_RATE  # 16000 Hz, ~16000 samples/chunk

import pyaudio

_MIC_ENV = os.getenv("MIC_DEVICE_INDEX", "").strip()
DEFAULT_DEVICE = int(_MIC_ENV) if _MIC_ENV else None


class MicrophoneManager:
    def __init__(self, device_index: int | None = DEFAULT_DEVICE):
        self._chunk = CHUNK_SAMPLES
        self._rate  = SAMPLE_RATE

        self._pa = pyaudio.PyAudio()
        dev_info = (
            self._pa.get_device_info_by_index(device_index)
            if device_index is not None
            else self._pa.get_default_input_device_info()
        )
        print(f"[mic] opening [{dev_info['index']}] {dev_info['name']} "
              f"@ {self._rate} Hz  chunk={self._chunk}")

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self._chunk,
        )

        self._mode       = "WAKE_WORD"
        self._wake_queue : queue.Queue[torch.Tensor] = queue.Queue(maxsize=4)
        self._cap_chunks : list[torch.Tensor] = []
        self._cap_need   = 0
        self._cap_done   = threading.Event()

        t = threading.Thread(target=self._reader, daemon=True)
        t.start()
        print(f"[mic] reader thread started (id={t.ident})")

    def _reader(self):
        print("[mic] reader loop alive")
        n = 0
        while True:
            try:
                raw = self._stream.read(self._chunk, exception_on_overflow=False)
            except Exception as e:
                print(f"[mic] stream error: {e}")
                continue

            # Convert exactly as SimpleWakeWords._record_chunk does
            samples = struct.unpack(f"{self._chunk}h", raw)
            chunk   = torch.tensor(samples, dtype=torch.float32) / 32768.0

            n += 1
            if n % 10 == 0:
                print(f"[mic] {n} chunks read, mode={self._mode}")

            if self._mode == "CAPTURE":
                self._cap_chunks.append(chunk)
                if len(self._cap_chunks) >= self._cap_need:
                    self._mode = "WAKE_WORD"
                    self._cap_done.set()
            else:
                # WAKE_WORD – drop oldest if queue is full rather than blocking
                if self._wake_queue.full():
                    try:
                        self._wake_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._wake_queue.put(chunk)

    # ── Public API ────────────────────────────────────────────────────────────

    def next_chunk(self) -> torch.Tensor:
        """Blocking: returns the next audio chunk for wake word detection."""
        return self._wake_queue.get()

    async def record(self, seconds: float) -> torch.Tensor:
        """
        Switch to CAPTURE mode, accumulate `seconds` of audio, return tensor.
        Call from SpeechCaptureAgent after the wake word fires.
        """
        import asyncio

        chunks_needed = max(1, round(seconds * self._rate / self._chunk))
        self._cap_chunks = []
        self._cap_need   = chunks_needed
        self._cap_done.clear()

        # Drain stale wake-word chunks so capture starts from now
        while not self._wake_queue.empty():
            try:
                self._wake_queue.get_nowait()
            except queue.Empty:
                break

        self._mode = "CAPTURE"
        print(f"[mic] capture mode – collecting {chunks_needed} chunks ({seconds}s)")
        await asyncio.to_thread(self._cap_done.wait)

        audio = torch.cat(self._cap_chunks)
        print(f"[mic] capture complete – {audio.shape[0]} samples")
        return audio
