"""
tools/microphone.py  -  Shared microphone resource manager.

Opens ONE sounddevice InputStream and feeds audio to the correct consumer
via a state machine:

  WAKE_WORD  (default)
    Callback pushes float32 torch chunks onto a small queue.
    WakeWordAgent drains it via next_chunk().

  CAPTURE
    Callback accumulates chunks until `seconds` of audio is collected,
    then fires a threading.Event.  SpeechCaptureAgent awaits that via record().

sounddevice is used (not PyAudio) so device indices match those reported by
setup_audio.py, which also uses sounddevice.

If the hardware sample rate differs from the 16 kHz the models expect
(set MIC_SAMPLE_RATE in .env via setup_audio.py), each chunk is resampled
before being placed on the queue.
"""

import os
import queue
import sys
import threading

import numpy as np
import torch
import sounddevice as sd

# SimpleWakeWords defines CHUNK_SAMPLES (16000) and SAMPLE_RATE (16000 Hz)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simple-wake-word'))
from SimpleWakeWords import CHUNK_SAMPLES, SAMPLE_RATE  # model wants 16 kHz

_MIC_ENV = os.getenv("MIC_DEVICE_INDEX", "").strip()
DEFAULT_DEVICE = int(_MIC_ENV) if _MIC_ENV else None

_MIC_RATE_ENV = os.getenv("MIC_SAMPLE_RATE", "").strip()
MIC_HW_RATE = int(_MIC_RATE_ENV) if _MIC_RATE_ENV else SAMPLE_RATE


class MicrophoneManager:
    def __init__(self, device_index: int | None = DEFAULT_DEVICE):
        self._model_rate = SAMPLE_RATE           # 16000 – what the models need
        self._hw_rate    = MIC_HW_RATE           # what the hardware runs at

        # One hardware chunk = one model chunk duration
        # e.g. hw=48000, model=16000 → hw_blocksize=48000 (1 s)
        self._model_chunk = CHUNK_SAMPLES
        ratio = self._hw_rate / self._model_rate
        self._hw_blocksize = round(self._model_chunk * ratio)

        dev_info = sd.query_devices(
            device_index if device_index is not None else sd.default.device[0]
        )
        print(f"[mic] opening [{dev_info['index']}] '{dev_info['name']}' "
              f"@ {self._hw_rate} Hz  blocksize={self._hw_blocksize}  "
              f"(model rate={self._model_rate} Hz)")

        self._mode       = "WAKE_WORD"
        self._wake_queue : queue.Queue[torch.Tensor] = queue.Queue(maxsize=4)
        self._cap_chunks : list[torch.Tensor] = []
        self._cap_need   = 0
        self._cap_done   = threading.Event()
        self._chunk_count = 0

        self._stream = sd.InputStream(
            device=device_index,
            samplerate=self._hw_rate,
            channels=1,
            dtype="float32",
            blocksize=self._hw_blocksize,
            callback=self._callback,
        )
        self._stream.start()
        print("[mic] sounddevice stream started")

    # ── Internal callback (runs in sounddevice's C thread) ────────────────────

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            print(f"[mic] {status}")

        raw = indata[:, 0].copy()  # (hw_blocksize,) float32 in [-1, 1]

        # Resample to model rate if needed
        if self._hw_rate != self._model_rate:
            from math import gcd
            from scipy.signal import resample_poly
            g   = gcd(self._model_rate, self._hw_rate)
            raw = resample_poly(raw, self._model_rate // g, self._hw_rate // g).astype(np.float32)

        chunk = torch.from_numpy(raw)  # already float32 normalised

        self._chunk_count += 1
        if self._chunk_count % 10 == 0:
            peak = float(np.abs(raw).max())
            print(f"[mic] {self._chunk_count} chunks, mode={self._mode}, peak={peak:.4f}")

        if self._mode == "CAPTURE":
            self._cap_chunks.append(chunk)
            if len(self._cap_chunks) >= self._cap_need:
                self._mode = "WAKE_WORD"
                self._cap_done.set()
        else:
            # WAKE_WORD – drop oldest if full so the detector never blocks
            if self._wake_queue.full():
                try:
                    self._wake_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._wake_queue.put_nowait(chunk)
            except queue.Full:
                pass

    # ── Public API ────────────────────────────────────────────────────────────

    def next_chunk(self) -> torch.Tensor:
        """Blocking: returns the next audio chunk for wake word detection."""
        return self._wake_queue.get()

    async def record(self, seconds: float) -> torch.Tensor:
        """Capture N seconds for Whisper, switch to CAPTURE mode, return tensor."""
        import asyncio

        chunks_needed = max(1, round(seconds * self._model_rate / self._model_chunk))
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
        print(f"[mic] capture complete – {audio.shape[0]} samples  "
              f"peak={audio.abs().max():.4f}")
        return audio
