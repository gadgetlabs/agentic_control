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

class MicrophoneManager:
    def __init__(self, device_index: int | None = None):
        # Read at instantiation time, not import time, so load_dotenv() has run
        if device_index is None:
            _v = os.getenv("MIC_DEVICE_INDEX", "").strip()
            device_index = int(_v) if _v else None

        _hw = os.getenv("MIC_SAMPLE_RATE", "").strip()
        hw_rate = int(_hw) if _hw else SAMPLE_RATE

        self._model_rate  = SAMPLE_RATE   # 16000 – what the models need
        self._hw_rate     = hw_rate       # what the hardware runs at
        self._model_chunk = CHUNK_SAMPLES # samples per chunk the model expects

        # Use a small callback blocksize (4096 samples ≈ 0.25 s at 16kHz) to
        # avoid sounddevice input overflow.  The callback accumulates these
        # mini-blocks into a full CHUNK_SAMPLES-sized buffer before handing
        # it to the wake word / capture consumers.
        CB_FRAMES         = 4096
        ratio             = self._hw_rate / self._model_rate
        self._hw_blocksize = round(CB_FRAMES * ratio)
        self._hw_per_chunk = round(self._model_chunk / CB_FRAMES)  # callbacks per model chunk
        self._cb_buf: list[np.ndarray] = []   # accumulator for mini-blocks

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
        if status and "overflow" not in str(status).lower():
            print(f"[mic] {status}")

        mini = indata[:, 0].copy()  # small block, float32 [-1, 1]

        # Resample mini-block to model rate if needed
        if self._hw_rate != self._model_rate:
            from math import gcd
            from scipy.signal import resample_poly
            g    = gcd(self._model_rate, self._hw_rate)
            mini = resample_poly(mini, self._model_rate // g, self._hw_rate // g).astype(np.float32)

        self._cb_buf.append(mini)

        # Only dispatch a full model chunk once we've accumulated enough mini-blocks
        if len(self._cb_buf) < self._hw_per_chunk:
            return

        raw   = np.concatenate(self._cb_buf)[:self._model_chunk]
        self._cb_buf = []

        chunk = torch.from_numpy(raw)

        self._chunk_count += 1
        if self._chunk_count % 5 == 0:
            peak = float(np.abs(raw).max())
            print(f"[mic] chunk #{self._chunk_count}  mode={self._mode}  peak={peak:.4f}")

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
