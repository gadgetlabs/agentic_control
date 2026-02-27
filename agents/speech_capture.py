"""
Speech Capture Agent
Opens a fresh microphone stream and records N seconds of audio
after the wake word fires. Returns a float32 torch tensor at 16 kHz.
"""

import asyncio

import numpy as np
import pyaudio
import torch

SAMPLE_RATE = 16_000
CHUNK       = 1_024


class SpeechCaptureAgent:
    def __init__(self, seconds: int = 3):
        self.seconds = seconds

    def _record(self) -> torch.Tensor:
        pa     = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16, channels=1,
            rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK,
        )
        n_chunks = int(SAMPLE_RATE / CHUNK * self.seconds)
        frames   = [np.frombuffer(stream.read(CHUNK), dtype=np.int16) for _ in range(n_chunks)]
        stream.stop_stream()
        stream.close()
        pa.terminate()

        audio = np.concatenate(frames).astype(np.float32) / 32_768.0
        return torch.from_numpy(audio)

    async def record(self) -> torch.Tensor:
        print(f"[capture] recording {self.seconds}s ...")
        return await asyncio.to_thread(self._record)
