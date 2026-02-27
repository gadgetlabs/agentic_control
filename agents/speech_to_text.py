"""
Speech-to-Text Agent
Transcribes a raw audio tensor to a string using faster-whisper.
Runs in a thread pool executor because Whisper is CPU-bound.
"""

import asyncio

import torch
from faster_whisper import WhisperModel


class SpeechToTextAgent:
    def __init__(self, model_size: str = "base"):
        # int8 quantisation keeps memory low on edge hardware
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print(f"[stt] loaded Whisper '{model_size}'")

    def _transcribe(self, audio: torch.Tensor) -> str:
        segments, _ = self._model.transcribe(audio.numpy(), beam_size=1)
        return " ".join(s.text for s in segments).strip()

    async def transcribe(self, audio: torch.Tensor) -> str:
        return await asyncio.to_thread(self._transcribe, audio)
