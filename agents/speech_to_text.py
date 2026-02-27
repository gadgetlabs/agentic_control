"""
Speech-to-Text Agent
Transcribes a raw audio tensor to a string using openai-whisper.
Runs in a thread pool executor because Whisper is CPU-bound.

Uses openai-whisper (pure PyTorch) rather than faster-whisper because
faster-whisper's ctranslate2 dependency lacks reliable ARM64/Jetson wheels.
"""

import asyncio

import torch
import whisper


class SpeechToTextAgent:
    def __init__(self, model_size: str = "base"):
        self._model = whisper.load_model(model_size)
        print(f"[stt] loaded Whisper '{model_size}'")

    def _transcribe(self, audio: torch.Tensor) -> str:
        result = self._model.transcribe(audio.numpy(), fp16=False)
        return result["text"].strip()

    async def transcribe(self, audio: torch.Tensor) -> str:
        return await asyncio.to_thread(self._transcribe, audio)
