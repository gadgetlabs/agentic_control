"""
Speech Capture Agent
Records N seconds of audio after the wake word fires.
Returns a float32 torch tensor at 16 kHz for Whisper.

Audio comes from the shared MicrophoneManager (tools/microphone.py) which
switches the mic from wake-word detection mode into capture mode, ensuring
only one consumer ever reads from the stream at a time.
"""

import torch
from tools.microphone import MicrophoneManager


class SpeechCaptureAgent:
    def __init__(self, seconds: int = 3, mic: MicrophoneManager = None):
        self.seconds = seconds
        self._mic    = mic

    async def record(self) -> torch.Tensor:
        print(f"[capture] recording {self.seconds}s ...")
        return await self._mic.record(self.seconds)
