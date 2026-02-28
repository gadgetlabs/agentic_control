"""
Text-to-Speech Agent
Speaks text aloud using Piper, a fast neural TTS engine built for edge devices.

The voice model is loaded once at startup. Synthesis produces raw PCM which
sounddevice plays directly - no temp files, no subprocesses.

Voice: en_GB-jenny_dioco-medium  (British English, female, ~60 MB)
Download via deploy.sh or manually:
  mkdir -p voices
  curl -L -o voices/en_GB-jenny_dioco-medium.onnx \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx
  curl -L -o voices/en_GB-jenny_dioco-medium.onnx.json \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx.json
"""

import asyncio
import os

import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_VOICE = os.path.join(_PROJECT_ROOT, "voices", "en_GB-jenny_dioco-medium.onnx")
VOICE_MODEL = os.getenv("PIPER_VOICE", _DEFAULT_VOICE)

_spk = os.getenv("SPEAKER_DEVICE_INDEX", "").strip()
SPEAKER_DEVICE = int(_spk) if _spk else None

_spk_rate = os.getenv("SPEAKER_SAMPLE_RATE", "").strip()
SPEAKER_SAMPLE_RATE = int(_spk_rate) if _spk_rate else None


class TextToSpeechAgent:
    def __init__(self):
        self._voice = PiperVoice.load(VOICE_MODEL)
        print(f"[tts] loaded piper voice from {VOICE_MODEL}")

    def _speak(self, text: str):
        from math import gcd
        from scipy.signal import resample_poly

        raw       = b"".join(self._voice.synthesize_stream_raw(text))
        audio_f32 = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        src_rate = self._voice.config.sample_rate
        dst_rate = SPEAKER_SAMPLE_RATE or src_rate   # setup_audio.py sets this

        if src_rate != dst_rate:
            g         = gcd(src_rate, dst_rate)
            audio_f32 = resample_poly(audio_f32, dst_rate // g, src_rate // g).astype(np.float32)

        sd.play(audio_f32, samplerate=dst_rate, device=SPEAKER_DEVICE)
        sd.wait()

    async def speak(self, text: str):
        print(f"[tts] {text!r}")
        await asyncio.to_thread(self._speak, text)
