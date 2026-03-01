"""
Text-to-Speech Agent
Speaks text aloud using Piper, a fast neural TTS engine built for edge devices.

The voice model is loaded once at startup. synthesize() writes a WAV into a
BytesIO buffer which is decoded and played via sounddevice – no temp files.

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
        print(f"[tts] speaker device={SPEAKER_DEVICE}  sample_rate={SPEAKER_SAMPLE_RATE}")

    def _speak(self, text: str):
        import io
        import wave
        from math import gcd
        from scipy.signal import resample_poly

        # piper's synthesize() only writes frames – it expects the wave writer
        # to already have channels/sampwidth/framerate set.
        src_rate = self._voice.config.sample_rate
        buf = io.BytesIO()
        wav = wave.open(buf, "wb")
        wav.setnchannels(1)
        wav.setsampwidth(2)          # 16-bit PCM
        wav.setframerate(src_rate)
        self._voice.synthesize(text, wav)
        wav.close()

        buf.seek(0)
        with wave.open(buf, "rb") as wav:
            raw = wav.readframes(wav.getnframes())

        audio_f32 = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        dst_rate  = SPEAKER_SAMPLE_RATE or src_rate   # set by setup_audio.py
        print(f"[tts] synthesised {len(audio_f32)} samples  src={src_rate}Hz → dst={dst_rate}Hz  "
              f"device={SPEAKER_DEVICE}  peak={np.abs(audio_f32).max():.4f}")

        if src_rate != dst_rate:
            g         = gcd(src_rate, dst_rate)
            audio_f32 = resample_poly(audio_f32, dst_rate // g, src_rate // g).astype(np.float32)

        sd.play(audio_f32, samplerate=dst_rate, device=SPEAKER_DEVICE)
        sd.wait()

    async def speak(self, text: str):
        print(f"[tts] {text!r}")
        await asyncio.to_thread(self._speak, text)
