"""
Text-to-Speech Agent
Speaks text aloud using Piper, a fast neural TTS engine built for edge devices.

Synthesis tries three approaches in order:
  1. synthesize_stream_raw()  – piper >= 1.2
  2. synthesize() + wave      – older piper (pre-configured wave writer)
  3. piper CLI subprocess     – fallback if both Python APIs fail

Voice: en_GB-jenny_dioco-medium  (British English, female, ~60 MB)
Download via deploy.sh or manually:
  mkdir -p voices
  curl -L -o voices/en_GB-jenny_dioco-medium.onnx \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx
  curl -L -o voices/en_GB-jenny_dioco-medium.onnx.json \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx.json
"""

import asyncio
import io
import os
import subprocess
import wave
from math import gcd

import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice

_PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_VOICE = os.path.join(_PROJECT_ROOT, "voices", "en_GB-jenny_dioco-medium.onnx")
# VOICE_MODEL is safe to read at module level – it doesn't change at runtime
VOICE_MODEL = os.getenv("PIPER_VOICE", _DEFAULT_VOICE)


class TextToSpeechAgent:
    def __init__(self):
        # Read device/rate at instantiation time so load_dotenv() has already run
        _spk      = os.getenv("SPEAKER_DEVICE_INDEX", "").strip()
        _spk_rate = os.getenv("SPEAKER_SAMPLE_RATE",  "").strip()
        self._device   = int(_spk)      if _spk      else None
        self._spk_rate = int(_spk_rate) if _spk_rate else None

        self._voice = PiperVoice.load(VOICE_MODEL)
        print(f"[tts] loaded piper voice from {VOICE_MODEL}")
        print(f"[tts] speaker device={self._device}  sample_rate={self._spk_rate}")

    # ── Synthesis helpers ─────────────────────────────────────────────────────

    def _synthesize_raw(self, text: str) -> bytes:
        """Return raw int16 PCM bytes, trying multiple piper API approaches."""

        # 1. Modern piper: synthesize_stream_raw() generator
        if hasattr(self._voice, "synthesize_stream_raw"):
            raw = b"".join(self._voice.synthesize_stream_raw(text))
            if raw:
                return raw

        # 2. Older piper: synthesize() writes to a pre-configured wave.Wave_write
        src_rate = self._voice.config.sample_rate
        buf = io.BytesIO()
        wav = wave.open(buf, "wb")
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(src_rate)
        try:
            self._voice.synthesize(text, wav)
        except Exception as e:
            print(f"[tts] synthesize() raised: {e}")
        wav.close()
        buf.seek(44)          # skip 44-byte PCM WAV header
        raw = buf.read()
        if raw:
            return raw

        # 3. Subprocess fallback: call the piper CLI directly
        print("[tts] Python API produced no audio – trying piper subprocess ...")
        result = subprocess.run(
            ["piper", "--model", VOICE_MODEL, "--output_raw", "--quiet"],
            input=text.encode("utf-8"),
            capture_output=True,
        )
        if result.stdout:
            return result.stdout

        print(f"[tts] piper subprocess also failed  "
              f"rc={result.returncode}  stderr={result.stderr.decode()[:200]}")
        return b""

    # ── Playback ──────────────────────────────────────────────────────────────

    def _speak(self, text: str):
        from scipy.signal import resample_poly

        raw = self._synthesize_raw(text)
        if not raw:
            print("[tts] no audio produced – skipping playback")
            return

        audio_f32 = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        src_rate  = self._voice.config.sample_rate
        dst_rate  = self._spk_rate or src_rate

        print(f"[tts] {len(audio_f32)} samples  {src_rate}→{dst_rate} Hz  "
              f"device={self._device}  peak={audio_f32.max():.4f}")

        if src_rate != dst_rate:
            g         = gcd(src_rate, dst_rate)
            audio_f32 = resample_poly(
                audio_f32, dst_rate // g, src_rate // g
            ).astype(np.float32)

        sd.play(audio_f32, samplerate=dst_rate, device=self._device)
        sd.wait()

    async def speak(self, text: str):
        print(f"[tts] {text!r}")
        await asyncio.to_thread(self._speak, text)
