"""
Text-to-Speech Agent
Speaks text aloud using Piper via subprocess with --output_raw.

Running piper as a subprocess is the most version-agnostic approach and
avoids any Python/ONNX API differences across piper-tts versions.

Voice: en_GB-jenny_dioco-medium  (British English, female, ~60 MB)
Download via deploy.sh or manually:
  mkdir -p voices
  curl -L -o voices/en_GB-jenny_dioco-medium.onnx \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx
  curl -L -o voices/en_GB-jenny_dioco-medium.onnx.json \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx.json
"""

import asyncio
import json
import os
import subprocess
from math import gcd

import numpy as np
import sounddevice as sd

_PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_VOICE = os.path.join(_PROJECT_ROOT, "voices", "en_GB-jenny_dioco-medium.onnx")


class TextToSpeechAgent:
    def __init__(self):
        # Read at instantiation time so load_dotenv() has already run
        _v        = os.getenv("PIPER_VOICE",          _DEFAULT_VOICE)
        _spk      = os.getenv("SPEAKER_DEVICE_INDEX", "").strip()
        _spk_rate = os.getenv("SPEAKER_SAMPLE_RATE",  "").strip()

        self._model  = os.path.abspath(_v)          # absolute path for subprocess
        self._device = int(_spk)      if _spk      else None
        self._spk_rate = int(_spk_rate) if _spk_rate else None

        # Read sample rate from the .onnx.json sidecar – no ONNX load needed
        config_path = self._model + ".json"
        with open(config_path) as f:
            self._src_rate = json.load(f)["audio"]["sample_rate"]

        print(f"[tts] voice={self._model}  src_rate={self._src_rate} Hz")
        print(f"[tts] speaker device={self._device}  playback_rate={self._spk_rate}")

    def _speak(self, text: str):
        from scipy.signal import resample_poly

        result = subprocess.run(
            ["python3", "-m", "piper",
             "--model",      self._model,
             "--output_raw"],
            input=text.encode("utf-8"),
            capture_output=True,
        )

        if not result.stdout:
            # Filter out known harmless ONNX/GPU noise so real errors are visible
            lines = result.stderr.decode(errors="replace").splitlines()
            real  = [l for l in lines if "onnxruntime" not in l and "GPU" not in l]
            print(f"[tts] piper produced no audio  rc={result.returncode}")
            if real:
                print("[tts] piper stderr:\n" + "\n".join(real))
            return

        audio_f32 = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        dst_rate   = self._spk_rate or self._src_rate

        print(f"[tts] {len(audio_f32)} samples  {self._src_rate}→{dst_rate} Hz  "
              f"device={self._device}  peak={audio_f32.max():.4f}")

        if self._src_rate != dst_rate:
            g         = gcd(self._src_rate, dst_rate)
            audio_f32 = resample_poly(
                audio_f32, dst_rate // g, self._src_rate // g
            ).astype(np.float32)

        sd.play(audio_f32, samplerate=dst_rate, device=self._device)
        sd.wait()

    async def speak(self, text: str):
        print(f"[tts] {text!r}")
        await asyncio.to_thread(self._speak, text)
