"""
Wake Word Agent
Runs a background thread that listens continuously on the microphone.
When the wake word is detected (cosine similarity > threshold), it signals
the async pipeline by putting True onto a queue.

Uses the internal helpers from SimpleWakeWords so we control the loop
ourselves - the public listen_for_wake_word() blocks forever with no callback.
"""

import asyncio
import os
import queue
import sys
import traceback
import threading

# simple-wake-word is not a pip package - deploy.sh clones it next to this project.
# This inserts its directory so Python can find the SimpleWakeWords module.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simple-wake-word'))

import pyaudio
import torch
import torch.nn.functional as F
from SimpleWakeWords import (
    enroll_wake_word,
    _audio_to_embedding,
    _open_mic_stream,
    _record_chunk,
    SIMILARITY_THRESHOLD,
)

MIC_DEVICE_INDEX = int(os.getenv("MIC_DEVICE_INDEX", "-1"))  # -1 = PyAudio default


def _list_audio_devices():
    """Print all available input devices so the user can pick the right one."""
    pa = pyaudio.PyAudio()
    print("[wake_word] ── available audio input devices ──")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"[wake_word]   [{i}] {info['name']}  "
                  f"(ch={info['maxInputChannels']}, "
                  f"rate={int(info['defaultSampleRate'])})")
    default = pa.get_default_input_device_info()
    print(f"[wake_word] default input: [{default['index']}] {default['name']}")
    print("[wake_word] ──────────────────────────────────")
    print("[wake_word] set MIC_DEVICE_INDEX=<n> in .env to override")
    pa.terminate()


class WakeWordAgent:
    def __init__(self, wake_word: str, embedding_file: str = "wake_word_embedding.pt"):
        _list_audio_devices()

        if os.path.exists(embedding_file):
            self._target = torch.load(embedding_file, weights_only=True)
            print(f"[wake_word] loaded embedding from {embedding_file}")
        else:
            print(f"[wake_word] enrolling '{wake_word}' ...")
            self._target = enroll_wake_word(wake_word)

        self._queue: queue.Queue = queue.Queue()

        t = threading.Thread(target=self._listen_loop, daemon=True)
        t.start()
        print(f"[wake_word] listener thread started (id={t.ident})")
        print(f"[wake_word] listening for '{wake_word}' (threshold={SIMILARITY_THRESHOLD:.2f}) ...")

    def _listen_loop(self):
        print("[wake_word] _listen_loop: thread alive, opening mic stream ...")
        try:
            device_index = MIC_DEVICE_INDEX if MIC_DEVICE_INDEX >= 0 else None
            _, stream = _open_mic_stream() if device_index is None else _open_mic_stream(device_index)
            print(f"[wake_word] mic stream open (device_index={device_index})")
        except Exception:
            print("[wake_word] FATAL: could not open mic stream:")
            traceback.print_exc()
            return

        chunk_n = 0
        while True:
            try:
                chunk = _record_chunk(stream)
                chunk_n += 1
                print(f"[wake_word] chunk #{chunk_n} recorded (len={len(chunk)})")
            except Exception:
                print(f"[wake_word] ERROR reading chunk #{chunk_n + 1}:")
                traceback.print_exc()
                break

            try:
                embedding  = _audio_to_embedding(chunk)
                similarity = F.cosine_similarity(
                    embedding.unsqueeze(0),
                    self._target.unsqueeze(0),
                ).item()
                print(f"[wake_word] chunk #{chunk_n}  sim={similarity:.3f}  threshold={SIMILARITY_THRESHOLD:.2f}")
            except Exception:
                print(f"[wake_word] ERROR computing similarity for chunk #{chunk_n}:")
                traceback.print_exc()
                continue

            if similarity > SIMILARITY_THRESHOLD:
                print(f"[wake_word] *** WAKE WORD DETECTED (sim={similarity:.3f}) ***")
                self._queue.put(True)

        print("[wake_word] _listen_loop exited — wake word detection stopped!")

    async def wait(self):
        """Suspend the pipeline until the wake word fires."""
        await asyncio.to_thread(self._queue.get)
