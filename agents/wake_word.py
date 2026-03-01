"""
Wake Word Agent
Runs a background thread that listens continuously on the microphone.
When the wake word is detected (cosine similarity > threshold), it signals
the async pipeline by putting True onto a queue.

Audio comes from the shared MicrophoneManager (tools/microphone.py) so the
mic stream is never opened twice.
"""

import asyncio
import os
import queue
import sys
import traceback
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simple-wake-word'))

import torch
import torch.nn.functional as F
from SimpleWakeWords import (
    enroll_wake_word,
    _audio_to_embedding,
    SIMILARITY_THRESHOLD,
)

from tools.microphone import MicrophoneManager


class WakeWordAgent:
    def __init__(self, wake_word: str, mic: MicrophoneManager,
                 embedding_file: str = "wake_word_embedding.pt"):
        self._mic = mic

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
        print("[wake_word] _listen_loop alive")
        chunk_n = 0
        while True:
            try:
                chunk = self._mic.next_chunk()
                chunk_n += 1
            except Exception:
                print("[wake_word] ERROR getting chunk:")
                traceback.print_exc()
                continue

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

    async def wait(self):
        """Suspend the pipeline until the wake word fires."""
        await asyncio.to_thread(self._queue.get)
