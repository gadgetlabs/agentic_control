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
import threading

# simple-wake-word is not a pip package - deploy.sh clones it next to this project.
# This inserts its directory so Python can find the SimpleWakeWords module.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simple-wake-word'))

import torch
import torch.nn.functional as F
from SimpleWakeWords import (
    enroll_wake_word,
    _audio_to_embedding,
    _open_mic_stream,
    _record_chunk,
    SIMILARITY_THRESHOLD,
)


class WakeWordAgent:
    def __init__(self, wake_word: str, embedding_file: str = "wake_word_embedding.pt"):
        if os.path.exists(embedding_file):
            self._target = torch.load(embedding_file, weights_only=True)
            print(f"[wake_word] loaded embedding from {embedding_file}")
        else:
            print(f"[wake_word] enrolling '{wake_word}' ...")
            self._target = enroll_wake_word(wake_word)

        self._queue: queue.Queue = queue.Queue()

        threading.Thread(target=self._listen_loop, daemon=True).start()
        print(f"[wake_word] listening for '{wake_word}' ...")

    def _listen_loop(self):
        _, stream = _open_mic_stream()
        while True:
            chunk      = _record_chunk(stream)
            embedding  = _audio_to_embedding(chunk)
            similarity = F.cosine_similarity(
                embedding.unsqueeze(0),
                self._target.unsqueeze(0),
            ).item()

            if similarity > SIMILARITY_THRESHOLD:
                print(f"[wake_word] detected! (sim={similarity:.2f})")
                self._queue.put(True)

    async def wait(self):
        """Suspend the pipeline until the wake word fires."""
        await asyncio.to_thread(self._queue.get)
