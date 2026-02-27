"""
Text-to-Speech Agent
Speaks a string aloud using pyttsx3 (offline, works on Linux/macOS/Windows).

pyttsx3 is not thread-safe so we initialise it inside the worker function
each time - this is slightly wasteful but simple and reliable.

For a better voice on the robot, swap pyttsx3 for piper-tts:
  pip install piper-tts
  piper --model en_US-lessac-medium --output_raw "hello" | aplay -r22050 -f S16_LE -c1
"""

import asyncio

import pyttsx3


class TextToSpeechAgent:
    def __init__(self, rate: int = 145):
        self.rate = rate

    def _speak(self, text: str):
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    async def speak(self, text: str):
        print(f"[tts] {text!r}")
        await asyncio.to_thread(self._speak, text)
