"""
main.py  -  CHAOS Robot Brain
─────────────────────────────────────────────────────────────────────────────
Pipeline (runs in a loop forever):

  [idle]
    │
    ▼  WakeWordAgent      background thread, wakes this loop on detection
    │
    ▼  SpeechCaptureAgent records N seconds of audio after wake word
    │
    ▼  SpeechToTextAgent  Whisper: audio tensor → text string
    │
    ▼  IntentAgent        Ollama: classify as "dialogue" or "action"
    │
    ├──► DialogueAgent    Ollama: generate spoken reply (no tools needed)
    │
    └──► PlanningAgent    Strands Agent: interprets command, calls @tool
              │              functions (drive_for / stop / set_emotion)
              │              directly - tool calls ARE the motor control
    │
    ▼  TextToSpeechAgent  speak the response aloud
    │
    ▼  [back to idle]

Ollama handles lightweight classify/converse tasks locally.
Strands + Anthropic handles the action loop that needs reliable tool calling.
─────────────────────────────────────────────────────────────────────────────
"""

import asyncio
import os

from dotenv import load_dotenv

import serial_reader
from agents.wake_word      import WakeWordAgent
from agents.speech_capture import SpeechCaptureAgent
from agents.speech_to_text import SpeechToTextAgent
from agents.intent         import IntentAgent
from agents.dialogue       import DialogueAgent
from agents.planning       import PlanningAgent
from agents.text_to_speech import TextToSpeechAgent

load_dotenv()

SERIAL_PORT  = os.getenv("SERIAL_PORT",  "/dev/ttyACM0")
WAKE_WORD    = os.getenv("WAKE_WORD",    "hey chaos")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


async def pipeline(wake, capture, stt, intent, dialogue, planning, tts):
    while True:
        # ── Idle ──────────────────────────────────────────────────────────
        print("\n[idle] waiting for wake word ...")
        await wake.wait()

        # ── Hear ──────────────────────────────────────────────────────────
        audio = await capture.record()
        text  = await stt.transcribe(audio)
        print(f"[stt]    {text!r}")

        if not text:
            continue

        # ── Think ─────────────────────────────────────────────────────────
        kind = await intent.classify(text)
        print(f"[intent] {kind}")

        if kind.get("type") == "dialogue":
            response = await dialogue.respond(text)
        else:
            response = await planning.plan(text)

        # ── Speak ─────────────────────────────────────────────────────────
        await tts.speak(response)


async def main():
    wake     = WakeWordAgent(WAKE_WORD)
    capture  = SpeechCaptureAgent(seconds=3)
    stt      = SpeechToTextAgent()
    intent   = IntentAgent(model=OLLAMA_MODEL)
    dialogue = DialogueAgent(model=OLLAMA_MODEL)
    planning = PlanningAgent()
    tts      = TextToSpeechAgent()

    await asyncio.gather(
        serial_reader.run(SERIAL_PORT),
        pipeline(wake, capture, stt, intent, dialogue, planning, tts),
    )


if __name__ == "__main__":
    asyncio.run(main())
