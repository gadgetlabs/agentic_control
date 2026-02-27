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
    ▼  IntentAgent        small LLM: classify as "dialogue" or "action"
    │
    ├──► DialogueAgent    small LLM: generate spoken reply
    │
    └──► PlanningAgent    small LLM: text → list of motor steps (JSON)
              │
              ▼
         MotorControlAgent  execute steps over CAN bus, await durations
    │
    ▼  TextToSpeechAgent  speak the response aloud
    │
    ▼  [back to idle]

Blocking operations (audio, Whisper, Ollama) run via asyncio.to_thread()
so the serial telemetry reader keeps flowing throughout.
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
from agents.motor_control  import MotorControlAgent
from agents.text_to_speech import TextToSpeechAgent

load_dotenv()

SERIAL_PORT  = os.getenv("SERIAL_PORT",  "/dev/ttyACM0")
WAKE_WORD    = os.getenv("WAKE_WORD",    "hey chaos")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


async def pipeline(wake, capture, stt, intent, dialogue, planning, motor, tts):
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
            plan     = await planning.plan(text)
            response = await motor.execute(plan)

        # ── Speak ─────────────────────────────────────────────────────────
        await tts.speak(response)


async def main():
    wake     = WakeWordAgent(WAKE_WORD)
    capture  = SpeechCaptureAgent(seconds=3)
    stt      = SpeechToTextAgent()
    intent   = IntentAgent(model=OLLAMA_MODEL)
    dialogue = DialogueAgent(model=OLLAMA_MODEL)
    planning = PlanningAgent(model=OLLAMA_MODEL)
    motor    = MotorControlAgent()
    tts      = TextToSpeechAgent()

    # Serial reader and the pipeline run concurrently.
    # Serial reader is an async task (I/O bound).
    # Pipeline suspends at await points, never blocks the loop.
    await asyncio.gather(
        serial_reader.run(SERIAL_PORT),
        pipeline(wake, capture, stt, intent, dialogue, planning, motor, tts),
    )


if __name__ == "__main__":
    asyncio.run(main())
