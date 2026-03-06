"""
main.py  -  CHAOS Robot Brain
─────────────────────────────────────────────────────────────────────────────
Pipeline (runs in a loop forever):

  [idle]
    │
    ▼  AudioCaptureAgent  single PyAudio loop: IDLE→wake word→LISTENING→return
    │                     (wake word detection + speech capture in one agent)
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

# load_dotenv() MUST run before importing any module that calls os.getenv()
# at module level (tools/microphone.py, agents/text_to_speech.py, etc.)
from dotenv import load_dotenv
load_dotenv()

import serial_reader
from agents.audio_capture  import AudioCaptureAgent
from agents.speech_to_text import SpeechToTextAgent
from agents.intent         import IntentAgent
from agents.dialogue       import DialogueAgent
from agents.planning       import PlanningAgent
from agents.text_to_speech import TextToSpeechAgent

SERIAL_PORT  = os.getenv("SERIAL_PORT",  "/dev/ttyACM0")
WAKE_WORD    = os.getenv("WAKE_WORD",    "didgeridoo")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


async def startup(tts: TextToSpeechAgent):
    """Announce we're alive with an emotion + spoken greeting."""
    from tools.emotion import set_emotion
    try:
        set_emotion("happy")
    except Exception as e:
        print(f"[startup] set_emotion failed (CAN not connected?): {e}")
    await tts.speak("Hey there, good looking!")


async def pipeline(audio_capture, stt, intent, dialogue, planning, tts):
    while True:
        # ── Idle + Capture ────────────────────────────────────────────────
        print("\n[idle] waiting for wake word ...")
        audio = await audio_capture.listen()

        # ── Transcribe ────────────────────────────────────────────────────
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
    audio_capture = AudioCaptureAgent(WAKE_WORD, speech_seconds=4)
    stt      = SpeechToTextAgent()
    intent   = IntentAgent(model=OLLAMA_MODEL)
    dialogue = DialogueAgent(model=OLLAMA_MODEL)
    planning = PlanningAgent(model=OLLAMA_MODEL)
    tts      = TextToSpeechAgent()

    await startup(tts)

    await asyncio.gather(
        serial_reader.run(SERIAL_PORT),
        pipeline(audio_capture, stt, intent, dialogue, planning, tts),
    )


if __name__ == "__main__":
    asyncio.run(main())
