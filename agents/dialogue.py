"""
Dialogue Agent
Generates a short spoken response for conversational commands.
The reply is passed to the TTS agent so it should be plain prose - no markdown.
"""

import asyncio

import ollama

SYSTEM = """
You are CHAOS, a small friendly wheeled robot.
Reply in 1-2 short sentences. Plain text only - no bullet points or markdown.
You will be read aloud.
""".strip()


class DialogueAgent:
    def __init__(self, model: str):
        self.model = model

    def _respond(self, text: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": text},
            ],
        )
        return response.message.content.strip()

    async def respond(self, text: str) -> str:
        return await asyncio.to_thread(self._respond, text)
