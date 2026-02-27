"""
Intent Agent
Asks a small local LLM to classify the transcribed command as either:
  "dialogue" - a question, greeting, or conversation
  "action"   - a physical task (movement, emotion, sensor query)

Uses ollama with format="json" so the response is always parseable.
"""

import asyncio
import json

import ollama

SYSTEM = """
Classify this robot voice command into one of two types.

Return ONLY valid JSON in one of these two forms:
  {"type": "dialogue"}
  {"type": "action"}

Use "dialogue" for: questions, greetings, jokes, general chat.
Use "action" for: movement, driving, turning, stopping, LED emotions, sensor queries.
""".strip()


class IntentAgent:
    def __init__(self, model: str):
        self.model = model

    def _classify(self, text: str) -> dict:
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": text},
            ],
            format="json",
        )
        return json.loads(response.message.content)

    async def classify(self, text: str) -> dict:
        return await asyncio.to_thread(self._classify, text)
