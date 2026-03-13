"""
Intent Agent
Asks an LLM to classify the transcribed command as either:
  "dialogue" - a question, greeting, or conversation
  "action"   - a physical task (movement, emotion, sensor query)

Uses litellm so the backend is swappable via LLM_MODEL env var.
"""

import asyncio
import json
import re

import litellm

SYSTEM = """
Classify this robot voice command into one of two types.

Return ONLY valid JSON in one of these two forms:
  {"type": "dialogue"}
  {"type": "action"}

Use "dialogue" for: questions, greetings, jokes, general chat.
Use "action" for: movement, driving, turning, stopping, LED emotions, sensor queries.
""".strip()


def _parse(content: str) -> dict:
    """Parse JSON from LLM response with fallbacks for messy output."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^}]+\}', content)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"type": "action" if "action" in content.lower() else "dialogue"}


class IntentAgent:
    def __init__(self, model: str):
        self.model = model

    def _classify(self, text: str) -> dict:
        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": text},
            ],
            response_format={"type": "json_object"},
        )
        return _parse(response.choices[0].message.content)

    async def classify(self, text: str) -> dict:
        return await asyncio.to_thread(self._classify, text)
