"""
Planning Agent
Converts a voice command into an ordered list of motor steps.

Each step is a dict:
  { "tool": str, "args": dict, "duration": float }

The LLM returns JSON. We parse it and hand the list to MotorControlAgent.
"""

import asyncio
import json

import ollama

SYSTEM = """
You are a robot motion planner. Convert the voice command into an action plan.

Return a JSON object with a "steps" array. Each step has:
  "tool"     : one of "drive", "stop", "set_emotion"
  "args"     : the function arguments
  "duration" : seconds to hold this state (0 for instant actions)

Tool signatures:
  drive(throttle: float -1.0..1.0, steering: float -1.0..1.0)
  stop()
  set_emotion(emotion: "idle" | "happy" | "thinking" | "sad" | "angry")

Example for "spin left then stop":
{
  "steps": [
    {"tool": "set_emotion", "args": {"emotion": "happy"},                     "duration": 0},
    {"tool": "drive",       "args": {"throttle": 0.0, "steering": -1.0},     "duration": 1.5},
    {"tool": "stop",        "args": {},                                        "duration": 0},
    {"tool": "set_emotion", "args": {"emotion": "idle"},                      "duration": 0}
  ]
}
""".strip()


class PlanningAgent:
    def __init__(self, model: str):
        self.model = model

    def _plan(self, text: str) -> list[dict]:
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": text},
            ],
            format="json",
        )
        data = json.loads(response.message.content)
        return data.get("steps", [])

    async def plan(self, text: str) -> list[dict]:
        print(f"[planning] generating plan for: {text!r}")
        return await asyncio.to_thread(self._plan, text)
