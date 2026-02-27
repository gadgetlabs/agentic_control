"""
Planning Agent  (Strands Agent + Qwen via Ollama)

A Strands Agent backed by a local Qwen model running in Ollama.
LiteLLM bridges between Strands' tool-calling interface and Ollama's API.

The agent receives the transcribed voice command, reasons about which tools
to call, then calls drive_for / stop / set_emotion directly.
The tool calls ARE the motor control - no separate execution step needed.
"""

import asyncio

from strands import Agent
from strands.models.litellm import LiteLLMModel

from tools import drive_for, stop, set_emotion, get_sensors

SYSTEM = """
You are the motor controller for CHAOS, a small wheeled robot.
When given a movement command, call the appropriate tools to execute it.

Guidelines:
- Set emotion to 'thinking' before a move, 'happy' when done
- Forward/reverse: drive_for with steering=0.0
- Turning: combine throttle with steering (e.g. throttle=0.3, steering=-0.8)
- Spin on the spot: throttle=0.0, full steering
- Always call stop() after timed moves as a safety measure
- Reply with one short sentence describing what you did
""".strip()


class PlanningAgent:
    def __init__(self, model: str):
        strands_model = LiteLLMModel(
            model_id=f"ollama/{model}",
            api_base="http://localhost:11434",
        )
        self._agent = Agent(
            model=strands_model,
            tools=[drive_for, stop, set_emotion, get_sensors],
            system_prompt=SYSTEM,
        )

    async def plan(self, text: str) -> str:
        print(f"[planning] {text!r}")
        result = await self._agent.invoke_async(text)
        return result.output
