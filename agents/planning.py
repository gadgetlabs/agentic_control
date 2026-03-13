"""
Planning Agent  (Strands Agent + LiteLLM)

Backed by whatever model LLM_MODEL points at.
For Ollama models (model starts with "ollama/"), api_base is set automatically
from OLLAMA_API_BASE env var (default: http://localhost:11434).
For cloud models (gemini/, claude/, gpt-*) no extra config is needed beyond
the provider's API key env var.
"""

import asyncio
import os

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
        client_args = {}
        if model.startswith("ollama/"):
            client_args["api_base"] = os.getenv(
                "OLLAMA_API_BASE", "http://localhost:11434"
            )
        strands_model = LiteLLMModel(model_id=model, client_args=client_args)
        self._agent = Agent(
            model=strands_model,
            tools=[drive_for, stop, set_emotion, get_sensors],
            system_prompt=SYSTEM,
        )

    async def plan(self, text: str) -> str:
        print(f"[planning] {text!r}")
        result = await self._agent.invoke_async(text)
        return str(result)
