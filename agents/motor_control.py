"""
Motor Control Agent
Executes an action plan produced by the Planning Agent.

Each step calls a robot_tools function then waits 'duration' seconds
before moving to the next step. asyncio.sleep keeps the event loop
free during waits (serial telemetry keeps flowing).
"""

import asyncio

from robot_tools import drive, stop, set_emotion, get_sensors

TOOL_MAP = {
    "drive":        drive,
    "stop":         stop,
    "set_emotion":  set_emotion,
    "get_sensors":  get_sensors,
}


class MotorControlAgent:
    async def execute(self, plan: list[dict]) -> str:
        if not plan:
            return "No actions to execute."

        print(f"[motor] executing {len(plan)} steps")

        for step in plan:
            tool     = step.get("tool", "")
            args     = step.get("args", {})
            duration = step.get("duration", 0)

            fn = TOOL_MAP.get(tool)
            if fn is None:
                print(f"[motor]   unknown tool: {tool!r}")
                continue

            result = fn(**args)
            print(f"[motor]   {tool}({args})  →  {result}")

            if duration > 0:
                await asyncio.sleep(duration)

        return f"Done - {len(plan)} steps completed."
