"""
tools/motion.py
Drive and stop tools for the Strands PlanningAgent.

CAN 0x100: drive command
  buf[0] = throttle as int8 (value * 100)
  buf[1] = steering as int8 (value * 100)
"""

import time

from strands import tool

from tools._can import send


@tool
def drive_for(throttle: float, steering: float, seconds: float) -> str:
    """
    Drive the robot for a set duration, then stop automatically.
    throttle: -1.0 (full reverse) to 1.0 (full forward)
    steering: -1.0 (full left)   to 1.0 (full right)
    seconds:  how long to drive before stopping
    """
    t = max(-100, min(100, int(throttle * 100)))
    s = max(-100, min(100, int(steering * 100)))
    send(0x100, [t & 0xFF, s & 0xFF, 0, 0, 0, 0, 0, 0])
    time.sleep(max(0.0, seconds))
    send(0x100, [0, 0, 0, 0, 0, 0, 0, 0])
    return f"drove throttle={throttle:.2f} steering={steering:.2f} for {seconds}s"


@tool
def stop() -> str:
    """Stop all motors immediately."""
    send(0x100, [0, 0, 0, 0, 0, 0, 0, 0])
    return "stopped"
