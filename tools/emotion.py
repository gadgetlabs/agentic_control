"""
tools/emotion.py
LED emotion tool for the Strands PlanningAgent.

Serial protocol (USB to Arduino):
  CMD,EMOTION,<code>

  0 idle     - dim white breathe
  1 happy    - green scanning sweep
  2 thinking - blue spinner
  3 sad      - deep blue pulse
  4 angry    - red alternating flash
"""

from strands import tool

import state_bus
import serial_reader

EMOTIONS = {"idle": 0, "happy": 1, "thinking": 2, "sad": 3, "angry": 4}


@tool
def set_emotion(emotion: str) -> str:
    """
    Set the robot's LED emotion.
    Options: idle, happy, thinking, sad, angry
    """
    code = EMOTIONS.get(emotion)
    if code is None:
        return f"unknown emotion '{emotion}'. options: {list(EMOTIONS)}"
    serial_reader.send_command(f"CMD,EMOTION,{code}")
    state_bus.set_current_emotion(emotion)
    return f"emotion={emotion}"
