"""
tools/emotion.py
LED emotion tool for the Strands PlanningAgent.

CAN 0x107: emotion command
  buf[0] = emotion code (0-4), maps to NeoPixel animations in chaos_hal.ino

  0 idle    - dim white breathe
  1 happy   - green scanning sweep
  2 thinking - blue spinner
  3 sad     - deep blue pulse
  4 angry   - red alternating flash
"""

from strands import tool

from tools._can import send

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
    send(0x107, [code])
    return f"emotion={emotion}"
