"""
robot_tools.py
Low-level robot I/O used by MotorControlAgent.

Set STUB_HARDWARE=1 in .env to run without the CAN bus (prints to console).
In stub mode the full pipeline still runs - you can test wake word, Whisper,
Ollama intent + planning on the Jetson before the robot is wired up.

Drive commands → CAN 0x100  (int8 throttle/steering scaled by 100)
Emotion        → CAN 0x107  (single byte, maps to NeoPixel animations)
"""

import os

from serial_reader import robot_state

STUB     = os.getenv("STUB_HARDWARE", "0") == "1"
EMOTIONS = {"idle": 0, "happy": 1, "thinking": 2, "sad": 3, "angry": 4}

if STUB:
    print("[robot_tools] STUB mode - CAN disabled, commands print to console")
    bus = None
else:
    import can
    # Linux / Jetson (socketcan):   channel="can0",        bustype="socketcan"
    # USB-CAN adapter (slcan):      channel="/dev/ttyUSB1", bustype="slcan"
    bus = can.interface.Bus(
        channel=os.getenv("CAN_CHANNEL", "can0"), bustype="socketcan"
    )


def _send_can(arb_id: int, data: list[int]):
    if STUB:
        print(f"[CAN stub]  0x{arb_id:03X}  {data}")
        return
    msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=False)
    bus.send(msg)


def drive(throttle: float, steering: float) -> str:
    t = max(-100, min(100, int(throttle * 100)))
    s = max(-100, min(100, int(steering * 100)))
    _send_can(0x100, [t & 0xFF, s & 0xFF, 0, 0, 0, 0, 0, 0])
    return f"drive(throttle={throttle:.2f}, steering={steering:.2f})"


def stop() -> str:
    _send_can(0x100, [0, 0, 0, 0, 0, 0, 0, 0])
    return "stopped"


def set_emotion(emotion: str) -> str:
    code = EMOTIONS.get(emotion)
    if code is None:
        return f"unknown emotion '{emotion}'"
    _send_can(0x107, [code])
    return f"emotion={emotion}"


def get_sensors() -> dict:
    return robot_state
