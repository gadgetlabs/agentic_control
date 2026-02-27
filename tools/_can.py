"""
tools/_can.py
Shared CAN bus connection used by motion.py and emotion.py.

Set STUB_HARDWARE=1 in .env to print frames to console instead of
sending real CAN messages - lets you test on Jetson without the robot wired up.
"""

import os

STUB = os.getenv("STUB_HARDWARE", "0") == "1"

if STUB:
    print("[tools] STUB mode - CAN disabled, commands print to console")
    _bus = None
else:
    import can
    _bus = can.interface.Bus(
        channel=os.getenv("CAN_CHANNEL", "can0"), bustype="socketcan"
    )


def send(arb_id: int, data: list[int]):
    if STUB:
        print(f"[CAN stub]  0x{arb_id:03X}  {data}")
        return
    msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=False)
    _bus.send(msg)
