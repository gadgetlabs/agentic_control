"""
tools/motion.py
Drive and stop tools for the Strands PlanningAgent.

Serial protocol (USB to Arduino):
  CMD,DRIVE,<throttle_int8>,<steering_int8>  — values are float*100, clamped ±100
  CMD,DRIVE,0,0                              — stop
"""

import time

from strands import tool

import serial_reader


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
    serial_reader.send_command(f"CMD,DRIVE,{t},{s}")
    time.sleep(max(0.0, seconds))
    serial_reader.send_command("CMD,DRIVE,0,0")
    return f"drove throttle={throttle:.2f} steering={steering:.2f} for {seconds}s"


@tool
def stop() -> str:
    """Stop all motors immediately."""
    serial_reader.send_command("CMD,DRIVE,0,0")
    return "stopped"
