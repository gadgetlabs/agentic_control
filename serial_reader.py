"""
serial_reader.py
Async task that populates robot_state from the Arduino's $-prefixed telemetry.

Set STUB_HARDWARE=1 in .env to generate fake sensor data instead of opening
a serial port - useful for testing on Jetson before the Arduino is connected.

Serial protocol (chaos_hal.ino, 115200 baud, 10 Hz):
  $IMU,ax,ay,az,gx,gy,gz
  $CMP,cx,cy,cz
  $ODO,linear,angular
  $RPM,left,right
  $LDR,rpm,d0,...,d359
"""

import asyncio
import math
import os
import random

STUB = os.getenv("STUB_HARDWARE", "0") == "1"

robot_state = {
    "imu":     {},
    "compass": {},
    "odom":    {},
    "rpm":     {},
    "lidar":   [],
}


def parse_line(line: str):
    parts = line.split(",")
    tag   = parts[0]

    if tag == "$IMU" and len(parts) == 7:
        robot_state["imu"] = {
            "ax": float(parts[1]), "ay": float(parts[2]), "az": float(parts[3]),
            "gx": float(parts[4]), "gy": float(parts[5]), "gz": float(parts[6]),
        }
    elif tag == "$CMP" and len(parts) == 4:
        robot_state["compass"] = {
            "x": float(parts[1]), "y": float(parts[2]), "z": float(parts[3]),
        }
    elif tag == "$ODO" and len(parts) == 3:
        robot_state["odom"] = {"linear": float(parts[1]), "angular": float(parts[2])}
    elif tag == "$RPM" and len(parts) == 3:
        robot_state["rpm"]  = {"left": float(parts[1]), "right": float(parts[2])}
    elif tag == "$LDR" and len(parts) > 2:
        robot_state["lidar"] = [int(d) for d in parts[2:]]


async def _stub_run():
    """Emit fake sensor data at 10 Hz so the rest of the pipeline has numbers."""
    print("[serial] STUB mode - generating fake sensor data at 10 Hz")
    t = 0.0
    while True:
        await asyncio.sleep(0.1)
        t += 0.1
        robot_state["imu"] = {
            "ax": round(random.gauss(0.0,  0.05), 3),
            "ay": round(random.gauss(0.0,  0.05), 3),
            "az": round(random.gauss(9.81, 0.02), 3),
            "gx": round(random.gauss(0.0,  0.01), 2),
            "gy": round(random.gauss(0.0,  0.01), 2),
            "gz": round(random.gauss(0.0,  0.01), 2),
        }
        robot_state["compass"] = {
            "x": round(200 + 10 * math.sin(t * 0.1), 1),
            "y": round( 50 + 10 * math.cos(t * 0.1), 1),
            "z": round(-400 + random.gauss(0, 2),     1),
        }
        robot_state["odom"] = {"linear": 0.0, "angular": 0.0}
        robot_state["rpm"]  = {"left":   0.0, "right":   0.0}


async def _serial_run(port: str):
    import serial
    import serial_asyncio
    while True:
        try:
            reader, _ = await serial_asyncio.open_serial_connection(url=port, baudrate=115200)
            print(f"[serial] connected on {port}")
            while True:
                raw  = await reader.readline()
                line = raw.decode(errors="ignore").strip()
                if line.startswith("$"):
                    parse_line(line)
        except serial.SerialException as e:
            if "[Errno 2]" in str(e):
                # Port doesn't exist - Arduino not plugged in. Fall back to stub
                # rather than spamming retries.
                print(f"[serial] {port} not found - falling back to stub mode")
                print(f"[serial] set SERIAL_PORT in .env and restart when Arduino is connected")
                await _stub_run()
                return
            # Other serial errors (e.g. device disconnected mid-run) - retry
            print(f"[serial] {port} lost ({e}) - retrying in 5 s ...")
            await asyncio.sleep(5)


async def run(port: str):
    if STUB:
        await _stub_run()
    else:
        await _serial_run(port)
