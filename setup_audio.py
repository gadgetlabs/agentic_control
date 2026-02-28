#!/usr/bin/env python3
"""
setup_audio.py  –  Interactive microphone and speaker selection.

Run this once before starting main.py to identify and test your audio devices.
Saves MIC_DEVICE_INDEX and SPEAKER_DEVICE_INDEX to .env so main.py picks them
up automatically.

Usage:
    python setup_audio.py
"""

import os
import re

import numpy as np
import sounddevice as sd


ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


# ── Device listing ────────────────────────────────────────────────────────────

def list_devices():
    devices = sd.query_devices()
    inputs  = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"]  > 0]
    outputs = [(i, d) for i, d in enumerate(devices) if d["max_output_channels"] > 0]
    return inputs, outputs


def print_devices(label: str, devices: list):
    print(f"\n{'─'*50}")
    print(f"  Available {label} devices")
    print(f"{'─'*50}")
    for idx, d in devices:
        ch   = d["max_input_channels"] if "input" in label else d["max_output_channels"]
        rate = int(d["default_samplerate"])
        print(f"  [{idx:2d}]  {d['name']:<40s}  {ch}ch  {rate}Hz")
    print(f"{'─'*50}")


# ── Device picking ────────────────────────────────────────────────────────────

def pick_device(label: str, devices: list, default_idx: int | None) -> int:
    valid = {idx for idx, _ in devices}
    prompt = f"\nEnter {label} device index"
    if default_idx is not None and default_idx in valid:
        prompt += f" [{default_idx}]"
    prompt += ": "

    while True:
        raw = input(prompt).strip()
        if raw == "" and default_idx is not None:
            return default_idx
        try:
            choice = int(raw)
            if choice in valid:
                return choice
        except ValueError:
            pass
        print(f"  Invalid – choose from: {sorted(valid)}")


# ── Testing ───────────────────────────────────────────────────────────────────

def test_mic(device_idx: int) -> bool:
    print(f"\n[test] Recording 2 seconds from device [{device_idx}] ...")
    print("       Speak or make noise now!")
    try:
        audio = sd.rec(
            int(2 * 16000),
            samplerate=16000,
            channels=1,
            dtype="float32",
            device=device_idx,
        )
        sd.wait()
        avg  = float(np.abs(audio).mean())
        peak = float(np.abs(audio).max())
        bar  = "█" * int(peak * 40)
        print(f"  avg={avg:.4f}  peak={peak:.4f}  {bar}")
        if peak < 0.001:
            print("  WARNING: level very low – mic may be muted or wrong device")
            return False
        print("  OK")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_speaker(device_idx: int) -> bool:
    rate = int(sd.query_devices(device_idx)["default_samplerate"])
    print(f"\n[test] Playing 440 Hz tone for 1 second on device [{device_idx}] (rate={rate} Hz) ...")
    try:
        t    = np.linspace(0, 1, rate, dtype="float32")
        tone = 0.3 * np.sin(2 * np.pi * 440 * t)
        sd.play(tone, samplerate=rate, device=device_idx)
        sd.wait()
        ans = input("  Did you hear the tone? [y/n]: ").strip().lower()
        return ans.startswith("y")
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


# ── .env persistence ──────────────────────────────────────────────────────────

def set_env_key(key: str, value: str):
    """Write or update a key=value line in .env (creates the file if absent)."""
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as f:
            content = f.read()
    else:
        content = ""

    pattern = rf"^{re.escape(key)}=.*$"
    new_line = f"{key}={value}"

    if re.search(pattern, content, flags=re.MULTILINE):
        content = re.sub(pattern, new_line, content, flags=re.MULTILINE)
    else:
        content = content.rstrip("\n") + f"\n{new_line}\n"

    with open(ENV_PATH, "w") as f:
        f.write(content)

    print(f"  saved  {new_line}  →  .env")


def get_current_index(key: str) -> int | None:
    if not os.path.exists(ENV_PATH):
        return None
    with open(ENV_PATH) as f:
        for line in f:
            if line.startswith(f"{key}="):
                try:
                    return int(line.split("=", 1)[1].strip())
                except ValueError:
                    pass
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n╔══════════════════════════════════╗")
    print("║  CHAOS Robot – Audio Setup       ║")
    print("╚══════════════════════════════════╝")
    print(f"\nsounddevice version: {sd.__version__}")
    print(f"Default input:  [{sd.default.device[0]}]")
    print(f"Default output: [{sd.default.device[1]}]")

    inputs, outputs = list_devices()

    # ── Microphone ────────────────────────────────────────────────────────────
    print_devices("INPUT (microphone)", inputs)
    current_mic = get_current_index("MIC_DEVICE_INDEX")
    mic_idx = pick_device("microphone INPUT", inputs, current_mic)
    mic_name = dict(inputs)[mic_idx]["name"]
    print(f"  selected: [{mic_idx}] {mic_name}")

    ok = test_mic(mic_idx)
    if not ok:
        retry = input("  Try a different device? [y/n]: ").strip().lower()
        if retry.startswith("y"):
            mic_idx = pick_device("microphone INPUT", inputs, None)
            mic_name = dict(inputs)[mic_idx]["name"]
            test_mic(mic_idx)

    # ── Speaker ───────────────────────────────────────────────────────────────
    print_devices("OUTPUT (speaker)", outputs)
    current_spk = get_current_index("SPEAKER_DEVICE_INDEX")
    spk_idx = pick_device("speaker OUTPUT", outputs, current_spk)
    spk_name = dict(outputs)[spk_idx]["name"]
    print(f"  selected: [{spk_idx}] {spk_name}")

    ok = test_speaker(spk_idx)
    if not ok:
        retry = input("  Try a different device? [y/n]: ").strip().lower()
        if retry.startswith("y"):
            spk_idx = pick_device("speaker OUTPUT", outputs, None)
            spk_name = dict(outputs)[spk_idx]["name"]
            test_speaker(spk_idx)

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n[save] Writing device indices to .env ...")
    set_env_key("MIC_DEVICE_INDEX", str(mic_idx))
    set_env_key("SPEAKER_DEVICE_INDEX", str(spk_idx))

    print(f"\nDone!")
    print(f"  Mic:     [{mic_idx}] {mic_name}")
    print(f"  Speaker: [{spk_idx}] {spk_name}")
    print(f"\nRun:  python main.py")


if __name__ == "__main__":
    main()
