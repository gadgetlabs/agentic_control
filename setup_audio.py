#!/usr/bin/env python3
"""
setup_audio.py  –  Interactive microphone and speaker selection.

Run this once before starting main.py to identify and test your audio devices.
Saves MIC_DEVICE_INDEX, SPEAKER_DEVICE_INDEX and their working sample rates
to .env so main.py picks them up automatically.

Usage:
    python setup_audio.py
"""

import contextlib
import os
import re

import numpy as np
import sounddevice as sd


ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

# Rates to probe in preference order
COMMON_RATES = [48000, 44100, 32000, 22050, 16000, 8000]


# ── Helpers ───────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _suppress_alsa_noise():
    """
    Redirect C-level stderr to /dev/null while probing sample rates.
    PortAudio/ALSA writes directly to fd 2 (not Python's sys.stderr),
    so a normal redirect won't work – we need to swap the file descriptor.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd   = os.dup(2)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    try:
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)


def probe_output_rate(device_idx: int) -> int:
    """Return the first sample rate ALSA will actually accept for output."""
    for rate in COMMON_RATES:
        try:
            with _suppress_alsa_noise():
                sd.check_output_settings(
                    device=device_idx, channels=1, dtype="float32", samplerate=rate
                )
            return rate
        except Exception:
            pass
    raise RuntimeError(
        f"Device [{device_idx}] rejected all rates {COMMON_RATES}. "
        "Check the device is not in use by another process."
    )


def probe_input_rate(device_idx: int) -> int:
    """Return the first sample rate ALSA will actually accept for input."""
    for rate in COMMON_RATES:
        try:
            with _suppress_alsa_noise():
                sd.check_input_settings(
                    device=device_idx, channels=1, dtype="float32", samplerate=rate
                )
            return rate
        except Exception:
            pass
    raise RuntimeError(
        f"Device [{device_idx}] rejected all input rates {COMMON_RATES}."
    )


# ── Device listing ────────────────────────────────────────────────────────────

def list_devices():
    devices = sd.query_devices()
    inputs  = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"]  > 0]
    outputs = [(i, d) for i, d in enumerate(devices) if d["max_output_channels"] > 0]
    return inputs, outputs


def print_devices(label: str, devices: list):
    print(f"\n{'─'*52}")
    print(f"  Available {label} devices")
    print(f"{'─'*52}")
    for idx, d in devices:
        ch   = d["max_input_channels"] if "input" in label.lower() else d["max_output_channels"]
        rate = int(d["default_samplerate"])
        print(f"  [{idx:2d}]  {d['name']:<40s}  {ch}ch")
    print(f"{'─'*52}")


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

def test_mic(device_idx: int) -> tuple[bool, int]:
    """Record 2 s and report levels. Returns (ok, working_rate)."""
    print(f"\n[test] Probing input rates for device [{device_idx}] ...")
    try:
        rate = probe_input_rate(device_idx)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return False, 0

    print(f"  Using {rate} Hz")
    print(f"[test] Recording 2 seconds – speak or make noise now!")
    try:
        audio = sd.rec(
            int(2 * rate), samplerate=rate, channels=1, dtype="float32", device=device_idx
        )
        sd.wait()
        avg  = float(np.abs(audio).mean())
        peak = float(np.abs(audio).max())
        bar  = "█" * min(int(peak * 50), 50)
        print(f"  avg={avg:.4f}  peak={peak:.4f}  |{bar:<50s}|")
        if peak < 0.001:
            print("  WARNING: level very low – mic may be muted or wrong device")
            return False, rate
        print("  OK")
        return True, rate
    except Exception as e:
        print(f"  ERROR: {e}")
        return False, rate


def test_speaker(device_idx: int) -> tuple[bool, int]:
    """Play a 440 Hz tone. Returns (heard, working_rate)."""
    print(f"\n[test] Probing output rates for device [{device_idx}] ...")
    try:
        rate = probe_output_rate(device_idx)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return False, 0

    print(f"  Using {rate} Hz")
    print(f"[test] Playing 440 Hz tone for 1 second ...")
    try:
        t    = np.linspace(0, 1, rate, dtype="float32")
        tone = 0.3 * np.sin(2 * np.pi * 440 * t)
        sd.play(tone, samplerate=rate, device=device_idx)
        sd.wait()
        ans = input("  Did you hear the tone? [y/n]: ").strip().lower()
        return ans.startswith("y"), rate
    except Exception as e:
        print(f"  ERROR: {e}")
        return False, rate


# ── .env persistence ──────────────────────────────────────────────────────────

def set_env_key(key: str, value: str):
    """Write or update a key=value line in .env (creates the file if absent)."""
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as f:
            content = f.read()
    else:
        content = ""

    pattern  = rf"^{re.escape(key)}=.*$"
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

    inputs, outputs = list_devices()

    # ── Microphone ────────────────────────────────────────────────────────────
    print_devices("INPUT (microphone)", inputs)
    current_mic = get_current_index("MIC_DEVICE_INDEX")
    mic_idx  = pick_device("microphone INPUT", inputs, current_mic)
    mic_name = dict(inputs)[mic_idx]["name"]
    print(f"  selected: [{mic_idx}] {mic_name}")

    ok, mic_rate = test_mic(mic_idx)
    if not ok:
        retry = input("  Try a different device? [y/n]: ").strip().lower()
        if retry.startswith("y"):
            mic_idx  = pick_device("microphone INPUT", inputs, None)
            mic_name = dict(inputs)[mic_idx]["name"]
            ok, mic_rate = test_mic(mic_idx)

    # ── Speaker ───────────────────────────────────────────────────────────────
    print_devices("OUTPUT (speaker)", outputs)
    current_spk = get_current_index("SPEAKER_DEVICE_INDEX")
    spk_idx  = pick_device("speaker OUTPUT", outputs, current_spk)
    spk_name = dict(outputs)[spk_idx]["name"]
    print(f"  selected: [{spk_idx}] {spk_name}")

    ok, spk_rate = test_speaker(spk_idx)
    if not ok:
        retry = input("  Try a different device? [y/n]: ").strip().lower()
        if retry.startswith("y"):
            spk_idx  = pick_device("speaker OUTPUT", outputs, None)
            spk_name = dict(outputs)[spk_idx]["name"]
            ok, spk_rate = test_speaker(spk_idx)

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n[save] Writing to .env ...")
    set_env_key("MIC_DEVICE_INDEX",    str(mic_idx))
    set_env_key("MIC_SAMPLE_RATE",     str(mic_rate))
    set_env_key("SPEAKER_DEVICE_INDEX", str(spk_idx))
    set_env_key("SPEAKER_SAMPLE_RATE",  str(spk_rate))

    print(f"\nDone!")
    print(f"  Mic:     [{mic_idx}] {mic_name}  @ {mic_rate} Hz")
    print(f"  Speaker: [{spk_idx}] {spk_name}  @ {spk_rate} Hz")
    print(f"\nRun:  python main.py")


if __name__ == "__main__":
    main()
