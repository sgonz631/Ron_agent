#!/usr/bin/env python3
"""
Standalone Piper TTS helper for Ronnor.

Usage from Python:
    import piper_tts
    piper_tts.speak_text("Hello from Ronnor")

Usage from terminal:
    python3 piper_tts.py "Hello from Ronnor"
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Optional


# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
PIPER_EXE = BASE_DIR / "piper" / "bin" / "piper"

# Change this if you want another default voice
DEFAULT_VOICE = BASE_DIR / "piper" / "voices" / "en_US-bryce-medium.onnx"

# Your Pi's espeak-ng data path
ESPEAK_DATA_PATH = "/usr/lib/aarch64-linux-gnu/espeak-ng-data"


# ---------- Text cleaning ----------
_EMOJI_RANGES = re.compile(
    "["
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"  # supplemental symbols/pictographs
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


def clean_text_for_piper(text: str) -> str:
    """Normalize text to safe UTF-8 input, remove emojis and control chars."""
    if not isinstance(text, str):
        text = str(text)

    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove emojis by range
    text = _EMOJI_RANGES.sub("", text)

    cleaned_chars = []
    for ch in text:
        category = unicodedata.category(ch)

        # Drop control / formatting / surrogate / private-use / unassigned
        if category.startswith("C"):
            # keep normal whitespace chars as spaces
            if ch in ("\n", "\r", "\t"):
                cleaned_chars.append(" ")
            continue

        # Drop "other symbols" (this removes many leftover emoji-like symbols)
        if category == "So":
            continue

        cleaned_chars.append(ch)

    text = "".join(cleaned_chars)

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Ensure valid UTF-8 roundtrip
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

    return text


# ---------- Validation ----------
def _validate_paths(voice_model: Path) -> None:
    if not PIPER_EXE.is_file():
        raise FileNotFoundError(f"Piper executable not found: {PIPER_EXE}")

    if not voice_model.is_file():
        raise FileNotFoundError(f"Piper voice model not found: {voice_model}")

    if shutil.which("aplay") is None:
        raise FileNotFoundError("aplay not found. Install it with: sudo apt install alsa-utils")

    if not Path(ESPEAK_DATA_PATH).is_dir():
        raise FileNotFoundError(f"espeak-ng data path not found: {ESPEAK_DATA_PATH}")


# ---------- TTS ----------
def speak_text(text: str, voice_model: Optional[str | Path] = None) -> bool:
    """
    Clean text and speak it with Piper.
    Returns True on success, False if text became empty.
    Raises on missing files or subprocess failure.
    """
    model_path = Path(voice_model) if voice_model else DEFAULT_VOICE
    _validate_paths(model_path)

    clean_text = clean_text_for_piper(text)
    if not clean_text:
        return False

    env = os.environ.copy()
    env["ESPEAK_DATA_PATH"] = ESPEAK_DATA_PATH

    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav_path = tmp.name

    try:
        # Generate WAV with Piper
        result = subprocess.run(
            [
                str(PIPER_EXE),
                "--model",
                str(model_path),
                "--output_file",
                wav_path,
            ],
            input=clean_text,
            text=True,
            capture_output=True,
            env=env,
            check=True,
        )

        # Play WAV
        subprocess.run(
            ["aplay", wav_path],
            env=env,
            check=True,
        )

        return True

    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else str(e)
        raise RuntimeError(f"Piper/aplay failed: {stderr}")

    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


# ---------- CLI ----------
def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python3 piper_tts.py "Your text here"', file=sys.stderr)
        return 1

    input_text = " ".join(sys.argv[1:])

    try:
        ok = speak_text(input_text)
        if not ok:
            print("No speakable text after cleaning.", file=sys.stderr)
            return 2
        return 0
    except Exception as e:
        print(f"TTS Error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
