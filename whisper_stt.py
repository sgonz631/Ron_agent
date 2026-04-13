#!/usr/bin/env python3
"""
Standalone Whisper STT helper for Ronnor.

Features:
- Records microphone audio to WAV
- Transcribes WAV with whisper.cpp
- Can be run standalone
- Can also be imported by agent_ron.py or another script
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
WHISPER_CLI = BASE_DIR / "whisper.cpp" / "build" / "bin" / "whisper-cli"

# Recommended for Raspberry Pi speed. You can change to base.en later.
WHISPER_MODEL = BASE_DIR / "whisper.cpp" / "models" / "ggml-tiny.en.bin"


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def _validate_paths() -> None:
    if not WHISPER_CLI.is_file():
        raise FileNotFoundError(f"Whisper CLI not found: {WHISPER_CLI}")
    if not WHISPER_MODEL.is_file():
        raise FileNotFoundError(f"Whisper model not found: {WHISPER_MODEL}")


def _clean_transcript(text: str) -> str:
    """
    Clean Whisper output into a more usable single-line text string.
    """
    if not isinstance(text, str):
        text = str(text)

    # Remove common timestamp lines if present
    text = re.sub(r"\[\d{2}:\d{2}:\d{2}\.\d{3} --> .*?\]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -------------------------------------------------------------------
# AUDIO RECORDING
# -------------------------------------------------------------------
def record_audio_to_wav(
    duration: int = 5,
    sample_rate: int = 16000,
    device: Optional[str] = None,
) -> str:
    """
    Records mono microphone audio to a temporary WAV file using arecord.
    Returns the WAV path.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav_path = tmp.name

    cmd = [
        "arecord",
        "-f", "S16_LE",
        "-r", str(sample_rate),
        "-c", "1",
        "-d", str(duration),
        wav_path,
    ]

    # Optional ALSA device override
    if device:
        cmd = ["arecord", "-D", device] + cmd[1:]

    subprocess.run(cmd, check=True)
    return wav_path


# -------------------------------------------------------------------
# WHISPER TRANSCRIPTION
# -------------------------------------------------------------------
def transcribe_file(wav_path: str, language: str = "en") -> str:
    """
    Transcribes an existing WAV file with whisper.cpp and returns clean text.
    """
    _validate_paths()

    cmd = [
        str(WHISPER_CLI),
        "-m", str(WHISPER_MODEL),
        "-f", str(wav_path),
        "-l", language,
        "-nt",        # no timestamps
        "-np",        # no progress
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    transcript = _clean_transcript(result.stdout)
    return transcript


def listen_and_transcribe(
    duration: int = 5,
    sample_rate: int = 16000,
    language: str = "en",
    device: Optional[str] = None,
) -> str:
    """
    Record from mic, transcribe, clean up temp file, return text.
    """
    wav_path = record_audio_to_wav(
        duration=duration,
        sample_rate=sample_rate,
        device=device,
    )

    try:
        return transcribe_file(wav_path, language=language)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


# -------------------------------------------------------------------
# STANDALONE TEST MODE
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("[WHISPER] Recording for 5 seconds...")
    try:
        text = listen_and_transcribe(duration=5)
        print(f"[WHISPER] Transcript: {text}")
    except Exception as e:
        print(f"[WHISPER] Error: {e}")