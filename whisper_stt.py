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
import audioop


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
    max_record_seconds: int = 12,
    sample_rate: int = 16000,
    silence_threshold: int = 700,
    silence_seconds_to_stop: float = 1.2,
    speech_start_timeout: float = 6.0,
    device: str | None = None,
) -> str:
    """
    Record from microphone until the user stops speaking.

    Behavior:
    - waits for speech to begin
    - once speech begins, keeps recording
    - stops after sustained silence
    - also stops at max_record_seconds as a safety cap

    Returns the WAV path.
    """
    import os
    import wave
    import tempfile
    import subprocess
    import time

    chunk_seconds = 0.25
    chunk_frames = int(sample_rate * chunk_seconds)
    bytes_per_sample = 2  # S16_LE mono
    chunk_bytes = chunk_frames * bytes_per_sample

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav_path = tmp.name

    cmd = [
        "arecord",
        "-q",
        "-f", "S16_LE",
        "-r", str(sample_rate),
        "-c", "1",
        "-t", "raw",
    ]

    if device:
        cmd = ["arecord", "-D", device] + cmd[1:]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    frames = []
    speech_started = False
    silence_chunks_needed = max(1, int(silence_seconds_to_stop / chunk_seconds))
    silence_chunk_count = 0

    start_time = time.time()
    speech_wait_start = time.time()

    try:
        while True:
            if proc.stdout is None:
                break

            chunk = proc.stdout.read(chunk_bytes)
            if not chunk:
                break

            rms = audioop.rms(chunk, 2)

            # Wait for speech to begin
            if not speech_started:
                if rms >= silence_threshold:
                    speech_started = True
                    frames.append(chunk)
                    silence_chunk_count = 0
                else:
                    # No speech yet: stop waiting after timeout
                    if time.time() - speech_wait_start > speech_start_timeout:
                        break
                    continue
            else:
                frames.append(chunk)

                if rms < silence_threshold:
                    silence_chunk_count += 1
                else:
                    silence_chunk_count = 0

                # Stop once we detect sustained silence after speech
                if silence_chunk_count >= silence_chunks_needed:
                    break

            # Safety cap
            if time.time() - start_time >= max_record_seconds:
                break

    finally:
        try:
            proc.terminate()
        except Exception:
            pass

        try:
            proc.wait(timeout=1)
        except Exception:
            pass

    # Save recorded frames as WAV
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))

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
    max_record_seconds: int = 10,
    sample_rate: int = 16000,
    language: str = "en",
    device: str | None = None,
) -> str:
    """
    Record until the user stops speaking, then transcribe.
    """
    wav_path = record_audio_to_wav(
        max_record_seconds=max_record_seconds,
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