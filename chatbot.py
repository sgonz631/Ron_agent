# CALL OLLAMA + PIPER TTS
import requests
import subprocess
import time
import os
from pathlib import Path

# Text-to-speech helper
import piper_tts
import whisper_stt


# -------------------------------------------------------------------
# OLLAMA CONFIG
# -------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_MODEL = "gemma3:4b"


# -------------------------------------------------------------------
# CHAT LOOP
# -------------------------------------------------------------------
def chat_with_ollama(shared_state):
    """
    Starts a terminal-based chat with Ollama.

    Behavior:
    - 'bye' -> ends only the current chat session
    - 'exit' -> closes the entire app
    - every Ollama response is sent to Piper for speech output
    - shared_state drives the face animations in agent_ron.py
    """
    print("[CHAT] Wake word detected. Starting terminal chat.")
    print("[CHAT] Type 'bye' to end chat, 'exit' to close Ronnor.\n")

    # Conversation history for Ollama
    messages = [
        {
            "role": "system",
            "content": "You are Ronnor, a concise helpful voice assistant running on a Raspberry Pi."
        }
    ]

    # Show listening face at the beginning of chat mode
    shared_state["expression"] = "listening"

    # Main chat loop
    while shared_state["running"]:
    # -----------------------------------------------------------
    # INPUT SELECTION
    # Default = voice
    # If space bar was pressed in agent_ron GUI, use text once
    # -----------------------------------------------------------

        if shared_state.get("force_text_input", False):
            shared_state["force_text_input"] = False
            user_text = input("You (text): ").strip()
        else:
            print("[VOICE] Speak now...")
            shared_state["expression"] = "listening"
            user_text = whisper_stt.listen_and_transcribe(duration=5).strip()
            if user_text:
                print(f"You (voice): {user_text}")

        command = user_text.lower()

        if not user_text:
            shared_state["expression"] = "listening"
            continue

        # -----------------------------------------------------------
        # END CHAT ONLY
        # -----------------------------------------------------------
        if command in {"bye", "stop", "quit"}:
            print("[CHAT] Ending chat mode.")

            # Optional spoken goodbye through Piper
            try:
                shared_state["expression"] = "speaking"
                piper_tts.speak_text("Goodbye! See you soon!")
            except Exception as e:
                print(f"[TTS] Goodbye speech failed: {e}")
            finally:
                shared_state["expression"] = "idle"

            shared_state["chat_active"] = False
            break

        # -----------------------------------------------------------
        # CLOSE ENTIRE APP
        # -----------------------------------------------------------
        if command == "exit":
            print("[SYSTEM] Shutting down Ronnor.")

            # Optional spoken shutdown message
            try:
                shared_state["expression"] = "speaking"
                piper_tts.speak_text("Shutting down.")
            except Exception as e:
                print(f"[TTS] Shutdown speech failed: {e}")
            finally:
                shared_state["expression"] = "idle"

            shared_state["chat_active"] = False
            shared_state["running"] = False
            break

        # Save the user's message into conversation memory
        messages.append({"role": "user", "content": user_text})

        try:
            # Show thinking face while Ollama is generating
            shared_state["expression"] = "thinking"

            response = requests.post(
                OLLAMA_CHAT_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False
                },
                timeout=300
            )
            response.raise_for_status()

            data = response.json()
            assistant_text = data["message"]["content"].strip()

             # Print response safely
            try:
                print(f"Ronnor: {assistant_text}")
            except UnicodeEncodeError:
                fallback_text = assistant_text.encode("ascii", errors="replace").decode("ascii")
                print(f"Ronnor: {fallback_text}")

              # Save assistant message to memory
            messages.append({"role": "assistant", "content": assistant_text})

            shared_state["expression"] = "speaking"
            try:
                piper_tts.speak_text(assistant_text)
            finally:
                if shared_state["running"]:
                    shared_state["expression"] = "listening"

        except requests.Timeout:
            print("[CHAT] Ollama took too long to respond. Please try again.")
            if shared_state["running"]:
                shared_state["expression"] = "listening"
            continue
        
        except requests.RequestException as e:
            print(f"[CHAT] Ollama request failed: {e}")
            shared_state["expression"] = "idle"
            shared_state["chat_active"] = False
            break

        except KeyError:
            print("[CHAT] Unexpected Ollama response format.")
            shared_state["expression"] = "idle"
            shared_state["chat_active"] = False
            break

        except KeyboardInterrupt:
            print("\n[CHAT] Interrupted by user.")
            shared_state["expression"] = "idle"
            shared_state["chat_active"] = False
            shared_state["running"] = False
            break        

        except Exception as e:
            # Catch any unexpected TTS or parsing issue
            print(f"[CHAT] Unexpected error: {e}")
            shared_state["expression"] = "idle"
            shared_state["chat_active"] = False
            break


# -------------------------------------------------------------------
# OLLAMA SERVER CHECK
# -------------------------------------------------------------------
def is_ollama_running():
    """
    Checks if the Ollama server is already running.
    """
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


# -------------------------------------------------------------------
# START OLLAMA SERVER
# -------------------------------------------------------------------
def start_ollama():
    """
    Starts Ollama server if it is not already running.
    """
    print("[OLLAMA] Starting Ollama server...")

    subprocess.Popen(["ollama", "serve"])

    # Wait up to ~20 seconds for the server to become available
    for _ in range(20):
        if is_ollama_running():
            print("[OLLAMA] Server is running.")
            return True
        time.sleep(1)

    raise RuntimeError("Ollama failed to start.")


# -------------------------------------------------------------------
# ENSURE MODEL EXISTS
# -------------------------------------------------------------------
def ensure_model():
    """
    Checks whether the selected Ollama model is installed.
    If not, pulls it automatically.
    """
    print(f"[OLLAMA] Checking model: {OLLAMA_MODEL}")

    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=10)
        response.raise_for_status()
        models = response.json().get("models", [])

        if any(m.get("name") == OLLAMA_MODEL for m in models):
            print("[OLLAMA] Model already installed.")
            return

    except Exception as e:
        print(f"[OLLAMA] Could not verify installed models: {e}")

    print(f"[OLLAMA] Pulling model {OLLAMA_MODEL}...")
    subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)


# -------------------------------------------------------------------
# WARM UP MODEL
# -------------------------------------------------------------------
def warmup_model():
    """
    Sends a quick request to Ollama so the model is loaded and ready
    before real interaction starts.
    """
    print("[OLLAMA] Warming up model...")

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        print("[OLLAMA] Model ready.")

    except Exception as e:
        print(f"[OLLAMA] Warmup failed: {e}")


# -------------------------------------------------------------------
# FULL SETUP
# -------------------------------------------------------------------
def setup_ollama():
    """
    Full setup pipeline:
    1. Start server if needed
    2. Ensure model is installed
    3. Warm it up
    """
    print("[OLLAMA] Initializing...")

    if not is_ollama_running():
        start_ollama()
    else:
        print("[OLLAMA] Server already running.")

    ensure_model()
    warmup_model()
    print("[OLLAMA] Setup complete.")


# -------------------------------------------------------------------
# STANDALONE TEST MODE
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Shared state for standalone testing without agent_ron.py
    test_state = {
        "expression": "idle",
        "running": True,
        "chat_active": True
    }

    setup_ollama()
    chat_with_ollama(test_state)