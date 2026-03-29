#CALL OLLAMA
import requests
import subprocess
import time
import sys

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_MODEL = "gemma3:4b"

def chat_with_ollama(shared_state):
    print("[CHAT] Wake word detected. Starting terminal chat.")
    print("[CHAT] Type 'exit' to stop chatting.\n")

    messages = [
        {
            "role": "system",
            "content": "You are Ronnor, a concise helpful voice assistant running on a Raspberry Pi."
        }
    ]

    shared_state["expression"] = "listening"

    while shared_state["running"]:
        user_text = input("You: ").strip()

        if not user_text:
            shared_state["expression"] = "listening"
            continue

        if user_text.lower() in {"exit", "quit", "stop"}:
            print("[CHAT] Ending chat mode.")
            shared_state["expression"] = "idle"
            shared_state["running"] = False
            break

        messages.append({"role": "user", "content": user_text})

        try:
            shared_state["expression"] = "thinking"

            response = requests.post(
                OLLAMA_CHAT_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()

            data = response.json()
            assistant_text = data["message"]["content"].strip()

            shared_state["expression"] = "speaking"

            safe_text = assistant_text.encode("latin-1", errors="replace").decode("latin-1")
            print(f"Ronnor: {safe_text}")

            messages.append({"role": "assistant", "content": assistant_text})

            shared_state["expression"] = "listening"

        except requests.RequestException as e:
            print(f"[CHAT] Ollama request failed: {e}")
            shared_state["expression"] = "idle"
            shared_state["running"] = False
            break

        except KeyError:
            print("[CHAT] Unexpected Ollama response format.")
            shared_state["expression"] = "idle"
            shared_state["running"] = False
            break


def is_ollama_running():
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def start_ollama():
    print("[OLLAMA] Starting Ollama server...")

    subprocess.Popen(
        ["ollama", "serve"]
    )

    for _ in range(20):
        if is_ollama_running():
            print("[OLLAMA] Server is running.")
            return True
        time.sleep(1)
    raise RuntimeError("Ollama failed to start.")


def ensure_model():
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


def warmup_model():
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

def setup_ollama():
    """
    Full setup pipeline:
    - start server if needed
    - ensure model is installed
    - warm up model
    """
    print("[OLLAMA] Initializing...")

    if not is_ollama_running():
        start_ollama()
    else:
        print("[OLLAMA] Server already running.")

    ensure_model()
    warmup_model()
    print("[OLLAMA] Setup complete.")


if __name__ == "__main__":
    setup_ollama()
    chat_with_ollama()
    #end
    