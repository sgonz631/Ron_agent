# CALL OLLAMA + PIPER TTS + WHISPER STT

import requests
import subprocess
import time
import re
import unicodedata
import json

# Text-to-speech helper
import piper_tts

# Speech-to-text helper
import whisper_stt

# Inventory management helper
import inventory_terminal as inventory

#time helpers for state tracking
from state_utils import set_expression, print_state_summary



# -------------------------------------------------------------------
# OLLAMA CONFIG
# -------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_MODEL = "gemma3:4b"


# -------------------------------------------------------------------
# TEXT NORMALIZATION / EXIT PHRASE DETECTION
# -------------------------------------------------------------------
def normalize_user_text(text: str) -> str:
    """
    Normalize text for intent matching:
    - lowercase
    - remove accents
    - remove asterisks
    - remove punctuation
    - collapse spaces
    """
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("*", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_end_chat_phrase(text: str) -> bool:
    """
    Returns True if the user said a phrase that should end the chat
    and return Ronnor to wake-word mode.

    Uses partial matching so it also works if Whisper adds extra words.
    """
    normalized = normalize_user_text(text)

    end_phrases = [
        "bye",
        "bye bye",
        "thatll be all",
        "that will be all",
    ]

    return any(phrase in normalized for phrase in end_phrases)

# -------------------------------------------------------------------
# SAFE PRINT
# -------------------------------------------------------------------
def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        fallback = text.encode("ascii", errors="replace").decode("ascii")
        print(fallback)

#caption shortening helper
def shorten_caption(text: str, max_len: int = 120) -> str:
    """
    Shortens long captions so they fit on screen better.
    """
    text = text.strip()
    return text if len(text) <= max_len else text[:max_len - 3] + "..."

# -------------------------------------------------------------------
# INVENTORY INTENT DETECTION
# -------------------------------------------------------------------
def is_inventory_request(text: str) -> bool:
    normalized = normalize_user_text(text)

    keywords = [
        "shoe", "shoes", "sneaker", "sneakers", "running shoes",
        "nike", "adidas", "puma", "reebok", "asics",
        "new balance", "converse", "vans", "under armour",
        "size", "budget", "price", "color", "black", "white", "red", "blue"
    ]

    return any(word in normalized for word in keywords)


# -------------------------------------------------------------------
# INVENTORY SESSION HANDLER
# -------------------------------------------------------------------
def handle_inventory_turn(user_text: str, shared_state) -> bool:
    """
    Returns True if the turn was handled by inventory mode.
    Returns False if normal Ollama chat should handle it.
    """
    inventory_mode = shared_state.get("inventory_mode", False)

    if not inventory_mode and not is_inventory_request(user_text):
        return False

    shared_state["inventory_mode"] = True

    # Initialize inventory state if missing
    if "inventory_filters" not in shared_state:
        shared_state["inventory_filters"] = inventory.empty_filters()

    if "inventory_history" not in shared_state:
        shared_state["inventory_history"] = []

    if "inventory_pending_field" not in shared_state:
        shared_state["inventory_pending_field"] = None

    filters = shared_state["inventory_filters"]
    history = shared_state["inventory_history"]
    pending_field = shared_state["inventory_pending_field"]

    history.append({"role": "user", "content": user_text})

    # If Ron asked a specific follow-up, parse directly
    if pending_field:
        extracted = inventory.extract_for_expected_field(user_text, pending_field)
        pending_field = None

        if not extracted:
            extracted = inventory.parse_with_ollama(user_text)
    else:
        extracted = inventory.parse_with_ollama(user_text)

    filters = inventory.merge_filters(filters, extracted)

    print("\n[DEBUG] Inventory filters:")
    print(json.dumps(filters, indent=2))

    missing = inventory.next_missing_field(filters)

    if missing:
        pending_field = missing
        reply = inventory.question_for_field(missing)
        shared_state["caption_text"] = f"RONNOR: {shorten_caption(reply)}"
        shared_state["caption_text"] = f"RONNOR: {reply}"
        shared_state["caption_speaker"] = "RONNOR"
        shared_state["caption_start_time"] = time.time()
        shared_state["caption_duration"] = piper_tts.estimate_speech_duration(reply)
        
        safe_print(f"Ronnor: {reply}")
        set_expression(shared_state, "speaking")

        try:
            piper_tts.speak_text(reply)
        finally:
            if shared_state["running"]:
                set_expression(shared_state, "listening")

        history.append({"role": "assistant", "content": reply})

        shared_state["inventory_filters"] = filters
        shared_state["inventory_history"] = history
        shared_state["inventory_pending_field"] = pending_field
        return True

    rows, match_type = inventory.search_inventory_relaxed(filters)

    reply = inventory.generate_natural_response(
        user_request=" ".join(msg["content"] for msg in history if msg["role"] == "user"),
        filters=filters,
        rows=rows,
        match_type=match_type,
    )

    safe_print(f"Ronnor: {reply}")
    set_expression(shared_state, "speaking")

    try:
        piper_tts.speak_text(reply)
    finally:
        if shared_state["running"]:
            set_expression(shared_state, "listening")

    if rows:
        followup = "Want to search for another pair?"
        
        #caption
        shared_state["caption_text"] = f"RONNOR: {followup}"
        shared_state["caption_speaker"] = "RONNOR"
        shared_state["caption_start_time"] = time.time()
        shared_state["caption_duration"] = piper_tts.estimate_speech_duration(followup)
        
        safe_print(f"Ronnor: {followup}")
        set_expression(shared_state, "speaking")
        
        try:
            piper_tts.speak_text(followup)
        finally:
            if shared_state["running"]:
                set_expression(shared_state, "listening")

        shared_state["inventory_mode"] = False
        shared_state["inventory_filters"] = inventory.empty_filters()
        shared_state["inventory_history"] = []
        shared_state["inventory_pending_field"] = None
    else:
        followup = "I can check similar options if you want, maybe another color, brand, or budget."
        #shared_state["caption_text"] = f"RONNOR: {shorten_caption(followup)}"
        
        shared_state["caption_text"] = f"RONNOR: {followup}"
        shared_state["caption_speaker"] = "RONNOR"
        shared_state["caption_start_time"] = time.time()
        shared_state["caption_duration"] = piper_tts.estimate_speech_duration(followup)

        safe_print(f"Ronnor: {followup}")
        set_expression(shared_state, "speaking")

        try:
            piper_tts.speak_text(followup)
        finally:
            if shared_state["running"]:
                set_expression(shared_state, "listening")

        shared_state["inventory_filters"] = filters
        shared_state["inventory_history"] = history
        shared_state["inventory_pending_field"] = None

    return True

# -------------------------------------------------------------------
# CHAT LOOP
# -------------------------------------------------------------------
def chat_with_ollama(shared_state):
    """
    Starts a chat session with Ollama.

    Behavior:
    - voice is the default input mode
    - space bar in agent_ron.py can force text input for the next turn
    - interrupt_requested lets text input take priority
    - 'bye' ends only the current chat session
    - 'exit' closes the whole app
    - every Ollama response is spoken through Piper
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

    shared_state.setdefault("inventory_mode", False)
    shared_state.setdefault("inventory_filters", inventory.empty_filters())
    shared_state.setdefault("inventory_history", [])
    shared_state.setdefault("inventory_pending_field", None)
    # Start in listening state
    shared_state["expression"] = "listening"

    # Main chat loop
    while shared_state["running"]:

        # -----------------------------------------------------------
        # GLOBAL INTERRUPT HANDLER
        # If space was pressed in the GUI, this flag is raised.
        # We clear it here and force the next input to be text.
        # -----------------------------------------------------------
        if shared_state.get("interrupt_requested", False):
            print("[CHAT] Interrupt detected.")
            shared_state["interrupt_requested"] = False
            shared_state["expression"] = "listening"

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

        # Ignore empty input
        if not user_text:
            shared_state["expression"] = "listening"
            continue

        # Show user caption and keep it until Ronnor replies
        shared_state["caption_text"] = f"USER: {shorten_caption(user_text)}"
        shared_state["caption_text"] = f"USER: {user_text}"
        shared_state["caption_speaker"] = "USER"
        shared_state["caption_start_time"] = 0.0
        shared_state["caption_duration"] = 0.0
       
        # -----------------------------------------------------------
        # END CHAT ONLY
        # This now works for typed commands and voice phrases
        # -----------------------------------------------------------
        if command in {"bye", "stop", "quit"} or is_end_chat_phrase(user_text):
            print("[CHAT] Ending chat mode.")

            # Optional spoken goodbye
            try:
                set_expression(shared_state, "speaking")
                piper_tts.speak_text("Goodbye! See you soon!")
            except Exception as e:
                print(f"[TTS] Goodbye speech failed: {e}")
            finally:
                set_expression(shared_state, "idle")

            shared_state["caption_text"] = ""
            shared_state["caption_speaker"] = ""
            shared_state["caption_start_time"] = 0.0
            shared_state["caption_duration"] = 0.0
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
                set_expression(shared_state, "idle")

            shared_state["caption_text"] = ""
            shared_state["caption_speaker"] = ""
            shared_state["caption_start_time"] = 0.0
            shared_state["caption_duration"] = 0.0
            shared_state["chat_active"] = False
            shared_state["running"] = False
            break
            

        # -----------------------------------------------------------
        # INVENTORY MODE
        # If the user is shopping for shoes, route to inventory logic
        # -----------------------------------------------------------
        try:
            if handle_inventory_turn(user_text, shared_state):
                continue
        except Exception as e:
            print(f"[INVENTORY] Error: {e}")
            shared_state["inventory_mode"] = False
            shared_state["inventory_filters"] = inventory.empty_filters()
            shared_state["inventory_history"] = []
            shared_state["inventory_pending_field"] = None
            shared_state["expression"] = "listening"

            #clear captions on errors
            shared_state["caption_text"] = ""
            shared_state["caption_speaker"] = ""
            shared_state["caption_start_time"] = 0.0
            shared_state["caption_duration"] = 0.0
            continue

        # Save user's message into conversation memory
        messages.append({"role": "user", "content": user_text})

        try:
            # -------------------------------------------------------
            # CHECK INTERRUPT BEFORE SENDING TO OLLAMA
            # -------------------------------------------------------
            if shared_state.get("interrupt_requested", False):
                print("[CHAT] Request interrupted before sending to Ollama.")
                shared_state["interrupt_requested"] = False
                shared_state["expression"] = "listening"
                continue

            # Show thinking face while Ollama is generating
            set_expression(shared_state, "thinking")

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
            
            # Show Ronnor caption and enable auto-scroll timing
            shared_state["caption_text"] = f"RONNOR: {assistant_text}"
            shared_state["caption_speaker"] = "RONNOR"
            shared_state["caption_start_time"] = time.time()
            shared_state["caption_duration"] = piper_tts.estimate_speech_duration(assistant_text)

            # Print response safely
            try:
                print(f"Ronnor: {assistant_text}")
            except UnicodeEncodeError:
                fallback_text = assistant_text.encode("ascii", errors="replace").decode("ascii")
                print(f"Ronnor: {fallback_text}")

            # Save assistant message to memory
            messages.append({"role": "assistant", "content": assistant_text})

            # -------------------------------------------------------
            # CHECK INTERRUPT BEFORE TTS
            # -------------------------------------------------------
            if shared_state.get("interrupt_requested", False):
                print("[CHAT] TTS skipped due to text-input interrupt.")
                shared_state["interrupt_requested"] = False
                shared_state["expression"] = "listening"
                continue

            # -------------------------------------------------------
            # SPEAK WITH PIPER
            # -------------------------------------------------------
            set_expression(shared_state, "speaking")
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
            set_expression(shared_state, "idle")
            shared_state["caption_text"] = ""
            shared_state["chat_active"] = False

            #clear captions on errors
            shared_state["caption_text"] = ""
            shared_state["caption_speaker"] = ""
            shared_state["caption_start_time"] = 0.0
            shared_state["caption_duration"] = 0.0
            break

        except KeyError:
            print("[CHAT] Unexpected Ollama response format.")
            set_expression(shared_state, "idle")
            shared_state["caption_text"] = ""
            shared_state["chat_active"] = False
            break

        except KeyboardInterrupt:
            print("\n[CHAT] Interrupted by user.")
            set_expression(shared_state, "idle")
            shared_state["caption_text"] = ""
            shared_state["chat_active"] = False
            shared_state["running"] = False
            break

        except Exception as e:
            print(f"[CHAT] Unexpected error: {e}")
            set_expression(shared_state, "idle")
            shared_state["caption_text"] = ""
            shared_state["chat_active"] = False

            #clear captions on errors
            shared_state["caption_text"] = ""
            shared_state["caption_speaker"] = ""
            shared_state["caption_start_time"] = 0.0
            shared_state["caption_duration"] = 0.0
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
        #clear captions on errors
        shared_state["caption_text"] = ""
        shared_state["caption_speaker"] = ""
        shared_state["caption_start_time"] = 0.0
        shared_state["caption_duration"] = 0.0
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
        #clear captions on errors
        shared_state["caption_text"] = ""
        shared_state["caption_speaker"] = ""
        shared_state["caption_start_time"] = 0.0
        shared_state["caption_duration"] = 0.0

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
        
        #clear captions on errors
        shared_state["caption_text"] = ""
        shared_state["caption_speaker"] = ""
        shared_state["caption_start_time"] = 0.0
        shared_state["caption_duration"] = 0.0


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
        "chat_active": True,
        "force_text_input": False,
        "interrupt_requested": False,
    }

    setup_ollama()
    chat_with_ollama(test_state)