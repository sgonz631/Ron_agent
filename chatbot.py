# CALL OLLAMA + PIPER TTS + WHISPER STT

import ronnor_inventory
import requests
import subprocess
import time
import re
import unicodedata

# Text-to-speech helper
import piper_tts

# Speech-to-text helper
import whisper_stt

# time helpers for state tracking
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
    - remove punctuation
    - collapse spaces
    """
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_end_chat_phrase(text: str) -> bool:
    """
    Returns True if the user said a phrase that should end the chat
    and return Ronnor to wake-word mode.
    """
    normalized = normalize_user_text(text)

    end_phrases = [
        "bye",
        "bye bye",
        "thank you",
        "thank you ron",
        "thatll be all",
        "that will be all",
    ]

    return any(phrase in normalized for phrase in end_phrases)


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def update_session_preferences(session_preferences: dict, filters: dict) -> None:
    if filters.get("brand"):
        session_preferences["brand"] = filters["brand"]

    if filters.get("size") is not None:
        session_preferences["size"] = filters["size"]

    if filters.get("color"):
        session_preferences["color"] = filters["color"]

    if filters.get("tags"):
        for tag in filters["tags"]:
            if tag not in session_preferences["tags"]:
                session_preferences["tags"].append(tag)


def merge_with_session_preferences(filters: dict, session_preferences: dict) -> dict:
    if filters.get("wants_promotions"):
        return {
            "brand": filters.get("brand", ""),
            "size": filters.get("size"),
            "color": filters.get("color", ""),
            "tags": list(filters.get("tags", [])),
            "wants_promotions": True,
        }

    merged = {
        "brand": filters.get("brand") or session_preferences.get("brand"),
        "size": filters.get("size") if filters.get("size") is not None else session_preferences.get("size"),
        "color": filters.get("color") or session_preferences.get("color"),
        "tags": list(filters.get("tags", [])),
        "wants_promotions": filters.get("wants_promotions", False),
    }

    for tag in session_preferences.get("tags", []):
        if tag not in merged["tags"]:
            merged["tags"].append(tag)

    return merged


def clean_text_for_tts(text: str) -> str:
    """
    Remove stage directions and awkward pause markers before TTS.
    """
    if not text:
        return ""

    text = re.sub(r"\((?:[^)]*)\)", " ", text)
    text = re.sub(r"\s*\.\.\.\s*", ". ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def set_caption(shared_state, speaker: str, text: str, duration: float = 0.0):
    """
    Update the on-screen caption in one place.
    speaker: "USER" or "RONNOR"
    duration: use 0.0 for user captions, or speech duration for Ronnor
    """
    shared_state["caption_text"] = f"{speaker}: {text}"
    shared_state["caption_speaker"] = speaker
    shared_state["caption_start_time"] = time.time() if duration > 0 else 0.0
    shared_state["caption_duration"] = duration

def clear_caption(shared_state):
    shared_state["caption_text"] = ""
    shared_state["caption_speaker"] = ""
    shared_state["caption_start_time"] = 0.0
    shared_state["caption_duration"] = 0.0

def should_reset_inventory_preferences(user_text: str) -> bool:
    text = normalize_user_text(user_text)

    reset_phrases = [
        "what shoes are on promotion",
        "what other shoes do you have",
        "now im interested in",
        "any size or style",
        "show me all",
        "start over",
        "something else",
        "other shoes",
    ]

    return any(phrase in text for phrase in reset_phrases)

# -------------------------------------------------------------------
# CHAT LOOP
# -------------------------------------------------------------------
def chat_with_ollama(shared_state):
    """
    Starts a chat session with Ollama.
    """
    print("[CHAT] Wake word detected. Starting terminal chat.")
    print("[CHAT] Type 'bye' to end chat, 'exit' to close Ronnor.\n")

    # Conversation history for normal chat with Ollama
    messages = [
        {
            "role": "system",
            "content": (
                "You are Ronnor, a concise and helpful voice assistant running on a Raspberry Pi. "
                "Respond naturally for spoken conversation. "
                "Keep answers brief unless the user asks for more detail. "
                "Do not include stage directions, sound effects, or parenthetical cues."
            )
        }
    ]

    # Session memory for inventory follow-ups
    session_preferences = {
        "brand": None,
        "size": None,
        "color": None,
        "tags": []
    }

    shared_state["expression"] = "listening"

    while shared_state["running"]:

        if shared_state.get("interrupt_requested", False):
            print("[CHAT] Interrupt detected.")
            shared_state["interrupt_requested"] = False
            shared_state["expression"] = "listening"

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
        # Show the user's speech as a caption on screen until Ronnor responds
        set_caption(shared_state, "USER", user_text)
        
        #BYE PHRASES - END CHAT BUT KEEP RUNNING FOR NEXT WAKE WORD
        if command in {"bye", "stop", "quit"} or is_end_chat_phrase(user_text):
            print("[CHAT] Ending chat mode.")
            try:
                set_expression(shared_state, "speaking")
                piper_tts.speak_text("Goodbye! See you soon!")
            except Exception as e:
                print(f"[TTS] Goodbye speech failed: {e}")
            finally:
                set_expression(shared_state, "idle")

            clear_caption(shared_state)
            shared_state["chat_active"] = False
            break

        if command == "exit":
            print("[SYSTEM] Shutting down Ronnor.")
            try:
                shared_state["expression"] = "speaking"
                piper_tts.speak_text("Shutting down.")
            except Exception as e:
                print(f"[TTS] Shutdown speech failed: {e}")
            finally:
                set_expression(shared_state, "idle")
            clear_caption(shared_state)
            shared_state["chat_active"] = False
            shared_state["running"] = False
            break

        # -------------------------------------------------------
        # INVENTORY TOOL ROUTING
        # -------------------------------------------------------
        inventory_data = None
        try:
            raw_filters = ronnor_inventory.get_inventory_filters(user_text)

            if raw_filters:
                if should_reset_inventory_preferences(user_text):
                    session_preferences = {
                        "brand": None,
                        "size": None,
                        "color": None,
                        "tags": []
                    }
                    
                merged_filters = merge_with_session_preferences(
                    raw_filters,
                    session_preferences
                )

                merged_rows = ronnor_inventory.rank_inventory_rows(
                    ronnor_inventory.search_inventory(merged_filters),
                    merged_filters
                )

                merged_context = ronnor_inventory.build_inventory_context(
                    user_text,
                    merged_filters,
                    merged_rows
                )

                inventory_data = {
                    "filters": merged_filters,
                    "rows": merged_rows,
                    "context": merged_context,
                }

                update_session_preferences(session_preferences, raw_filters)

        except Exception as e:
            print(f"[INVENTORY] Inventory lookup failed: {e}")

        if inventory_data:
            try:
                set_expression(shared_state, "thinking")

                inventory_prompt_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are Ronnor, a helpful in-store shoe assistant. "
                            "Speak naturally and briefly, like a real store associate. "
                            "Use only the inventory facts provided. "
                            "Do not invent products, sizes, prices, availability, or promotions. "
                            "Recommend the best matches first instead of listing everything. "
                            "Keep responses to 1 to 3 short sentences. "
                            "If the request is broad, suggest one helpful way to narrow it down, such as size, style, or brand. "
                            "If the user already gave a preference like size or brand, respect it. "
                            "Avoid repeating the full user request. "
                            "Do not include stage directions, sound effects, or parenthetical cues."
                        )
                    },
                    {
                        "role": "user",
                        "content": inventory_data["context"]
                    }
                ]

                response = requests.post(
                    OLLAMA_CHAT_URL,
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": inventory_prompt_messages,
                        "stream": False
                    },
                    timeout=120
                )
                response.raise_for_status()

                data = response.json()
                inventory_reply = data["message"]["content"].strip()

            except Exception as e:
                print(f"[INVENTORY] Natural phrasing failed, using fallback: {e}")

                rows = inventory_data["rows"]
                if not rows:
                    inventory_reply = "I could not find any matching shoes in inventory."
                else:
                    top = rows[:3]
                    names = [f"{r[0]} {r[1]} in size {r[2]}" for r in top]
                    inventory_reply = (
                        f"I found {len(rows)} matching options, including "
                        + ", ".join(names)
                        + "."
                    )
            set_caption(shared_state, "RONNOR",inventory_reply,piper_tts.estimate_speech_duration(inventory_reply))

            try:
                #set_caption(shared_state, "RONNOR",inventory_reply,piper_tts.estimate_speech_duration(inventory_reply))
                print(f"Ronnor: {inventory_reply}")
            except UnicodeEncodeError:
                fallback_text = inventory_reply.encode("ascii", errors="replace").decode("ascii")
                print(f"Ronnor: {fallback_text}")

            messages.append({"role": "assistant", "content": inventory_reply})

            if shared_state.get("interrupt_requested", False):
                print("[CHAT] TTS skipped due to text-input interrupt.")
                shared_state["interrupt_requested"] = False
                shared_state["expression"] = "listening"
                continue

            set_expression(shared_state, "speaking")
            try:
                piper_tts.speak_text(clean_text_for_tts(inventory_reply))
            finally:
                if shared_state["running"]:
                    shared_state["expression"] = "listening"

            continue

        # Normal Ollama chat path
        messages.append({"role": "user", "content": user_text})

        try:
            if shared_state.get("interrupt_requested", False):
                print("[CHAT] Request interrupted before sending to Ollama.")
                shared_state["interrupt_requested"] = False
                shared_state["expression"] = "listening"
                continue

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

            set_caption(
                shared_state,
                "RONNOR",
                assistant_text,
                piper_tts.estimate_speech_duration(assistant_text)
            )

            try:
                print(f"Ronnor: {assistant_text}")
            except UnicodeEncodeError:
                fallback_text = assistant_text.encode("ascii", errors="replace").decode("ascii")
                print(f"Ronnor: {fallback_text}")

            messages.append({"role": "assistant", "content": assistant_text})

            if shared_state.get("interrupt_requested", False):
                print("[CHAT] TTS skipped due to text-input interrupt.")
                shared_state["interrupt_requested"] = False
                shared_state["expression"] = "listening"
                continue

            set_expression(shared_state, "speaking")
            try:
                piper_tts.speak_text(clean_text_for_tts(assistant_text))
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
            clear_caption(shared_state)
            shared_state["chat_active"] = False
            break

        except KeyError:
            print("[CHAT] Unexpected Ollama response format.")
            set_expression(shared_state, "idle")
            clear_caption(shared_state)
            shared_state["chat_active"] = False
            break

        except KeyboardInterrupt:
            print("\n[CHAT] Interrupted by user.")
            set_expression(shared_state, "idle")
            clear_caption(shared_state)
            shared_state["chat_active"] = False
            shared_state["running"] = False
            break

        except Exception as e:
            print(f"[CHAT] Unexpected error: {e}")
            clear_caption(shared_state)
            set_expression(shared_state, "idle")
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
    test_state = {
        "expression": "idle",
        "running": True,
        "chat_active": True,
        "force_text_input": False,
        "interrupt_requested": False,
        "caption_text": "",
        "caption_start_time": 0.0,
        "caption_duration": 0.0,
        "caption_speaker": "",
    }

    setup_ollama()
    chat_with_ollama(test_state)
