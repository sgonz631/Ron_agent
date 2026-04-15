import os
import sys
import signal

os.environ["LANG"] = "en_US.UTF-8"
os.environ["LANGUAGE"] = "en_US.UTF-8"
os.environ["LC_CTYPE"] = "en_US.UTF-8"
os.environ.pop("LC_ALL", None)

import sys
import pygame
import threading #one thread for Ollama and another for the images
 
# Wake word model
from testwakeword import WakeWordDetector

#Call Ollama Chatbot
import chatbot

#TTS
#import piper_tts

#Time Helpers
from state_utils import set_expression, print_state_summary

#for captions
import time
import textwrap

FACES_ROOT = "/home/pi/Ronnor/RONNOR/faces/faces - Copy"  
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 480
FRAME_DELAY_MS = 700
BACKGROUND_COLOR = (0, 0, 0)
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

#captions
CAPTION_FONT_SIZE = 28
CAPTION_BOX_HEIGHT = 140
CAPTION_PADDING = 16
CAPTION_TEXT_COLOR = (255, 255, 255)
CAPTION_BOX_COLOR = (0, 0, 0)
CAPTION_BOX_ALPHA = 180
CAPTION_MAX_VISIBLE_LINES = 3

def load_face_folders(root_folder):
    expressions = {}

    if not os.path.exists(root_folder):
        print(f"Error: folder '{root_folder}' does not exist.")
        sys.exit(1)

    for folder_name in sorted(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, folder_name)

        if os.path.isdir(folder_path):
            images = []
            for file_name in sorted(os.listdir(folder_path)):
                if file_name.lower().endswith(SUPPORTED_EXTENSIONS):
                    images.append(os.path.join(folder_path, file_name))

            if images:
                expressions[folder_name] = images

    if not expressions:
        print("Error: no valid expression folders with images were found.")
        sys.exit(1)

    return expressions


def load_and_scale_image(image_path, screen_size):
    image = pygame.image.load(image_path).convert_alpha()
    img_w, img_h = image.get_size()
    screen_w, screen_h = screen_size

    scale = min(screen_w / img_w, screen_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    image = pygame.transform.smoothscale(image, (new_w, new_h))
    return image


def draw_centered(screen, image, bg_color):
    screen.fill(bg_color)
    rect = image.get_rect(center=screen.get_rect().center)
    screen.blit(image, rect)

def draw_caption(screen, shared_state):
    """
    Draw one caption at a time.
    If the caption has more than 3 lines and Ronnor is speaking,
    auto-scroll downward as the speech progresses.
    """
    caption_text = shared_state.get("caption_text", "")
    if not caption_text:
        return

    screen_width, screen_height = screen.get_size()
    font = pygame.font.SysFont(None, CAPTION_FONT_SIZE)

    # Wrap text into multiple lines
    max_chars_per_line = 42
    lines = textwrap.wrap(caption_text, width=max_chars_per_line)

    # Create caption background
    caption_surface = pygame.Surface((screen_width, CAPTION_BOX_HEIGHT), pygame.SRCALPHA)
    caption_surface.fill((*CAPTION_BOX_COLOR, CAPTION_BOX_ALPHA))

    total_lines = len(lines)
    visible_lines = CAPTION_MAX_VISIBLE_LINES

    # Default: start from the first line
    start_line = 0

    # Auto-scroll only while Ronnor is speaking and caption is long
    if (
        shared_state.get("caption_speaker") == "RONNOR"
        and total_lines > visible_lines
        and shared_state.get("caption_duration", 0) > 0
    ):
        elapsed = time.time() - shared_state.get("caption_start_time", 0)
        duration = shared_state.get("caption_duration", 1)

        progress = max(0.0, min(elapsed / duration, 1.0))

        max_start_line = total_lines - visible_lines
        start_line = int(progress * max_start_line)

    # Draw only the visible lines
    y = CAPTION_PADDING
    for line in lines[start_line:start_line + visible_lines]:
        text_surface = font.render(line, True, CAPTION_TEXT_COLOR)
        caption_surface.blit(text_surface, (CAPTION_PADDING, y))
        y += font.get_linesize() + 4

    screen.blit(caption_surface, (0, screen_height - CAPTION_BOX_HEIGHT))

def launch_GUI(shared_state):
    pygame.init()
    pygame.display.set_caption("Ron Face Viewer")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    expressions = load_face_folders(FACES_ROOT)

    if "idle" not in expressions:
        print("Error: 'idle' folder not found.")
        pygame.quit()
        return

    frame_index = 0
    last_frame_change = pygame.time.get_ticks()
    current_expression = shared_state["expression"]

    current_image = load_and_scale_image(
        expressions[current_expression][frame_index],
        (SCREEN_WIDTH, SCREEN_HEIGHT)
    )

    running = True
    while running and shared_state["running"]:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                shared_state["running"] = False

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
                shared_state["running"] = False

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                print("[INPUT] Space bar pressed -> switching to text input.")
                shared_state["interrupt_requested"] = True
                shared_state["force_text_input"] = True
                shared_state["expression"] = "listening"
               
        new_expression = shared_state["expression"]

        if new_expression not in expressions:
            new_expression = "idle"

        if new_expression != current_expression:
            current_expression = new_expression
            frame_index = 0
            current_image = load_and_scale_image(
                expressions[current_expression][frame_index],
                (SCREEN_WIDTH, SCREEN_HEIGHT)
            )

        if now - last_frame_change >= FRAME_DELAY_MS:
            frame_index = (frame_index + 1) % len(expressions[current_expression])
            current_image = load_and_scale_image(
                expressions[current_expression][frame_index],
                (SCREEN_WIDTH, SCREEN_HEIGHT)
            )
            last_frame_change = now

        draw_centered(screen, current_image, BACKGROUND_COLOR)
        draw_caption(screen, shared_state)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

def main():
    chatbot.setup_ollama()

    shared_state = {
        "expression": "idle",
        "running": True,
        "chat_active": False,
        "force_text_input": False,
        "interrupt_requested": False,
        "caption_text": "",
        "caption_start_time": 0.0,
        "caption_duration": 0.0,
        "caption_speaker": "",
    }

    # -----------------------------------------------------------
    # SAFE TERMINAL EXIT HANDLER
    # Ctrl+C or kill signal will shut the app down cleanly
    # -----------------------------------------------------------
    def safe_shutdown(signum=None, frame=None):
        print("\n[SYSTEM] Safe shutdown requested from terminal...")
        shared_state["running"] = False
        shared_state["chat_active"] = False
        shared_state["interrupt_requested"] = True
        shared_state["expression"] = "idle"

    signal.signal(signal.SIGINT, safe_shutdown)   # Ctrl+C
    signal.signal(signal.SIGTERM, safe_shutdown)  # kill / system stop

    gui_thread = threading.Thread(target=launch_GUI, args=(shared_state,))
    gui_thread.start()

    detector = WakeWordDetector(exact_word=True)

    try:
        while shared_state["running"]:
            shared_state["expression"] = "idle"
            print("[SYSTEM] Waiting for wake word...")

            result = detector.detect()

            if result == "WAKE" and shared_state["running"]:
                shared_state["chat_active"] = True
                chatbot.chat_with_ollama(shared_state)
                shared_state["chat_active"] = False
                shared_state["expression"] = "idle"

    except KeyboardInterrupt:
        # Extra protection if Ctrl+C reaches here directly
        safe_shutdown()

    finally:
        shared_state["running"] = False
        shared_state["chat_active"] = False
        shared_state["expression"] = "idle"

        # Wait briefly for GUI thread to exit cleanly
        if gui_thread.is_alive():
            gui_thread.join(timeout=2)

        print("[SYSTEM] Shutting down.")
        print_state_summary(shared_state)

if __name__ == "__main__":
    main()

    # For testing on cmd: py -3.11 face_viewer_Copy.py
    # For testing terminal vscode: python face_viewer_Copy.py
