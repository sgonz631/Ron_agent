import os
import sys

import os

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


FACES_ROOT = "/home/pi/Ronnor/RONNOR/faces/faces - Copy"  
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 480
FRAME_DELAY_MS = 700
BACKGROUND_COLOR = (0, 0, 0)
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


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
    pygame.display.flip()


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
        clock.tick(30)

    pygame.quit()

def main():
    chatbot.setup_ollama()

    shared_state = {
        "expression": "idle",
        "running": True,
        "chat_active": False
    }

    gui_thread = threading.Thread(target=launch_GUI, args=(shared_state,), daemon=True)
    gui_thread.start()

    detector = WakeWordDetector(exact_word=True)

    while shared_state["running"]:
        shared_state["expression"] = "idle"
        print("[SYSTEM] Waiting for wake word...")

        result = detector.detect()

        if result == "WAKE" and shared_state["running"]:
            shared_state["chat_active"] = True
            chatbot.chat_with_ollama(shared_state)
            shared_state["chat_active"] = False
            shared_state["expression"] = "idle"

    print("[SYSTEM] Shutting down.")
        

if __name__ == "__main__":
    main()

    # For testing on cmd: py -3.11 face_viewer_Copy.py
    # For testing terminal vscode: python face_viewer_Copy.py
