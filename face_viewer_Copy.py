import os
import sys
import pygame

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


def main():
    pygame.init()
    pygame.display.set_caption("Ron Face Viewer")

    # windowed mode first for testing
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    expressions = load_face_folders(FACES_ROOT)
    expression_names = list(expressions.keys())

    print("Loaded states:", expression_names)

    expression_index = 0
    frame_index = 0
    autoplay = True
    last_frame_change = pygame.time.get_ticks()

    current_expression = expression_names[expression_index]
    current_image = load_and_scale_image(
        expressions[current_expression][frame_index],
        (SCREEN_WIDTH, SCREEN_HEIGHT)
    )

    running = True
    while running:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RIGHT:
                    frame_index = (frame_index + 1) % len(expressions[current_expression])
                elif event.key == pygame.K_LEFT:
                    frame_index = (frame_index - 1) % len(expressions[current_expression])
                elif event.key == pygame.K_DOWN:
                    expression_index = (expression_index + 1) % len(expression_names)
                    current_expression = expression_names[expression_index]
                    frame_index = 0
                    print("State:", current_expression)
                elif event.key == pygame.K_UP:
                    expression_index = (expression_index - 1) % len(expression_names)
                    current_expression = expression_names[expression_index]
                    frame_index = 0
                    print("State:", current_expression)
                elif event.key == pygame.K_SPACE:
                    autoplay = not autoplay
                    print("Autoplay:", autoplay)

                current_image = load_and_scale_image(
                    expressions[current_expression][frame_index],
                    (SCREEN_WIDTH, SCREEN_HEIGHT)
                )

        if autoplay and now - last_frame_change >= FRAME_DELAY_MS:
            frame_index = (frame_index + 1) % len(expressions[current_expression])
            current_image = load_and_scale_image(
                expressions[current_expression][frame_index],
                (SCREEN_WIDTH, SCREEN_HEIGHT)
            )
            last_frame_change = now

        draw_centered(screen, current_image, BACKGROUND_COLOR)
        clock.tick(30)

    pygame.quit()
    sys.exit()

#Wake Up Word#
class WakeWordDetector:
    WAKE_WORD_MODEL = "WakeupWord/Hi_Ron.onnx"
    WAKE_WORD_THRESHOLD = 0.5
    
    def __init__(self, model_path=None, threshold=None, input_device=None, exact_word=True):
        """
        Initialize the wake word detector.
        
        Args:
            model_path: Path to .onnx wake word model (default: ./wakeword.onnx)
            threshold: Detection threshold (default: 0.5)
            input_device: Audio input device name/index (default: None = system default)
            exact_word: If True, only accept detections above threshold (default: True)
        """
        self.model_path = model_path or self.WAKE_WORD_MODEL
        self.threshold = threshold or self.WAKE_WORD_THRESHOLD
        self.input_device = input_device
        self.exact_word = exact_word
        self.oww_model = None
        
        self.load_model()
    
    def load_model(self):
        """Load the OpenWakeWord model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Wake word model not found: {self.model_path}")
        
        print(f"[INIT] Loading Wake Word Model: {self.model_path}")
        try:
            self.oww_model = Model(wakeword_model_paths=[self.model_path])
            print("[INIT] Wake Word Loaded.")
        except TypeError:
            # Fallback for older API
            try:
                self.oww_model = Model(wakeword_models=[self.model_path])
                print("[INIT] Wake Word Loaded (New API).")
            except Exception as e:
                raise RuntimeError(f"Failed to load wake word model: {e}")
    
    def detect(self):
        """
        Run wake word detection loop.
        Only returns "WAKE" when the EXACT wake word is detected above threshold.
        Shows "Waiting for the right wake up word..." while listening.
        
        Returns: "WAKE" when exact wake word detected
        """
        if self.oww_model is None:
            raise RuntimeError("Wake word model not loaded")
        
        self.oww_model.reset()
        
        CHUNK_SIZE = 1280
        OWW_SAMPLE_RATE = 16000
        
        # Get native sample rate
        try:
            device_info = sd.query_devices(kind='input')
            native_rate = int(device_info['default_samplerate'])
        except:
            native_rate = 48000
        
        # Determine if resampling is needed
        use_resampling = (native_rate != OWW_SAMPLE_RATE)
        input_rate = native_rate if use_resampling else OWW_SAMPLE_RATE
        input_chunk_size = int(CHUNK_SIZE * (input_rate / OWW_SAMPLE_RATE)) if use_resampling else CHUNK_SIZE
        
        print("[WAKE WORD] Waiting for the right wake up word...")
        
        try:
            with sd.InputStream(samplerate=input_rate, channels=1, dtype='int16',
                              blocksize=input_chunk_size, device=self.input_device) as stream:
                while True:
                    # Check for stdin input (optional CLI interrupt)
                    #rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
                    #if rlist:
                     #   sys.stdin.readline()
                     #   return "CLI"
                    
                    # Read audio chunk
                    data, _ = stream.read(input_chunk_size)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    
                    # Resample if needed
                    if use_resampling:
                        audio_data = scipy.signal.resample(audio_data, CHUNK_SIZE).astype(np.int16)
                    
                    # Predict
                    prediction = self.oww_model.predict(audio_data)
                    
                    # Check threshold for EXACT wake word match
                    for mdl in self.oww_model.prediction_buffer.keys():
                        confidence = list(self.oww_model.prediction_buffer[mdl])[-1]
                        
                        # Only accept if confidence exceeds threshold
                        if confidence > self.threshold:
                            model_name = mdl
                            print(f"\n[WAKE WORD] ✓ EXACT WAKE WORD DETECTED!")
                            print(f"[WAKE WORD] Model: {model_name}")
                            print(f"[WAKE WORD] Confidence: {confidence:.4f} (threshold: {self.threshold})")
                            self.oww_model.reset()
                            return "WAKE"
                        else:
                            # Show confidence for debugging (optional)
                            if confidence > 0.1:  # Only show if there's some detection
                                print(f"[WAKE WORD] False trigger detected - confidence: {confidence:.4f} (need: >{self.threshold})")
        
        except KeyboardInterrupt:
            print("\n[WAKE WORD] Detection interrupted by user")
            return "INTERRUPTED"
        except Exception as e:
            print(f"[WAKE WORD] Stream Error: {e}")
            raise



if __name__ == "__main__":
    main()
    detector = WakeWordDetector(exact_word=True)
    
    print("Starting wake word detection...")
    print("=" * 50)
    
    result = detector.detect()
    
    if result == "WAKE":
        print("=" * 50)
        print("✓ Correct wake word detected! Proceeding...")
    # For testing on cmd: py -3.11 face_viewer_Copy.py
    # For testing terminal vscode: python face_viewer_Copy.py
