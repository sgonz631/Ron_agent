import os
import numpy as np
import sounddevice as sd
import scipy.signal

from openwakeword.model import Model


class WakeWordDetector:
    WAKE_WORD_MODEL = os.path.join(os.path.dirname(__file__), "WakeupWord", "Hi_Ron.onnx")
    WAKE_WORD_THRESHOLD = 0.5

    def __init__(self, model_path=None, threshold=None, input_device=None, exact_word=True):
        """
        Initialize the wake word detector.

        Args:
            model_path: Path to .onnx or .tflite wake word model
            threshold: Detection threshold
            input_device: Audio input device name/index
            exact_word: If True, only accept detections above threshold
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

        ext = os.path.splitext(self.model_path)[1].lower()

        try:
            if ext == ".onnx":
                self.oww_model = Model(
                    wakeword_models=[self.model_path],
                    inference_framework="onnx"
                )
            elif ext == ".tflite":
                self.oww_model = Model(
                    wakeword_models=[self.model_path]
                )
            else:
                raise ValueError(f"Unsupported model format: {ext}")

            print("[INIT] Wake Word Loaded.")

        except Exception as e:
            raise RuntimeError(f"Failed to load wake word model: {e}")

    def detect(self):
        """
        Run wake word detection loop.
        Only returns "WAKE" when the exact wake word is detected above threshold.
        """
        if self.oww_model is None:
            raise RuntimeError("Wake word model not loaded")

        self.oww_model.reset()

        CHUNK_SIZE = 1280
        OWW_SAMPLE_RATE = 16000

        try:
            device_info = sd.query_devices(self.input_device, 'input') if self.input_device is not None else sd.query_devices(kind='input')
            native_rate = int(device_info["default_samplerate"])
        except Exception:
            native_rate = 48000

        use_resampling = native_rate != OWW_SAMPLE_RATE
        input_rate = native_rate if use_resampling else OWW_SAMPLE_RATE
        input_chunk_size = int(CHUNK_SIZE * (input_rate / OWW_SAMPLE_RATE)) if use_resampling else CHUNK_SIZE

        print("[WAKE WORD] Waiting for the right wake up word...")

        try:
            with sd.InputStream(
                samplerate=input_rate,
                channels=1,
                dtype="int16",
                blocksize=input_chunk_size,
                device=self.input_device
            ) as stream:
                while True:
                    data, _ = stream.read(input_chunk_size)
                    audio_data = np.frombuffer(data, dtype=np.int16)

                    if use_resampling:
                        audio_data = scipy.signal.resample(audio_data, CHUNK_SIZE).astype(np.int16)

                    self.oww_model.predict(audio_data)

                    for mdl in self.oww_model.prediction_buffer.keys():
                        confidence = list(self.oww_model.prediction_buffer[mdl])[-1]

                        if confidence > self.threshold:
                            print("[WAKE WORD] EXACT WAKE WORD DETECTED!")
                            print(f"[WAKE WORD] Model: {mdl}")
                            print(f"[WAKE WORD] Confidence: {confidence:.4f} (threshold: {self.threshold})")
                            self.oww_model.reset()
                            return "WAKE"
                        elif confidence > 0.1:
                            print(f"[WAKE WORD] False trigger detected - confidence: {confidence:.4f} (need: >{self.threshold})")

        except KeyboardInterrupt:
            print("\n[WAKE WORD] Detection interrupted by user")
            return "INTERRUPTED"
        except Exception as e:
            print(f"[WAKE WORD] Stream Error: {e}")
            raise


if __name__ == "__main__":
    detector = WakeWordDetector(exact_word=True)

    print("Starting wake word detection...")
    print("=" * 50)

    result = detector.detect()

    if result == "WAKE":
        print("=" * 50)
        print("[OK] Correct wake word detected! Proceeding...")