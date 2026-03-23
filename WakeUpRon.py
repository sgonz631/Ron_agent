import os
import numpy as np
import sounddevice as sd
import scipy.signal
import select
import sys

# Wake word model
import openwakeword
from openwakeword.model import Model

class WakeWordDetector:
    WAKE_WORD_MODEL = "/home/pi/Ronnor/RONNOR/WakeupWord/Hi_Ron.onnx"
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


# Example usage
if __name__ == "__main__":
    detector = WakeWordDetector(exact_word=True)
    
    print("Starting wake word detection...")
    print("=" * 50)
    
    result = detector.detect()
    
    if result == "WAKE":
        print("=" * 50)
        print("✓ Correct wake word detected! Proceeding...")