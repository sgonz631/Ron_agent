import wave
from piper import PiperVoice

if __name__ == "__main__":
    voice = PiperVoice.load("/home/pi/Ronnor/RONNOR/voices/en_US-bryce-medium.onnx")
    with wave.open("test.wav", "wb") as wav_file:
        voice.synthesize_wav("Welcome to the world of speech synthesis!", wav_file)
        