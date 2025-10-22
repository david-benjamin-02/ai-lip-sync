from pydub import AudioSegment

audio_path = "output.wav"  # Ensure this is the correct path

try:
    audio = AudioSegment.from_wav(audio_path)
    print("Audio successfully loaded!")
except Exception as e:
    print("Error loading audio:", e)
