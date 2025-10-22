import gradio as gr
import torch
import whisper
from deep_translator import GoogleTranslator
from TTS.api import TTS
import subprocess

# Load models
whisper_model = whisper.load_model("base")
translator = GoogleTranslator(source='auto', target='en')

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_enabled = device == "cuda"

# Load TTS model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=gpu_enabled)

def process_video(video):
    video_path = video  # ✅ FIXED: video is a string (file path)

    # Extract audio using FFmpeg
    audio_path = "output_audio.wav"
    ffmpeg_cmd = ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", "-y", audio_path]
    subprocess.run(ffmpeg_cmd, check=True)  # ✅ FIXED: safer subprocess call

    # Transcribe audio
    result = whisper_model.transcribe(audio_path)
    text = result["text"]
    lang = result["language"]

    # Translate text
    translated_text = translator.translate(text)

    # Convert text to speech
    tts_output = "output_synth.wav"
    tts.tts_to_file(text=translated_text, speaker_wav=audio_path, file_path=tts_output, language="en")

    return translated_text, tts_output

# Gradio UI
iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=[gr.Text(label="Transcribed & Translated Text"), gr.Audio(label="Synthesized Speech")]
)

iface.launch()
