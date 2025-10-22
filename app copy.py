import gradio as gr
import torch
import whisper
import subprocess
import os
from deep_translator import GoogleTranslator
from TTS.api import TTS

def resize_video(video_path):
    """Resizes the video to 720p."""
    resized_video = "resized_video.mp4"
    cmd = ["ffmpeg", "-i", video_path, "-vf", "scale=-1:720", "-y", resized_video]
    subprocess.run(cmd, check=True)
    return resized_video

def extract_audio(video_path):
    """Extracts audio from video."""
    audio_path = "output_audio.wav"
    cmd = ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", "-y", audio_path]
    subprocess.run(cmd, check=True)
    return audio_path

def transcribe_audio(audio_path):
    """Transcribes audio to text using Whisper."""
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"], result["language"]

def translate_text(text, target_lang):
    """Translates text using GoogleTranslator."""
    translator = GoogleTranslator(source='auto', target=target_lang)
    return translator.translate(text)

def synthesize_speech(text, audio_path, lang):
    """Converts text to speech using Coqui TTS."""
    output_audio = "output_synth.wav"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_enabled = device == "cuda"
    tts = TTS("tts_models/multilingual/multi-dataset/your_tts", gpu=gpu_enabled)
    tts.tts_to_file(text=text, speaker_wav=audio_path, file_path=output_audio, language=lang)
    return output_audio

def lip_sync(video_path, audio_path, quality="normal"):
    """Performs lip-syncing on the video using Wav2Lip or Video-Retalking."""
    output_video = "output_lipsync.mp4"
    if quality == "high":
        cmd = ["python", "video-retalking/inference.py", "--face", video_path, "--audio", audio_path, "--outfile", output_video]
    else:
        cmd = ["python", "Wav2Lip/inference.py", "--checkpoint_path", "Wav2Lip/checkpoints/wav2lip_gan.pth", "--face", video_path, "--audio", audio_path, "--outfile", output_video]
    subprocess.run(cmd, check=True)
    return output_video

def process_video(video, target_language, resize, lip_sync_quality):
    """Processes the video by extracting audio, transcribing, translating, synthesizing, and lip-syncing."""
    video_path = video
    if resize:
        video_path = resize_video(video)
    
    audio_path = extract_audio(video_path)
    transcribed_text, detected_lang = transcribe_audio(audio_path)
    translated_text = translate_text(transcribed_text, target_language)
    tts_output = synthesize_speech(translated_text, audio_path, target_language)
    lipsynced_video = lip_sync(video_path, tts_output, lip_sync_quality)
    
    return translated_text, tts_output, lipsynced_video

iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Dropdown(label="Target Language", choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn"], value="en"),
        gr.Checkbox(label="Resize to 720p", value=False),
        gr.Radio(label="Lip Sync Quality", choices=["normal", "high"], value="normal")
    ],
    outputs=[
        gr.Text(label="Translated Text"),
        gr.Audio(label="Synthesized Speech"),
        gr.Video(label="Lip-Synced Video")
    ]
)

iface.launch()
