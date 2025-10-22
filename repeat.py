import gradio as gr
import torch
import whisper
import subprocess
import os
from deep_translator import GoogleTranslator
from TTS.api import TTS
from pydub import AudioSegment
from TTS.tts.models.xtts import XttsArgs
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig

torch.serialization.add_safe_globals(
    [XttsArgs, XttsConfig, XttsAudioConfig, BaseDatasetConfig]
)

# Load Whisper model globally
WHISPER_MODEL = whisper.load_model("base")

def extract_audio(video_path):
    """Extracts audio from video ensuring it is in WAV format."""
    audio_path = "output_audio.wav"
    if os.path.exists(audio_path):
        os.remove(audio_path)

    cmd = [
        "ffmpeg", "-i", video_path,
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", audio_path,
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return audio_path
    except subprocess.CalledProcessError:
        return None

def transcribe_audio(audio_path):
    """Transcribes audio to text using Whisper while preserving timestamps."""
    try:
        result = WHISPER_MODEL.transcribe(audio_path, word_timestamps=True)
        return result["text"], result["segments"], result["language"]
    except Exception:
        return None, None, None

def translate_text(text, target_lang):
    """Translates text using GoogleTranslator."""
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception:
        return None

def synthesize_speech(text, audio_path, lang, timestamps):
    """Generates speech while ensuring silence and timing match the original audio."""
    output_audio = "output_synth.wav"
    temp_audio = "temp_synth.wav"

    if os.path.exists(output_audio):
        os.remove(output_audio)

    try:
        print(f"🟢 Running TTS for language: {lang}")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        
        print("✅ Model loaded, generating speech...")
        tts.tts_to_file(text=text, speaker_wav=audio_path, file_path=temp_audio, language=lang)
        
        original_audio = AudioSegment.from_wav(audio_path)
        generated_speech = AudioSegment.from_wav(temp_audio)
        
        final_audio = AudioSegment.silent(duration=len(original_audio))
        for segment in timestamps:
            start_time = int(segment["start"] * 1000)
            speech_part = generated_speech[: int(segment["end"] * 1000 - start_time)]
            final_audio = final_audio.overlay(speech_part, position=start_time)
        
        final_audio.export(output_audio, format="wav")
        return output_audio
    except Exception as e:
        print(f"❌ Error in TTS: {e}")
        return None

def apply_wav2lip(original_video, translated_audio):
    """Applies Wav2Lip to synchronize translated audio with lip movements."""
    python_exec = r"D:\django\ai-lipsync\Wav2Lip\venv_wav\Scripts\python.exe"
    output_video = "lip_synced_video.mp4"
    cmd = [
        python_exec, "Wav2Lip/inference.py",
        "--checkpoint_path", "Wav2Lip/checkpoints/wav2lip_gan.pth",
        "--face", original_video,
        "--audio", translated_audio,
        "--outfile", output_video
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return output_video
    except subprocess.CalledProcessError:
        return None

def process_video(video, target_language):
    """Main pipeline: extracts audio, transcribes, translates, synthesizes, and lip-syncs."""
    video_path = video if isinstance(video, str) else video.name
    print(f"🟢 Processing video: {video_path}")

    audio_path = extract_audio(video_path)
    if audio_path is None:
        return "Error extracting audio", None, None

    transcribed_text, timestamps, detected_lang = transcribe_audio(audio_path)
    print(f"🟢 Transcription: {transcribed_text}")
    if transcribed_text is None:
        return "Error transcribing audio", None, None

    translated_text = translate_text(transcribed_text, target_language)
    print(f"🟢 Translation: {translated_text}")
    if translated_text is None:
        return "Error translating text", None, None

    tts_output = synthesize_speech(translated_text, audio_path, target_language, timestamps)
    print(f"🟢 TTS Output: {tts_output}")
    if tts_output is None:
        return translated_text, None, None

    final_video = apply_wav2lip(video_path, tts_output)
    print(f"✅ Final video with lip-synced translated audio: {final_video}")

    return translated_text, tts_output, final_video

iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Dropdown(label="Target Language", choices=["en", "es", "fr", "de"], value="en"),
    ],
    outputs=[
        gr.Text(label="Translated Text"),
        gr.Audio(label="Synchronized Speech"),
        gr.Video(label="Final Lip-Synced Video")
    ],
)

iface.launch()
