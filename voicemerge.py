import gradio as gr
import torch
import whisper
import subprocess
import os
from deep_translator import GoogleTranslator
from TTS.api import TTS
from pydub import AudioSegment, silence
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
        "ffmpeg",
        "-i",
        video_path,
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-y",
        audio_path,
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


def get_duration(file_path):
    """Returns the duration of an audio or video file."""
    cmd = [
        "ffprobe",
        "-i",
        file_path,
        "-show_entries",
        "format=duration",
        "-v",
        "quiet",
        "-of",
        "csv=p=0",
    ]
    return float(subprocess.check_output(cmd).decode("utf-8").strip())


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
        tts.tts_to_file(
            text=text,
            speaker_wav=audio_path,
            file_path=temp_audio,
            language=lang,
        )

        # Load original audio to get silent regions
        original_audio = AudioSegment.from_wav(audio_path)
        generated_speech = AudioSegment.from_wav(temp_audio)

        final_audio = AudioSegment.silent(duration=len(original_audio))

        for segment in timestamps:
            start_time = int(segment["start"] * 1000)  # Convert to milliseconds
            speech_part = generated_speech[: int(segment["end"] * 1000 - start_time)]
            final_audio = final_audio.overlay(speech_part, position=start_time)

        final_audio.export(output_audio, format="wav")
        return output_audio

    except Exception as e:
        print(f"❌ Error in TTS: {e}")
        return None


def merge_audio_to_video(video_path, audio_path):
    """Merges the translated audio into the video and removes the original audio."""
    final_video = "translated_video.mp4"

    cmd_replace_audio = [
        "ffmpeg",
        "-i", video_path,  # Input video
        "-i", audio_path,   # Translated audio
        "-map", "0:v:0",    # Keep only the video stream from original
        "-map", "1:a:0",    # Use the translated audio
        "-c:v", "copy",     # Copy video codec (no re-encoding)
        "-c:a", "aac",      # Encode audio in AAC format
        "-b:a", "192k",     # Set audio bitrate
        "-y", final_video,  # Output file
    ]

    try:
        subprocess.run(cmd_replace_audio, check=True)
        return final_video
    except subprocess.CalledProcessError:
        return None


def process_video(video, target_language):
    """Main pipeline: extracts audio, transcribes, translates, synthesizes, and synchronizes."""
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

    final_video = merge_audio_to_video(video_path, tts_output)
    print(f"✅ Final video with translated audio: {final_video}")

    return translated_text, tts_output, final_video


iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Dropdown(label="Target Language", choices=["en", "es", "fr", "de"], value="en"),
    ],
    outputs=[gr.Text(label="Translated Text"), gr.Audio(label="Synchronized Speech"), gr.Video(label="Final Video")],
)

iface.launch()
