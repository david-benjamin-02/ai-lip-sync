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


def resize_video(video_path):
    """Resizes video to 720p."""
    resized_video = "resized_video.mp4"
    cmd = ["ffmpeg", "-i", video_path, "-vf", "scale=-1:720", "-y", resized_video]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        return None
    return resized_video


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
    """Transcribes audio to text using Whisper."""
    try:
        result = WHISPER_MODEL.transcribe(audio_path)
        return result["text"], result["language"]
    except Exception:
        return None, None


def translate_text(text, target_lang):
    """Translates text using GoogleTranslator."""
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception:
        return None


def validate_wav(audio_path):
    """Validates and ensures correct WAV format."""
    temp_path = "temp_valid.wav"
    cmd = ["ffmpeg", "-i", audio_path, "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", "-y", temp_path]
    subprocess.run(cmd, check=True)
    os.replace(temp_path, audio_path)


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


def synthesize_speech(text, audio_path, lang):
    """Converts text to speech."""
    output_audio = "output_synth.wav"

    if os.path.exists(output_audio):
        os.remove(output_audio)

    try:
        print(f"🟢 Running TTS for language: {lang}")

        # ✅ Load the model with `weights_only=False` to prevent weight errors
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

        print("✅ Model loaded, generating speech...")

        # ✅ Generate speech (no need for tokenizer, let `tts_to_file` handle text)
        tts.tts_to_file(
            text=text,
            speaker_wav=audio_path,  
            file_path=output_audio, 
            language=lang
        )

        print(f"🟢 Speech synthesized successfully: {output_audio}")
        return output_audio

    except Exception as e:
        print(f"❌ Error in TTS: {e}")
        return None
  

def match_audio_to_video(audio_path, video_path):
    """Adjusts the synthesized audio to match the video duration."""
    adjusted_audio = "output_synth_adjusted.wav"
    temp_audio = "temp_output.wav"  

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        print("❌ Error: Extracted audio is empty or missing.")
        return None

    # Validate WAV format
    validate_wav(audio_path)

    video_duration = get_duration(video_path)
    audio_duration = get_duration(audio_path)

    if audio_duration == 0:
        print("❌ Error: Audio duration is zero.")
        return None

    speed_factor = video_duration / audio_duration

    if 0.5 <= speed_factor <= 2.0:  # Reasonable speed adjustment
        print(f"✅ Adjusting speed with atempo: {speed_factor:.2f}")
        cmd_adjust = [
            "ffmpeg",
            "-i",
            audio_path,
            "-filter:a",
            f"atempo={speed_factor:.2f}",
            "-y",
            temp_audio,
        ]
        subprocess.run(cmd_adjust, check=True)

    elif speed_factor > 2.0:  # Extreme speed-up
        print(f"⚠️ High speed-up detected: {speed_factor:.2f}. Using resampling.")
        resampled_audio = "resampled_audio.wav"
        new_sample_rate = int(24000 * speed_factor)
        cmd_resample = [
            "ffmpeg",
            "-i",
            audio_path,
            "-ar",
            str(new_sample_rate),
            "-y",
            resampled_audio,
        ]
        subprocess.run(cmd_resample, check=True)

        # Bring back to 24kHz for compatibility
        cmd_final = ["ffmpeg", "-i", resampled_audio, "-ar", "24000", "-y", temp_audio]
        subprocess.run(cmd_final, check=True)

        os.remove(resampled_audio)

    else:  # Extreme slow down
        print("⚠️ Extreme slow-down detected. Using padding/trimming.")
        audio = AudioSegment.from_wav(audio_path)

        if audio_duration > video_duration:
            print(f"🔹 Trimming audio from {audio_duration}s to {video_duration}s")
            audio = audio[: int(video_duration * 1000)]
        else:
            print(f"🔹 Padding audio from {audio_duration}s to {video_duration}s")
            silence = AudioSegment.silent(duration=(video_duration - audio_duration) * 1000)
            audio = audio + silence

        audio.export(temp_audio, format="wav")

    os.replace(temp_audio, adjusted_audio)
    return adjusted_audio


def process_video(video, target_language, resize):
    """Main pipeline: extracts audio, transcribes, translates, synthesizes, and synchronizes."""
    video_path = video if isinstance(video, str) else video.name
    print(f"🟢 Processing video: {video_path}")

    if resize:
        video_path = resize_video(video_path)
        if video_path is None:
            return "Error resizing video", None, None

    audio_path = extract_audio(video_path)
    if audio_path is None:
        return "Error extracting audio", None, None

    transcribed_text, detected_lang = transcribe_audio(audio_path)
    print(f"🟢 Transcription: {transcribed_text}")
    if transcribed_text is None:
        return "Error transcribing audio", None, None

    translated_text = translate_text(transcribed_text, target_language)
    print(f"🟢 Translation: {translated_text}")
    if translated_text is None:
        return "Error translating text", None, None

    tts_output = synthesize_speech(translated_text, audio_path, target_language)
    print(f"🟢 TTS Output: {tts_output}")
    if tts_output is None:
        return translated_text, None, None

    adjusted_audio = match_audio_to_video(tts_output, video_path)
    print(f"🟢 Adjusted Audio: {adjusted_audio}")
    if adjusted_audio is None:
        return translated_text, None, None

    print("✅ Process completed successfully!")
    return translated_text, adjusted_audio, video_path


iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Dropdown(label="Target Language", choices=["en", "es", "fr", "de"], value="en"),
        gr.Checkbox(label="Resize to 720p", value=False),
    ],
    outputs=[gr.Text(label="Translated Text"), gr.Audio(label="Synchronized Speech"), gr.Video(label="Final Video")],
)

iface.launch()
