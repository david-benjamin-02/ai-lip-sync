import gradio as gr
import torch
import whisper
import subprocess
import os
import sys
from deep_translator import GoogleTranslator
from TTS.api import TTS
from TTS.tts.models.xtts import XttsArgs
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig

# torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig])
torch.serialization.add_safe_globals(
    [XttsArgs, XttsConfig, XttsAudioConfig, BaseDatasetConfig]
)

# Load Whisper model globally for efficiency
WHISPER_MODEL = whisper.load_model("base")


def resize_video(video_path):
    """Resizes the video to 720p."""
    resized_video = "resized_video.mp4"
    cmd = ["ffmpeg", "-i", video_path, "-vf", "scale=-1:720", "-y", resized_video]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        return f"Error resizing video: {e}"
    return resized_video


def extract_audio(video_path):
    """Extracts audio from video."""
    audio_path = "output_audio.wav"
    if os.path.exists(audio_path):
        os.remove(audio_path)
    cmd = ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", "-y", audio_path]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        return f"Error extracting audio: {e}"
    return audio_path


def transcribe_audio(audio_path):
    """Transcribes audio to text using Whisper."""
    try:
        result = WHISPER_MODEL.transcribe(audio_path)
        return result["text"], result["language"]
    except Exception as e:
        return f"Error transcribing audio: {e}", None


def translate_text(text, target_lang):
    """Translates text using GoogleTranslator with error handling."""
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception as e:
        return f"Error translating text: {e}"


def synthesize_speech(text, audio_path, lang):
    """Converts text to speech using Coqui TTS."""
    output_audio = "output_synth.wav"
    if os.path.exists(output_audio):
        os.remove(output_audio)

    try:
        # ✅ Initialize the TTS model (No need for .load_model())
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

        # ✅ Directly synthesize speech (No separate load_model call)
        tts.tts_to_file(
            text=text, speaker_wav=audio_path, file_path=output_audio, language=lang
        )

        return output_audio
    except Exception as e:
        print(f"❌ Error in speech synthesis: {e}")  # Debugging output
        return "Error in speech synthesis"


def lip_sync(video_path, audio_path, output_video):
    python_exec = r"D:\django\ai-lipsync\Wav2Lip\venv_wav\Scripts\python.exe"
    wav2lip_script = "Wav2Lip/inference.py"
    checkpoint_path = "Wav2Lip/checkpoints/wav2lip_gan.pth"

    cmd = [
        python_exec, wav2lip_script,
        "--checkpoint_path", checkpoint_path,
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", output_video,
        "--face_det_batch_size", "1"  # ✅ Reduce memory usage
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if "Face not detected!" in result.stderr:
            print("⚠️ No face detected! Merging audio with original video instead.")
            output_video = merge_audio_with_video(video_path, audio_path)  # ✅ Merge audio & video
        elif result.returncode != 0:
            print(f"❌ Wav2Lip failed! Error: {result.stderr}")
            return None  # Indicate failure
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

    return output_video  # ✅ Return the output video path

def merge_audio_with_video(video_path, audio_path):
    output_path = "output_no_lipsync.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac", "-strict", "experimental",
        output_path
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print("✅ Audio merged successfully!")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Error merging audio: {e}")
        return None
    
    
def process_video(video, target_language, resize, lip_sync_quality):
    """Processes the video: extracts audio, transcribes, translates, synthesizes, and lip-syncs."""
    video_path = video if isinstance(video, str) else video.name

    if resize:
        video_path = resize_video(video_path)
        if "Error" in video_path:
            return video_path, None, None  # ✅ Return None instead of error strings

    audio_path = extract_audio(video_path)
    if "Error" in audio_path:
        return audio_path, None, None

    transcribed_text, detected_lang = transcribe_audio(audio_path)
    if "Error" in transcribed_text:
        return transcribed_text, None, None

    translated_text = translate_text(transcribed_text, target_language)
    if "Error" in translated_text:
        return translated_text, None, None

    tts_output = synthesize_speech(translated_text, audio_path, target_language)
    if "Error" in tts_output:
        return translated_text, None, None  # ✅ Ensuring correct return values

    lipsynced_video = lip_sync(video_path, tts_output, lip_sync_quality)
    if "Error" in lipsynced_video:  # ✅ Check if lipsync failed
        return translated_text, None, None  # ✅ Avoid passing error string to Gradio

    return translated_text, tts_output, lipsynced_video  # ✅ Ensure valid return values


iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Dropdown(
            label="Target Language",
            choices=[
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "pl",
                "tr",
                "ru",
                "nl",
                "cs",
                "ar",
                "zh-cn",
            ],
            value="en",
        ),
        gr.Checkbox(label="Resize to 720p", value=False),
        gr.Radio(label="Lip Sync Quality", choices=["normal", "high"], value="normal"),
    ],
    outputs=[
        gr.Text(label="Translated Text"),
        gr.Audio(label="Synthesized Speech"),
        gr.Video(label="Lip-Synced Video"),
    ],
)

iface.launch()
# https://youtube.com/shorts/yn4ke4wr6oA?si=JJ5bDae5_0aWqZTx
