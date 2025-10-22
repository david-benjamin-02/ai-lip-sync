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

torch.serialization.add_safe_globals([XttsArgs, XttsConfig, XttsAudioConfig, BaseDatasetConfig])

WHISPER_MODEL = whisper.load_model("base")

def extract_audio(video_path):
    audio_path = "output_audio.wav"
    if os.path.exists(audio_path):
        os.remove(audio_path)
    cmd = ["ffmpeg", "-i", video_path, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", audio_path]
    try:
        subprocess.run(cmd, check=True)
        return audio_path
    except subprocess.CalledProcessError:
        return None

def transcribe_audio(audio_path):
    try:
        result = WHISPER_MODEL.transcribe(audio_path, word_timestamps=True)
        return result["text"], result["segments"], result["language"]
    except Exception:
        return None, None, None

def translate_text(text, target_lang):
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception:
        return None

def synthesize_speech_with_alignment(text, audio_path, lang, timestamps):
    output_audio = "output_synth.wav"
    temp_resampled_audio = "output_synth_converted.wav"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

    original_audio = AudioSegment.from_wav(audio_path)
    final_audio = AudioSegment.silent(duration=len(original_audio))

    print("🟢 Synthesizing and aligning segments...")
    for segment in timestamps:
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        duration_ms = end_ms - start_ms

        translated_segment = GoogleTranslator(source="auto", target=lang).translate(segment["text"])

        try:
            temp_segment_path = "temp_segment.wav"
            tts.tts_to_file(
                text=translated_segment,
                speaker_wav=audio_path,
                file_path=temp_segment_path,
                language=lang,
            )
            gen_audio = AudioSegment.from_wav(temp_segment_path)

            if len(gen_audio) > duration_ms:
                gen_audio = gen_audio[:duration_ms]
            else:
                silence = AudioSegment.silent(duration=duration_ms - len(gen_audio))
                gen_audio = gen_audio + silence

            final_audio = final_audio.overlay(gen_audio, position=start_ms)
        except Exception as e:
            print(f"❌ Error synthesizing segment: {e}")
            continue

    # Export synthesized audio
    final_audio.export(output_audio, format="wav")

    # Convert to 16000 Hz for Wav2Lip
    convert_cmd = [
        "ffmpeg", "-y", "-i", output_audio, "-ar", "16000", temp_resampled_audio
    ]
    subprocess.run(convert_cmd, check=True)

    return temp_resampled_audio

def apply_wav2lip(original_video, translated_audio):
    python_exec = r"D:\django\ai-lipsync\Wav2Lip\venv_wav\Scripts\python.exe"
    output_video = "lip_synced_video.mp4"

    if not os.path.exists(original_video):
        print(f"❌ Original video not found: {original_video}")
        return None
    if not os.path.exists(translated_audio):
        print(f"❌ Translated audio not found: {translated_audio}")
        return None

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
    except subprocess.CalledProcessError as e:
        print("❌ Wav2Lip failed:", e)
        return None

def process_video(video, target_language):
    
    video_path = video if isinstance(video, str) else video.name

# If video was uploaded through Gradio, we must save it
    if not os.path.exists(video_path):
        with open("input_video.mp4", "wb") as f:
            f.write(video.read())
        video_path = "input_video.mp4"

    print(f"🟢 Processing video: {video_path}")

    audio_path = extract_audio(video_path)
    if audio_path is None:
        return "Error extracting audio", None, None

    transcribed_text, timestamps, detected_lang = transcribe_audio(audio_path)
    if transcribed_text is None:
        return "Error transcribing audio", None, None

    translated_text = translate_text(transcribed_text, target_language)
    if translated_text is None:
        return "Error translating text", None, None

    aligned_audio = synthesize_speech_with_alignment(translated_text, audio_path, target_language, timestamps)
    if aligned_audio is None:
        return translated_text, None, None

    final_video = apply_wav2lip(video_path, aligned_audio)
    if final_video is None:
        return translated_text, aligned_audio, "Lip-sync failed. Ensure face is visible throughout video."

    return translated_text, aligned_audio, final_video

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
