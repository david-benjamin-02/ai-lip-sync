import gradio as gr
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
import pickle
from moviepy.editor import concatenate_videoclips, VideoFileClip, CompositeVideoClip

# Fix pickle issues
pickle.loads(pickle.dumps(XttsArgs()))
pickle.loads(pickle.dumps(XttsConfig()))
pickle.loads(pickle.dumps(XttsAudioConfig()))
pickle.loads(pickle.dumps(BaseDatasetConfig()))

WHISPER_MODEL = whisper.load_model("base")

# Step 1: Extract Audio
def extract_audio(video_path):
    audio_path = "output_audio.wav"
    if os.path.exists(audio_path):
        os.remove(audio_path)
    cmd = ["ffmpeg", "-i", video_path, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", audio_path]
    subprocess.run(cmd, check=True)
    return audio_path

# Step 2: Transcribe Audio
def transcribe_audio(audio_path):
    result = WHISPER_MODEL.transcribe(audio_path, word_timestamps=True)
    return result["text"], result["segments"], result["language"]

# Step 3: Translate

def translate_text(text, target_lang):
    return GoogleTranslator(source="auto", target=target_lang).translate(text)

# Step 4: TTS Synthesis

def synthesize_speech_with_alignment(text, audio_path, lang, timestamps):
    output_audio = "output_synth.wav"
    temp_resampled_audio = "output_synth_converted.wav"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

    original_audio = AudioSegment.from_wav(audio_path)
    final_audio = AudioSegment.silent(duration=len(original_audio))

    for segment in timestamps:
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        duration_ms = end_ms - start_ms

        translated_segment = GoogleTranslator(source="auto", target=lang).translate(segment["text"])
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
            gen_audio += silence

        final_audio = final_audio.overlay(gen_audio, position=start_ms)

    final_audio.export(output_audio, format="wav")
    convert_cmd = ["ffmpeg", "-y", "-i", output_audio, "-ar", "16000", temp_resampled_audio]
    subprocess.run(convert_cmd, check=True)
    return temp_resampled_audio

# Step 5: Apply Wav2Lip

def apply_wav2lip(original_video, translated_audio):
    python_exec = r"D:\\django\\ai-lipsync\\Wav2Lip\\venv_wav\\Scripts\\python.exe"
    output_video = "lip_synced_video.mp4"
    cmd = [
        python_exec, "Wav2Lip/inference.py",
        "--checkpoint_path", "Wav2Lip/checkpoints/wav2lip_gan.pth",
        "--face", original_video,
        "--audio", translated_audio,
        "--outfile", output_video
    ]
    subprocess.run(cmd, check=True)
    return output_video

# Sign Language Generation
def transcribe_sign_audio(audio_path):
    result = WHISPER_MODEL.transcribe(audio_path)
    return result["text"]

def text_to_gloss(text):
    translated = GoogleTranslator(source='auto', target='en').translate(text)
    gloss = translated.upper().replace(".", "").replace(",", "").split()
    normalized = [word.replace("'", "").capitalize() for word in gloss]
    return normalized

def generate_sign_video(gloss_list):
    clips = []
    for word in gloss_list:
        path = f"sign_clips/{word.lower()}.mp4"
        if os.path.exists(path):
            clips.append(VideoFileClip(path).resize((320, 240)))
    if not clips:
        return None
    final = concatenate_videoclips(clips, method="compose")
    output_path = "sign_output.mp4"
    final.write_videofile(output_path, codec="libx264", audio=False)
    return output_path

def audio_to_sign_language(audio_file):
    audio_path = audio_file if isinstance(audio_file, str) else "input_audio.wav"
    if not isinstance(audio_file, str):
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())
    text = transcribe_sign_audio(audio_path)
    glosses = text_to_gloss(text)
    sign_video = generate_sign_video(glosses)
    return text, sign_video

# Overlay sign video

def overlay_sign_language_on_lipsync(lipsynced_video_path, sign_video_path, output_path="final_output.mp4"):
    main_clip = VideoFileClip(lipsynced_video_path)
    sign_clip = VideoFileClip(sign_video_path).resize(height=main_clip.h // 4)
    position = (main_clip.w - sign_clip.w - 10, main_clip.h - sign_clip.h - 10)
    sign_clip = sign_clip.set_position(position).set_start(0).set_duration(main_clip.duration)
    final = CompositeVideoClip([main_clip, sign_clip])
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")
    return output_path

# Main pipeline
def process_video(video, target_language):
    video_path = video if isinstance(video, str) else video.name
    if not os.path.exists(video_path):
        with open("input_video.mp4", "wb") as f:
            f.write(video.read())
        video_path = "input_video.mp4"

    audio_path = extract_audio(video_path)
    transcribed_text, timestamps, detected_lang = transcribe_audio(audio_path)
    translated_text = translate_text(transcribed_text, target_language)
    aligned_audio = synthesize_speech_with_alignment(translated_text, audio_path, target_language, timestamps)
    lip_video = apply_wav2lip(video_path, aligned_audio)
    _, sign_video = audio_to_sign_language(audio_path)
    if not sign_video:
        return translated_text, aligned_audio, "Sign language video generation failed."
    final_output = overlay_sign_language_on_lipsync(lip_video, sign_video)
    return translated_text, aligned_audio, final_output

iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Dropdown(label="Target Language", choices=["en", "es", "fr", "de"], value="en"),
    ],
    outputs=[
        gr.Text(label="Translated Text"),
        gr.Audio(label="Synchronized Speech"),
        gr.Video(label="Final Video with Sign Language")
    ],
    title="Lip Sync + Sign Language Translation"
)

iface.launch()