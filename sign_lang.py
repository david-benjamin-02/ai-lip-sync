import gradio as gr
import whisper
import os
from moviepy.editor import concatenate_videoclips, VideoFileClip
from deep_translator import GoogleTranslator

# Load Whisper model once
whisper_model = whisper.load_model("base")

# Convert audio to text
def transcribe_audio(audio_path):
    audio = whisper_model.transcribe(audio_path)
    return audio["text"]

# Convert to sign glosses
def text_to_gloss(text):
    translated = GoogleTranslator(source='auto', target='en').translate(text)
    gloss = translated.upper().replace(".", "").replace(",", "").split()
    normalized_glosses = [word.replace("'", "").capitalize() for word in gloss]
    return normalized_glosses

# Generate video by combining sign clips
def generate_sign_video(gloss_list):
    clips = []
    for i, word in enumerate(gloss_list):
        file = f"sign_clips/{word.lower()}.mp4"
        print(f"🔍 [{i+1}/{len(gloss_list)}] Checking: {file}")
        if os.path.exists(file):
            try:
                clip = VideoFileClip(file).resize((320, 240))
                clips.append(clip)
                print(f"✅ Added: {file}")
            except Exception as e:
                print(f"⚠️ Error loading {file}: {e}")
        else:
            print(f"❌ Not found: {file}")

    if not clips:
        return None

    final_clip = concatenate_videoclips(clips, method="compose")
    output_path = "sign_output.mp4"
    final_clip.write_videofile(output_path, codec="libx264", audio=False)
    final_clip.close()
    return output_path

# Main function
def audio_to_sign_language(audio_file):
    if isinstance(audio_file, str):
        audio_path = audio_file
    else:
        audio_path = "input_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())

    print("🟢 Transcribing...")
    text = transcribe_audio(audio_path)
    print("🟢 Text:", text)

    gloss_list = text_to_gloss(text)
    print("🟢 Glosses:", gloss_list)

    print("🟢 Generating sign language video...")
    sign_video = generate_sign_video(gloss_list)

    if not sign_video:
        return text, None

    return text, sign_video

# Gradio Interface
iface = gr.Interface(
    fn=audio_to_sign_language,
    inputs=[gr.Audio(type="filepath", label="Upload Audio")],
    outputs=[
        gr.Text(label="Transcribed Text"),
        gr.Video(label="Sign Language Video")
    ],
    title="Audio to Sign Language Converter"
)

iface.launch()
