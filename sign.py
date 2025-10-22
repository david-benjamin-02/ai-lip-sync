import subprocess

def generate_sign_language_video_placeholder(duration_sec=10):
    print("🟢 Generating sign language video placeholder...")

    font_path = "C\\:/Windows/Fonts/arial.ttf"  # Note the slash style for FFmpeg
    output_path = "sign_language_video.mp4"

    # Use no quotes around the text, just escape spaces with '\ '
    drawtext_filter = (
        f"drawtext=fontfile={font_path}:"
        f"text='Sign\\ Language\\ Placeholder':"
        f"fontcolor=white:fontsize=24:x=(w-text_w)/2:y=(h-text_h)/2"
    )

    ffmpeg_cmd = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', f'color=c=black:s=320x240:d={duration_sec}',
        '-vf', drawtext_filter,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y', output_path
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print("✅ Placeholder video generated:", output_path)
    except subprocess.CalledProcessError as e:
        print("❌ Error creating sign language placeholder video:", e)

generate_sign_language_video_placeholder()
