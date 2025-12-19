import os
import subprocess
import shutil
from pathlib import Path

# --- 1. SETUP ĐƯỜNG DẪN FFMPEG (Giống WhisperX của bạn) ---
BASE_DIR = Path(__file__).resolve().parent
ffmpeg_bin_path = BASE_DIR / "ffmpeg-8.0.1-essentials_build" / "bin"

# Thêm vào PATH để các filter của ffmpeg có thể tìm thấy nhau
os.environ["PATH"] = str(ffmpeg_bin_path) + os.pathsep + os.environ["PATH"]
FFMPEG_EXE = str(ffmpeg_bin_path / "ffmpeg.exe")

def check_ffmpeg():
    if os.path.exists(FFMPEG_EXE):
        print(f"✅ Đã tìm thấy FFmpeg GPU tại: {FFMPEG_EXE}")
        return True
    else:
        print(f"❌ Không tìm thấy FFmpeg tại: {ffmpeg_bin_path}")
        return False

# --- 2. HÀM XỬ LÝ CHÍNH ---
def create_video_gpu(filename):
    mp3_input = f"{filename}.mp3"
    srt_input = f"{filename}_word.srt"
    mp4_output = f"{filename}_word.mp4"

    if not os.path.exists(mp3_input) or not os.path.exists(srt_input):
        print(f"❌ Thiếu file đầu vào cho: {filename}")
        return

    print(f"🚀 Đang bắt đầu render bằng GPU (NVENC): {filename}...")

    # Xử lý đường dẫn Subtitles cho FFmpeg (Quan trọng: FFmpeg trên Windows cần escape dấu : và \)
    # Ví dụ: C:\path\sub.srt -> C\\:/path/sub.srt
    srt_path_fixed = srt_input.replace("\\", "/").replace(":", "\\:")

    # Lệnh FFmpeg tối ưu:
    # -f lavfi -i color: Tạo nền đen
    # -c:v h264_nvenc: Dùng GPU NVIDIA để mã hóa video
    # -c:a copy: Giữ nguyên định dạng audio, không tốn thời gian convert lại
    cmd = [
        FFMPEG_EXE, "-y",
        "-f", "lavfi", "-i", "color=c=black:s=1920x1080:r=24", # Tạo nền đen Full HD
        "-i", mp3_input,                                      # Input nhạc
        "-vf", f"subtitles='{srt_path_fixed}':force_style='FontSize=24,PrimaryColour=&H00FFFFFF,Alignment=2'", # Chèn Sub
        "-c:v", "h264_nvenc",                                 # TĂNG TỐC GPU TẠI ĐÂY
        "-preset", "p7",                                      # p7 là chất lượng cao nhất của NVENC
        "-c:a", "copy",                                       # Copy audio gốc (siêu nhanh)
        "-shortest",                                          # Kết thúc video khi hết nhạc
        mp4_output
    ]

    try:
        # Chạy lệnh và hiển thị log đơn giản
        subprocess.run(cmd, check=True)
        print(f"\n✨ HOÀN TẤT! Video đã được lưu tại: {mp4_output}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi thực thi FFmpeg: {e}")

# --- 3. CHẠY ---
if __name__ == "__main__":
    if check_ffmpeg():
        # Tên file gốc (không bao gồm đuôi)
        MY_FILE = "How to Talk to Anyone with Confidence  English Podcast For Learning English  English Leap Podcast"
        
        create_video_gpu(MY_FILE)