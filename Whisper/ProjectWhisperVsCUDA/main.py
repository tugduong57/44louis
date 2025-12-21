import os
import sys
import shutil
from pathlib import Path
import time  # Đo giờ
import warnings # Tắt cảnh báo
import logging # Tắt log hệ thống
import contextlib # Dùng để chặn output cứng đầu

# --- 0. CẤU HÌNH TẮT CẢNH BÁO (Làm sạch màn hình) ---
# Tắt warnings python
warnings.simplefilter("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

# Tắt logging của các thư viện con (Cấu hình tận gốc)
logging.getLogger().setLevel(logging.ERROR) # Tắt toàn bộ Info log hệ thống
for logger_name in ["whisperx", "lightning", "pytorch_lightning", "pyannote", "speechbrain"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Hàm chặn output thừa (những dòng chữ đỏ không thể tắt bằng logging)
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        old_stdout = sys.stdout
        sys.stderr = devnull
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stderr = old_stderr
            sys.stdout = old_stdout

# 1. Setup FFmpeg
BASE_DIR = Path(__file__).resolve().parent
ffmpeg_bin_path = BASE_DIR / "resourse4whisper" / "ffmpeg-8.0.1-essentials_build" / "bin"
os.environ["PATH"] = str(ffmpeg_bin_path) + os.pathsep + os.environ["PATH"]

ffmpeg_exe = shutil.which("ffmpeg")
if ffmpeg_exe:
    print(f"✅ FFmpeg Ready.") 
else:
    print(f"❌ Error: Không tìm thấy FFmpeg tại {ffmpeg_bin_path}")

import torch
import whisperx
import gc
from typing import Optional

# --- 2. FIX LỖI PYTORCH 2.6 ---
def setup_torch():
    _original_torch_load = torch.load
    def new_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False 
        return _original_torch_load(*args, **kwargs)
    torch.load = new_torch_load
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

setup_torch()

def format_timestamp(seconds: float) -> str:
    x = int(seconds)
    msec = int((seconds - x) * 1000)
    hours = x // 3600
    minutes = (x % 3600) // 60
    seconds = x % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{msec:03d}"

# --- 3. CLASS CHÍNH ---
class WhisperTranscriber:
    def __init__(self, model_size="small", device=None, compute_type="int8", batch_size=16):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        print(f"🔹 Model: {model_size} | Device: {self.device} | Type: {compute_type}")
        try:
            # Thử load "im lặng" trước
            with suppress_output():
                self.model = whisperx.load_model(
                    model_size,                     # large-v2, medium, small 
                    self.device,                    # "cuda", "cpu"
                    compute_type=compute_type       # "float16" - mặc định, "int8": cho GPU yếu, "float32": bắt buộc trên CPU
                    # language = "vi", "en", None   # nếu biết chắc -> tiết kiệm thời gian load
                    '''
                    # Tăng độ chính xác & Cung cấp ngữ cảnh
                    asr_options={
                        "beam_size": 10,    # Tìm kiếm kỹ hơn (mặc định 5)
                        "initial_prompt": "Hội thoại tiếng Việt, chủ đề công nghệ thông tin, lập trình Python." 
                    },
                    
                    # Tinh chỉnh cắt giọng nói (nếu thấy bị mất chữ đầu câu thì giảm vad_onset)

                    Bạn nên thêm một hàm tiền xử lý (pre-process) dùng FFmpeg để tạo ra file tạm thời đã được chuẩn hóa, sau đó mới đưa vào Whisper.
                    
                    vad_options={
                        "vad_onset": 0.4,   # Nhạy hơn mặc định một chút
                        "vad_offset": 0.35
                    }
                    Xác suất (Prob)
                      1.0 |          ________ đỉnh cao trào (nói to)
                          |         /        \
                      0.8 |        /          \
                          |       /            \
                      0.5 |------/--------------\------------------ (Mặc định Onset) -> Bắt đầu cắt TẠI ĐÂY
                      0.4 |-----/----------------\----------------- (Onset Tùy chỉnh) -> Bắt đầu cắt SỚM HƠN (Lấy được chữ đầu)
                          |    /                  \
                      0.3 |---/--------------------\--------------- (Offset) -> Kết thúc cắt (Cho phép giọng nhỏ dần)
                          |  /                      \
                      0.0 |_/                        \____________
                    Time:  (Tiếng thở/nhỏ)   (Nói rõ)      (Nói nhỏ/kết câu)
                    '''
                    )
        except Exception: 
            # Thử load lại "công khai" để hiện lỗi hoặc thanh download
            self.model = whisperx.load_model(model_size, self.device, compute_type=compute_type)
            
        print(f"✅ Model ({model_size}) Ready.")

    def transcribe_file(self, audio_path: str, language: Optional[str] = None):
        if not os.path.exists(audio_path):
            print(f"❌ File not found: {audio_path}")
            return False

        print(f"🎧 Processing: {os.path.basename(audio_path)}")
        
        # B1. Transcribe
        audio = whisperx.load_audio(audio_path)
        result = self.model.transcribe(audio, batch_size=self.batch_size, language=language)
        # self.model.transcribe(audio, ...)
        # task="translate",       # Dịch thẳng sang tiếng Anh (nếu cần)
        # num_workers=4,          # Dùng 4 luồng CPU để chuẩn bị dữ liệu nhanh hơn
        # print_progress=False    # Tắt thanh loading bar mặc định của thư viện

        # B2. Align
        try:
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
            del model_a; del metadata; gc.collect(); torch.cuda.empty_cache()
        except Exception:
            return False

        # B3. Save
        self._save_srt(result["segments"], audio_path, is_word_level=False)
        self._save_srt(result["segments"], audio_path, is_word_level=True)
        return True

    def _save_srt(self, segments, audio_path, is_word_level=False):
        suffix = "_word.srt" if is_word_level else ".srt"
        output_file = model_size + "_" + audio_path.rsplit('.', 1)[0] + suffix
        with open(output_file, "w", encoding="utf-8") as f:
            counter = 1
            if is_word_level:
                for seg in segments:
                    if 'words' not in seg: continue
                    for w in seg['words']:
                        if 'start' in w and 'end' in w:
                            f.write(f"{counter}\n{format_timestamp(w['start'])} --> {format_timestamp(w['end'])}\n{w['word'].strip()}\n\n")
                            counter += 1
            else:
                for seg in segments:
                    f.write(f"{counter}\n{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n{seg['text'].strip()}\n\n")
                    counter += 1

    def close(self):
        del self.model; gc.collect(); torch.cuda.empty_cache()

# --- 4. HÀM LOG ---
def write_log(file_name, size_mb, duration, model_size, compute_type, batch_size):
    with open("Log_Time4Mp3_2_SRT.txt", "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] File: {file_name[file_name.find('.'):]} | Size: {size_mb:.2f} MB | Time: {duration:.2f}s | Model: {model_size} | Compute_type: {compute_type} | Batch_size: {batch_size}\n")
    print(f"📝 Log saved: history_log.txt | Time: {duration:.2f}s")

# --- 5. CHẠY ---
if __name__ == "__main__":
    MY_AUDIO = r"How to Talk to Anyone with Confidence  English Podcast For Learning English  English Leap Podcast.mp4"
    
    compute_type="int8";  
    batch_size=16;  
    model_size="small";

    start = time.time()
    app = WhisperTranscriber(compute_type=compute_type, batch_size=batch_size, model_size=model_size)
    #    compute_type = "int8" ; batch_size = 16 (hoặc float16 và 32)
    # Model         |   VRAM    |   Tốc độ      |   Độ chính xác    
    # small         |   ~2GB    |   nhanh       |   Khá
    # medium        |   ~5GB    |   vừa phải    |   Tốt (bắt buộc với Tiếng Việt)
    # large-v2      |   ~8-10GB |   chậm        |   Rất tốt
    # large-v3      |   ~10GB   |   chậm nhất   |   Tốt nhất
    

    if app.transcribe_file(MY_AUDIO, language="en"):
        duration = time.time() - start
        size_mb = os.path.getsize(MY_AUDIO) / 1048576 if os.path.exists(MY_AUDIO) else 0
        write_log(os.path.basename(MY_AUDIO), size_mb, duration, model_size, compute_type, batch_size)
        
    app.close()