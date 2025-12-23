import streamlit as st
import socket
import time
import pandas as pd
import numpy as np
import os
import sys
import shutil
import subprocess  # Cần thiết cho FFmpeg
from pathlib import Path
import warnings
import logging
import contextlib
import torch
import whisperx
import gc
import re
from typing import Optional
from datetime import datetime

# --- 0. CẤU HÌNH TRANG & MÔI TRƯỜNG ---
st.set_page_config(
    page_title="WhisperX Multi-User Tool",
    page_icon="🎙️",
    layout="wide"
)

# --- CẤU HÌNH TẮT CẢNH BÁO ---
warnings.simplefilter("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
logging.getLogger().setLevel(logging.ERROR)
for logger_name in ["whisperx", "lightning", "pytorch_lightning", "pyannote", "speechbrain"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# --- SETUP ĐƯỜNG DẪN FFMPEG ---
BASE_DIR = Path(__file__).resolve().parent
ffmpeg_bin_path = BASE_DIR / "resourse4whisper" / "ffmpeg-8.0.1-essentials_build" / "bin"
FFMPEG_EXE = str(ffmpeg_bin_path / "ffmpeg.exe")
os.environ["PATH"] = str(ffmpeg_bin_path) + os.pathsep + os.environ["PATH"]

def check_ffmpeg():
    return os.path.exists(FFMPEG_EXE)

# --- HÀM HỖ TRỢ HỆ THỐNG ---
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stderr, old_stdout = sys.stderr, sys.stdout
        sys.stderr, sys.stdout = devnull, devnull
        try: yield
        finally:
            sys.stderr, sys.stdout = old_stderr, old_stdout

def setup_torch():
    _original_torch_load = torch.load
    def new_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False 
        return _original_torch_load(*args, **kwargs)
    torch.load = new_torch_load

setup_torch()

def format_timestamp(seconds: float) -> str:
    x = int(seconds)
    msec = int((seconds - x) * 1000)
    hours, minutes, seconds = x // 3600, (x % 3600) // 60, x % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{msec:03d}"

# --- LOGIC GỘP SUBTITLE THEO MAX TIME & DẤU CÂU ---
def merge_segments_logic(segments, max_time=15.0):
    """
    Gộp các segment dựa trên dấu chấm câu hoặc đạt ngưỡng thời gian max_time.
    """
    merged_segments = []
    if not segments: return merged_segments

    curr_start = segments[0]['start']
    curr_text = []
    
    for i, seg in enumerate(segments):
        curr_text.append(seg['text'].strip())
        duration = seg['end'] - curr_start
        
        # Điều kiện ngắt: có dấu câu kết thúc HOẶC quá thời gian
        ends_with_punc = seg['text'].strip().endswith(('.', '?', '!'))
        is_too_long = duration >= max_time
        
        if ends_with_punc or is_too_long or i == len(segments) - 1:
            merged_segments.append({
                'start': curr_start,
                'end': seg['end'],
                'text': " ".join(curr_text)
            })
            if i < len(segments) - 1:
                curr_start = segments[i+1]['start']
                curr_text = []
                
    return merged_segments

# --- HÀM QUẢN LÝ USER & FILE ---
ROOT_WORKSPACE = "user_workspaces"

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name).strip().replace(" ", "_")

def get_user_workspace(username):
    safe_name = sanitize_filename(username) or "default_user"
    user_path = os.path.join(ROOT_WORKSPACE, safe_name)
    os.makedirs(user_path, exist_ok=True)
    return user_path

def save_uploaded_file(uploaded_file, user_path):
    try:
        file_path = os.path.join(user_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception:
        return None

# --- LOGIC RENDER VIDEO GPU ---
def render_video_gpu(audio_input, srt_input, output_path, status_container=None):
    if not os.path.exists(audio_input) or not os.path.exists(srt_input):
        if status_container: status_container.error(f"❌ Thiếu file đầu vào")
        return False

    srt_path_fixed = srt_input.replace("\\", "/").replace(":", "\\:")
    cmd = [
        FFMPEG_EXE, "-y",
        "-f", "lavfi", "-i", "color=c=black:s=1920x1080:r=24",
        "-i", audio_input,
        "-vf", f"subtitles='{srt_path_fixed}':force_style='FontSize=24,PrimaryColour=&H00FFFFFF,Alignment=2'",
        "-c:v", "h264_nvenc", "-preset", "p1", "-c:a", "copy", "-shortest",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        if status_container: status_container.error(f"❌ Lỗi FFmpeg: {e}")
        return False

# --- LOGIC WHISPER ---
@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size, device, compute_type):
    with suppress_output():
        return whisperx.load_model(model_size, device, compute_type=compute_type)

class WhisperTranscriber:
    def __init__(self, model_size="small", device=None, compute_type="int8", batch_size=16):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model_size = model_size
        self.compute_type = compute_type

    def transcribe_process(self, model, audio_path: str, language=None, status_container=None, max_merge_time=15.0):
        if not os.path.exists(audio_path): return False, 0, None

        start_time = time.time()
        audio = whisperx.load_audio(audio_path)
        
        if status_container: status_container.write("🎧 Đang Transcribe...")
        result = model.transcribe(audio, batch_size=self.batch_size, language=language)

        if status_container: status_container.write("⏳ Đang Align...")
        try:
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
            del model_a; del metadata; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            if status_container: status_container.warning(f"⚠️ Align lỗi: {e}")

        if status_container: status_container.write("💾 Đang xử lý và lưu các phiên bản SRT...")
        
        # 1. Lưu bản Word-level (Karaoke)
        word_srt_path = self._save_srt(result["segments"], audio_path, is_word_level=True)
        
        # 2. Lưu bản segment gốc từ WhisperX
        self._save_srt(result["segments"], audio_path, is_word_level=False)
        
        # 3. MỚI: Gộp theo logic Max Time và lưu bản Final
        merged_segs = merge_segments_logic(result["segments"], max_time=max_merge_time)
        final_srt_path = self._save_srt(merged_segs, audio_path, suffix="_DaGopTheoMaxTime.srt")
        
        duration = time.time() - start_time
        return True, duration, word_srt_path, final_srt_path

    def _save_srt(self, segments, audio_path, is_word_level=False, suffix=None):
        if suffix is None:
            suffix = "_word.srt" if is_word_level else ".srt"
            
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_file = os.path.join(os.path.dirname(audio_path), f"{self.model_size}_{base_name}{suffix}")
        
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
                    if 'start' in seg and 'end' in seg:
                        f.write(f"{counter}\n{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n{seg['text'].strip()}\n\n")
                        counter += 1
        return output_file

def write_log_user(user_path, file_name, size_mb, duration, model_size, compute_type):
    log_file = os.path.join(user_path, "user_log.txt")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"File: {file_name} | Size: {size_mb:.2f} MB | Time: {duration:.2f}s | Model: {model_size} | Type: {compute_type} [{time.strftime('%Y-%m-%d %H:%M:%S')}] \n ")

# --- UI STREAMLIT ---
with st.sidebar:
    st.header("👤 Định danh")
    tester_name_input = st.text_input("Nhập tên/ID của bạn:", "Duong")
    user_workspace = get_user_workspace(tester_name_input)
    st.caption(f"📂 Workspace: `{user_workspace}`")
    
    st.divider()
    st.header("⚙️ Whisper Config")
    model_size = st.selectbox("Model Size:", ["small", "medium", "large-v2", "large-v3"], index=0)
    compute_type = st.selectbox("Compute Type:", ["int8", "float16", "float32"], index=0)
    language_opt = st.selectbox("Ngôn ngữ (Optional):", ["Auto Detect", "vi", "en"])
    
    # MỚI: Tùy chỉnh Max Time gộp sub
    # max_merge_time = st.slider("Max Time gộp Sub (giây):", 5.0, 30.0, 15.0)
    # \
    max_merge_time = st.slider(
        "Max Time gộp Sub (giây):", 
        min_value=8.0, 
        max_value=15.0, 
        value=11.0, 
        step=0.5,
        help="Giới hạn thời gian tối đa cho một đoạn sub đã gộp."
    )

    st.divider()
    if torch.cuda.is_available():
        st.success(f"GPU: {torch.cuda.get_device_name(0)}")
    if check_ffmpeg(): st.success("✅ FFmpeg Ready")

st.title(f"🎙️ WhisperX Multi-User Tool")
tab1, tab2, tab3 = st.tabs(["📤 Upload & Chạy", "📂 Quản lý File & Kết quả", "📝 User Logs"])

with tab1:
    col1, col2 = st.columns([2, 1])
    file_path_to_process = None
    with col1:
        uploaded_file = st.file_uploader("Chọn file Audio/Video", type=['mp3', 'wav', 'mp4', 'm4a', 'mkv'])
        if uploaded_file:
            saved_path = save_uploaded_file(uploaded_file, user_workspace)
            if saved_path:
                st.success(f"✅ Đã upload: `{os.path.basename(saved_path)}`")
                file_path_to_process = saved_path

    if 'transcribe_success' not in st.session_state: st.session_state.transcribe_success = False

    run_btn = st.button("🚀 BẮT ĐẦU XỬ LÝ", type="primary", use_container_width=True, disabled=(file_path_to_process is None))

    if run_btn and file_path_to_process:
        with st.status("Đang xử lý...", expanded=True) as status:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = load_whisper_model(model_size, device, compute_type)
                app = WhisperTranscriber(model_size=model_size, device=device, compute_type=compute_type)
                
                lang_arg = None if language_opt == "Auto Detect" else language_opt
                success, duration, word_srt, final_srt = app.transcribe_process(
                    model, file_path_to_process, language=lang_arg, status_container=status, max_merge_time=max_merge_time
                )
                
                if success:
                    size_mb = os.path.getsize(file_path_to_process) / 1048576
                    write_log_user(user_workspace, os.path.basename(file_path_to_process), size_mb, duration, model_size, compute_type)
                    st.session_state.transcribe_success = True
                    st.session_state.current_srt_path = word_srt
                    st.session_state.final_srt_path = final_srt
                    st.session_state.current_audio_path = file_path_to_process
                    status.update(label="✅ Xử lý hoàn tất!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")

    if st.session_state.transcribe_success:
        st.divider()
        st.subheader("🎥 Render Video")
        col_v1, col_v2 = st.columns(2)
        
        # Nút render bản Word-level
        if col_v1.button("🎞️ Render Karaoke (Từng chữ)"):
            out_vid = os.path.join(user_workspace, f"karaoke_{int(time.time())}.mp4")
            if render_video_gpu(st.session_state.current_audio_path, st.session_state.current_srt_path, out_vid):
                st.video(out_vid)
        
        # Nút render bản Final (Đã gộp câu)
        if col_v2.button("🎞️ Render Bản Final (Gộp câu)"):
            out_vid_final = os.path.join(user_workspace, f"final_{int(time.time())}.mp4")
            if render_video_gpu(st.session_state.current_audio_path, st.session_state.final_srt_path, out_vid_final):
                st.video(out_vid_final)

# === TAB 2 & 3 (Giữ nguyên logic cũ của bạn) ===
with tab2:
    st.subheader(f"📂 Workspace: {tester_name_input}")
    if st.button("🔄 Làm mới"): st.rerun()
    if os.path.exists(user_workspace):
        files = sorted([f for f in os.listdir(user_workspace) if os.path.isfile(os.path.join(user_workspace, f))], 
                       key=lambda x: os.path.getmtime(os.path.join(user_workspace, x)), reverse=True)
        for f in files:
            f_path = os.path.join(user_workspace, f)
            c1, c2 = st.columns([3, 1])
            c1.write(f"📄 {f}")
            with open(f_path, "rb") as fd:
                c2.download_button("⬇️ Tải", fd, file_name=f, key=f"dl_{f}")
            if f.endswith(".mp4"): st.video(f_path)
            elif f.endswith((".srt", ".txt")): 
                with st.expander("Xem nội dung"):
                    st.code(open(f_path, "r", encoding="utf-8").read())
            st.divider()

with tab3:
    user_log_file = os.path.join(user_workspace, "user_log.txt")
    if st.button("Xóa Log"): 
        open(user_log_file, "w").close()
        st.rerun()
    if os.path.exists(user_log_file):
        for line in reversed(open(user_log_file, "r", encoding="utf-8").readlines()):
            st.code(line.strip(), language="bash")