import streamlit as st
import socket
import time
import pandas as pd
import numpy as np
import os
import sys
import shutil
from pathlib import Path
import warnings
import logging
import contextlib
import torch
import whisperx
import gc
import re
from typing import Optional

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
        old_stderr = sys.stderr
        old_stdout = sys.stdout
        sys.stderr = devnull
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stderr = old_stderr
            sys.stdout = old_stdout

# Setup FFmpeg
BASE_DIR = Path(__file__).resolve().parent
ffmpeg_bin_path = BASE_DIR / "resourse4whisper" / "ffmpeg-8.0.1-essentials_build" / "bin"
os.environ["PATH"] = str(ffmpeg_bin_path) + os.pathsep + os.environ["PATH"]

# Fix PyTorch 2.6
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
    hours = x // 3600
    minutes = (x % 3600) // 60
    seconds = x % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{msec:03d}"

# --- HÀM QUẢN LÝ USER & FILE ---
ROOT_WORKSPACE = "user_workspaces"

def sanitize_filename(name):
    """Làm sạch tên để tạo folder an toàn"""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip().replace(" ", "_")

def get_user_workspace(username):
    """Tạo và lấy đường dẫn thư mục làm việc của user"""
    safe_name = sanitize_filename(username)
    if not safe_name: safe_name = "default_user"
    user_path = os.path.join(ROOT_WORKSPACE, safe_name)
    if not os.path.exists(user_path):
        os.makedirs(user_path)
    return user_path

def save_uploaded_file(uploaded_file, user_path):
    """Lưu file upload vào thư mục user"""
    try:
        file_path = os.path.join(user_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        return None

# --- LOGIC WHISPER (ĐƯỢC CACHE) ---
@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size, device, compute_type):
    with suppress_output():
        try:
            model = whisperx.load_model(
                model_size, 
                device, 
                compute_type=compute_type
            )
        except Exception:
             model = whisperx.load_model(model_size, device, compute_type=compute_type)
    return model

class WhisperTranscriber:
    def __init__(self, model_size="small", device=None, compute_type="int8", batch_size=16):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model_size = model_size
        self.compute_type = compute_type

    def transcribe_process(self, model, audio_path: str, language: Optional[str] = None, status_container=None):
        if not os.path.exists(audio_path):
            if status_container: status_container.error(f"❌ File not found: {audio_path}")
            return False, 0

        start_time = time.time()
        
        # B1. Transcribe
        if status_container: status_container.write("🎧 Đang Transcribe (Nhận dạng giọng nói)...")
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=self.batch_size, language=language)

        # B2. Align
        if status_container: status_container.write("⏳ Đang Align (Đồng bộ thời gian)...")
        try:
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
            del model_a; del metadata; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            if status_container: status_container.warning(f"⚠️ Align lỗi (nhưng vẫn có sub): {e}")

        # B3. Save
        if status_container: status_container.write("💾 Đang lưu file SRT...")
        self._save_srt(result["segments"], audio_path, is_word_level=False)
        self._save_srt(result["segments"], audio_path, is_word_level=True)
        
        duration = time.time() - start_time
        return True, duration

    def _save_srt(self, segments, audio_path, is_word_level=False):
        suffix = "_word.srt" if is_word_level else ".srt"
        # Output lưu ngay tại thư mục chứa file audio (tức là thư mục user)
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
                    f.write(f"{counter}\n{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n{seg['text'].strip()}\n\n")
                    counter += 1

def write_log_user(user_path, file_name, size_mb, duration, model_size, compute_type):
    """Ghi log riêng cho từng user"""
    log_file = os.path.join(user_path, "user_log.txt")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"File: {file_name} | Size: {size_mb:.2f} MB | Time: {duration:.2f}s | Model: {model_size} | Type: {compute_type} [{time.strftime('%Y-%m-%d %H:%M:%S')}] \n ")

# --- UI STREAMLIT ---

# 1. SIDEBAR
with st.sidebar:
    st.header("👤 Định danh (User Workspace)")
    tester_name_input = st.text_input("Nhập tên/ID của bạn:", "Duong")
    
    # Tạo/Lấy thư mục user ngay lập tức
    user_workspace = get_user_workspace(tester_name_input)
    st.caption(f"📂 Workspace: `{user_workspace}`")
    
    st.divider()
    
    st.header("⚙️ Whisper Config")
    model_size = st.selectbox("Model Size:", ["small", "medium", "large-v2", "large-v3"], index=0)
    compute_type = st.selectbox("Compute Type:", ["int8", "float16", "float32"], index=0)
    # batch_size = st.slider("Batch Size:", 1, 32, 16)
    batch_size = 16
    language_opt = st.selectbox("Ngôn ngữ (Optional):", ["Auto Detect", "vi", "en"])
    
    st.divider()
    st.info(f"Server IP: **{get_ip()}**")
    
    if torch.cuda.is_available():
        st.success(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        st.caption(f"VRAM: {vram:.2f} GB")
    else:
        st.error("Running on CPU")

# 2. MAIN CONTENT
st.title(f"🎙️ WhisperX Multi-User Tool")
st.markdown(f"Chào **{tester_name_input}**, upload file MP3/Video để bắt đầu xử lý trong không gian làm việc riêng.")

tab1, tab2, tab3 = st.tabs(["📤 Upload & Chạy", "📂 Quản lý File & Kết quả", "📝 User Logs"])

# === TAB 1: UPLOAD & RUN ===
with tab1:
    col1, col2 = st.columns([2, 1])
    
    file_path_to_process = None

    with col1:
        uploaded_file = st.file_uploader("Chọn file Audio/Video (MP3, MP4, WAV...)", type=['mp3', 'wav', 'mp4', 'm4a', 'mkv'])
        
        if uploaded_file is not None:
            # Lưu file vào thư mục user
            saved_path = save_uploaded_file(uploaded_file, user_workspace)
            if saved_path:
                st.success(f"✅ Đã upload thành công: `{os.path.basename(saved_path)}`")
                file_path_to_process = saved_path
            else:
                st.error("Lỗi khi lưu file.")

    with col2:
        st.write("### Trạng thái")
        if file_path_to_process:
            st.info("Sẵn sàng xử lý file vừa upload.")
        else:
            st.warning("Vui lòng upload file.")

    st.write("---")
    
    run_btn = st.button("🚀 BẮT ĐẦU XỬ LÝ (Start Transcribe)", type="primary", use_container_width=True, disabled=(file_path_to_process is None))

    if run_btn and file_path_to_process:
        with st.status("Đang khởi tạo quy trình...", expanded=True) as status:
            
            # B1. Load Model
            st.write(f"🔌 Đang load model `{model_size}` (Cache enabled)...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            try:
                model = load_whisper_model(model_size, device, compute_type)
                st.write("✅ Model Ready!")
                
                # B2. App Logic
                app = WhisperTranscriber(model_size=model_size, device=device, compute_type=compute_type, batch_size=batch_size)
                
                # B3. Run Transcribe
                lang_arg = None if language_opt == "Auto Detect" else language_opt
                success, duration = app.transcribe_process(model, file_path_to_process, language=lang_arg, status_container=status)
                
                if success:
                    # B4. Write Log to User Folder
                    size_mb = os.path.getsize(file_path_to_process) / 1048576
                    write_log_user(user_workspace, os.path.basename(file_path_to_process), size_mb, duration, model_size, compute_type)
                    
                    status.update(label="✅ Xử lý hoàn tất!", state="complete", expanded=False)
                    st.balloons()
                    st.success(f"Xong! Thời gian: {duration:.2f}s. Kiểm tra Tab 'Quản lý File' để tải kết quả.")
                
            except Exception as e:
                status.update(label="❌ Có lỗi xảy ra!", state="error")
                st.error(f"Lỗi chi tiết: {str(e)}")

# === TAB 2: FILE MANAGER ===
with tab2:
    st.subheader(f"📂 File trong thư mục: {tester_name_input}")
    
    # Liệt kê file trong folder user
    if os.path.exists(user_workspace):
        files = os.listdir(user_workspace)
        files = [f for f in files if os.path.isfile(os.path.join(user_workspace, f))]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(user_workspace, x)), reverse=True)
        
        if not files:
            st.info("Chưa có file nào.")
        else:
            # Tạo bảng hiển thị file
            for f in files:
                f_path = os.path.join(user_workspace, f)
                col_f1, col_f2, col_f3 = st.columns([3, 1, 1])
                
                with col_f1:
                    if f.endswith(".srt"):
                        st.markdown(f"📄 **{f}**")
                    elif f.endswith(".txt"):
                         st.markdown(f"📝 **{f}**")
                    else:
                        st.markdown(f"🎵 **{f}**")
                        
                with col_f2:
                    # Nút Download
                    with open(f_path, "rb") as file_data:
                        st.download_button(f"⬇️ Tải về", file_data, file_name=f, key=f"dl_{f}")
                
                with col_f3:
                    # Nút Xem trước (chỉ cho file text/srt)
                    if f.endswith(".srt") or f.endswith(".txt"):
                        if st.button("👁️ Xem", key=f"view_{f}"):
                            with open(f_path, "r", encoding="utf-8") as file_read:
                                st.text_area(f"Nội dung: {f}", file_read.read(), height=300)
                
                st.divider()
    else:
        st.error("Không tìm thấy thư mục user.")

# === TAB 3: USER LOGS ===
with tab3:
    st.subheader("📜 Nhật ký hoạt động (User Log)")
    user_log_file = os.path.join(user_workspace, "user_log.txt")
    
    col_l1, col_l2 = st.columns([4, 1])
    with col_l2:
        if st.button("Xóa Log User"):
            with open(user_log_file, "w") as f: f.write("")
            st.rerun()
            
    if os.path.exists(user_log_file):
        with open(user_log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in reversed(lines):
                st.code(line.strip(), language="bash")
    else:
        st.info("Chưa có log hoạt động.")