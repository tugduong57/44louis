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
import random

# --- THƯ VIỆN AUTOMATION ---
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pyperclip

# --- 0. CẤU HÌNH TRANG & MÔI TRƯỜNG ---
st.set_page_config(
    page_title="WhisperX & Gemini Automation",
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

# ==============================================================================
# PHẦN LOGIC XỬ LÝ SRT MỚI (TÍCH HỢP TỪ USER)
# ==============================================================================

def get_duration_in_seconds(time_range_str):
    """Tính khoảng thời gian (giây) từ chuỗi SRT '00:00:01,000 --> 00:00:05,000'"""
    try:
        start_str, end_str = time_range_str.split(' --> ')
        start_str = start_str.replace(',', '.')
        end_str = end_str.replace(',', '.')
        time_format = "%H:%M:%S.%f"
        t_start = datetime.strptime(start_str, time_format)
        t_end = datetime.strptime(end_str, time_format)
        duration = t_end - t_start
        return duration.total_seconds()
    except ValueError:
        return 0.0

def parse_srt_data(text_content): 
    """Tách nội dung file SRT thành 3 list: ids, timestamps, contents"""
    indices = []
    timestamps = []
    contents = []
    
    blocks = text_content.strip().split('\n\n')
    
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            indices.append(lines[0].strip())
            timestamps.append(lines[1].strip())
            content_text = " ".join([line.strip() for line in lines[2:]])
            contents.append(content_text)
            
    return indices, timestamps, contents

def save_to_srt(filename, ids, times, texts):
    """Ghi dữ liệu ra file SRT chuẩn"""
    with open(filename, "w", encoding='utf-8') as out:
        for i in range(len(ids)):
            out.write(str(i + 1) + '\n')
            out.write(f"{times[i]}\n") 
            out.write(texts[i] + '\n')
            out.write('\n')
    return filename

def merge_blocks_by_sentence(ids, times, texts, max_time=15.0):
    """BƯỚC 1: Gộp các block lại cho đến khi gặp dấu chấm câu hoặc > max_time"""
    new_ids, new_times, new_texts = [], [], []
    
    buf_text = []
    buf_dur = 0
    buf_start = None
    buf_end = None
    buf_id = None

    for i in range(len(ids)):
        curr_text = texts[i]
        curr_time = times[i]
        curr_dur = get_duration_in_seconds(curr_time)
        try:
            curr_start, curr_end = curr_time.split(' --> ')
        except: continue # Skip lỗi format

        if not buf_text:
            buf_id = ids[i]
            buf_start = curr_start

        buf_text.append(curr_text)
        buf_dur += curr_dur
        buf_end = curr_end 

        ends_with_dot = curr_text.strip().endswith(('.', '?', '!'))
        is_too_long = buf_dur >= max_time

        if ends_with_dot or is_too_long:
            new_ids.append(buf_id)
            new_times.append(f"{buf_start} --> {buf_end}")
            new_texts.append(" ".join(buf_text))
            
            buf_text = []
            buf_dur = 0
            buf_start = None

    if buf_text:
        new_ids.append(buf_id)
        new_times.append(f"{buf_start} --> {buf_end}")
        new_texts.append(" ".join(buf_text))

    return new_ids, new_times, new_texts

def merge_blocks_by_duration(ids, times, texts, max_time=11.0):
    """BƯỚC 2: Gộp các block đã xử lý câu để tối ưu thời gian hiển thị"""
    final_ids, final_times, final_texts = [], [], []
    durations = [get_duration_in_seconds(t) for t in times]

    buf_text = []
    buf_dur = 0
    buf_start = None
    buf_end = None
    buf_id = None

    for i in range(len(ids)):
        curr_id = ids[i]
        curr_dur = durations[i]
        curr_text = texts[i]
        try:
            curr_start, curr_end = times[i].split(' --> ')
        except: continue

        if (buf_dur + curr_dur > max_time) and (len(buf_text) > 0):
            final_ids.append(buf_id)
            final_times.append(f"{buf_start} --> {buf_end}")
            final_texts.append(" ".join(buf_text))

            buf_id = curr_id
            buf_dur = curr_dur
            buf_text = [curr_text]
            buf_start = curr_start
            buf_end = curr_end
        else:
            if buf_id is None:
                buf_id = curr_id
                buf_start = curr_start
            
            buf_dur += curr_dur
            buf_text.append(curr_text)
            buf_end = curr_end

    if buf_text:
        final_ids.append(buf_id)
        final_times.append(f"{buf_start} --> {buf_end}")
        final_texts.append(" ".join(buf_text))

    return final_ids, final_times, final_texts

# --- LOGIC CHUYỂN TXT SANG EXCEL ---
def txt_to_excel(input_file, output_file):
    if not os.path.exists(input_file): return False
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = re.compile(r'(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3})\s+(\{.*\})')
    data_list = []
    matches = pattern.findall(content)

    for match in matches:
        data_list.append({
            "ID": match[0],
            "Thời gian": match[1],
            "Data JSON": match[2]
        })

    if data_list:
        df = pd.DataFrame(data_list)
        df.to_excel(output_file, index=False, engine='openpyxl')
        return True
    return False

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

    def transcribe_process(self, model, audio_path: str, language=None, status_container=None, max_merge_time=11.0):
        if not os.path.exists(audio_path): return False, 0, None, None
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
        
        # 1. Lưu bản Word Level (gốc)
        word_srt_path = self._save_srt(result["segments"], audio_path, is_word_level=True)
        
        # 2. Lưu bản Raw Sentence (cơ bản từ Whisper) để làm input cho thuật toán merge
        raw_srt_path = self._save_srt(result["segments"], audio_path, is_word_level=False, suffix=".raw.srt")
        
        # 3. ÁP DỤNG LOGIC GỘP MỚI
        if status_container: status_container.write("🔄 Đang chạy thuật toán tối ưu sub (Step 1 & 2)...")
        try:
            # Đọc file raw vừa tạo
            with open(raw_srt_path, "r", encoding='utf-8') as f:
                raw_content = f.read()
            
            # Parse nội dung
            ids_raw, times_raw, texts_raw = parse_srt_data(raw_content)
            
            # BƯỚC 1: Gộp theo câu (Max 15s)
            ids_s1, times_s1, texts_s1 = merge_blocks_by_sentence(ids_raw, times_raw, texts_raw, max_time=15.0)
            
            # BƯỚC 2: Gộp tối ưu thời gian (Theo config UI)
            ids_final, times_final, texts_final = merge_blocks_by_duration(ids_s1, times_s1, texts_s1, max_time=max_merge_time)
            
            # Lưu file Final
            base_final_name = raw_srt_path.replace(".raw.srt", "_Final_Merged.srt")
            final_srt_path = save_to_srt(base_final_name, ids_final, times_final, texts_final)
            
        except Exception as e:
            if status_container: status_container.error(f"⚠️ Lỗi thuật toán merge: {e}. Dùng bản gốc.")
            final_srt_path = raw_srt_path # Fallback nếu lỗi

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

# --- GEMINI AUTOMATION LOGIC ---
PROFILE_DIR = os.path.join(os.getcwd(), "ChromeProfile_Gemini")

def split_srt_blocks(file_path):
    if not os.path.exists(file_path): return []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content.split('\n\n')

def wait_for_gemini_finish(driver, timeout=120):
    wait = WebDriverWait(driver, timeout)
    try:
        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label*='Send'], button[aria-label*='Gửi']")))
        time.sleep(2) 
        return True
    except Exception:
        return False

def run_gemini_automation(input_srt_path, output_txt_path, chunk_size, gemini_url, status_callback):
    if not os.path.exists(PROFILE_DIR): os.makedirs(PROFILE_DIR)
    options = uc.ChromeOptions()
    options.add_argument(f'--user-data-dir={PROFILE_DIR}')
    options.add_argument('--profile-directory=Default')
    options.add_argument('--no-first-run')
    options.add_argument('--password-store=basic')

    status_callback("🚀 Đang khởi động trình duyệt...")
    try:
        driver = uc.Chrome(options=options)
        window_width, window_height = 800, 800
        screen_width = driver.execute_script("return window.screen.availWidth")
        driver.set_window_size(window_width, window_height)
        driver.set_window_position(screen_width - window_width, 0)
        
        status_callback(f"🔗 Đang truy cập Gemini: {gemini_url}")
        driver.get(gemini_url)
        wait = WebDriverWait(driver, 30)

        try:
            status_callback("🔐 Đang kiểm tra đăng nhập...")
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='textbox']")))
        except:
            status_callback("⚠️ CẢNH BÁO: Chưa đăng nhập! Bạn có 45s để đăng nhập thủ công.")
            time.sleep(45)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='textbox']")))

        blocks = split_srt_blocks(input_srt_path)
        chunks = [blocks[i:i + chunk_size] for i in range(0, len(blocks), chunk_size)]
        status_callback(f"📄 Tổng {len(blocks)} đoạn sub. Chia thành {len(chunks)} nhóm.")

        # Xóa file cũ nếu có trước khi ghi mới
        if os.path.exists(output_txt_path): os.remove(output_txt_path)

        with open(output_txt_path, "a", encoding="utf-8") as f_out:
            for index, chunk in enumerate(chunks):
                status_callback(f"📤 Đang gửi nhóm {index + 1}/{len(chunks)}...")
                prompt_text = "\n\n".join(chunk)
                pyperclip.copy(prompt_text)
                
                prompt_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='textbox']")))
                prompt_box.click()
                time.sleep(1)
                prompt_box.send_keys(Keys.CONTROL, 'v')
                time.sleep(1)
                prompt_box.send_keys(Keys.ENTER)
                
                status_callback(f"⏳ Đang đợi Gemini trả lời nhóm {index + 1}...")
                if wait_for_gemini_finish(driver):
                    time.sleep(2)
                    responses = driver.find_elements(By.CSS_SELECTOR, ".model-response-text")
                    if responses:
                        last_reply = responses[-1].text
                        f_out.write(last_reply + "\n\n") 
                        f_out.flush()
                        status_callback(f"✅ Đã lưu nhóm {index + 1}")
                    else:
                        status_callback(f"❌ Không tìm thấy phản hồi nhóm {index + 1}")
                else:
                    status_callback(f"❌ Quá thời gian chờ nhóm {index + 1}")
                time.sleep(random.uniform(3, 5))

        status_callback("🎉 Hoàn tất Automation! Trình duyệt sẽ đóng sau 5s.")
        time.sleep(5)
        return True
    except Exception as e:
        status_callback(f"❌ Lỗi Automation: {str(e)}")
        return False
    finally:
        try: driver.quit()
        except: pass

# --- UI STREAMLIT ---
with st.sidebar:
    st.header("👤 Định danh")
    tester_name_input = st.text_input("Nhập tên/ID của bạn:", "Test")
    user_workspace = get_user_workspace(tester_name_input)
    st.caption(f"📂 Workspace: `{user_workspace}`")
    
    st.divider()
    st.header("⚙️ Whisper Config")
    model_size = st.selectbox("Model Size:", ["small", "medium", "large-v2", "large-v3"], index=0)
    compute_type = st.selectbox("Compute Type:", ["int8", "float16", "float32"], index=0)
    language_opt = st.selectbox("Ngôn ngữ (Optional):", ["Auto Detect", "vi", "en"])
    
    max_merge_time = st.slider("Max Time gộp Sub (giây):", 8.0, 15.0, 11.0, 0.5)
    
    st.divider()
    st.header("🤖 Gemini Config")
    chunk_size = st.number_input("SRT Chunk Size:", min_value=5, max_value=100, value=20)
    gemini_link = st.text_input("GEM URL:", "https://gemini.google.com/gem/10SIxAUluVNmVtZ16NRIYSqW8B_-Wy7nL?usp=sharing")

    if torch.cuda.is_available():
        st.success(f"GPU: {torch.cuda.get_device_name(0)}")
    if check_ffmpeg(): st.success("✅ FFmpeg Ready")

st.title(f"🎙️ WhisperX & Gemini Tool")
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Whisper", "📂 Files", "📝 Logs", "🤖 Gemini Auto"])

# --- TAB 1: WHISPER ---
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

    run_btn = st.button("🚀 BẮT ĐẦU WHISPER", type="primary", use_container_width=True, disabled=(file_path_to_process is None))

    if run_btn and file_path_to_process:
        with st.status("Đang xử lý Whisper...", expanded=True) as status:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = load_whisper_model(model_size, device, compute_type)
                app = WhisperTranscriber(model_size=model_size, device=device, compute_type=compute_type)
                lang_arg = None if language_opt == "Auto Detect" else language_opt
                
                # Truyền max_merge_time vào transcribe_process
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
                    status.update(label="✅ Whisper & Merge Hoàn tất!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")

# --- TAB 2: FILES ---
with tab2:
    st.subheader(f"📂 Workspace: {tester_name_input}")
    
    col_top1, col_top2 = st.columns([1, 5])
    if col_top1.button("🔄 Làm mới"): 
        st.rerun()

    if os.path.exists(user_workspace):
        files = sorted([f for f in os.listdir(user_workspace) if os.path.isfile(os.path.join(user_workspace, f))], 
                       key=lambda x: os.path.getmtime(os.path.join(user_workspace, x)), reverse=True)
        
        if not files:
            st.info("Thư mục trống.")
        
        for f in files:
            f_path = os.path.join(user_workspace, f)
            c1, c2 = st.columns([3, 1])
            
            c1.write(f"📄 {f}")
            
            with open(f_path, "rb") as fd:
                c2.download_button("⬇️ Tải", fd, file_name=f, key=f"dl_{f}", use_container_width=True)
            
            # if c3.button("🗑️ Xóa", key=f"del_{f}", use_container_width=True):
            #     try:
            #         os.remove(f_path)
            #         st.toast(f"✅ Đã xóa file: {f}")
            #         time.sleep(0.5)
            #         st.rerun()
            #     except Exception as e:
            #         st.error(f"Lỗi khi xóa: {e}")
            
            if f.endswith(".mp4"):
                st.video(f_path)
            elif f.endswith((".srt", ".txt")):
                with st.expander(f"Xem nhanh nội dung:"):
                    try:
                        with open(f_path, "r", encoding="utf-8") as f_preview:
                            st.code(f_preview.read())
                    except:
                        st.write("Không thể đọc nội dung file này.")

            elif f.endswith(".xlsx"):
                with st.expander(f"📊 Xem nhanh bảng tính:"):
                    try:
                        # Đọc file excel bằng pandas
                        df_preview = pd.read_excel(f_path)
                        # Hiển thị dạng bảng tương tác
                        st.dataframe(df_preview, use_container_width=True)
                        st.caption(f"Số dòng: {len(df_preview)} | Số cột: {len(df_preview.columns)}")
                    except Exception as e:
                        st.error(f"⚠️ Không thể đọc file Excel này: {str(e)}")
            
            st.divider()

with tab3:
    user_log_file = os.path.join(user_workspace, "user_log.txt")
    if os.path.exists(user_log_file):
        for line in reversed(open(user_log_file, "r", encoding="utf-8").readlines()):
            st.code(line.strip(), language="bash")

# --- TAB 4: GEMINI AUTOMATION ---
with tab4:
    st.header("🤖 Tự động gửi SRT sang Gemini & Xuất Excel")
    
    if os.path.exists(user_workspace):
        all_files = sorted([f for f in os.listdir(user_workspace) if f.endswith(".srt")], 
                           key=lambda x: os.path.getmtime(os.path.join(user_workspace, x)), reverse=True)
    else: all_files = []

    if not all_files:
        st.warning("⚠️ Không tìm thấy file SRT nào.")
    else:
        # Tự động chọn file _Final_Merged nếu có
        default_index = 0
        for i, name in enumerate(all_files):
            if "Final_Merged" in name:
                default_index = i
                break
        
        selected_srt_name = st.selectbox("🎯 Chọn file SRT để xử lý:", all_files, index=default_index)
        input_srt_path = os.path.join(user_workspace, selected_srt_name)
        
        base_name = os.path.splitext(selected_srt_name)[0]
        timestamp_str = int(time.time())
        output_gemini_filename = f"Gemini_{base_name}_{timestamp_str}.txt"
        output_excel_filename = f"Excel_{base_name}_{timestamp_str}.xlsx"
        
        output_gemini_path = os.path.join(user_workspace, output_gemini_filename)
        output_excel_path = os.path.join(user_workspace, output_excel_filename)

        if st.button("🚀 CHẠY AUTOMATION & XUẤT EXCEL", type="primary", use_container_width=True):
            status_text = st.empty()
            def update_ui(msg): status_text.text(f"🤖 {msg}")

            success_auto = run_gemini_automation(
                input_srt_path=input_srt_path,
                output_txt_path=output_gemini_path,
                chunk_size=chunk_size,
                gemini_url=gemini_link,
                status_callback=update_ui
            )
            
            if success_auto:
                update_ui("📊 Đang chuyển đổi sang Excel...")
                success_excel = txt_to_excel(output_gemini_path, output_excel_path)
                
                if success_excel:
                    st.success(f"✅ Hoàn tất! Đã tạo file Excel.")
                    
                    col_dl1, col_dl2 = st.columns(2)
                    with open(output_gemini_path, "rb") as f_txt:
                        col_dl1.download_button("⬇️ Tải file TXT", f_txt, file_name=output_gemini_filename, use_container_width=True)
                    
                    with open(output_excel_path, "rb") as f_xl:
                        col_dl2.download_button("⬇️ TẢI FILE EXCEL (.xlsx)", f_xl, file_name=output_excel_filename, type="primary", use_container_width=True)
                    
                    df_preview = pd.read_excel(output_excel_path)
                    st.dataframe(df_preview, use_container_width=True)
                else:
                    st.warning("⚠️ Đã lấy được dữ liệu TXT nhưng không tìm thấy cấu trúc phù hợp để chuyển sang Excel (Regex không khớp).")
                    with open(output_gemini_path, "rb") as f_txt:
                        st.download_button("⬇️ Vẫn tải file TXT", f_txt, file_name=output_gemini_filename)
            else:
                st.error("❌ Automation Gemini thất bại.")