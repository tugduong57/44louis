import os
import re
import time
import concurrent.futures
from tqdm import tqdm  # Cần cài đặt: pip install tqdm
from google import genai
from google.api_core import exceptions
from google.genai import types

# --- CẤU HÌNH ---
API_KEY = "AIzaSyARBZds9gF9-d4MYYe1accItEzgpKt3I-I"  # <-- Dán key mới của bạn vào đây
MODEL_NAME = "gemini-2.5-flash-lite"
MAX_WORKERS = 1  # Số luồng chạy cùng lúc. Flash Lite khá nhanh, 5-8 là ổn định.
BATCH_SIZE = 12 # Số block SRT trong 1 lần gửi

# Khởi tạo client
client = genai.Client(api_key=API_KEY)

def split_srt_blocks(content):
    """Chia nội dung SRT thành list các block."""
    content = content.strip()
    blocks = re.split(r'\n\s*\n', content)
    # Lọc bỏ các block rỗng nếu có
    return [b for b in blocks if b.strip()]

def prompt_batch(concept, batch_content, batch_index, total_batches):
    """
    Hàm xử lý 1 batch. Trả về tuple (index, response_text) để sau này sắp xếp lại.
    """
    final_prompt = (
        f"{concept}\n\n"
        f"--- BẮT ĐẦU DỮ LIỆU PART {batch_index}/{total_batches} ---\n"
        f"{batch_content}\n"
        f"--- KẾT THÚC DỮ LIỆU PART {batch_index}/{total_batches} ---"
    )

    # Retry mechanism
    for attempt in range(6): # Thử tối đa 6 lần
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=final_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", # ép trả về dạng json 
                    temperature=0.1 # Giảm độ sáng tạo xuống thấp nhất để ổn định
                )
            )
            # Trả về index và text để sort sau này
            return (batch_index, response.text)
            
        except exceptions.ResourceExhausted:
            wait_time = (attempt + 1) * 10 + 5 # 15s, 25s, 35s...
            # Dùng tqdm.write để không bị vỡ thanh process bar
            tqdm.write(f"⚠️ [Batch {batch_index}] Hết Quota (429). Đợi {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            tqdm.write(f"❌ [Batch {batch_index}] Lỗi: {e}")
            time.sleep(5)
            
    return (batch_index, f"[FAILED BATCH {batch_index}]")

def process_srt_multithread(srt_path, concept_path, output_path):
    # 1. Đọc dữ liệu
    print(f"📖 Đang đọc file: {srt_path}")
    try:
        with open(srt_path, "r", encoding='utf-8') as f:
            contentOfSRT = f.read()
        with open(concept_path, "r", encoding='utf-8') as f:
            concept = f.read()
    except FileNotFoundError as e:
        print(f"❌ Lỗi file: {e}")
        return

    # 2. Chia batch
    all_blocks = split_srt_blocks(contentOfSRT)
    total_blocks = len(all_blocks)
    batches = [all_blocks[i:i + BATCH_SIZE] for i in range(0, total_blocks, BATCH_SIZE)]
    total_batches = len(batches)
    
    print(f"📊 Tổng block: {total_blocks} | Tổng batch: {total_batches}")
    print(f"🚀 Bắt đầu xử lý song song với {MAX_WORKERS} luồng...")

    # 3. Chuẩn bị dữ liệu cho multithreading
    # Tạo list các arguments để truyền vào hàm
    tasks = []
    for i, batch in enumerate(batches, 1):
        batch_str = "\n\n".join(batch)
        tasks.append((concept, batch_str, i, total_batches))

    results = []

    # 4. Thực thi Đa luồng
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit các task vào executor
        # future_to_index map giữa future object và index batch (để debug nếu cần)
        future_to_batch = {
            executor.submit(prompt_batch, concept, batch_str, idx, total): idx 
            for (concept, batch_str, idx, total) in tasks
        }

        # Sử dụng tqdm để hiển thị tiến trình hoàn thành
        for future in tqdm(concurrent.futures.as_completed(future_to_batch), total=total_batches, desc="Tiến độ"):
            try:
                # result ở đây là tuple (batch_index, text) từ hàm prompt_batch
                data = future.result()
                results.append(data)
            except Exception as exc:
                idx = future_to_batch[future]
                print(f"Batch {idx} generated an exception: {exc}")

    # 5. Sắp xếp kết quả (Quan trọng!)
    # Vì chạy song song nên kết quả trả về lộn xộn, cần sort lại theo batch_index
    print("\n🔄 Đang sắp xếp lại thứ tự các phần...")
    results.sort(key=lambda x: x[0]) 

    # 6. Lưu file
    print(f"💾 Đang lưu vào file: {output_path}")
    with open(output_path, "w", encoding='utf-8') as f_out:
        # Chỉ lấy phần text (x[1]) để ghi
        final_text = "\n\n--- PART BREAK ---\n\n".join([x[1] for x in results])
        f_out.write(final_text)

    print("✅ Hoàn tất!")

# --- PHẦN THỰC THI ---
if __name__ == "__main__":
    name_file = "50 2 Intro_Final"
    
    file_srt = name_file + ".srt"
    file_concept = "prompt4_SRT.txt"
    file_out = name_file + "_Response_MultiThread.txt"

    # Chạy hàm chính
    process_srt_multithread(file_srt, file_concept, file_out)