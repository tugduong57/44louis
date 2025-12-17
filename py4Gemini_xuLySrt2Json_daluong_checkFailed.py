import os
import re
import time
import concurrent.futures
from tqdm import tqdm
from google import genai
from google.api_core import exceptions
from google.genai import types

# --- CẤU HÌNH ---
# API_KEY = "AIzaSyARBZds9gF9-d4MYYe1accItEzgpKt3I-I"
API_KEY = "AIzaSyDqhnmMFbFwSIUve9CdvPn4u5PXT-OZwAo"
MODEL_NAME = "gemini-2.5-flash-lite"
MAX_WORKERS = 5 
BATCH_SIZE = 57 

# Thư mục lưu file tạm (Quan trọng cho tính năng Resume)
TEMP_DIR = "temp_batches_data" 

client = genai.Client(api_key=API_KEY)

def split_srt_blocks(content):
    content = content.strip()
    blocks = re.split(r'\n\s*\n', content)
    return [b for b in blocks if b.strip()]

def save_temp_batch(batch_index, content):
    """Lưu kết quả của từng batch ra file riêng lẻ."""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    
    file_path = os.path.join(TEMP_DIR, f"batch_{batch_index}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

def check_existing_batch(batch_index):
    """
    Kiểm tra xem batch này đã chạy xong chưa.
    Trả về: (Có tồn tại không?, Nội dung nếu có)
    """
    file_path = os.path.join(TEMP_DIR, f"batch_{batch_index}.txt")
    
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        # Kiểm tra nội dung: Nếu rỗng hoặc báo lỗi thì coi như chưa chạy
        if not content or f"[FAILED BATCH {batch_index}]" in content:
            return False, None
            
        # Nếu OK
        return True, content
    
    return False, None

def prompt_batch(concept, batch_content, batch_index, total_batches):
    final_prompt = (
        f"{concept}\n\n"
        f"--- BẮT ĐẦU DỮ LIỆU PART {batch_index}/{total_batches} ---\n"
        f"{batch_content}\n"
        f"--- KẾT THÚC DỮ LIỆU PART {batch_index}/{total_batches} ---"
    )

    for attempt in range(6): 
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=final_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    max_output_tokens=8192
                )
            )
            
            result_text = response.text
            # ✅ LƯU NGAY VÀO FILE TẠM KHI THÀNH CÔNG
            save_temp_batch(batch_index, result_text)
            
            # Nếu dùng Free Tier, nên sleep nhẹ 1 chút
            time.sleep(2) 
            
            return (batch_index, result_text)
            
        except exceptions.ResourceExhausted:
            wait_time = (attempt + 1) * 10 + 5
            tqdm.write(f"⚠️ [Batch {batch_index}] Hết Quota (429). Đợi {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            tqdm.write(f"❌ [Batch {batch_index}] Lỗi: {e}")
            time.sleep(5)
            
    # Nếu thất bại toàn tập
    fail_msg = f"[FAILED BATCH {batch_index}]"
    save_temp_batch(batch_index, fail_msg) # Lưu lỗi để lần sau biết mà chạy lại
    return (batch_index, fail_msg)

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

    # 3. KIỂM TRA & LỌC CÁC BATCH CẦN CHẠY
    tasks = []
    cached_results = []
    
    print("🔍 Đang kiểm tra dữ liệu cũ...")
    
    # Tạo thư mục tạm nếu chưa có
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    for i, batch in enumerate(batches, 1):
        # Kiểm tra xem file tạm đã có và hợp lệ chưa
        exists, content = check_existing_batch(i)
        
        if exists:
            # Nếu đã có, đưa vào list kết quả luôn, KHÔNG cần chạy lại
            # tqdm.write(f"✅ Batch {i} đã có dữ liệu -> Skip.")
            cached_results.append((i, content))
        else:
            # Nếu chưa có hoặc lỗi, thêm vào danh sách cần chạy
            batch_str = "\n\n".join(batch)
            tasks.append((concept, batch_str, i, total_batches))

    print(f"⏭️  Đã bỏ qua (Skip): {len(cached_results)} batch.")
    print(f"🚀 Cần xử lý: {len(tasks)} batch với {MAX_WORKERS} luồng...")

    # 4. Thực thi các batch còn thiếu
    new_results = []
    if tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_batch = {
                executor.submit(prompt_batch, concept, batch_str, idx, total): idx 
                for (concept, batch_str, idx, total) in tasks
            }

            for future in tqdm(concurrent.futures.as_completed(future_to_batch), total=len(tasks), desc="Tiến độ"):
                try:
                    data = future.result()
                    new_results.append(data)
                except Exception as exc:
                    idx = future_to_batch[future]
                    print(f"Batch {idx} generated an exception: {exc}")
    else:
        print("🎉 Tất cả các batch đều đã hoàn thành từ trước!")

    # 5. Gộp kết quả (Cũ + Mới)
    print("\n🔄 Đang gộp và sắp xếp dữ liệu...")
    final_results = cached_results + new_results
    final_results.sort(key=lambda x: x[0]) 

    # 6. Lưu file cuối cùng
    print(f"💾 Đang lưu file tổng hợp: {output_path}")
    with open(output_path, "w", encoding='utf-8') as f_out:
        final_text = "\n\n--- PART BREAK ---\n\n".join([x[1] for x in final_results])
        f_out.write(final_text)

    print("✅ Hoàn tất toàn bộ quy trình!")

# --- PHẦN THỰC THI ---
if __name__ == "__main__":
    name_file = "50 2 Intro_Final"
    file_srt = name_file + ".srt"
    file_concept = "prompt4_SRT.txt"
    file_out = name_file + "_Response_MultiThread_2.txt"

    process_srt_multithread(file_srt, file_concept, file_out)