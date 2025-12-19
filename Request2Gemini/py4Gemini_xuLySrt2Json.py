import os
# import google.generativeai as genai
import re
from google import genai
import google.api_core.exceptions
import time
from google.api_core import exceptions

# --- PHẦN KHỞI TẠO MODEL (chỉ chạy 1 lần) ---
print("Khởi tạo model")

# api_key = os.environ['GOOGLE_API_KEY']
# genai.configure(api_key="AIzaSyARBZds9gF9-d4MYYe1accItEzgpKt3I-I")
client = genai.Client(api_key="AIzaSyARBZds9gF9-d4MYYe1accItEzgpKt3I-I")

# model = genai.GenerativeModel(model_name='models/gemini-2.5-flash-lite')

def prompt2gemini(concept, content):
    target_model = "gemini-2.5-flash-lite"
    prompt = concept + content;
    # response = model.generate_content(prompt)

    response = client.models.generate_content(
        model=target_model,
        contents=f"{concept}\n\nDữ liệu: {content}"
    )

    return response.text

def split_srt_blocks(content):
    """
    Chia nội dung SRT thành list các block dựa trên dòng trống (\n\n).
    Sử dụng Regex để xử lý trường hợp có nhiều dòng trống liên tiếp.
    """
    # Xóa khoảng trắng thừa đầu/cuối file
    content = content.strip()
    # Tách theo 2 dấu xuống dòng trở lên
    blocks = re.split(r'\n\s*\n', content)
    return blocks

def prompt_batch(concept, batch_content, batch_index, total_batches):
    """
    Gửi 1 batch (100 blocks) tới Gemini và nhận phản hồi.
    """
    target_model = "gemini-2.5-flash-lite"
    
    # Tạo prompt rõ ràng cho model
    final_prompt = (
        f"{concept}\n\n"
        f"--- BẮT ĐẦU DỮ LIỆU PART {batch_index}/{total_batches} ---\n"
        f"{batch_content}\n"
        f"--- KẾT THÚC DỮ LIỆU PART {batch_index}/{total_batches} ---"
    )

    print(f" >> Đang xử lý Batch {batch_index}/{total_batches}...")

    # Cơ chế thử lại (Retry) nếu gặp lỗi Quota
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=target_model,
                contents=final_prompt
            )
            return response.text
        except exceptions.ResourceExhausted:
            wait_time = (attempt + 1) * 20  # Đợi 20s, 40s, 60s...
            print(f"    ! Hết Quota (429). Đợi {wait_time}s rồi thử lại...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"    ! Lỗi không xác định: {e}")
            return f"[ERROR AT BATCH {batch_index}]"
            
    return "[FAILED AFTER RETRIES]"

# --- PHẦN THỰC THI CHÍNH ---   
if __name__ == "__main__":
    # Đổi tên file dưới đây thành file của bạn
    name_file = "50 2 Intro_Final"

    file_to_save = name_file + "_Respone.txt"
    file_to_prompt = name_file + ".srt"
    file_concept = "prompt4_SRT.txt"


    # with open(file_to_prompt, "r", encoding='utf-8') as inp1:
    #     contentOfSRT = inp1.read()
    # with open(file_concept, "r", encoding='utf-8') as inp2:
    #     concept  = inp2.read()
    # respone4SRT = prompt2gemini(contentOfSRT, concept)

    with open(file_to_prompt, "r", encoding='utf-8') as f:
        contentOfSRT = f.read()
    
    with open(file_concept, "r", encoding='utf-8') as f:
        concept = f.read()
        
    # 3. Tách block SRT
    all_blocks = split_srt_blocks(contentOfSRT)
    total_blocks = len(all_blocks)
    print(f"Tổng số block tìm thấy: {total_blocks}")

    # 4. Chia thành các nhóm 100 (Chunking)
    BATCH_SIZE = 100
    full_response = []
    
    # Tạo danh sách các batch
    batches = [all_blocks[i:i + BATCH_SIZE] for i in range(0, total_blocks, BATCH_SIZE)]
    total_batches = len(batches)

    print(f"Sẽ chia làm {total_batches} lần gửi (requests).")

    # 5. Gửi từng batch
    for i, batch in enumerate(batches, 1):
        # Gộp 100 block thành 1 chuỗi string để gửi
        batch_str = "\n\n".join(batch)
        # print(batch_str)
        # # Gọi API
        result = prompt_batch(concept, batch_str, i, total_batches)
        full_response.append(result)
        
        # Nghỉ ngắn giữa các lần gửi thành công để tránh spam server (dù chưa lỗi)
        if i < total_batches:
            time.sleep(2) 

    # 6. Lưu kết quả
    print("-" * 30)
    print("Đang lưu file...")
    with open(file_to_save, "a", encoding='utf-8') as f_out:
        # Ghi nối các kết quả lại, ngăn cách bằng dòng kẻ
        f_out.write("\n\n--- PART BREAK ---\n\n".join(full_response))
        
    print(f"Hoàn tất! Kết quả đã lưu tại: {file_to_save}")


