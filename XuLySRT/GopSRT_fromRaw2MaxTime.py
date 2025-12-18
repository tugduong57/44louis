from datetime import datetime

# ==============================================================================
# PHẦN 1: CÁC HÀM TIỆN ÍCH (HELPER FUNCTIONS)
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
            # Tự động đánh lại số thứ tự (1, 2, 3...)
            out.write(str(i + 1) + '\n')
            
            # Ghi thời gian (Kèm duration bên cạnh để debug nếu cần)
            duration = get_duration_in_seconds(times[i])
            out.write(f"{times[i]} {duration}\n") 
            
            out.write(texts[i] + '\n')
            out.write('\n')
    print(f"-> Đã ghi file: {filename}")

# ==============================================================================
# PHẦN 2: CÁC HÀM XỬ LÝ LOGIC CHÍNH (CORE LOGIC)
# ==============================================================================

def merge_blocks_by_sentence(ids, times, texts, max_time=15.0):
    """
    BƯỚC 1: Gộp các block lại cho đến khi gặp dấu chấm câu (.) 
    hoặc tổng thời gian vượt quá max_time.
    """
    new_ids, new_times, new_texts = [], [], []
    
    # Buffer tạm
    buf_text = []
    buf_dur = 0
    buf_start = None
    buf_end = None
    buf_id = None

    for i in range(len(ids)):
        curr_text = texts[i]
        curr_time = times[i]
        curr_dur = get_duration_in_seconds(curr_time)
        curr_start, curr_end = curr_time.split(' --> ')

        # Khởi tạo buffer nếu mới bắt đầu nhóm
        if not buf_text:
            buf_id = ids[i]
            buf_start = curr_start

        # Cộng dồn
        buf_text.append(curr_text)
        buf_dur += curr_dur
        buf_end = curr_end # Luôn cập nhật điểm cuối

        # Điều kiện ngắt nhóm:
        # 1. Kết thúc bằng dấu chấm
        ends_with_dot = curr_text.strip().endswith(('.', '?', '!'))
        # 2. Quá thời gian cho phép của 1 câu
        is_too_long = buf_dur >= max_time

        if ends_with_dot or is_too_long:
            new_ids.append(buf_id)
            new_times.append(f"{buf_start} --> {buf_end}")
            new_texts.append(" ".join(buf_text))
            
            # Reset buffer
            buf_text = []
            buf_dur = 0
            buf_start = None

    # Xử lý phần dư (nếu còn trong buffer)
    if buf_text:
        new_ids.append(buf_id)
        new_times.append(f"{buf_start} --> {buf_end}")
        new_texts.append(" ".join(buf_text))

    return new_ids, new_times, new_texts


def merge_blocks_by_duration(ids, times, texts, max_time=11.0):
    """
    BƯỚC 2: Gộp các block (đã xử lý câu) lại với nhau để tối ưu thời gian hiển thị.
    Cố gắng gộp sao cho tổng thời gian <= max_time.
    """
    final_ids, final_times, final_texts = [], [], []

    # Tính lại duration cho danh sách đầu vào
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
        curr_start, curr_end = times[i].split(' --> ')

        # Kiểm tra nếu cộng thêm block này thì có bị lố thời gian không?
        if (buf_dur + curr_dur > max_time) and (len(buf_text) > 0):
            # --- CHỐT NHÓM CŨ ---
            final_ids.append(buf_id)
            final_times.append(f"{buf_start} --> {buf_end}")
            final_texts.append(" ".join(buf_text))

            # --- BẮT ĐẦU NHÓM MỚI ---
            buf_id = curr_id
            buf_dur = curr_dur
            buf_text = [curr_text]
            buf_start = curr_start
            buf_end = curr_end
        else:
            # --- GỘP TIẾP VÀO NHÓM HIỆN TẠI ---
            if buf_id is None:
                buf_id = curr_id
                buf_start = curr_start
            
            buf_dur += curr_dur
            buf_text.append(curr_text)
            buf_end = curr_end

    # Xử lý phần dư cuối cùng
    if buf_text:
        final_ids.append(buf_id)
        final_times.append(f"{buf_start} --> {buf_end}")
        final_texts.append(" ".join(buf_text))

    return final_ids, final_times, final_texts

# ==============================================================================
# PHẦN 3: MAIN (CHƯƠNG TRÌNH CHÍNH)
# ==============================================================================

if __name__ == "__main__":
    input_filename = "50. Introversion vs.srt"
    output_filename = "50. Introversion vs_Final.srt"

    print("--- BẮT ĐẦU XỬ LÝ ---")

    # 1. Đọc file
    with open(input_filename, "r", encoding='utf-8') as inp:
        raw_data = inp.read()
    
    ids_raw, times_raw, texts_raw = parse_srt_data(raw_data)
    print(f"Input: {len(ids_raw)} dòng.")

    # 2. BƯỚC 1: Gộp theo câu (Dấu chấm hoặc 15s)
    ids_s1, times_s1, texts_s1 = merge_blocks_by_sentence(
        ids_raw, times_raw, texts_raw, max_time=15.0
    )
    print(f"Bước 1 (Gộp câu): Còn {len(ids_s1)} dòng.")

    # 3. BƯỚC 2: Gộp tối ưu thời gian (11s)
    ids_final, times_final, texts_final = merge_blocks_by_duration(
        ids_s1, times_s1, texts_s1, max_time=11.0
    )
    print(f"Bước 2 (Gộp thời gian): Kết quả {len(ids_final)} dòng.")

    # 4. Ghi file kết quả
    save_to_srt(output_filename, ids_final, times_final, texts_final)

    # 5. In kiểm tra vài dòng cuối
    print("\n--- PREVIEW KẾT QUẢ ---")
    for i in range(min(5, len(ids_final))):
        d_text = (texts_final[i][:50] + '...') if len(texts_final[i]) > 50 else texts_final[i]
        dur = get_duration_in_seconds(times_final[i])
        print(f"[{i+1}] {dur:.2f}s | {d_text}")