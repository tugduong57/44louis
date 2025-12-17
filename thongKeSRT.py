import os
from datetime import datetime

# ==============================================================================
# 1. HÀM HỖ TRỢ TÍNH TOÁN THỜI GIAN
# ==============================================================================
def get_duration_in_seconds(time_range_str):
    """
    Tính khoảng thời gian (giây) từ chuỗi SRT '00:00:01,000 --> 00:00:05,000'
    """
    try:
        start_str, end_str = time_range_str.split(' --> ')
        
        # Chuẩn hóa format (thay dấu phẩy bằng dấu chấm cho python dễ parse)
        start_str = start_str.strip().replace(',', '.')
        end_str = end_str.strip().replace(',', '.')
        
        time_format = "%H:%M:%S.%f"
        t_start = datetime.strptime(start_str, time_format)
        t_end = datetime.strptime(end_str, time_format)
        
        duration = t_end - t_start
        return duration.total_seconds()
    except Exception as e:
        # print(f"Lỗi parse thời gian: {time_range_str} | {e}")
        return 0.0

# ==============================================================================
# 2. MODULE ĐỌC FILE VÀ LẤY DURATION
# ==============================================================================
def extract_durations_from_file(file_path):
    """
    Đọc file SRT, parse dòng timestamp để tính duration.
    Không phụ thuộc vào con số ghi sẵn ở cuối dòng.
    """
    durations = []
    
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Tách các block bằng 2 dấu xuống dòng
    blocks = content.split('\n\n')
    
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 2:
            # Dòng chứa time gốc: "00:00:00,066 --> 00:00:05,572 5.506" (hoặc không có 5.506)
            raw_time_line = lines[1].strip()
            
            # Kiểm tra xem dòng này có phải dòng chứa thời gian hợp lệ không
            if " --> " in raw_time_line:
                # Xử lý làm sạch dòng thời gian để loại bỏ các số thừa ở cuối (nếu có)
                # Tách thành: ['00:00:00,066', '00:00:05,572 5.506']
                parts = raw_time_line.split(' --> ')
                
                start_time = parts[0].strip()
                # Lấy phần đầu tiên của vế sau để loại bỏ " 5.506" nếu tồn tại
                end_time = parts[1].strip().split(' ')[0] 
                
                # Tạo chuỗi chuẩn: "00:00:00,066 --> 00:00:05,572"
                clean_time_str = f"{start_time} --> {end_time}"
                
                # Tính toán
                dur = get_duration_in_seconds(clean_time_str)
                durations.append(dur)
                
    return durations

# ==============================================================================
# 3. MODULE THỐNG KÊ (PHÂN TÍCH DỮ LIỆU)
# ==============================================================================
def analyze_duration_distribution(durations):
    """
    Thống kê số lượng và phần trăm theo từng khoảng 1 giây (bins).
    """
    total_count = len(durations)
    if total_count == 0:
        return None, []

    # 1. Tìm thời gian lớn nhất
    max_val = max(durations)
    max_bin_index = int(max_val)
    
    # Tạo bins động
    bins = [0] * (max_bin_index + 1)

    # 2. Phân loại
    for d in durations:
        idx = int(d)
        if 0 <= idx <= max_bin_index:
            bins[idx] += 1

    # 3. Tính toán phần trăm
    stats = []
    for i in range(len(bins)):
        count = bins[i]
        percent = (count / total_count) * 100
        label = f"{i}s - {i+1}s"
        
        stats.append({
            "label": label,
            "count": count,
            "percent": percent
        })
        
    return total_count, stats

# ==============================================================================
# 4. HÀM HIỂN THỊ BÁO CÁO
# ==============================================================================
def print_statistics_report(total, stats, range_start=3, range_end=7):
    print(f"\n{'='*60}")
    print(f" BÁO CÁO THỐNG KÊ (Dynamic Range)")
    print(f" Tổng số blocks: {total}")
    print(f"{'='*60}")
    print(f"{'KHOẢNG THỜI GIAN':<18} | {'SL':<5} | {'TỶ LỆ (%)':<10} | {'BIỂU ĐỒ'}")
    print("-" * 75)

    # In từng dòng chi tiết
    for item in stats:
        # Ẩn các dòng 0% để gọn báo cáo (tùy chọn)
        # if item['count'] == 0: continue
        
        bar_length = int(item['percent'] / 2) 
        if bar_length == 0 and item['count'] > 0: bar_length = 1 
        bar_chart = '#' * bar_length
        
        print(f"{item['label']:<18} | {item['count']:<5} | {item['percent']:<6.2f}%    | {bar_chart}")
    
    print("-" * 75)
    
    # --- LOGIC TÍNH TỔNG TRONG KHOẢNG [A, B] ---
    safe_end = min(range_end, len(stats))
    target_stats = stats[range_start : safe_end]
    
    range_sum_percent = sum(item['percent'] for item in target_stats)
    range_sum_count = sum(item['count'] for item in target_stats)
    
    print(f"\n>>> TỔNG HỢP KHOẢNG TÙY CHỌN [{range_start}s - {range_end}s]:")
    print(f"    - Tổng số lượng: {range_sum_count} blocks")
    print(f"    - Tổng tỷ lệ   : {range_sum_percent:.2f}%")
    print(f"{'='*60}\n")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":

    # Đổi tên file dưới đây thành file của bạn
    # file_to_check = "50 2 Intro.srt" 
    file_to_check = "50 2 Intro_Final.srt"
    TARGET_A = 6  # Giây bắt đầu
    TARGET_B = 12 # Giây kết thúc

    dur_list = extract_durations_from_file(file_to_check)
    
    if dur_list:
        total_items, statistics = analyze_duration_distribution(dur_list)
        print_statistics_report(total_items, statistics, range_start=TARGET_A, range_end=TARGET_B)
    else:
        print("Không có dữ liệu.")