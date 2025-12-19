import json
import pandas as pd
import os

def json_to_excel(input_path, output_path):
    print(f"📖 Đang đọc file: {input_path}")
    
    if not os.path.exists(input_path):
        print("❌ Không tìm thấy file đầu vào!")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Tách các phần dựa trên dấu phân cách bạn đã set trong code đa luồng
    # Dùng regex hoặc string split. Ở đây dùng split chuỗi cố định.
    raw_blocks = content.split('--- PART BREAK ---')
    
    all_items = []
    
    print("⚙️ Đang xử lý dữ liệu JSON...")
    
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
            
        try:
            # Parse chuỗi JSON thành List các Dict
            data = json.loads(block)
            
            # Nếu kết quả là list (do prompt yêu cầu trả về list), ta mở rộng danh sách tổng
            if isinstance(data, list):
                all_items.extend(data)
            else:
                # Nếu lỡ nó trả về 1 object đơn lẻ
                all_items.append(data)
                
        except json.JSONDecodeError as e:
            print(f"⚠️ Lỗi parse JSON ở một block (bỏ qua): {e}")
            # Mẹo: In ra một đoạn nhỏ để debug nếu cần
            # print(block[:100])

    # 2. Chuẩn bị dữ liệu cho DataFrame
    excel_rows = []
    
    for idx, item in enumerate(all_items, 1):
        # --- Xử lý STT ---
        # Ưu tiên lấy scene_id trong JSON, nếu không có thì lấy số thứ tự tự tăng
        stt = item.get('scene_id', idx)
        
        # --- Xử lý Thời gian ---
        # Kiểm tra xem JSON có chứa timestamp từ prompt không, hay chỉ có duration
        # Bạn có thể điều chỉnh logic này tùy vào key thực tế trong JSON của bạn
        time_display = "N/A"
        if 'start_time' in item and 'end_time' in item:
            time_display = f"{item['start_time']} --> {item['end_time']}"
        elif 'timestamp' in item:
            time_display = item['timestamp']
        elif 'duration' in item:
            time_display = f"Duration: {item['duration']}s"
        
        # --- Xử lý Nội dung JSON ---
        # Convert ngược dict thành string JSON để bỏ vào ô Excel
        json_content = json.dumps(item, ensure_ascii=False)
        
        row = {
            "STT": stt,
            "Mốc thời gian": time_display,
            "Nội dung Json": json_content
        }
        excel_rows.append(row)

    # 3. Xuất ra Excel
    if excel_rows:
        df = pd.DataFrame(excel_rows)
        
        # Lưu file
        df.to_excel(output_path, index=False)
        print(f"✅ Đã xuất thành công {len(excel_rows)} dòng ra file: {output_path}")
    else:
        print("❌ Không trích xuất được dữ liệu nào.")

# --- CHẠY ---
if __name__ == "__main__":
    # Tên file giống như file output ở bước trước
    name_file = "50 2 Intro_Final"
    input_file = name_file + "_Response_MultiThread.txt"
    output_file = name_file + "_Final_Excel.xlsx"
    
    json_to_excel(input_file, output_file)