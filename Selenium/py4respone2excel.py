import re
import pandas as pd

def txt_to_excel(input_file, output_file):
    # Đọc nội dung file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex để tìm: STT, Thời gian và Khối JSON
    # Pattern giải thích:
    # (\d+) : Cột ID (số)
    # (\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}) : Cột thời gian
    # (\{.*\}) : Cột JSON Data (tất cả nằm trong ngoặc nhọn)
    pattern = re.compile(r'(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3})\s+(\{.*\})')

    data_list = []
    
    # Tìm tất cả các dòng khớp với cấu trúc
    matches = pattern.findall(content)

    for match in matches:
        data_list.append({
            "ID": match[0],
            "Thời gian": match[1],
            "Data JSON": match[2]
        })

    # Chuyển thành DataFrame
    df = pd.DataFrame(data_list)

    # Xuất ra Excel
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"Đã trích xuất thành công {len(df)} dòng vào file: {output_file}")

# Thực thi
input_filename = 'ket_qua_gemini.txt'  # File văn bản đầu vào
output_filename = 'Excel_Prompts.xlsx'
txt_to_excel(input_filename, output_filename)