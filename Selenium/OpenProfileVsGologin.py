import os
import json
import zipfile
import subprocess

# 1. Cấu hình các đường dẫn (Kiểm tra lại xem đúng chưa nhé)
BASE_DIR = r"D:\DHB tools\DHB GoLogin Manager v1.10\Data"
BROWSER_EXE = os.path.join(BASE_DIR, r"Browser\orbita-browser-141\chrome.exe")
ZIP_PROFILE = os.path.join(BASE_DIR, r"Profile\testSelenium\694a044b6a186148e603a0a6.zip")
EXTRACT_TO = os.path.join(BASE_DIR, r"Profile Running\694a044b6a186148e603a0a6")

def mo_profile_chuan():
    # Bước 1: Tắt trình duyệt cũ để tránh lỗi chiếm quyền file
    # os.system("taskkill /f /im chrome.exe >nul 2>&1")

    # Bước 2: Giải nén dữ liệu từ file ZIP vào thư mục Running
    if os.path.exists(ZIP_PROFILE):
        print(f"--- Đang giải nén dữ liệu Profile... ---")
        with zipfile.ZipFile(ZIP_PROFILE, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_TO)
    else:
        print("Lỗi: Không tìm thấy file ZIP dữ liệu!")
        return

    # Bước 3: Thiết lập tham số chạy từ dữ liệu JSON bạn gửi
    # Ở đây tôi lấy User-Agent và các thông số chống giả dạng
    args = [
        BROWSER_EXE,
        f'--user-data-dir={EXTRACT_TO}',
        '--profile-directory=Default',
        '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.7390.54 Safari/537.36',
        '--font-masking-mode=2',
        '--disable-encryption',
        '--restore-last-session',
        '--no-first-run'
    ]

    # Bước 4: Chạy trình duyệt
    try:
        subprocess.Popen(args)
        print("--- Mở thành công! Trình duyệt sẽ có sẵn Đăng nhập ---")
    except Exception as e:
        print(f"Lỗi khi chạy trình duyệt: {e}")

if __name__ == "__main__":
    mo_profile_chuan()