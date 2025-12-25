import os
import time
import random
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pyperclip
from selenium.webdriver.common.keys import Keys

# --- CẤU HÌNH BIẾN ---
CHUNK_SIZE = 20  # Số cụm (đoạn SRT) mỗi lần gửi
INPUT_FILE = "prompt4Gem.txt"
OUTPUT_FILE = "ket_qua_gemini.txt"

def split_srt_blocks(file_path):
    """Hàm đọc file và tách thành từng cụm SRT dựa trên dòng trống"""
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    # Tách các cụm dựa trên 2 lần xuống dòng (cấu trúc chuẩn của SRT)
    blocks = content.split('\n\n')
    return blocks

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Thư mục chứa Profile sẽ được tạo ngay tại thư mục chạy script
PROFILE_DIR = os.path.join(os.getcwd(), "ChromeProfile_Gemini")
if not os.path.exists(PROFILE_DIR):
    os.makedirs(PROFILE_DIR)

def human_typing(element, text):
    """Giả lập gõ phím như người thật"""
    for char in text:
        element.send_keys(char)
        time.sleep(random.uniform(0.05, 0.15))

def wait_for_gemini_finish(driver, timeout=120):
    """Đợi cho đến khi Gemini ngừng tạo văn bản"""
    print("--- Đang chờ Gemini hoàn tất phản hồi... ---")
    wait = WebDriverWait(driver, timeout)
    
    try:
        # Cách 1: Đợi cho đến khi nút 'Gửi' (Send message) có thể tương tác trở lại
        # Khi đang tạo, nút này thường bị ẩn hoặc thay thế bằng nút 'Dừng'
        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label*='Send'], button[aria-label*='Gửi']")))
        
        # Đợi thêm một chút ngắn để đảm bảo text đã render hết vào DOM
        time.sleep(2) 
        return True
    except Exception as e:
        print(f"Lỗi khi đợi phản hồi: {e}")
        return False

def main():
    options = uc.ChromeOptions()
    
    # Chỉ định thư mục Profile
    options.add_argument(f'--user-data-dir={PROFILE_DIR}')
    options.add_argument('--profile-directory=Default')

    # Thiết lập kích thước cửa sổ (Ví dụ: Rộng 800px, Cao 600px)
    options.add_argument('--window-size=800,600')
    
    # Một số cấu hình giúp trình duyệt nhẹ và thật hơn
    options.add_argument('--no-first-run')
    options.add_argument('--no-service-autorun')
    options.add_argument('--password-store=basic')

    print(f"--- Đang khởi tạo Chrome với Profile tại: {PROFILE_DIR} ---")
    

    # undetected_chromedriver tự động tải driver phù hợp với phiên bản Chrome máy bạn
    driver = uc.Chrome(options=options)

    # input()

    try:
        # 1. Truy cập Gemini
        print("--- Đang truy cập Gemini... ---")
        driver.get("https://gemini.google.com/gem/10SIxAUluVNmVtZ16NRIYSqW8B_-Wy7nL?usp=sharing")
        
        # Đợi trang tải xong
        wait = WebDriverWait(driver, 30)
        
        # KIỂM TRA ĐĂNG NHẬP: 
        # Nếu chưa đăng nhập, chương trình sẽ dừng 30s để bạn đăng nhập thủ công (chỉ cần làm 1 lần)
        try:
            prompt_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='textbox']")))
        except:
            print("!!! CẢNH BÁO: Có thể bạn chưa đăng nhập Google. Hãy đăng nhập trong cửa sổ trình duyệt vừa mở.")
            time.sleep(45) # Chờ bạn đăng nhập thủ công
            prompt_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='textbox']")))

        # --- 2. Đọc Prompt từ file ---
        prompt_file = "prompt4Gem.txt"
        if os.path.exists(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                my_prompt = f.read().strip()
        else:
            my_prompt = "Viết 1 đoạn ngắn về lập trình Python."

        # --- 3. Đưa nội dung vào Clipboard (Tương đương việc nhấn Ctrl+C hoặc chuẩn bị cho Win+V) ---
        pyperclip.copy(my_prompt)
        print(f"--- Đã sao chép vào clipboard: {my_prompt[:50]}... ---")

        # --- 4. Gửi nội dung bằng Ctrl + V ---
        prompt_box.click()
        time.sleep(1)

        # Giả lập nhấn Ctrl + V để dán nội dung
        # Lưu ý: Trên Mac, bạn có thể cần dùng Keys.COMMAND thay vì Keys.CONTROL
        prompt_box.send_keys(Keys.CONTROL, 'v')

        time.sleep(1)
        
        time.sleep(1)
        prompt_box.send_keys(Keys.ENTER)

        # 4. Chờ và lấy kết quả
        print("--- Đang chờ Gemini trả lời... ---")
        time.sleep(15) # Bạn có thể viết hàm check để đợi chính xác hơn
        # Sử dụng hàm đợi thông minh thay vì time.sleep cố định
        if wait_for_gemini_finish(driver):
            # Lấy tất cả các phản hồi
            responses = driver.find_elements(By.CSS_SELECTOR, ".model-response-text")
            
            if responses:
                # Lấy phản hồi mới nhất (phần tử cuối cùng)
                last_reply = responses[-1].text
                
                # Lưu vào file
                with open("ket_qua_gemini.txt", "w", encoding="utf-8") as f:
                    f.write(last_reply)
                print("--- Đã lưu kết quả thành công! ---")
            else:
                print("--- Không tìm thấy nội dung phản hồi dù đã đợi xong. ---")
        else:
            print("--- Quá thời gian chờ (Timeout) hoặc có lỗi xảy ra. ---")
        
        responses = driver.find_elements(By.CSS_SELECTOR, ".model-response-text")
        if responses:
            last_reply = responses[-1].text
            with open("ket_qua_gemini.txt", "w", encoding="utf-8") as f:
                f.write(last_reply)
            print("--- Đã lưu kết quả thành công! ---")
        else:
            print("--- Không tìm thấy phản hồi. ---")

    except Exception as e:
        print(f"Lỗi: {e}")
    
    finally:
        print("--- Hoàn tất. Trình duyệt sẽ đóng sau 5 giây ---")
        time.sleep(5)
        input()
        # driver.quit()

if __name__ == "__main__":
    main()