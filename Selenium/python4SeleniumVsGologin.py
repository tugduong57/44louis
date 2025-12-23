import os
import time
import random
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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

def main():
    options = uc.ChromeOptions()
    
    # Chỉ định thư mục Profile
    options.add_argument(f'--user-data-dir={PROFILE_DIR}')
    options.add_argument('--profile-directory=Default')
    
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
        driver.get("https://gemini.google.com/")
        
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

        # 2. Đọc Prompt từ file
        prompt_file = "input_prompt.txt"
        if os.path.exists(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                my_prompt = f.read().strip()
        else:
            my_prompt = "Viết 1 đoạn ngắn về lập trình Python."

        # 3. Gửi Prompt
        print(f"--- Đang gửi nội dung: {my_prompt[:50]}... ---")
        prompt_box.click()
        time.sleep(1)
        
        human_typing(prompt_box, my_prompt)
        time.sleep(1)
        prompt_box.send_keys(Keys.ENTER)

        # 4. Chờ và lấy kết quả
        print("--- Đang chờ Gemini trả lời... ---")
        time.sleep(15) # Bạn có thể viết hàm check để đợi chính xác hơn
        
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
        driver.quit()

if __name__ == "__main__":
    main()