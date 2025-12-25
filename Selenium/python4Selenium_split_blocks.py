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
    
    # Một số cấu hình giúp trình duyệt nhẹ và thật hơn
    options.add_argument('--no-first-run')
    options.add_argument('--no-service-autorun')
    options.add_argument('--password-store=basic')

    print(f"--- Đang khởi tạo Chrome với Profile tại: {PROFILE_DIR} ---")

    # undetected_chromedriver tự động tải driver phù hợp với phiên bản Chrome máy bạn
    driver = uc.Chrome(options=options)

        
    # 1. Định nghĩa kích thước cửa sổ bạn muốn
    window_width = 800
    window_height = 600

    # 2. Thiết lập kích thước trước
    driver.set_window_size(window_width, window_height)

    # 3. Lấy độ rộng của màn hình hiện tại (sử dụng JavaScript)
    screen_width = driver.execute_script("return window.screen.availWidth")

    # 4. Tính toán vị trí X để sát mép phải
    # X = Tổng độ rộng màn hình - Độ rộng trình duyệt
    x_position = screen_width - window_width
    y_position = 0  # Sát mép trên

    # 5. Di chuyển cửa sổ
    driver.set_window_position(x_position, y_position)

    print(f"--- Đã di chuyển cửa sổ sang góc trên phải (X={x_position}, Y={y_position}) ---")

    input()

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
        blocks = split_srt_blocks(INPUT_FILE)
        total_blocks = len(blocks)
        print(f"--- Tổng cộng có {total_blocks} cụm. Gửi mỗi lần {CHUNK_SIZE} cụm. ---")

        # Chia nhỏ danh sách blocks thành các nhóm (chunks)
        chunks = [blocks[i:i + CHUNK_SIZE] for i in range(0, total_blocks, CHUNK_SIZE)]

        # Mở file output ở chế độ 'a' (append - viết tiếp) để không bị ghi đè
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
            
            for index, chunk in enumerate(chunks):
                print(f"--- Đang xử lý nhóm {index + 1}/{len(chunks)} ---")
                
                # Ghép các cụm trong nhóm lại thành 1 chuỗi văn bản
                prompt_text = "\n\n".join(chunk)
                
                # 1. Đưa vào Clipboard
                pyperclip.copy(prompt_text)
                
                # 2. Tìm và click vào khung soạn thảo (phải tìm lại mỗi lần để đảm bảo DOM ổn định)
                prompt_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='textbox']")))
                prompt_box.click()
                time.sleep(1)
                
                # 3. Dán và Gửi
                prompt_box.send_keys(Keys.CONTROL, 'v')
                time.sleep(1)
                prompt_box.send_keys(Keys.ENTER)
                
                # 4. Chờ Gemini trả lời xong
                print("    Đang chờ Gemini phản hồi...")
                if wait_for_gemini_finish(driver):
                    # Lấy phản hồi cuối cùng
                    time.sleep(2) # Đợi thêm một chút cho UI ổn định
                    responses = driver.find_elements(By.CSS_SELECTOR, ".model-response-text")
                    if responses:
                        last_reply = responses[-1].text
                        
                        # Ghi vào file kèm dấu phân cách để dễ theo dõi
                        f_out.write(f"\n--- PHẢN HỒI NHÓM {index + 1} ---\n")
                        f_out.write(last_reply)
                        f_out.write("\n\n" + "="*50 + "\n")
                        f_out.flush() # Đẩy dữ liệu vào file ngay lập tức
                        print(f"    --- Đã lưu nhóm {index + 1} thành công! ---")
                    else:
                        print(f"    !!! Lỗi: Không tìm thấy văn bản phản hồi ở nhóm {index + 1}")
                else:
                    print(f"    !!! Lỗi: Quá thời gian chờ ở nhóm {index + 1}")
                    
                # Nghỉ một chút trước khi gửi nhóm tiếp theo để tránh bị spam block
                time.sleep(5)

    except Exception as e:
        print(f"Lỗi: {e}")
    
    finally:
        print("--- Hoàn tất. Trình duyệt sẽ đóng sau 5 giây ---")
        time.sleep(5)
        input()
        # driver.quit()

if __name__ == "__main__":
    main()