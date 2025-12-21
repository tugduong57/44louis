import streamlit as st
import socket
import time
import pandas as pd
import numpy as np

# --- 0. CẤU HÌNH TRANG (LAYOUT & CONFIG) ---
st.set_page_config(
    page_title="Internal Tool Super Test",
    page_icon="🛠️",
    layout="wide" # Chế độ màn hình rộng
)

# Hàm lấy IP LAN (Giữ nguyên từ bài trước)
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

# --- 1. SIDEBAR (THANH BÊN) - KHU VỰC CẤU HÌNH ---
with st.sidebar:
    st.header("👤 Thông tin Tester")
    tester_name = st.text_input("Tên của bạn:", "Tester A")
    
    st.divider() # Đường kẻ ngang
    
    st.header("⚙️ Cài đặt Server")
    server_env = st.selectbox("Chọn môi trường:", ["Development", "Staging", "Production"])
    debug_mode = st.toggle("Bật chế độ Debug", value=False)
    
    st.info(f"Server IP: **{get_ip()}**")

# --- 2. MAIN CONTENT (GIAO DIỆN CHÍNH) ---
st.title(f"🚀 Control Panel - Xin chào, {tester_name}!")
st.markdown(f"Đang kết nối tới môi trường: `{server_env}`")

# Chia Tab để tổ chức giao diện gọn gàng
tab1, tab2, tab3 = st.tabs(["🎮 Điều khiển", "📊 Dữ liệu & Báo cáo", "📝 Logs hệ thống"])

# === TAB 1: NHẬP LIỆU & TƯƠNG TÁC (INPUT WIDGETS) ===
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Tham số đầu vào")
        # Slider chọn số lượng
        num_threads = st.slider("Số luồng xử lý (Threads):", min_value=1, max_value=10, value=4)
        # Nhập số cụ thể
        retry_count = st.number_input("Số lần thử lại nếu lỗi:", min_value=0, max_value=5, value=3)

    with col2:
        st.subheader("2. Upload File Config")
        # Upload file (CSV, TXT, JSON...)
        uploaded_file = st.file_uploader("Tải lên file kịch bản (.csv, .txt)", type=['csv', 'txt'])
        if uploaded_file is not None:
            st.success(f"Đã nhận file: {uploaded_file.name}")

    st.write("---")
    
    # Nút bấm kích hoạt hành động
    run_btn = st.button("🚀 CHẠY SCRIPT XỬ LÝ", type="primary", use_container_width=True)

    # === PHẦN STATUS ELEMENTS (TRẠNG THÁI) ===
    if run_btn:
        with st.status("Đang khởi tạo tiến trình...", expanded=True) as status:
            st.write("🔌 Đang kết nối API...")
            time.sleep(1)
            
            st.write(f"⚙️ Đang chạy với {num_threads} luồng...")
            time.sleep(1)
            
            st.write("📂 Đang phân tích file upload...")
            # Thanh tiến trình (Progress Bar)
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02) # Giả lập xử lý
                progress_bar.progress(i + 1)
            
            status.update(label="✅ Xử lý hoàn tất!", state="complete", expanded=False)
        
        st.success("Script đã chạy thành công! Vui lòng kiểm tra tab 'Dữ liệu'.")
        if debug_mode:
            st.warning("Debug Mode đang bật: Log chi tiết đã được ghi lại.")

# === TAB 2: HIỂN THỊ DỮ LIỆU (DATA DISPLAY) ===
with tab2:
    st.subheader("Kết quả phân tích")
    
    # Tạo dữ liệu giả lập
    data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['CPU Usage', 'Memory Usage', 'Disk I/O']
    )
    
    # Chia cột để hiển thị Bảng và Biểu đồ cạnh nhau
    d_col1, d_col2 = st.columns([1, 2])
    
    with d_col1:
        st.caption("Bảng dữ liệu chi tiết (Interactive Dataframe)")
        st.dataframe(data, height=300) # Bảng có thể scroll, sort
        
    with d_col2:
        st.caption("Biểu đồ giám sát thời gian thực")
        st.line_chart(data) # Vẽ biểu đồ đường cực nhanh

    # Hiển thị Metrics (Chỉ số quan trọng)
    st.write("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Tổng Request", "1,024", "+5%")
    m2.metric("Thời gian phản hồi", "45ms", "-12ms")
    m3.metric("Lỗi hệ thống", "0", "Normal")

# === TAB 3: JSON & CODE ===
with tab3:
    st.subheader("Cấu hình hiện tại (JSON View)")
    config_data = {
        "tester": tester_name,
        "environment": server_env,
        "threads": num_threads,
        "retry": retry_count,
        "ip": get_ip()
    }
    st.json(config_data) # Hiển thị JSON đẹp mắt
    
    st.subheader("Log Backend")
    st.code("""
    [INFO] 2023-10-25 10:00:01 - Connection established
    [INFO] 2023-10-25 10:00:02 - User authorized
    [WARN] Low memory warning on Thread-2
    """, language="bash")