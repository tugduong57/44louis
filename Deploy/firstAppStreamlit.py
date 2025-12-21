import streamlit as st
import socket

# Hàm lấy IP máy hiện tại để hiển thị cho tiện
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Không cần kết nối internet thật, chỉ để lấy IP LAN
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

# Giao diện Streamlit
st.title("🛠️ Internal Tool Test")
st.write(f"Server IP: **{get_ip()}**") # Hiển thị IP để bạn gửi cho Tester

st.write("---")

st.info("Tester hãy nhấn nút bên dưới để chạy lệnh.")

# Nút bấm kích hoạt script
if st.button('Chạy Script Python'):
    # --- Khu vực code xử lý backend của bạn ---
    st.success("✅ Đã nhận lệnh! Script đang chạy trên máy Server...")
    print("Log: Tester đã kích hoạt script thành công!") 
    # ------------------------------------------
    # http://192.168.1.20:8501