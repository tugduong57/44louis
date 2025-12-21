Project: tạo Video Youtube (hơn 10 phút) từ file mp3 kịch bản

- Tạo project, sử dụng api của người dùng

- Từ Kịch bản --- Gửi prompt 2 gemini để tạo từng phân cảnh cho Veo3
				--- Dùng Veo3 tạo từng video khoảng 8s (giới hạn veo3-fast)
				  --- Cho phép người dùng xem list veo3 được tạo ra, sửa từng file (prompt lại ...), sửa promt ...
				  		và merge thành video

- Flow hiện tại: Từ mp3 -> SRT -- Làm sạch --> Gemini --> (Chưa làm: ) --> Excel --> Veo3 --> Video