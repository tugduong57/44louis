Mô hình Speech to Text mạnh mẽ nhất
- Qua API
	cung cấp mô hình whisper-1:
+ Hỗ trợ Mp3, Mp4, ...
+ Đa ngôn ngữ
	tính năng đặc biệt
+ audio -> text
+ dịch từ bất kỳ ngôn ngữ nào sang English
+ timestamp: Xuất file có mốc thời gian tới từng mili giây

- Qua Web
	qua ChatGPT: tải mp3 lên -- Whisper --> Văn bản

- Do API tính phí, nguyên cứu sử dụng Whisper Local
+ Cấu hình máy hiện tại, 3070 Ti, 8GB VRAM, -> dùng Faster-Whisper
	Cấu hình khuyến nghị cho bạn:
	Model: large-v3 hoặc large-v3-turbo
	Compute type: float16 (giúp chạy nhanh và tiết kiệm VRAM)
	Device: cuda

+ WhisperX: (audio 2 srt, model không phân được Speaker A vs Speaker B)
Audio file
 ↓
Whisper (ASR)
 ↓
CTC Forced Alignment
 ↓
Word-level timestamps
 ↓
Xuất JSON / SRT / ASS


