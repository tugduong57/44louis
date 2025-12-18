import whisperx
import gc

device = "cuda" # hoặc "cpu" nếu không có GPU
audio_file = "audio.mp3"
batch_size = 16 # giảm xuống nếu bị tràn bộ nhớ GPU (VRAM)
compute_type = "float16" # dùng "int8" nếu chạy trên CPU hoặc GPU yếu

# 1. Tải model và Transcribe
model = whisperx.load_model("large-v2", device, compute_type=compute_type)
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)

# 2. Căn chỉnh thời gian (Alignment)
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print(result["segments"]) # Kết quả đã có thời gian chính xác từng từ