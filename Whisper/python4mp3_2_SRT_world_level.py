import torch
import whisperx
import gc
import os
import json 
from collections import defaultdict
from typing import Any, List, Dict, Optional

# --- 1. SETUP & FIX LỖI PYTORCH 2.6 ---
def setup_environment():
    _original_torch_load = torch.load
    def new_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False 
        return _original_torch_load(*args, **kwargs)
    torch.load = new_torch_load

    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig
    from omegaconf.base import ContainerMetadata
    
    torch.serialization.add_safe_globals([
        ListConfig, DictConfig, ContainerMetadata, Any, defaultdict,
        list, dict, tuple, set, int, float, bool, str, type, object
    ])
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

setup_environment()

def format_timestamp(seconds: float) -> str:
    x = int(seconds)
    msec = int((seconds - x) * 1000)
    hours = x // 3600
    minutes = (x % 3600) // 60
    seconds = x % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{msec:03d}"

# --- 2. MAIN CLASS ---
class WhisperTranscriber:
    def __init__(self, model_size="small", device=None, compute_type="int8", batch_size=16):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        print(f"🔹 Khởi tạo model ({model_size}) trên {self.device}...")
        self.model = whisperx.load_model(model_size, self.device, compute_type=compute_type)

    def transcribe_file(self, audio_path: str, language: Optional[str] = None):
        """
        Chạy 1 lần, xuất ra 2 file SRT:
        1. file.srt (Theo câu)
        2. file_word.srt (Theo từ)
        """
        if not os.path.exists(audio_path):
            print(f"❌ Không tìm thấy file: {audio_path}")
            return

        print(f"\n🎧 Đang xử lý: {os.path.basename(audio_path)}")
        
        # B1. Transcribe
        audio = whisperx.load_audio(audio_path)
        result = self.model.transcribe(audio, batch_size=self.batch_size, language=language)
        
        detected_lang = result["language"]
        print(f"   -> Ngôn ngữ: {detected_lang}")
        
        # B2. Align (Căn chỉnh)
        try:
            model_a, metadata = whisperx.load_align_model(language_code=detected_lang, device=self.device)
            result = whisperx.align(
                result["segments"], 
                model_a, 
                metadata, 
                audio, 
                self.device, 
                return_char_alignments=False 
            )
            del model_a; del metadata; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️ Lỗi Align: {e}")
            return # Nếu align lỗi thì không xuất được word level chính xác

        # B3. Xuất 2 loại file
        print(f"💾 Đang lưu kết quả...")
        self._save_sentence_srt(result["segments"], audio_path)
        self._save_word_srt(result["segments"], audio_path)
        print("✅ HOÀN TẤT CẢ 2 FILE!")

    def _save_sentence_srt(self, segments, audio_path):
        """Lưu SRT theo CÂU (Dùng để xem video/Youtube)"""
        output_file = audio_path.rsplit('.', 1)[0] + ".srt"
        with open(output_file, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, start=1):
                start = format_timestamp(segment['start'])
                end = format_timestamp(segment['end'])
                text = segment['text'].strip()
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
        print(f"   -> Đã lưu File CÂU: {os.path.basename(output_file)}")

    def _save_word_srt(self, segments, audio_path):
        """Lưu SRT theo TỪ (Dùng cho Premiere Pro)"""
        output_file = audio_path.rsplit('.', 1)[0] + "_word.srt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            counter = 1
            for seg in segments:
                if 'words' not in seg: continue
                for word_info in seg['words']:
                    if 'start' in word_info and 'end' in word_info:
                        start = format_timestamp(word_info['start'])
                        end = format_timestamp(word_info['end'])
                        text = word_info['word'].strip()
                        
                        f.write(f"{counter}\n{start} --> {end}\n{text}\n\n")
                        counter += 1
                        
        print(f"   -> Đã lưu File TỪ:  {os.path.basename(output_file)}")

    def close(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

# --- 3. CHẠY ---
if __name__ == "__main__":
    # Thay đường dẫn file của bạn
    MY_AUDIO = r"C:\Users\Admin\Desktop\2351170588\44louis\Whisper\ProjectWhisperVsConda\Audio\test4.mp3"
    
    # Khởi tạo
    app = WhisperTranscriber(model_size="small") 
    
    try:
        app.transcribe_file(MY_AUDIO, language="en") 
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
    finally:
        app.close()