import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import glob
import time

def record(filename, duration=2, fs=16000):
    """Ghi âm và lưu file WAV"""
    print("🎤 Bắt đầu ghi âm ...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, fs, audio)
    print("✅ Đã lưu:", filename)
    return audio

labels = ["batDen", "tatDen", "batQuat", "tatQuat", "tatTatCa", "batTatCa"]

# Tạo thư mục cho từng nhãn
for label in labels:
    os.makedirs(f"data/{label}", exist_ok=True)

def next_filename(label, base="data"):
    """Tìm tên file tiếp theo không bị trùng"""
    folder = f"{base}/{label}"
    files = glob.glob(f"{folder}/sample*.wav")

    indices = []
    for f in files:
        try:
            num = int(os.path.basename(f).replace("sample", "").replace(".wav", ""))
            indices.append(num)
        except:
            pass

    next_index = max(indices) + 1 if indices else 1
    return f"{folder}/sample{next_index}.wav"

def check_audio_quality(audio, min_amplitude=0.1):
    """Kiểm tra chất lượng âm thanh"""
    if np.max(np.abs(audio)) < min_amplitude:
        return False
    return True

def batch_record(label, num_samples=30, duration=2):
    """Thu thập nhiều mẫu liên tục"""
    print(f"=== Bắt đầu ghi {num_samples} mẫu cho nhãn '{label}' ===")
    for i in range(num_samples):
        print(f"\n🎧 Mẫu {i+1}/{num_samples}")
        filename = next_filename(label)
        while True:
            audio = record(filename, duration=duration)
            if check_audio_quality(audio):
                break
            print("⚠️ Âm lượng quá nhỏ, vui lòng thử lại")
        time.sleep(0.5)  # nghỉ 0.5s giữa các mẫu (đỡ bị chồng âm)
    print("✅ Hoàn tất ghi âm tất cả mẫu!")

if __name__ == "__main__":
    label = "tatQuat"   # đổi label khi cần
    batch_record(label)
