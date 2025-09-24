import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import glob

def record(filename, duration=2, fs=16000):
    """Ghi âm và lưu file WAV"""
    print("🎤 Bắt đầu ghi âm ...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, fs, audio)
    print("✅ Đã lưu:", filename)

    return audio  # Trả về dữ liệu âm thanh để kiểm tra chất lượng

labels = ["batDen", "tatDen", "batQuat", "tatQuat"]

# Tạo thư mục cho từng nhãn
for label in labels:
    os.makedirs(f"data/{label}", exist_ok=True)

def next_filename(label, base="data"):
    """Tìm tên file tiếp theo không bị trùng"""
    folder = f"{base}/{label}"
    files = glob.glob(f"{folder}/sample*.wav")

    # Lấy số thứ tự từ tên file
    indices = []
    for f in files:
        try:
            num = int(os.path.basename(f).replace("sample", "").replace(".wav", ""))
            indices.append(num)
        except:
            pass

    # Nếu chưa có file thì bắt đầu từ 1
    next_index = max(indices) + 1 if indices else 1
    return f"{folder}/sample{next_index}.wav"

def check_audio_quality(audio, min_amplitude=0.1):
    """Kiểm tra chất lượng âm thanh"""
    if np.max(np.abs(audio)) < min_amplitude:
        return False
    return True

def batch_record(label, num_samples=5):
    """Thu thập nhiều mẫu liên tiếp"""
    for i in range(num_samples):
        filename = next_filename(label)
        while True:
            audio = record(filename)
            if check_audio_quality(audio):
                break
            print("⚠️ Âm lượng quá nhỏ, vui lòng thử lại")

if __name__ == "__main__":
    # Ví dụ: ghi âm cho nhãn "tatQuat"
    label = "batQuat"   # đổi thành batDen / tatDen / batQuat / tatQuat khi cần
    filename = next_filename(label)
    record(filename)
