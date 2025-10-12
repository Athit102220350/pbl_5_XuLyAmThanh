import os
import librosa
import soundfile as sf
import numpy as np

# đường dẫn gốc
root_path = r"C:\Users\souva\OneDrive\Documents\PBL6-TEST\jo_voice"

# duyệt từng thư mục lệnh
for command_folder in os.listdir(root_path):
    command_path = os.path.join(root_path, command_folder)
    if not os.path.isdir(command_path):
        continue

    # duyệt từng file trong thư mục
    for file_name in os.listdir(command_path):
        if not file_name.lower().endswith((".wav", ".flac", ".mp3" , ".m4a")):
            continue
        file_path = os.path.join(command_path, file_name)

        # đọc file, giữ nguyên sr gốc
        y, sr = librosa.load(file_path, sr=None)
        # chuyển về 16kHz
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
        # chuyển stereo sang mono nếu cần
        if len(y_resampled.shape) > 1:
            y_resampled = np.mean(y_resampled, axis=1)
        # lưu đè lên file cũ
        sf.write(file_path, y_resampled, 16000)

print("Xử lý xong tất cả file audio.")
