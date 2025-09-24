# preprocessing_with_augmentation.py
import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATASET_PATH = "data"   # thư mục chứa các folder con (mỗi folder = 1 label)
OUTPUT_FEATURES = "features.npy"
OUTPUT_LABELS = "labels.npy"

# ---- Hàm tăng cường dữ liệu ----
def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise

def shift_time(data, shift_max=0.2):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(data))
    return np.roll(data, shift)

def change_pitch(data, sr, n_steps=2):
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)

def change_speed(data, speed_factor=1.2):
    return librosa.effects.time_stretch(y=data, rate=speed_factor)


# ---- Hàm trích xuất đặc trưng MFCC ----
def extract_features(data, sr, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# ---- Kiểm tra chất lượng âm thanh ----
def check_audio_quality(audio, min_amplitude=0.1):
    """Kiểm tra chất lượng âm thanh"""
    if np.max(np.abs(audio)) < min_amplitude:
        return False
    return True

# ---- Pipeline xử lý dữ liệu ----
features, labels = [], []

for label in os.listdir(DATASET_PATH):
    class_dir = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(class_dir):
        continue

    print(f"🔍 Đang xử lý lớp: {label}")

    for file in os.listdir(class_dir):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(class_dir, file)

        try:
            y, sr = librosa.load(file_path, sr=16000, mono=True)

            # Kiểm tra chất lượng âm thanh
            if not check_audio_quality(y):
                print(f"⚠️ Âm thanh {file_path} có chất lượng kém, bỏ qua.")
                continue

            # 1. Gốc
            features.append(extract_features(y, sr))
            labels.append(label)

            # 2. Nhiễu
            y_noise = add_noise(y)
            features.append(extract_features(y_noise, sr))
            labels.append(label)

            # 3. Pitch shift
            y_pitch = change_pitch(y, sr, n_steps=2)
            features.append(extract_features(y_pitch, sr))
            labels.append(label)

            # 4. Speed change
            y_speed = change_speed(y, speed_factor=1.2)
            features.append(extract_features(y_speed, sr))
            labels.append(label)

        except Exception as e:
            print(f"Lỗi khi xử lý {file_path}: {e}")

# ---- Chuẩn hóa dữ liệu ----
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ---- Chia tập train/test ----
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, labels, test_size=0.2, random_state=42
)

# ---- Lưu dữ liệu ra file numpy ----
np.save(OUTPUT_FEATURES, np.array(features))
np.save(OUTPUT_LABELS, np.array(labels))

print(f"✅ Đã lưu đặc trưng vào {OUTPUT_FEATURES}, {OUTPUT_LABELS}")
