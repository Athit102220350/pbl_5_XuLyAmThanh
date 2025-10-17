import os
import librosa
import numpy as np
import pickle
import random

# ==== Cấu hình ====
data_dir = r"C:\Users\souva\OneDrive\Documents\DUT_PROJECT\PBL5-TEST\data_train"
save_dir = r"C:\Users\souva\OneDrive\Documents\DUT_PROJECT\PBL5-TEST\features"

SR = 16000
N_MFCC = 13

# ==== Hàm trích đặc trưng + tăng cường dữ liệu ====
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    features = []

    # --- 1️⃣ Bản gốc ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    features.append(np.mean(mfcc.T, axis=0))

    # --- 2️⃣ Thêm nhiễu ---
    noise = np.random.randn(len(y))
    y_noise = y + 0.005 * noise
    y_noise = np.clip(y_noise, -1.0, 1.0)
    mfcc_noise = librosa.feature.mfcc(y=y_noise, sr=sr, n_mfcc=N_MFCC)
    features.append(np.mean(mfcc_noise.T, axis=0))

    # --- 3️⃣ Kéo giãn thời gian ---
    rate = random.uniform(0.9, 1.1)
    y_stretch = librosa.effects.time_stretch(y, rate=rate)
    y_stretch = np.clip(y_stretch, -1.0, 1.0)
    mfcc_stretch = librosa.feature.mfcc(y=y_stretch, sr=sr, n_mfcc=N_MFCC)
    features.append(np.mean(mfcc_stretch.T, axis=0))

    # --- 4️⃣ Đổi cao độ ---
    n_steps = random.randint(-2, 2)
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    y_pitch = np.clip(y_pitch, -1.0, 1.0)
    mfcc_pitch = librosa.feature.mfcc(y=y_pitch, sr=sr, n_mfcc=N_MFCC)
    features.append(np.mean(mfcc_pitch.T, axis=0))

    return features

# ==== Xử lý từng tập ====
def process_set(set_type):
    X, y = [], []
    set_dir = os.path.join(data_dir, set_type)
    for label in os.listdir(set_dir):
        label_dir = os.path.join(set_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for file in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file)
            try:
                feats = extract_features(file_path)
                for f in feats:
                    X.append(f)
                    y.append(label)
            except Exception as e:
                print("⚠️ Lỗi với file:", file_path, e)
    return np.array(X), np.array(y)

# ==== Trích xuất toàn bộ ====
X_train, y_train = process_set("train")
X_val, y_val     = process_set("val")
X_test, y_test   = process_set("test")

print("✅ Train:", X_train.shape, len(y_train))
print("✅ Val:", X_val.shape, len(y_val))
print("✅ Test:", X_test.shape, len(y_test))

# ==== Lưu ra file ====
os.makedirs(save_dir, exist_ok=True)
with open(os.path.join(save_dir, "features_train.pkl"), "wb") as f:
    pickle.dump((X_train, y_train), f)
with open(os.path.join(save_dir, "features_val.pkl"), "wb") as f:
    pickle.dump((X_val, y_val), f)
with open(os.path.join(save_dir, "features_test.pkl"), "wb") as f:
    pickle.dump((X_test, y_test), f)

print("🎯 Đã lưu xong features_train.pkl, features_val.pkl, features_test.pkl")
