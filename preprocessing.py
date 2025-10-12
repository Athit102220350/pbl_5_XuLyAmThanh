import os
import librosa
import numpy as np
import pickle

# ==== Cấu hình ====
data_dir = r"C:\Users\souva\OneDrive\Documents\DUT_PROJECT\PBL5-TEST\data_train"
save_dir = r"C:\Users\souva\OneDrive\Documents\DUT_PROJECT\PBL5-TEST\features"

SR = 16000           # Tần số mẫu (Hz)
N_MFCC = 13          # Số lượng MFCC
MAX_DURATION = 3.0   # Thời lượng tối đa (giây) muốn chuẩn hóa
MAX_LEN = int(SR * MAX_DURATION)  # Tổng số mẫu tương ứng

# ==== Hàm chuẩn hóa độ dài + trích đặc trưng ====
def extract_features(file_path):
    # Đọc file
    y, sr = librosa.load(file_path, sr=SR)
    
    # Chuẩn hóa độ dài
    if len(y) > MAX_LEN:
        y = y[:MAX_LEN]  # Cắt nếu dài hơn
    else:
        y = np.pad(y, (0, max(0, MAX_LEN - len(y))))  # Đệm 0 nếu ngắn hơn
    
    # Trích xuất MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    
    # Lấy trung bình theo trục thời gian
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

# ==== Hàm xử lý từng tập ====
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
                features = extract_features(file_path)
                X.append(features)
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

with open(os.path.join(save_dir, "features_3_train.pkl"), "wb") as f:
    pickle.dump((X_train, y_train), f)

with open(os.path.join(save_dir, "features_3_val.pkl"), "wb") as f:
    pickle.dump((X_val, y_val), f)

with open(os.path.join(save_dir, "features_3_test.pkl"), "wb") as f:
    pickle.dump((X_test, y_test), f)

print("🎯 Đã lưu xong features_train.pkl, features_val.pkl, features_test.pkl")
