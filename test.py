import librosa
import numpy as np
import joblib

# ==== B1: Load model đã train ====
model_dir = r"C:\Users\souva\OneDrive\Documents\PBL5-TEST\model"
clf = joblib.load(f"{model_dir}/svm_model.pkl")
scaler = joblib.load(f"{model_dir}/scaler.pkl")
le = joblib.load(f"{model_dir}/label_encoder.pkl")

# ==== B2: Hàm trích MFCC từ file mới ====
SR = 16000
N_MFCC = 13

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc

# ==== B3: Dự đoán file audio ====
file_audio = r"C:\Users\souva\OneDrive\Documents\PBL5-TEST\data\tatQuat\jo_2.wav"  # đổi sang file của bạn
features = extract_features(file_audio)
features_scaled = scaler.transform([features])  # nhớ reshape 1 sample
pred = clf.predict(features_scaled)
pred_label = le.inverse_transform(pred)

print(f"🎯 File {file_audio} được dự đoán là: {pred_label[0]}")
