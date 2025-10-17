import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import joblib

# ==== B1: Load model ====
model_dir = r"C:\Users\souva\OneDrive\Documents\DUT_PROJECT\PBL5-TEST\model"
clf = joblib.load(f"{model_dir}/svm_model3.pkl")
scaler = joblib.load(f"{model_dir}/scaler3.pkl")
le = joblib.load(f"{model_dir}/label_encoder3.pkl")

# ==== B2: Hàm trích MFCC ====
SR = 16000
N_MFCC = 13     

def extract_features(y, sr=SR):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc

# ==== B3: Ghi âm trực tiếp ====
duration = 2 # giây
print("🎙️ Bắt đầu ghi âm...")
recording = sd.rec(int(duration * SR), samplerate=SR, channels=1)
sd.wait()
y = recording.flatten()

# ==== B4: Dự đoán ====
features = extract_features(y)
features_scaled = scaler.transform([features])
pred = clf.predict(features_scaled)
pred_label = le.inverse_transform(pred)

print(f"🎯 Dự đoán trực tiếp: {pred_label[0]}")
