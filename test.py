# test_realtime.py
import sounddevice as sd
import numpy as np
import librosa
from tensorflow.keras.models import load_model

DURATION = 2      # thời gian ghi âm (giây)
FS = 16000        # tần số mẫu
N_MFCC = 20       # phải giống lúc train

# Load model
model = load_model("model.h5")  # đường dẫn model bạn đã train
label2idx = {"batDen":0, "batQuat":1, "tatDen":2, "tatQuat":3}
idx2label = {v:k for k,v in label2idx.items()}

def record_audio(duration=DURATION, fs=FS):
    print("🎤 Nói đi, đang ghi âm...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    audio = audio.flatten()
    return audio

def extract_features(y, sr=FS, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

def predict_with_confidence(model, features, confidence_threshold=0.7):
    """Dự đoán với ngưỡng tin cậy"""
    probs = model.predict(features)[0]
    max_prob = np.max(probs)
    pred_idx = np.argmax(probs)
    
    if max_prob >= confidence_threshold:
        return idx2label[pred_idx], max_prob
    return "unknown", max_prob

# ==== MAIN ====
y_audio = record_audio()
features = extract_features(y_audio)
features = np.expand_dims(features, axis=0)  # reshape (1, N_MFCC)

# Dự đoán
pred_label, confidence = predict_with_confidence(model, features)
if confidence < 0.7:
    pred_label = "unknown"

print(f"✅ Nhận diện: {pred_label}")
