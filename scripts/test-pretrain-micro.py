import torch
import sounddevice as sd
import numpy as np
from transformers import AutoProcessor, AutoModelForCTC

# Đường dẫn tới checkpoint
model_path = "./results/checkpoint-315"

# Load model & processor
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForCTC.from_pretrained(model_path)

# Cấu hình ghi âm
sr = 16000   # sample rate (phải khớp với model)
duration = 5  # số giây ghi âm

print("🎙️ Đang ghi âm... Nói gì đó nhé!")
audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
sd.wait()
print("✅ Ghi âm xong!")

# Chuyển sang dạng 1D numpy array
speech = np.squeeze(audio)

# Chuẩn bị input
inputs = processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)

# Dự đoán
with torch.no_grad():
    logits = model(**inputs).logits

# Giải mã
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]

print("🗣️ Kết quả nhận dạng:", transcription)
