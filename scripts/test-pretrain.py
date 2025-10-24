import torch
from transformers import AutoProcessor, AutoModelForCTC
import librosa

# Đường dẫn tới checkpoint bạn đã train
model_path = "./results/checkpoint-315"

# Load processor và model
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForCTC.from_pretrained(model_path)

# Đọc file âm thanh
speech, sr = librosa.load("./data/batDen/binh_1.wav", sr=16000)

# Chuẩn bị dữ liệu đầu vào
inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)

# Dự đoán
with torch.no_grad():
    logits = model(**inputs).logits

# Lấy nhãn dự đoán
predicted_ids = torch.argmax(logits, dim=-1)

# Giải mã thành text
transcription = processor.batch_decode(predicted_ids)[0]

print("👉 Văn bản dự đoán:", transcription)
