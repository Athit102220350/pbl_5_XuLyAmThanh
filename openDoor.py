""""

import speech_recognition as sr

# Khởi tạo recognizer
recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("🎤 Nói lệnh (ví dụ: 'mở cửa')...")
    # Điều chỉnh độ ồn môi trường
    recognizer.adjust_for_ambient_noise(source, duration=1)
    audio = recognizer.listen(source)

try:
    # Nhận diện giọng nói (Google API)
    text = recognizer.recognize_google(audio, language="vi-VN")
    print("👉 Bạn nói:", text)

    # Kiểm tra lệnh
    if "mở cửa" in text.lower():
        print("✅ Lệnh hợp lệ → Mở cửa!")
        # Ở đây bạn thêm code điều khiển servo/relay
    else:
        print("❌ Lệnh không đúng.")
except sr.UnknownValueError:
    print("❌ Không hiểu giọng nói.")
except sr.RequestError:
    print("❌ Không kết nối được API Google.")
open

"""