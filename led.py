"""
import serial
import speech_recognition as sr

arduino = serial.Serial('COM5' ,9600)
recognizer = sr.Recognizer()

while True:
    with sr.Microphone() as source:
        print("🎤 Nói lệnh (ví dụ: 'bật đèn' hoặc 'tắt đèn')...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language="vi-VN")
        print(" ban noi :", text)

        if "bật đèn" in text.lower():
            arduino.write(b'1 ')# gui ky tu 1
            print("Đèn đã được bật.")
        elif "tắt đèn" in text.lower():
            arduino.write(b'0 ')# gui ky tu 0
            print("Đèn đã được tắt.")
        else:
            print("Lệnh không hợp lệ.")
    except:
        print(" ko nhan dien duoc. ")
"""