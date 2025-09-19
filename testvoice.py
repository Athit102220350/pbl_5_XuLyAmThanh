# testvoice.py
import os
import numpy as np
import librosa
import joblib
import RPi.GPIO as GPIO
import time

MODEL_FILENAME = "voice_command_model.pkl"
DATA_DIR = "audio_dataset"

# ตั้งค่า GPIO
GPIO.setmode(GPIO.BCM)
LED_PIN = 17
GPIO.setup(LED_PIN, GPIO.OUT)

class VoiceCommandTester:
    def __init__(self, model_filename=MODEL_FILENAME):
        self.model_filename = model_filename
        self.model = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_filename):
            print(f"Error: Model file '{self.model_filename}' not found.")
            return None
        return joblib.load(self.model_filename)

    def _extract_features(self, file_path):
        try:
            audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
        return mfccs

    def predict(self, file_path):
        if self.model is None:
            return "Model not loaded"

        print(f"\nTesting file: {file_path}")
        features = self._extract_features(file_path)
        if features is not None:
            features = features.reshape(1, -1)
            prediction = self.model.predict(features)
            return prediction[0]
        return None

if __name__ == "__main__":
    tester = VoiceCommandTester()

    test_dir = os.path.join(DATA_DIR, "turn_on")
    if os.path.exists(test_dir) and os.listdir(test_dir):
        test_file = os.path.join(test_dir, os.listdir(test_dir)[0])
        prediction = tester.predict(test_file)

        if prediction == 1:
            print("🔊 คำสั่ง: เปิดไฟ")
            GPIO.output(LED_PIN, GPIO.HIGH)
        elif prediction == 0:
            print("🔊 คำสั่ง: ปิดไฟ")
            GPIO.output(LED_PIN, GPIO.LOW)
        else:
            print("Prediction failed.")
    else:
        print("No test files found.")
