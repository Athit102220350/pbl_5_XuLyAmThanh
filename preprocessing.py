# processtrainvoice.py
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

DATA_DIR = "audio_dataset"
MODEL_FILENAME = "voice_command_model.pkl"

class VoiceCommandTrainer:
    def __init__(self, data_dir=DATA_DIR, model_filename=MODEL_FILENAME):
        self.data_dir = data_dir
        self.model_filename = model_filename
        self.turn_on_dir = os.path.join(self.data_dir, "turn_on")
        self.turn_off_dir = os.path.join(self.data_dir, "turn_off")

    def _extract_features(self, file_path):
        try:
            audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
        return mfccs

    def load_dataset(self):
        features, labels = [], []

        print("Processing 'turn_on' files...")
        for filename in os.listdir(self.turn_on_dir):
            if filename.endswith(".wav"):
                file_path = os.path.join(self.turn_on_dir, filename)
                mfccs = self._extract_features(file_path)
                if mfccs is not None:
                    features.append(mfccs)
                    labels.append(1)  # 1 = เปิดไฟ

        print("Processing 'turn_off' files...")
        for filename in os.listdir(self.turn_off_dir):
            if filename.endswith(".wav"):
                file_path = os.path.join(self.turn_off_dir, filename)
                mfccs = self._extract_features(file_path)
                if mfccs is not None:
                    features.append(mfccs)
                    labels.append(0)  # 0 = ปิดไฟ

        return np.array(features), np.array(labels)

    def train_and_save_model(self):
        if not os.path.exists(self.turn_on_dir) or not os.path.exists(self.turn_off_dir):
            print("Dataset folders not found. Please create 'turn_on' and 'turn_off'.")
            return

        X, y = self.load_dataset()
        if len(X) == 0:
            print("No audio files found.")
            return

        print(f"Loaded {len(X)} samples.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training model...")
        model = SVC(kernel='linear')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.2f}")

        joblib.dump(model, self.model_filename)
        print(f"Model saved to {self.model_filename}")

if __name__ == "__main__":
    trainer = VoiceCommandTrainer()
    trainer.train_and_save_model()
