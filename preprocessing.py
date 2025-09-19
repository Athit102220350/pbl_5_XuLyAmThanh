# This file contains the logic for processing audio data and training the
# machine learning model.
# Run this script first to create the 'sound_classifier_model.pkl' file.

# --- 1. Import necessary libraries ---
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# --- 2. Define constants ---
DATA_DIR = "audio_dataset"
MODEL_FILENAME = "sound_classifier_model.pkl"

# --- 3. AudioClassifierTrainer Class ---
class AudioClassifierTrainer:
    """
    A class to handle the entire training pipeline:
    - Loading and processing audio data
    - Training the machine learning model
    - Saving the trained model to a file
    """
    def __init__(self, data_dir=DATA_DIR, model_filename=MODEL_FILENAME):
        self.data_dir = data_dir
        self.model_filename = model_filename
        self.voice_dir = os.path.join(self.data_dir, "voice")
        self.noise_dir = os.path.join(self.data_dir, "background_noise")

    def _extract_features(self, file_path):
        """
        Extracts features from an audio file. This is a private helper method.
        """
        try:
            audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        except Exception as e:
            print(f"Error encountered while parsing file: {file_path}")
            return None
        return mfccs

    def load_dataset(self):
        """
        Loads audio files from the defined directories and extracts features.
        """
        features = []
        labels = []

        print("Processing voice files...")
        for filename in os.listdir(self.voice_dir):
            if filename.endswith(".wav"):
                file_path = os.path.join(self.voice_dir, filename)
                mfccs = self._extract_features(file_path)
                if mfccs is not None:
                    features.append(mfccs)
                    labels.append(1)  # Label for voice

        print("Processing background noise files...")
        for filename in os.listdir(self.noise_dir):
            if filename.endswith(".wav"):
                file_path = os.path.join(self.noise_dir, filename)
                mfccs = self._extract_features(file_path)
                if mfccs is not None:
                    features.append(mfccs)
                    labels.append(0)  # Label for noise

        return np.array(features), np.array(labels)

    def train_and_save_model(self):
        """
        Main method to train the model and save it to a file.
        """
        if not os.path.exists(self.voice_dir) or not os.path.exists(self.noise_dir):
            print("Required folder structure not found.")
            print(f"Please create '{self.data_dir}' with subfolders 'voice' and 'background_noise'.")
            return

        X, y = self.load_dataset()

        if len(X) == 0:
            print("No audio files found. Please add .wav files to your folders.")
            return

        print(f"Loaded {len(X)} audio files.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training the model...")
        model = SVC(kernel='linear')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained successfully. Accuracy on test data: {accuracy:.2f}")

        joblib.dump(model, self.model_filename)
        print(f"Model saved to {self.model_filename}")

# --- 4. Main Execution Block for Training ---
if __name__ == "__main__":
    trainer = AudioClassifierTrainer()
    trainer.train_and_save_model()
