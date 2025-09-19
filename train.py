# This file is used to test a trained machine learning model.
# It loads the model created by processtrainsignal.py and makes a prediction.

# --- 1. Import necessary libraries ---
import os
import numpy as np
import librosa
import joblib

# --- 2. Define constants and file paths ---
MODEL_FILENAME = "sound_classifier_model.pkl"
DATA_DIR = "audio_dataset"

# --- 3. AudioClassifierTester Class ---
class AudioClassifierTester:
    """
    A class to load a saved model and make predictions on new audio files.
    """
    def __init__(self, model_filename=MODEL_FILENAME):
        self.model_filename = model_filename
        self.model = self._load_model()
    
    def _load_model(self):
        """
        Loads the trained model from the file.
        """
        if not os.path.exists(self.model_filename):
            print(f"Error: Model file '{self.model_filename}' not found.")
            return None
        return joblib.load(self.model_filename)

    def _extract_features(self, file_path):
        """
        Helper method to extract features from a single new audio file.
        """
        try:
            audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        except Exception as e:
            print(f"Error encountered while processing file: {file_path}")
            return None
        return mfccs

    def predict(self, file_path):
        """
        Makes a prediction on a single audio file.
        Returns 1 for a 'clap' and 0 for 'background noise'.
        """
        if self.model is None:
            return "Model not loaded. Please train a model first."

        print(f"\nMaking a prediction on: {file_path}")
        features = self._extract_features(file_path)

        if features is not None:
            # Reshape the features to a 2D array for the model
            features = features.reshape(1, -1)
            prediction = self.model.predict(features)
            return prediction[0]
        else:
            return "Could not extract features from the audio file."

# --- 4. Main Execution Block for Testing ---
if __name__ == "__main__":
    tester = AudioClassifierTester()

    # Simulate a test file by picking the first file from the clap directory.
    # NOTE: You must have run processtrainsignal.py at least once to create the model.
    # You must also have audio files in your 'audio_dataset/claps' folder.
    clap_dir = os.path.join(DATA_DIR, "claps")
    if os.path.exists(clap_dir) and os.listdir(clap_dir):
        test_file_path = os.path.join(clap_dir, os.listdir(clap_dir)[0])
        prediction = tester.predict(test_file_path)
        
        if prediction == 1:
            print("Prediction: A clap sound was detected.")
        elif prediction == 0:
            print("Prediction: Background noise was detected.")
        else:
            print(prediction) # Print the error message
    else:
        print("\nCould not find a test file. Please ensure you have audio files in your 'audio_dataset/claps' folder and have trained the model first.")
