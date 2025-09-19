import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os

def record(filename, duration=2, fs=16000):
    print(" bat dau ghi am ....")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1,dtype='int16')
    sd.wait()
    wav.write(filename, fs,audio)
    print(" da luu:", filenae)

labels = ["batDen","tatDen","batQuat", "tatQuat"]

for i, label in labels :
    os.makedirs(f"data/{label}",exist_ok =True)

record("data/batDen/sample1.wav")

