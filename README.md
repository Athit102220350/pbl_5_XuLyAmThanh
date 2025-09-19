# create file venc
# pip install scikit-learn
# pip install --upgrade scikit-learn


* หลักการทำงานของการ เทรนเสียง

เก็บ Dataset

โฟลเดอร์ voice/ → เก็บเสียงที่ต้องการให้โมเดลรู้จัก เช่น เสียงพูด “เปิดไฟ”, “ปิดไฟ” หรือคำสั่งต่าง ๆ

โฟลเดอร์ background_noise/ → เก็บเสียงรบกวน เช่น เสียงลม เสียงพัดลม เสียงคนคุย เสียงรถ ฯลฯ
👉 จุดประสงค์คือให้โมเดลแยกให้ออกว่า อะไรคือ “เสียงจริง” (voice) และอะไรคือ “เสียงรบกวน” (noise)

Preprocessing (การประมวลผลเบื้องต้น)
หลักการทำงานของการ เทรนเสียง

เก็บ Dataset

โฟลเดอร์ voice/ → เก็บเสียงที่ต้องการให้โมเดลรู้จัก เช่น เสียงพูด “เปิดไฟ”, “ปิดไฟ” หรือคำสั่งต่าง ๆ

โฟลเดอร์ background_noise/ → เก็บเสียงรบกวน เช่น เสียงลม เสียงพัดลม เสียงคนคุย เสียงรถ ฯลฯ
👉 จุดประสงค์คือให้โมเดลแยกให้ออกว่า อะไรคือ “เสียงจริง” (voice) และอะไรคือ “เสียงรบกวน” (noise)

Preprocessing (การประมวลผลเบื้องต้น)

เสียงที่บันทึกมามักอยู่ในรูป .wav หรือ .mp3

จะถูกตัดเป็น segment (เช่น 1 วินาทีต่อคลิป)

แปลงจาก waveform → spectrogram / MFCC (Mel-frequency cepstral coefficients) ซึ่งเป็นการแปลงเสียงให้กลายเป็นภาพ (เหมือนกราฟความถี่ตามเวลา)

Training Model

ใช้โครงข่ายประสาทเทียม เช่น CNN (Convolutional Neural Network) หรือ RNN (LSTM/GRU)

ป้อน voice และ background_noise ให้โมเดลเรียนรู้ว่า เสียงแบบไหนคือเสียงสั่งงาน และเสียงแบบไหนคือ noise

โมเดลจะเรียนรู้ pattern ของเสียงพูด เช่น ความถี่ของเสียงพูดมนุษย์ ต่างจากเสียงรบกวน
เสียงที่บันทึกมามักอยู่ในรูป .wav หรือ .mp3

จะถูกตัดเป็น segment (เช่น 1 วินาทีต่อคลิป)

แปลงจาก waveform → spectrogram / MFCC (Mel-frequency cepstral coefficients) ซึ่งเป็นการแปลงเสียงให้กลายเป็นภาพ (เหมือนกราฟความถี่ตามเวลา)

Training Model

ใช้โครงข่ายประสาทเทียม เช่น CNN (Convolutional Neural Network) หรือ RNN (LSTM/GRU)

ป้อน voice และ background_noise ให้โมเดลเรียนรู้ว่า เสียงแบบไหนคือเสียงสั่งงาน และเสียงแบบไหนคือ noise

โมเดลจะเรียนรู้ pattern ของเสียงพูด เช่น ความถี่ของเสียงพูดมนุษย์ ต่างจากเสียงรบกวน



# project structure
project/
├── processtrainsignal.py
├── testsignal.py
└── audio_dataset/
    ├── voice/
    │   └── file1.wav
    └── background_noise/
        └── file2.wav
