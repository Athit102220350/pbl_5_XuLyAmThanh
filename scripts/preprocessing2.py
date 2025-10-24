import os
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm

# ==== Cấu hình ====
data_dir = r"C:\Users\souva\OneDrive\Documents\DUT_PROJECT\PBL6-TEST\data_train"
save_dir = r"C:\Users\souva\OneDrive\Documents\DUT_PROJECT\PBL6-TEST\features"
os.makedirs(save_dir, exist_ok=True)

# ==== Hàm xử lý 1 tập dữ liệu (train / val / test) ====
def process_split(split_name):
    split_dir = os.path.join(data_dir, split_name)
    rows = []

    if not os.path.exists(split_dir):
        print(f"⚠️ Không tìm thấy thư mục: {split_dir}")
        return

    for label in tqdm(os.listdir(split_dir), desc=f"📂 Đang xử lý {split_name}"):
        label_dir = os.path.join(split_dir, label)
        if not os.path.isdir(label_dir):
            continue

        for file in os.listdir(label_dir):
            if file.lower().endswith(".wav"):
                src_path = os.path.join(label_dir, file)
                try:
                    # Đọc file âm thanh và ép về 16kHz, mono
                    y, sr = librosa.load(src_path, sr=16000, mono=True)
                    y = librosa.util.normalize(y)

                    # Thư mục lưu file đã xử lý
                    save_subdir = os.path.join(save_dir, split_name, label)
                    os.makedirs(save_subdir, exist_ok=True)
                    save_path = os.path.join(save_subdir, file)

                    # Lưu file
                    sf.write(save_path, y, 16000, subtype='PCM_16')

                    # Lưu thông tin vào danh sách
                    rows.append({
                        "path": save_path,
                        "text": label
                    })

                except Exception as e:
                    print(f"⚠️ Lỗi khi xử lý {src_path}: {e}")

    # Ghi file CSV cho tập này
    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, f"{split_name}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ Đã tạo {csv_path} ({len(df)} mẫu)")

# ==== Chạy cho từng tập ====
for split in ["train", "val", "test"]:
    process_split(split)

print("🎯 Hoàn tất toàn bộ tiền xử lý!")
