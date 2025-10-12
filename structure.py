import os
import glob
import shutil
from sklearn.model_selection import train_test_split

# Thư mục gốc
data_dir = r"C:\Users\souva\OneDrive\Documents\DUT_PROJECT\PBL5-TEST\data"
output_dir = r"C:\Users\souva\OneDrive\Documents\DUT_PROJECT\PBL5-TEST\data_train"

# Tạo thư mục train/val/test
for folder in ["train", "val", "test"]:
    for label in os.listdir(data_dir):
        os.makedirs(os.path.join(output_dir, folder, label), exist_ok=True)

# Duyệt từng class (batDen, tatDen, ...)
for label in os.listdir(data_dir):
    files = glob.glob(os.path.join(data_dir, label, "*"))

    # Chia train (70%), temp (30%)
    train_files, temp_files = train_test_split(files, test_size=0.3, random_state=42)

    # Chia tiếp temp thành val (15%) và test (15%)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    # Copy file sang thư mục mới
    for f in train_files:
        shutil.copy(f, os.path.join(output_dir, "train", label))
    for f in val_files:
        shutil.copy(f, os.path.join(output_dir, "val", label))
    for f in test_files:
        shutil.copy(f, os.path.join(output_dir, "test", label))

print("✅ Đã chia dữ liệu xong!")
