import os
from collections import Counter
import matplotlib.pyplot as plt

data_dir = r"C:\Users\souva\OneDrive\Documents\PBL5-TEST\data_train"

# Đếm tổng số file trong mỗi lớp (tổng cả train/val/test)
counts = Counter()

for set_type in ["train", "val", "test"]:
    set_dir = os.path.join(data_dir, set_type)
    for label in os.listdir(set_dir):
        label_dir = os.path.join(set_dir, label)
        n_files = len([f for f in os.listdir(label_dir) if f.endswith((".wav", ".m4a"))])
        counts[label] += n_files

# Vẽ bar chart
plt.figure(figsize=(6,4))
plt.bar(counts.keys(), counts.values(), color='skyblue')
plt.title("Tổng số lượng mẫu dữ liệu theo lớp")
plt.xlabel("Lệnh")
plt.ylabel("Số lượng mẫu")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
