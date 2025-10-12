import os

root_path = r"C:\Users\souva\Downloads\tattatca (1)\tattatca"

for idx, file_name in enumerate(os.listdir(root_path), start=1):
    file_path = os.path.join(root_path, file_name)
    if not os.path.isfile(file_path):
        continue  # bỏ qua thư mục

    ext = os.path.splitext(file_name)[1]
    new_name = f"jo_{idx}{ext}"
    new_path = os.path.join(root_path, new_name)
    os.rename(file_path, new_path)
    print(f"Đã đổi tên: {file_name} → {new_name}")

print("✅ Xong việc đổi tên tất cả file.")
