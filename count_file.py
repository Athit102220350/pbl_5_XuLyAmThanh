import os

root_path = r"C:\Users\souva\OneDrive\Documents\DUT_PROJECT\PBL6-TEST\data"

# Duyệt qua tất cả thư mục con
for folder_name in os.listdir(root_path):
    folder_path = os.path.join(root_path, folder_name)
    if not os.path.isdir(folder_path):
        continue  # bỏ qua file

    # Lấy danh sách file (chỉ tính file, bỏ thư mục con khác nếu có)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    count = len(files)

    print(f"📁 {folder_name}: {count} file")

print("✅ Đã đếm xong tất cả thư mục.")
