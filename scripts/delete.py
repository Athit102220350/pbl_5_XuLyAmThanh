import os

# Đường dẫn dataset
data_dir = r"C:\Users\souva\OneDrive\Documents\PBL5-TEST\data"

count = 0

# Duyệt qua toàn bộ thư mục con
for root, _, files in os.walk(data_dir):
    for file in files:
        if file.lower().endswith(".m4a"):
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print("🗑️ Đã xóa:", file_path)
                count += 1
            except Exception as e:
                print("❌ Lỗi khi xóa:", file_path, e)

print(f"\n✅ Đã xóa tổng cộng {count} file .m4a")
