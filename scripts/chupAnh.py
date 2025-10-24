import cv2
import os

# Tạo thư mục để lưu ảnh
save_path = "faces"
os.makedirs(save_path, exist_ok=True)

# Mở camera (0 = camera mặc định)
cap = cv2.VideoCapture(0)

count = 0
total_images = 59

while count < total_images:
    ret, frame = cap.read()
    if not ret:
        print("Không lấy được khung hình!")
        break

    # Hiển thị khung hình
    cv2.imshow("Capture Face", frame)

    # Lưu ảnh
    filename = os.path.join(save_path, f"face_{count+1}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Đã lưu {filename}")

    count += 1

    # Nhấn phím q để thoát sớm
    if cv2.waitKey(300) & 0xFF == ord('q'):
        break

print("Hoàn thành chụp 59 ảnh.")
cap.release()
cv2.destroyAllWindows()
