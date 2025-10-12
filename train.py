import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

# ==== B1: Load đặc trưng đã trích xuất từ Bước 3 ====
with open(r"features\features_3_train.pkl", "rb") as f:
    X_train, y_train = pickle.load(f)

with open(r"features\features_3_val.pkl", "rb") as f:
    X_val, y_val = pickle.load(f)

with open(r"features\features_3_test.pkl", "rb") as f:
    X_test, y_test = pickle.load(f)

# Convert sang numpy
X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)

print("Train size:", len(y_train))
print("Val size:", len(y_val))
print("Test size:", len(y_test))

# ==== B2: Encode nhãn ====
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

# ==== B3: Chuẩn hóa dữ liệu ====
scaler = StandardScaler() #tạo một đối tượng chuẩn hoá dữ liệu
#Chuyển các danh sách/structure sang numpy arrays để thuận tiện xử lý số học và tương thích với scikit-learn.
X_train_scaled = scaler.fit_transform(X_train) 
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ==== B4: Huấn luyện SVM ====
clf = SVC(kernel="linear", probability=True)
clf.fit(X_train_scaled, y_train_enc)

# ==== B5: Đánh giá trên validation ====
val_pred = clf.predict(X_val_scaled)
print("\n📊 Validation accuracy:", accuracy_score(y_val_enc, val_pred))

# ==== B6: Đánh giá trên test ====
test_pred = clf.predict(X_test_scaled)
print("\n📊 Test accuracy:", accuracy_score(y_test_enc, test_pred))
print(classification_report(y_test_enc, test_pred, target_names=le.classes_))

# ==== B7: Xuất confusion matrix ====
cm = confusion_matrix(y_test_enc, test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

plt.figure(figsize=(6,6))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix - Test Set")
plt.show()

# ==== B8: Lưu model, scaler, encoder ====
model_dir = r"C:\Users\souva\OneDrive\Documents\DUT_PROJECT\PBL5-TEST\model"
os.makedirs(model_dir, exist_ok=True)

joblib.dump(clf, os.path.join(model_dir, "svm_model3.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "scaler3.pkl"))
joblib.dump(le, os.path.join(model_dir, "label_encoder3.pkl"))

print(f"\n✅ Đã lưu SVM model, scaler và encoder vào thư mục: {model_dir}")
