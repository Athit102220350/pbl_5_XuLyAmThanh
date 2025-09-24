import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Tham số
N_MFCC = 20  # Số lượng MFCC features
NUM_CLASSES = 4  # Số lượng lớp (batDen, tatDen, batQuat, tatQuat)
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def create_model(input_shape, num_classes):
    """Tạo model neural network"""
    model = Sequential([
        # Layer 1
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        
        # Layer 2
        Dense(128, activation='relu'),
        Dropout(0.3),
        
        # Layer 3
        Dense(64, activation='relu'),
        Dropout(0.2),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """Vẽ đồ thị quá trình training"""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def train_model():
    """Huấn luyện model"""
    # Load dữ liệu đã được tiền xử lý
    print("Loading data...")
    features = np.load('features.npy')
    labels = np.load('labels.npy')
    
    # Chuyển đổi nhãn thành số
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Chia tập train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels_encoded, 
        test_size=0.2, 
        random_state=42,
        stratify=labels_encoded
    )
    
    # Chuyển nhãn sang dạng one-hot encoding
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat = to_categorical(y_val, NUM_CLASSES)
    
    # Tạo model
    print("Creating model...")
    model = create_model(N_MFCC, NUM_CLASSES)
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        ),
        ModelCheckpoint(
            'model_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Training
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Vẽ đồ thị training
    plot_training_history(history)
    
    # Lưu model cuối cùng
    model.save('model.h5')
    print("\nModel saved as 'model.h5'")
    
    # Đánh giá trên tập validation
    val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
    print(f"\nValidation accuracy: {val_acc*100:.2f}%")
    
    return model, history

if __name__ == "__main__":
    # Kiểm tra xem đã có dữ liệu tiền xử lý chưa
    if not (os.path.exists('features.npy') and os.path.exists('labels.npy')):
        print("Error: Không tìm thấy file features.npy và labels.npy")
        print("Hãy chạy preprocessing.py trước!")
        exit(1)
    
    # Train model
    model, history = train_model()