import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocessor import prepare_data
from model_factory import create_lstm_model

# --- KONFIGURASI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "commodity_lstm_model.keras")
BATCH_SIZE = 512
EPOCHS = 20

def run_training():
    """Proses utama pelatihan model dengan penanganan imbalance."""
    # 1. Load Data
    X, y, classes = prepare_data()
    num_classes = len(classes)
    
    # 2. Split Data (80% Train, 20% Test)
    print("[INFO] Membagi data menjadi train dan test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Hitung Bobot Kelas (Class Weighting) untuk Mengatasi Imbalance
    print("[INFO] Menghitung bobot kelas untuk mengatasi imbalance...")
    # y_train berisi label integer
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: weights[i] for i in range(len(weights))}
    print(f"Bobot Kelas: {class_weights}")
    for i, name in enumerate(classes):
        print(f"  - Kelas {name} (ID {i}): {class_weights[i]:.4f}")

    # 4. Create Model
    print("[INFO] Membangun arsitektur model...")
    input_shape = (X.shape[1], X.shape[2]) # (SEQ_LENGTH, N_FEATURES)
    model = create_lstm_model(input_shape, num_classes)
    model.summary()
    
    # 5. Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)
    ]
    
    # 6. Training dengan Class Weights
    print(f"[INFO] Memulai pelatihan dengan Class Weighting (Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # 7. Evaluation
    print("\n[INFO] Mengevaluasi model pada data test...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # 8. Simpan Grafik Training (Opsional)
    print("[INFO] Menyimpan grafik history pelatihan...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.savefig(os.path.join(BASE_DIR, 'models', 'training_history.png'))
    print("[SUCCESS] Pelatihan selesai. Model dan grafik telah disimpan.")

if __name__ == "__main__":
    run_training()
