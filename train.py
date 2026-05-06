import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocessor import prepare_data
from model_factory import create_lstm_model

# --- KONFIGURASI ---
MODEL_PATH = "models/commodity_lstm_model.keras"
BATCH_SIZE = 64
EPOCHS = 30

def run_training():
    """Proses utama pelatihan model."""
    # 1. Load Data
    X, y, classes = prepare_data()
    num_classes = len(classes)
    
    # 2. Split Data (80% Train, 20% Test)
    print("[INFO] Membagi data menjadi train dan test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Create Model
    print("[INFO] Membangun arsitektur model...")
    input_shape = (X.shape[1], X.shape[2]) # (SEQ_LENGTH, N_FEATURES)
    model = create_lstm_model(input_shape, num_classes)
    model.summary()
    
    # 4. Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True)
    ]
    
    # 5. Training
    print(f"[INFO] Memulai pelatihan (Epochs: {EPOCHS})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. Evaluation
    print("\n[INFO] Mengevaluasi model pada data test...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # 7. Simpan Grafik Training (Opsional)
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
    
    plt.savefig('models/training_history.png')
    print("[SUCCESS] Pelatihan selesai. Model dan grafik telah disimpan.")

if __name__ == "__main__":
    run_training()
