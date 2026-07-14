"""
train_v2.py — Training script untuk Model LSTM v2 (Dual-Input Hybrid)

Perbedaan dari train.py (v1):
- Menggunakan prepare_data_v2() yang menghasilkan dua input:
    X_seq    (N, 30, 3) : sequence iklim harian
    X_static (N, 6)     : fitur lahan statis (elevasi, curah hujan, pH, liat, pasir, debu)
- Menggunakan create_lstm_model_v2() dengan arsitektur Functional API dual-input
- Menyimpan model ke commodity_lstm_model_v2.keras (tidak mengganti model v1)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import modul lokal
from preprocessor import prepare_data_v2
from model_factory import create_lstm_model_v2

# --- KONFIGURASI ---
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "commodity_lstm_model_v2.keras")
BATCH_SIZE = 128
EPOCHS     = 20  # Lebih banyak epoch karena arsitektur lebih kompleks


def run_training_v2():
    """Proses pelatihan model LSTM v2 dengan fitur tanah & topografi lengkap."""

    # 1. Load Data (dual-input)
    X_seq, X_static, y, classes = prepare_data_v2()
    num_classes = len(classes)

    print(f"\n[INFO] Jumlah kelas: {num_classes} — {list(classes)}")
    print(f"[INFO] Input sequential : {X_seq.shape}")
    print(f"[INFO] Input static     : {X_static.shape}")

    # 2. Split Data (80% Train, 20% Test) — seed sama dengan v1 agar split konsisten
    print("\n[INFO] Membagi data menjadi train dan test (80/20)...")
    (X_seq_train, X_seq_test,
     X_static_train, X_static_test,
     y_train, y_test) = train_test_split(
        X_seq, X_static, y,
        test_size=0.2, random_state=42
    )

    print(f"  Train: {len(y_train)} samples | Test: {len(y_test)} samples")

    # 3. Hitung Bobot Kelas untuk mengatasi imbalance
    print("\n[INFO] Menghitung bobot kelas...")
    weights      = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: weights[i] for i in range(len(weights))}
    for i, name in enumerate(classes):
        print(f"  - {name} (ID {i}): {class_weights[i]:.4f}")

    # 4. Bangun Model v2
    print("\n[INFO] Membangun arsitektur model LSTM v2 (Dual-Input Hybrid)...")
    seq_input_shape   = (X_seq.shape[1], X_seq.shape[2])   # (30, 3)
    n_static_features = X_static.shape[1]                  # 6

    model = create_lstm_model_v2(seq_input_shape, n_static_features, num_classes)
    model.summary()

    # 5. Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # 6. Training
    print(f"\n[INFO] Memulai pelatihan (max {EPOCHS} epochs, batch {BATCH_SIZE})...")
    print("[INFO] Input: [X_seq, X_static] -> menggunakan dua branch model\n")

    history = model.fit(
        [X_seq_train, X_static_train],
        y_train,
        validation_data=([X_seq_test, X_static_test], y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # 7. Evaluasi Akhir
    print("\n[INFO] Mengevaluasi model v2 pada data test...")
    test_loss, test_acc = model.evaluate(
        [X_seq_test, X_static_test], y_test, verbose=0
    )
    print(f"[RESULT] Test Accuracy : {test_acc*100:.2f}%")
    print(f"[RESULT] Test Loss     : {test_loss:.4f}")

    # 8. Simpan Grafik Training History
    eval_dir = os.path.join(BASE_DIR, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'],     label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model v2 — Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'],     label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model v2 — Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    history_path = os.path.join(eval_dir, "training_history_v2.png")
    plt.savefig(history_path, dpi=150)
    plt.close()

    print(f"\n[SUCCESS] Model v2 tersimpan di: {MODEL_PATH}")
    print(f"[SUCCESS] Grafik training tersimpan di: {history_path}")
    print("\n[NEXT STEP] Jalankan evaluate_v2.py untuk laporan perbandingan lengkap.")


if __name__ == "__main__":
    run_training_v2()
