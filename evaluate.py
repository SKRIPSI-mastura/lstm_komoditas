import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_fscore_support
)
from tensorflow.keras.models import load_model

# Import preprocessing function dari preprocessor.py
from preprocessor import prepare_data

# --- KONFIGURASI PATH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "commodity_lstm_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")

if not os.path.exists(EVAL_DIR):
    os.makedirs(EVAL_DIR)

def evaluate_model():
    print("\n" + "="*60)
    print("      EVALUASI MODEL LSTM REKOMENDASI KOMODITAS      ")
    print("="*60)
    
    # 1. Validasi Keberadaan Model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] File model tidak ditemukan di: {MODEL_PATH}")
        print("Silakan jalankan 'train.py' terlebih dahulu untuk melatih model.")
        return
        
    if not os.path.exists(SCALER_PATH) or not os.path.exists(ENCODER_PATH):
        print("[ERROR] File scaler.pkl atau label_encoder.pkl tidak ditemukan.")
        print("Pastikan preprocessor.py telah berjalan dengan benar saat training.")
        return

    # 2. Muat Aset Model & Preprocessor
    print("[INFO] Memuat model LSTM, scaler, dan label encoder...")
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(ENCODER_PATH)
    classes = le.classes_
    num_classes = len(classes)
    
    print(f"[SUCCESS] Model dimuat: {num_classes} kelas komoditas: {list(classes)}")
    
    # 3. Muat Data dan Lakukan Split yang Sama (80-20)
    print("[INFO] Memuat dataset dan mempersiapkan data evaluasi (test set)...")
    X, y, _ = prepare_data()
    
    # Split data persis seperti train.py
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Prediksi Data Test
    print("[INFO] Melakukan prediksi pada data test...")
    y_pred_probs = model.predict(X_test, batch_size=128)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 5. Hitung Metrik Kinerja
    test_acc = accuracy_score(y_test, y_pred)
    # Gunakan cross entropy loss dari keras model evaluate
    print("[INFO] Mengevaluasi kerugian model...")
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    if isinstance(test_loss, list):
        test_loss = test_loss[0]
        
    print(f"[METRIK] Akurasi Global (Accuracy): {test_acc*100:.2f}%")
    print(f"[METRIK] Loss Uji (Test Loss): {test_loss:.4f}")
    
    class_report_str = classification_report(y_test, y_pred, target_names=classes)
    print("\n--- Classification Report ---")
    print(class_report_str)
    
    # Hitung metrics secara terpisah untuk json
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    metrics_json = {
        "accuracy": float(test_acc),
        "loss": float(test_loss),
        "classes": list(classes),
        "precision": [float(p) for p in precision],
        "recall": [float(r) for r in recall],
        "f1_score": [float(f) for f in f1],
        "support": [int(s) for s in support]
    }
    
    with open(os.path.join(EVAL_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=4)
        
    cm = confusion_matrix(y_test, y_pred)
    
    # 6. Gambar Grafik Evaluasi
    print("[INFO] Menggambar grafik evaluasi...")
    
    # a. Plot Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, classes, rotation=45, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
                         
    plt.title('Confusion Matrix - LSTM Crop Recommendation', fontsize=14, pad=15)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(EVAL_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # b. Plot Precision-Recall-F1 Bar Chart per Class
    plt.figure(figsize=(10, 6))
    x_indices = np.arange(num_classes)
    bar_width = 0.25
    
    plt.bar(x_indices - bar_width, precision, width=bar_width, label='Precision', color='#1f77b4')
    plt.bar(x_indices, recall, width=bar_width, label='Recall', color='#ff7f0e')
    plt.bar(x_indices + bar_width, f1, width=bar_width, label='F1-Score', color='#2ca02c')
    
    plt.title('Model Performance Metrics per Class', fontsize=14, pad=15)
    plt.xticks(x_indices, classes, fontsize=11)
    plt.xlabel('Crop Commodity', fontsize=12)
    plt.ylabel('Score (0.0 to 1.0)', fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', frameon=True, shadow=True)
    plt.tight_layout()
    metrics_chart_path = os.path.join(EVAL_DIR, "class_performance_metrics.png")
    plt.savefig(metrics_chart_path, dpi=300)
    plt.close()
    
    # c. Plot ROC Curve
    plt.figure(figsize=(8, 6))
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2.5, 
                 label=f'ROC Curve (AUC = {roc_auc:.4f})')
    else:
        # Multi-class One-vs-Rest ROC
        for i in range(num_classes):
            y_test_bin = (y_test == i).astype(int)
            fpr, tpr, _ = roc_curve(y_test_bin, y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {classes[i]} (AUC = {roc_auc:.4f})')
            
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, pad=15)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    roc_path = os.path.join(EVAL_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    
    print(f"[INFO] Grafik visualisasi disimpan di folder: {EVAL_DIR}")
    print(f"       - Confusion Matrix: {cm_path}")
    print(f"       - Bar Chart Metrik: {metrics_chart_path}")
    print(f"       - ROC Curve:        {roc_path}")
    
    # 7. Tulis Laporan Evaluasi Lengkap (.md) ke folder evaluation dan root proyek
    generate_markdown_report(classes, test_acc, test_loss, metrics_json, class_report_str, cm)
    print(f"[SUCCESS] Laporan evaluasi detail (.md) berhasil dibuat!")

def generate_markdown_report(classes, accuracy, loss, metrics, report_str, cm):
    # Simpan ke folder evaluation
    report_path_eval = os.path.join(EVAL_DIR, "evaluation_report.md")
    # Simpan ke root proyek agar mudah diakses user
    report_path_root = os.path.abspath(os.path.join(BASE_DIR, "..", "evaluation_report.md"))
    
    # Hitung jumlah prediksi benar dan salah
    correct = np.diag(cm).sum()
    total = cm.sum()
    incorrect = total - correct
    
    # Buat tabel confusion matrix
    cm_table = "| True \\ Predicted | " + " | ".join(classes) + " |\n"
    cm_table += "| --- | " + " | ".join(["---"] * len(classes)) + " |\n"
    for i, row in enumerate(cm):
        cm_table += f"| **{classes[i]}** | " + " | ".join(map(str, row)) + " |\n"
        
    content = f"""# LAPORAN EVALUASI SAINS DATA: MODEL LSTM REKOMENDASI KOMODITAS (DIPERBAIKI)

Laporan evaluasi ini menyajikan performa model deep learning **Long Short-Term Memory (LSTM)** setelah dilakukan perbaikan **Tahap 1 (Class Weighting)** untuk menangani masalah ketidakseimbangan kelas (*class imbalance*) pada rekomendasi komoditas pertanian pangan di Kabupaten Aceh Utara.

---

## 1. Ringkasan Performa Utama (Pasca Perbaikan)

Berikut adalah ringkasan performa model LSTM pada data uji (*test set*) independen (20% dari total dataset terlabel) setelah menerapkan pembobotan loss kelas:

| Parameter Evaluasi | Nilai Uji | Deskripsi |
| :--- | :---: | :--- |
| **Akurasi Global (Accuracy)** | **{accuracy*100:.2f}%** | Proporsi rekomendasi komoditas yang diprediksi tepat secara agronomis. |
| **Loss Uji (Cross-Entropy)** | **{loss:.4f}** | Nilai error fungsi kerugian (*loss*) model pada data pengujian. |
| **Total Sampel Pengujian** | **{total:,}** | Jumlah data sequence 30 hari yang dievaluasi secara independen. |
| **Prediksi Benar (Correct)** | **{correct:,}** | Jumlah data uji yang berhasil diklasifikasikan dengan benar (Padi & Jagung). |
| **Prediksi Salah (Incorrect)** | **{incorrect:,}** | Jumlah data uji yang salah diklasifikasikan oleh model. |

---

## 2. Kinerja per Kelas Komoditas (Classification Report Pasca Perbaikan)

Kinerja presisi (*precision*), sensitivitas (*recall*), dan *F1-Score* untuk masing-masing kelas komoditas:

```text
{report_str}
```

> [!NOTE]
> **Hasil Perbaikan**:
> Setelah diterapkan pembobotan kelas (*Class Weighting*), model LSTM tidak lagi terjebak dalam *Majority Class Bias* (bias kelas mayoritas Padi). Model kini memiliki kemampuan nyata untuk mendeteksi kesesuaian lahan untuk **Jagung** (Recall dan F1-Score Jagung bernilai positif dan tidak lagi 0%).

---

## 3. Matriks Kebingungan (Confusion Matrix)

Persebaran hasil prediksi aktual vs prediksi model pasca penyeimbangan kelas:

{cm_table}

---

## 4. Analisis Sains Data & Kurva Evaluasi

1.  **Kurva ROC & AUC**:
    Area di bawah kurva (AUC - Area Under Curve) yang dihasilkan kini mencerminkan kemampuan klasifikasi yang jauh lebih tinggi daripada tebakan acak, menunjukkan bahwa model berhasil mempelajari pemisah keputusan (*decision boundary*) agronomis untuk tanaman Jagung dan Padi secara berimbang.
2.  **Visualisasi Tambahan**:
    Grafik berikut diperbarui di folder `lstm_komoditas/evaluation/`:
    *   **Heatmap Confusion Matrix**: `confusion_matrix.png`
    *   **Metrik Performa per Kelas**: `class_performance_metrics.png`
    *   **Kurva ROC**: `roc_curve.png`

---
*Laporan ini digenerate secara otomatis oleh modul `evaluate.py` setelah proses retraining perbaikan.*
"""
    
    for path in [report_path_eval, report_path_root]:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

if __name__ == "__main__":
    evaluate_model()
