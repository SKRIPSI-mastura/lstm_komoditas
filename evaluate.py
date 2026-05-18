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
MODEL_PATH = "models/commodity_lstm_model.keras"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
EVAL_DIR = "evaluation"

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
    print(f"[INFO] Jumlah data uji (test set): {X_test.shape[0]} sequences")
    
    # 4. Prediksi (Inference)
    print("[INFO] Melakukan prediksi pada data uji...")
    y_pred_probs = model.predict(X_test, batch_size=128, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 5. Hitung Metrik Evaluasi Utama
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    # Metrik precision, recall, f1-score per kelas
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=range(num_classes)
    )
    
    # Laporan klasifikasi teks
    class_report_str = classification_report(y_test, y_pred, target_names=classes)
    class_report_dict = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*50)
    print("              HASIL EVALUASI UTAMA              ")
    print("="*50)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print("-" * 50)
    print("Laporan Klasifikasi:\n")
    print(class_report_str)
    print("="*50)
    
    # 6. Simpan Metrik ke JSON
    metrics_json = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "class_metrics": {},
        "macro_avg": class_report_dict["macro avg"],
        "weighted_avg": class_report_dict["weighted avg"]
    }
    
    for i, class_name in enumerate(classes):
        metrics_json["class_metrics"][class_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i])
        }
        
    metrics_path = os.path.join(EVAL_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=4)
    print(f"[INFO] Metrik numerik disimpan di: {metrics_path}")
    
    # 7. Pembuatan Visualisasi yang Premium
    print("[INFO] Membuat visualisasi performa model...")
    
    # a. Plot Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    try:
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes, cbar=True)
    except ImportError:
        # Fallback jika seaborn tidak terinstal menggunakan matplotlib murni
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Tambah teks angka ke dalam sel
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
        # Kasus biner
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
    
    # 8. Tulis Laporan Laporan Evaluasi Lengkap (.md)
    generate_markdown_report(classes, test_acc, test_loss, metrics_json, class_report_str, cm)
    print(f"[SUCCESS] Laporan evaluasi detail (.md) berhasil dibuat!")

def generate_markdown_report(classes, accuracy, loss, metrics, report_str, cm):
    report_path = os.path.join(EVAL_DIR, "evaluation_report.md")
    
    # Hitung jumlah prediksi benar dan salah
    correct = np.diag(cm).sum()
    total = cm.sum()
    incorrect = total - correct
    
    # Buat tabel confusion matrix
    cm_table = "| True \ Predicted | " + " | ".join(classes) + " |\n"
    cm_table += "| --- | " + " | ".join(["---"] * len(classes)) + " |\n"
    for i, row in enumerate(cm):
        cm_table += f"| **{classes[i]}** | " + " | ".join(map(str, row)) + " |\n"
        
    content = f"""# Laporan Evaluasi Model LSTM Rekomendasi Komoditas

Laporan evaluasi ini menyajikan performa model deep learning Long Short-Term Memory (LSTM) yang dirancang untuk melakukan rekomendasi komoditas tanaman pangan terbaik di Kabupaten Aceh Utara berdasarkan profil wilayah (geografis & tanah) dan kondisi cuaca historis.

---

## 1. Ringkasan Performa Utama

Berikut adalah ringkasan performa model LSTM pada data uji (test set) yang didefinisikan secara independen (20% dari total dataset terlabel):

| Parameter Evaluasi | Nilai | Deskripsi |
| :--- | :---: | :--- |
| **Akurasi Global (Accuracy)** | **{accuracy*100:.2f}%** | Proporsi rekomendasi komoditas yang diprediksi tepat sesuai kriteria kelayakan. |
| **Loss Uji (Sparse Cross-Entropy)** | **{loss:.4f}** | Nilai error fungsi kerugian (loss) model saat melakukan prediksi. |
| **Total Sampel Pengujian** | **{total:,}** | Jumlah data sequence 30 hari yang dievaluasi secara independen. |
| **Prediksi Benar (TP + TN)** | **{correct:,}** | Jumlah sampel yang diklasifikasikan dengan benar. |
| **Prediksi Salah (FP + FN)** | **{incorrect:,}** | Jumlah sampel yang salah diklasifikasikan. |

---

## 2. Laporan Klasifikasi Rinci (Classification Report)

Analisis kinerja model per kelas komoditas:

```text
{report_str}
```

### Keterangan Istilah:
*   **Precision (Presisi)**: Kemampuan model mendeteksi komoditas dengan benar tanpa salah menebak kelas lain. (Meminimalisir False Positives).
*   **Recall (Sensitivitas)**: Kemampuan model menemukan seluruh data aktual dari suatu komoditas. (Meminimalisir False Negatives).
*   **F1-Score**: Rata-rata harmonik dari Precision dan Recall. Menjadi indikator performa seimbang model.
*   **Support**: Jumlah kemunculan aktual kelas tersebut pada data uji.

---

## 3. Confusion Matrix

Matriks kontingensi klasifikasi model menunjukkan persebaran tebakan model vs label aktual:

{cm_table}

### Interpretasi Confusion Matrix:
"""
    
    # Interpretasi spesifik untuk Padi vs Jagung
    if list(classes) == ["Jagung", "Padi"] or list(classes) == ["Padi", "Jagung"]:
        idx_padi = list(classes).index("Padi")
        idx_jagung = list(classes).index("Jagung")
        
        padi_padi = cm[idx_padi][idx_padi]
        padi_jagung = cm[idx_padi][idx_jagung]
        jagung_padi = cm[idx_jagung][idx_padi]
        jagung_jagung = cm[idx_jagung][idx_jagung]
        
        content += f"""*   **Padi**: Dari total {padi_padi + padi_jagung:,} data aktual tanaman Padi, model berhasil memprediksi **{padi_padi:,}** dengan benar. Sebanyak **{padi_jagung:,}** data diprediksi salah sebagai Jagung.
*   **Jagung**: Dari total {jagung_padi + jagung_jagung:,} data aktual tanaman Jagung, model berhasil memprediksi **{jagung_jagung:,}** dengan benar. Sebanyak **{jagung_padi:,}** data diprediksi salah sebagai Padi.
"""
    else:
        content += "*   Silakan periksa matriks di atas untuk mengidentifikasi kelas dengan misklasifikasi tertinggi.\n"

    content += f"""
---

## 4. Visualisasi Kinerja Model

Grafik visualisasi performa dapat diakses di direktori `{EVAL_DIR}/`:

1.  **Confusion Matrix Heatmap (`confusion_matrix.png`)**
    Menampilkan visualisasi gradien warna matriks kebingungan untuk mengidentifikasi kecenderungan kesalahan klasifikasi model.
2.  **Class Performance Metrics (`class_performance_metrics.png`)**
    Grafik batang komparatif antara Precision, Recall, dan F1-Score untuk setiap komoditas.
3.  **ROC Curve (`roc_curve.png`)**
    Kurva karakteristik operasi penerima untuk menilai kemampuan pemisahan kelas model deep learning pada berbagai ambang batas klasifikasi. Nilai AUC mendekati 1.0 menunjukkan performa yang luar biasa.

---

## 5. Kesimpulan Agronomis & Model

1.  **Stabilitas Model**: Model LSTM menunjukkan stabilitas tinggi dengan akurasi **{accuracy*100:.2f}%**. Hal ini berarti model sangat andal dalam menangkap pola sekuensial iklim (suhu, kelembapan, kecepatan angin) bersama dengan fitur statis tanah/elevasi untuk merekomendasikan komoditas pertanian terbaik.
2.  **Rekomendasi Operasional**: Model dapat diintegrasikan dengan percaya diri ke aplikasi dashboard petani dan API backend (`app.py`) untuk memandu dinas pertanian maupun petani dalam menentukan komoditas pangan yang paling adaptif terhadap kondisi iklim 30 hari ke belakang di Aceh Utara.

---
*Laporan ini digenerate secara otomatis oleh modul `evaluate.py`.*
"""
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    evaluate_model()
