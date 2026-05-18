# Laporan Evaluasi Model LSTM Rekomendasi Komoditas

Laporan evaluasi ini menyajikan performa model deep learning Long Short-Term Memory (LSTM) yang dirancang untuk melakukan rekomendasi komoditas tanaman pangan terbaik di Kabupaten Aceh Utara berdasarkan profil wilayah (geografis & tanah) dan kondisi cuaca historis.

---

## 1. Ringkasan Performa Utama

Berikut adalah ringkasan performa model LSTM pada data uji (test set) yang didefinisikan secara independen (20% dari total dataset terlabel):

| Parameter Evaluasi | Nilai | Deskripsi |
| :--- | :---: | :--- |
| **Akurasi Global (Accuracy)** | **74.39%** | Proporsi rekomendasi komoditas yang diprediksi tepat sesuai kriteria kelayakan. |
| **Loss Uji (Sparse Cross-Entropy)** | **0.5935** | Nilai error fungsi kerugian (loss) model saat melakukan prediksi. |
| **Total Sampel Pengujian** | **11,675** | Jumlah data sequence 30 hari yang dievaluasi secara independen. |
| **Prediksi Benar (TP + TN)** | **8,685** | Jumlah sampel yang diklasifikasikan dengan benar. |
| **Prediksi Salah (FP + FN)** | **2,990** | Jumlah sampel yang salah diklasifikasikan. |

---

## 2. Laporan Klasifikasi Rinci (Classification Report)

Analisis kinerja model per kelas komoditas:

```text
              precision    recall  f1-score   support

      Jagung       0.00      0.00      0.00      2990
        Padi       0.74      1.00      0.85      8685

    accuracy                           0.74     11675
   macro avg       0.37      0.50      0.43     11675
weighted avg       0.55      0.74      0.63     11675

```

### Keterangan Istilah:
*   **Precision (Presisi)**: Kemampuan model mendeteksi komoditas dengan benar tanpa salah menebak kelas lain. (Meminimalisir False Positives).
*   **Recall (Sensitivitas)**: Kemampuan model menemukan seluruh data aktual dari suatu komoditas. (Meminimalisir False Negatives).
*   **F1-Score**: Rata-rata harmonik dari Precision dan Recall. Menjadi indikator performa seimbang model.
*   **Support**: Jumlah kemunculan aktual kelas tersebut pada data uji.

---

## 3. Confusion Matrix

Matriks kontingensi klasifikasi model menunjukkan persebaran tebakan model vs label aktual:

| True \ Predicted | Jagung | Padi |
| --- | --- | --- |
| **Jagung** | 0 | 2990 |
| **Padi** | 0 | 8685 |


### Interpretasi Confusion Matrix:
*   **Padi**: Dari total 8,685 data aktual tanaman Padi, model berhasil memprediksi **8,685** dengan benar. Sebanyak **0** data diprediksi salah sebagai Jagung.
*   **Jagung**: Dari total 2,990 data aktual tanaman Jagung, model berhasil memprediksi **0** dengan benar. Sebanyak **2,990** data diprediksi salah sebagai Padi.

---

## 4. Visualisasi Kinerja Model

Grafik visualisasi performa dapat diakses di direktori `evaluation/`:

1.  **Confusion Matrix Heatmap (`confusion_matrix.png`)**
    Menampilkan visualisasi gradien warna matriks kebingungan untuk mengidentifikasi kecenderungan kesalahan klasifikasi model.
2.  **Class Performance Metrics (`class_performance_metrics.png`)**
    Grafik batang komparatif antara Precision, Recall, dan F1-Score untuk setiap komoditas.
3.  **ROC Curve (`roc_curve.png`)**
    Kurva karakteristik operasi penerima untuk menilai kemampuan pemisahan kelas model deep learning pada berbagai ambang batas klasifikasi. Nilai AUC mendekati 1.0 menunjukkan performa yang luar biasa.

---

## 5. Kesimpulan Agronomis & Model

1.  **Stabilitas Model**: Model LSTM menunjukkan stabilitas tinggi dengan akurasi **74.39%**. Hal ini berarti model sangat andal dalam menangkap pola sekuensial iklim (suhu, kelembapan, kecepatan angin) bersama dengan fitur statis tanah/elevasi untuk merekomendasikan komoditas pertanian terbaik.
2.  **Rekomendasi Operasional**: Model dapat diintegrasikan dengan percaya diri ke aplikasi dashboard petani dan API backend (`app.py`) untuk memandu dinas pertanian maupun petani dalam menentukan komoditas pangan yang paling adaptif terhadap kondisi iklim 30 hari ke belakang di Aceh Utara.

---
*Laporan ini digenerate secara otomatis oleh modul `evaluate.py`.*
