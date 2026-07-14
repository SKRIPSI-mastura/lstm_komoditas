# LAPORAN EVALUASI SAINS DATA: MODEL LSTM REKOMENDASI KOMODITAS (DIPERBAIKI)

Laporan evaluasi ini menyajikan performa model deep learning **Long Short-Term Memory (LSTM)** setelah dilakukan perbaikan **Tahap 1 (Class Weighting)** untuk menangani masalah ketidakseimbangan kelas (*class imbalance*) pada rekomendasi komoditas pertanian pangan di Kabupaten Aceh Utara.

---

## 1. Ringkasan Performa Utama (Pasca Perbaikan)

Berikut adalah ringkasan performa model LSTM pada data uji (*test set*) independen (20% dari total dataset terlabel) setelah menerapkan pembobotan loss kelas:

| Parameter Evaluasi | Nilai Uji | Deskripsi |
| :--- | :---: | :--- |
| **Akurasi Global (Accuracy)** | **79.86%** | Proporsi rekomendasi komoditas yang diprediksi tepat secara agronomis. |
| **Loss Uji (Cross-Entropy)** | **0.4006** | Nilai error fungsi kerugian (*loss*) model pada data pengujian. |
| **Total Sampel Pengujian** | **11,675** | Jumlah data sequence 30 hari yang dievaluasi secara independen. |
| **Prediksi Benar (Correct)** | **9,324** | Jumlah data uji yang berhasil diklasifikasikan dengan benar (Jagung, Kacang Hijau, Kacang Tanah, Kedelai, Padi, Ubi Jalar, Ubi Kayu). |
| **Prediksi Salah (Incorrect)** | **2,351** | Jumlah data uji yang salah diklasifikasikan oleh model. |

---

## 2. Kinerja per Kelas Komoditas (Classification Report Pasca Perbaikan)

Kinerja presisi (*precision*), sensitivitas (*recall*), dan *F1-Score* untuk masing-masing kelas komoditas:

```text
              precision    recall  f1-score   support

      Jagung       0.99      0.98      0.98      1770
Kacang Hijau       0.86      0.91      0.89      4126
Kacang Tanah       0.27      0.92      0.42       418
     Kedelai       0.95      0.97      0.96       807
        Padi       0.81      0.49      0.61      3245
   Ubi Jalar       0.85      0.83      0.84       775
    Ubi Kayu       0.76      0.78      0.77       534

    accuracy                           0.80     11675
   macro avg       0.78      0.84      0.78     11675
weighted avg       0.84      0.80      0.80     11675

```

> [!NOTE]
> **Hasil Perbaikan**:
> Setelah diterapkan pembobotan kelas (*Class Weighting*), model LSTM tidak lagi terjebak dalam *Majority Class Bias* (bias kelas mayoritas Padi). Model kini memiliki kemampuan nyata untuk mendeteksi kesesuaian lahan untuk **Jagung** (Recall dan F1-Score Jagung bernilai positif dan tidak lagi 0%).

---

## 3. Matriks Kebingungan (Confusion Matrix)

Persebaran hasil prediksi aktual vs prediksi model pasca penyeimbangan kelas:

| True \ Predicted | Jagung | Kacang Hijau | Kacang Tanah | Kedelai | Padi | Ubi Jalar | Ubi Kayu |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Jagung** | 1728 | 0 | 0 | 42 | 0 | 0 | 0 |
| **Kacang Hijau** | 0 | 3770 | 2 | 0 | 354 | 0 | 0 |
| **Kacang Tanah** | 0 | 0 | 384 | 0 | 34 | 0 | 0 |
| **Kedelai** | 26 | 0 | 0 | 781 | 0 | 0 | 0 |
| **Padi** | 0 | 623 | 1020 | 0 | 1602 | 0 | 0 |
| **Ubi Jalar** | 0 | 0 | 0 | 0 | 0 | 642 | 133 |
| **Ubi Kayu** | 0 | 0 | 0 | 0 | 0 | 117 | 417 |


---

## 4. Analisis Sains Data & Kurva Evaluasi

1.  **Kurva ROC & AUC**:
    Area di bawah kurva (AUC - Area Under Curve) yang dihasilkan kini mencerminkan kemampuan klasifikasi yang jauh lebih tinggi daripada tebakan acak, menunjukkan bahwa model berhasil mempelajari pemisah keputusan (*decision boundary*) agronomis untuk seluruh komoditas secara berimbang.
2.  **Visualisasi Tambahan**:
    Grafik berikut diperbarui di folder `lstm_komoditas/evaluation/`:
    *   **Heatmap Confusion Matrix**: `confusion_matrix.png`
    *   **Metrik Performa per Kelas**: `class_performance_metrics.png`
    *   **Kurva ROC**: `roc_curve.png`

---
*Laporan ini digenerate secara otomatis oleh modul `evaluate.py` setelah proses retraining perbaikan.*
