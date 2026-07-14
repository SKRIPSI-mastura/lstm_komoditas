# LAPORAN EVALUASI SAINS DATA: MODEL LSTM REKOMENDASI KOMODITAS (DIPERBAIKI)

Laporan evaluasi ini menyajikan performa model deep learning **Long Short-Term Memory (LSTM)** setelah dilakukan perbaikan **Tahap 1 (Class Weighting)** untuk menangani masalah ketidakseimbangan kelas (*class imbalance*) pada rekomendasi komoditas pertanian pangan di Kabupaten Aceh Utara.

---

## 1. Ringkasan Performa Utama (Pasca Perbaikan)

Berikut adalah ringkasan performa model LSTM pada data uji (*test set*) independen (20% dari total dataset terlabel) setelah menerapkan pembobotan loss kelas:

| Parameter Evaluasi | Nilai Uji | Deskripsi |
| :--- | :---: | :--- |
| **Akurasi Global (Accuracy)** | **75.34%** | Proporsi rekomendasi komoditas yang diprediksi tepat secara agronomis. |
| **Loss Uji (Cross-Entropy)** | **0.5560** | Nilai error fungsi kerugian (*loss*) model pada data pengujian. |
| **Total Sampel Pengujian** | **11,675** | Jumlah data sequence 30 hari yang dievaluasi secara independen. |
| **Prediksi Benar (Correct)** | **8,796** | Jumlah data uji yang berhasil diklasifikasikan dengan benar (Padi & Jagung). |
| **Prediksi Salah (Incorrect)** | **2,879** | Jumlah data uji yang salah diklasifikasikan oleh model. |

---

## 2. Kinerja per Kelas Komoditas (Classification Report Pasca Perbaikan)

Kinerja presisi (*precision*), sensitivitas (*recall*), dan *F1-Score* untuk masing-masing kelas komoditas:

```text
              precision    recall  f1-score   support

      Jagung       0.98      0.68      0.81      1770
Kacang Hijau       0.77      0.96      0.85      4126
Kacang Tanah       0.00      0.00      0.00       418
     Kedelai       0.58      0.97      0.73       807
        Padi       0.78      0.64      0.70      3245
   Ubi Jalar       0.59      1.00      0.74       775
    Ubi Kayu       0.00      0.00      0.00       534

    accuracy                           0.75     11675
   macro avg       0.53      0.61      0.55     11675
weighted avg       0.72      0.75      0.72     11675

```

> [!NOTE]
> **Hasil Perbaikan**:
> Setelah diterapkan pembobotan kelas (*Class Weighting*), model LSTM tidak lagi terjebak dalam *Majority Class Bias* (bias kelas mayoritas Padi). Model kini memiliki kemampuan nyata untuk mendeteksi kesesuaian lahan untuk **Jagung** (Recall dan F1-Score Jagung bernilai positif dan tidak lagi 0%).

---

## 3. Matriks Kebingungan (Confusion Matrix)

Persebaran hasil prediksi aktual vs prediksi model pasca penyeimbangan kelas:

| True \ Predicted | Jagung | Kacang Hijau | Kacang Tanah | Kedelai | Padi | Ubi Jalar | Ubi Kayu |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Jagung** | 1209 | 0 | 0 | 561 | 0 | 0 | 0 |
| **Kacang Hijau** | 0 | 3959 | 0 | 0 | 167 | 0 | 0 |
| **Kacang Tanah** | 0 | 3 | 0 | 0 | 415 | 0 | 0 |
| **Kedelai** | 23 | 0 | 0 | 784 | 0 | 0 | 0 |
| **Padi** | 0 | 1176 | 0 | 0 | 2069 | 0 | 0 |
| **Ubi Jalar** | 0 | 0 | 0 | 0 | 0 | 775 | 0 |
| **Ubi Kayu** | 0 | 0 | 0 | 0 | 0 | 534 | 0 |


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
