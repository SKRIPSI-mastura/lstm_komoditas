# LAPORAN EVALUASI: PENGEMBANGAN MODEL LSTM v2 DENGAN FITUR LAHAN LENGKAP

Laporan ini membandingkan performa **Model LSTM v1** (baseline) dengan **Model LSTM v2** yang menggunakan
arsitektur *Dual-Input Hybrid* dengan penambahan fitur tekstur tanah (tanah liat, pasir, debu)
pada klasifikasi komoditas pertanian di Kabupaten Aceh Utara.

---

## 1. Perubahan Arsitektur Model

| Aspek | Model v1 (Baseline) | Model v2 (Fitur Lengkap) |
| :--- | :--- | :--- |
| **Tipe Arsitektur** | Sequential LSTM | Dual-Input Hybrid (Functional API) |
| **Jumlah Fitur Input** | 6 fitur | 9 fitur |
| **Fitur Sequential (LSTM)** | T2M, RH2M, WS2M, elevasi, pH, curah_hujan | T2M, RH2M, WS2M |
| **Fitur Statis (Dense Branch)** | *(tidak ada)* | elevasi, curah_hujan, pH, liat, pasir, debu |
| **Penanganan Imbalance** | Class Weighting | Class Weighting |
| **File Model** | `commodity_lstm_model.keras` | `commodity_lstm_model_v2.keras` |

### Arsitektur Dual-Input v2

```
Input_seq  (30, 3) ── LSTM(128) ── Dropout ── LSTM(64) ── Dropout ──┐
                                                                      ├── Concat ── Dense(64) ── Dropout ── Dense(32) ── Softmax(7)
Input_static  (6,) ── Dense(32) ── BatchNorm ── Dense(16) ───────────┘
```

> [!NOTE]
> Fitur statis (tanah & topografi) dipisahkan dari sequence LSTM karena nilainya **konstan per kecamatan**
> (tidak berubah harian). Memasukkannya sebagai time-step berulang akan menyebabkan redundansi.
> Arsitektur dual-input memungkinkan model belajar representasi temporal iklim dan representasi
> spasial lahan secara **independen**, lalu menggabungkannya untuk keputusan akhir.

---

## 2. Ringkasan Perbandingan Performa Utama

| Parameter | Model v1 (Baseline) | Model v2 (Fitur Lengkap) | Delta |
| :--- | :---: | :---: | :---: |
| **Akurasi Global** | 79.86% | **98.31%** | **+18.45%** |
| **Loss Uji** | 0.4006 | **0.0388** | -0.3618 |
| **Macro ROC-AUC** | *(tidak dihitung)* | **0.9997** | — |
| **Total Sampel Uji** | 11,675 | **11,675** | — |
| **Prediksi Benar** | 9,324 | **11,478** | +2,154 |
| **Prediksi Salah** | 2,351 | **197** | -2,154 |

---

## 3. Perbandingan Kinerja per Kelas Komoditas

| Kelas | Prec-v1 | Prec-v2 | ΔPrec | Rec-v1 | Rec-v2 | ΔRec | F1-v1 | F1-v2 | ΔF1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Jagung** | 0.99 | 1.00 | +0.01 | 0.98 | 1.00 | +0.02 | 0.98 | 1.00 | +0.02 |
| **Kacang Hijau** | 0.86 | 0.99 | +0.13 | 0.91 | 0.99 | +0.08 | 0.89 | 0.99 | +0.10 |
| **Kacang Tanah** | 0.27 | 0.86 | +0.59 | 0.92 | 0.90 | -0.02 | 0.42 | 0.88 | +0.46 |
| **Kedelai** | 0.95 | 1.00 | +0.05 | 0.97 | 1.00 | +0.03 | 0.96 | 1.00 | +0.04 |
| **Padi** | 0.81 | 0.97 | +0.16 | 0.49 | 0.97 | +0.48 | 0.61 | 0.97 | +0.36 |
| **Ubi Jalar** | 0.85 | 0.99 | +0.14 | 0.83 | 0.99 | +0.16 | 0.84 | 0.99 | +0.15 |
| **Ubi Kayu** | 0.76 | 0.99 | +0.23 | 0.78 | 0.98 | +0.20 | 0.77 | 0.99 | +0.22 |


> [!NOTE]
> **Fokus analisis** pada kelas bermasalah di v1:
> - **Kacang Tanah** — F1 baseline 0.42 (precision rendah 0.27, banyak false positive)
> - **Padi** — F1 baseline 0.61 (recall rendah 0.49, banyak prediksi salah ke Kacang Hijau/Tanah)

---

## 4. Classification Report Lengkap (Model v2)

```text
              precision    recall  f1-score   support

      Jagung       1.00      1.00      1.00      1761
Kacang Hijau       0.99      0.99      0.99      4595
Kacang Tanah       0.86      0.90      0.88       260
     Kedelai       1.00      1.00      1.00       816
        Padi       0.97      0.97      0.97      2934
   Ubi Jalar       0.99      0.99      0.99       791
    Ubi Kayu       0.99      0.98      0.99       518

    accuracy                           0.98     11675
   macro avg       0.97      0.98      0.97     11675
weighted avg       0.98      0.98      0.98     11675
```

---

## 5. Matriks Kebingungan (Model v2)

| True \ Predicted | Jagung | Kacang Hijau | Kacang Tanah | Kedelai | Padi | Ubi Jalar | Ubi Kayu |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Jagung** | 1760 | 0 | 0 | 1 | 0 | 0 | 0 |
| **Kacang Hijau** | 0 | 4531 | 0 | 0 | 64 | 0 | 0 |
| **Kacang Tanah** | 0 | 0 | 234 | 0 | 26 | 0 | 0 |
| **Kedelai** | 4 | 0 | 0 | 812 | 0 | 0 | 0 |
| **Padi** | 0 | 52 | 37 | 0 | 2845 | 0 | 0 |
| **Ubi Jalar** | 0 | 0 | 0 | 0 | 0 | 787 | 4 |
| **Ubi Kayu** | 0 | 0 | 0 | 0 | 0 | 9 | 509 |


---

## 6. Fitur yang Ditambahkan dan Dampaknya

### 6.1 Fitur Baru: Tekstur Tanah (% Clay, Sand, Silt)

| Fitur | Relevansi Agronomis |
| :--- | :--- |
| **Tanah Liat (%)** | Padi membutuhkan liat tinggi (>35%) untuk menahan air sawah. Kedelai juga toleran liat. |
| **Tanah Pasir (%)** | Kacang Tanah tumbuh optimal di tanah berpasir (drainase cepat, aerasi akar baik). |
| **Tanah Debu (%)** | Ubi Jalar & Ubi Kayu preferensi tanah gembur dengan debu sedang. |

### 6.2 Fitur Statis yang Sudah Ada (Tetap Digunakan)

| Fitur | Keterangan |
| :--- | :--- |
| **Elevasi (m dpl)** | Pengaruh terhadap suhu dan jenis komoditas (dataran rendah vs tinggi) |
| **Curah Hujan Tahunan (mm)** | Kebutuhan air rata-rata tahunan per komoditas |
| **pH Tanah** | Kesesuaian kemasaman untuk masing-masing komoditas |

---

## 7. Visualisasi

Grafik berikut tersimpan di folder `lstm_komoditas/evaluation/`:

| File | Deskripsi |
| :--- | :--- |
| `confusion_matrix_v2.png` | Heatmap confusion matrix model v2 |
| `class_performance_metrics_v2.png` | Grafik perbandingan Precision/Recall/F1 v1 vs v2 per kelas |
| `roc_curve_v2.png` | ROC Curve multi-kelas model v2 (One-vs-Rest) |
| `training_history_v2.png` | Grafik training & validation loss/accuracy model v2 |

---

*Laporan digenerate otomatis oleh `evaluate_v2.py` — Model LSTM v2 (Dual-Input Hybrid)*
