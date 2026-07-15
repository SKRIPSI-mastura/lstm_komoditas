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
| **Akurasi Global** | 79.86% | **84.99%** | **+5.13%** |
| **Loss Uji** | 0.4006 | **0.3336** | -0.0670 |
| **Macro ROC-AUC** | *(tidak dihitung)* | **0.9833** | — |
| **Total Sampel Uji** | 11,675 | **11,675** | — |
| **Prediksi Benar** | 9,324 | **9,922** | +598 |
| **Prediksi Salah** | 2,351 | **1,753** | -598 |

---

## 3. Perbandingan Kinerja per Kelas Komoditas

| Kelas | Prec-v1 | Prec-v2 | ΔPrec | Rec-v1 | Rec-v2 | ΔRec | F1-v1 | F1-v2 | ΔF1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Jagung** | 0.99 | 0.98 | -0.01 | 0.98 | 0.98 | +0.00 | 0.98 | 0.98 | +0.00 |
| **Kacang Hijau** | 0.86 | 0.86 | -0.00 | 0.91 | 0.91 | +0.00 | 0.89 | 0.89 | -0.00 |
| **Kacang Tanah** | 0.27 | 0.46 | +0.19 | 0.92 | 0.78 | -0.14 | 0.42 | 0.58 | +0.16 |
| **Kedelai** | 0.95 | 0.96 | +0.01 | 0.97 | 0.96 | -0.01 | 0.96 | 0.96 | +0.00 |
| **Padi** | 0.81 | 0.83 | +0.02 | 0.49 | 0.69 | +0.20 | 0.61 | 0.76 | +0.15 |
| **Ubi Jalar** | 0.85 | 0.90 | +0.05 | 0.83 | 0.77 | -0.06 | 0.84 | 0.83 | -0.01 |
| **Ubi Kayu** | 0.76 | 0.72 | -0.04 | 0.78 | 0.87 | +0.09 | 0.77 | 0.79 | +0.02 |


> [!NOTE]
> **Fokus analisis** pada kelas bermasalah di v1:
> - **Kacang Tanah** — F1 baseline 0.42 (precision rendah 0.27, banyak false positive)
> - **Padi** — F1 baseline 0.61 (recall rendah 0.49, banyak prediksi salah ke Kacang Hijau/Tanah)

---

## 4. Classification Report Lengkap (Model v2)

```text
              precision    recall  f1-score   support

      Jagung       0.98      0.98      0.98      1770
Kacang Hijau       0.86      0.91      0.89      4126
Kacang Tanah       0.46      0.78      0.58       418
     Kedelai       0.96      0.96      0.96       807
        Padi       0.83      0.69      0.76      3245
   Ubi Jalar       0.90      0.77      0.83       775
    Ubi Kayu       0.72      0.87      0.79       534

    accuracy                           0.85     11675
   macro avg       0.82      0.85      0.83     11675
weighted avg       0.86      0.85      0.85     11675
```

---

## 5. Matriks Kebingungan (Model v2)

| True \ Predicted | Jagung | Kacang Hijau | Kacang Tanah | Kedelai | Padi | Ubi Jalar | Ubi Kayu |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Jagung** | 1739 | 0 | 0 | 31 | 0 | 0 | 0 |
| **Kacang Hijau** | 0 | 3774 | 0 | 0 | 352 | 0 | 0 |
| **Kacang Tanah** | 0 | 0 | 325 | 0 | 93 | 0 | 0 |
| **Kedelai** | 29 | 0 | 0 | 778 | 0 | 0 | 0 |
| **Padi** | 0 | 624 | 375 | 0 | 2246 | 0 | 0 |
| **Ubi Jalar** | 0 | 0 | 0 | 0 | 0 | 593 | 182 |
| **Ubi Kayu** | 0 | 0 | 0 | 0 | 0 | 67 | 467 |


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
