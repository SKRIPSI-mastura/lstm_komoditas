# MVP Plan: Model LSTM Komoditas Tanaman Pangan

Rencana ini merinci langkah-langkah untuk membangun versi MVP (Minimum Viable Product) dari model LSTM yang dapat memprediksi atau merekomendasikan komoditas tanaman pangan berdasarkan data iklim dan profil wilayah.

## 1. Tujuan (Objective)
Membangun model Deep Learning (LSTM) yang menerima input urutan data iklim (time-series) dan fitur statis (tanah/elevasi) untuk menghasilkan output klasifikasi komoditas tanaman pangan yang paling sesuai.

## 2. Persiapan Data (Data Preparation)
*   **Dataset Utama**: `iklim_semua_kecamatan_aceh_utara_2020_2025.csv`
*   **Fitur Statis**: pH Tanah, Elevasi (mdpl), dan Curah Hujan Tahunan.
*   **Target (Output)**: Salah satu dari 7 komoditas (Padi, Jagung, Kedelai, Kacang Hijau, Kacang Tanah, Ubi Jalar, Ubi Kayu).

> [!IMPORTANT]
> Karena data historis "komoditas apa yang ditanam" belum tersedia secara eksplisit dalam CSV, kita akan menggunakan **Heuristic Labeling** (menggunakan logika skoring di `main.py`) untuk menghasilkan label target pada data historis guna melatih model.

## 3. Alur Kerja Teknis (Technical Workflow)

### A. Preprocessing & Feature Engineering
1.  **Integrasi Data**: Menggabungkan data iklim harian dengan data profil kecamatan (pH, Elevasi).
2.  **Pembersihan**: Menangani nilai kosong (Imputation) terutama pada kolom `PRECTOT`.
3.  **Scaling**: Menggunakan `MinMaxScaler` untuk menormalkan rentang data iklim.
4.  **Sequencing**: Membuat window data (misalnya: data 30 hari terakhir) sebagai input untuk LSTM.

### B. Arsitektur Model (MVP)
*   **Input Layer**: Menerima tensor (Batch, Time_Steps, Features).
*   **LSTM Layers**: 2-3 layer LSTM untuk menangkap pola temporal cuaca.
*   **Dense Layer**: Fully connected layers untuk integrasi fitur statis.
*   **Output Layer**: Dense layer dengan aktivasi **Softmax** (7 neuron) untuk probabilitas masing-masing komoditas.

### C. Strategi Pelatihan
*   **Loss Function**: `categorical_crossentropy`.
*   **Optimizer**: `Adam`.
*   **Metrics**: `Accuracy`.

## 4. Arsitektur Program Python (Detailed Plan)
Untuk memastikan kode modular dan mudah dikembangkan, program akan dibagi menjadi beberapa modul berikut:

1.  **`data_loader.py`**: Modul untuk membaca semua file CSV di folder `data/` dan menggabungkannya berdasarkan nama Kecamatan.
2.  **`labeler.py`**: Modul khusus untuk menghasilkan *target labels*. Karena kita tidak punya data riwayat asli, modul ini akan menghitung komoditas terbaik untuk setiap baris data iklim menggunakan kriteria agronomis.
3.  **`preprocessor.py`**: Modul untuk normalisasi data (Scaling) dan pembentukan urutan waktu (Sequencing/Windowing) agar data siap masuk ke model LSTM.
4.  **`model_factory.py`**: Modul yang berisi definisi arsitektur neural network (Layer LSTM, Dense, Dropout).
5.  **`train.py`**: Script utama untuk menjalankan proses pelatihan model dan menyimpan hasil model (`.keras`).
6.  **`predict.py`**: Script untuk melakukan inferensi atau prediksi komoditas berdasarkan data terbaru yang diinput atau diambil dari sistem.

## 5. Tahapan Pengerjaan (Checklist)
- [x] **Step 1**: Implementasi `data_loader.py` dan validasi data gabungan.
- [ ] **Step 2**: Implementasi `labeler.py` untuk membuat dataset pelatihan (`train_data.csv`).
- [ ] **Step 3**: Implementasi `preprocessor.py` untuk menyiapkan tensor X dan y.
- [ ] **Step 4**: Pembuatan arsitektur di `model_factory.py` dan script `train.py`.
- [ ] **Step 5**: Integrasi hasil prediksi ke dalam script `predict.py`.

## 5. Pertanyaan Terbuka (Open Questions)
*   **Gambar Input**: Anda menyebutkan "input seperti gambar", apakah yang dimaksud adalah struktur tensor data atau ada format gambar (seperti peta/grafik) yang ingin dijadikan input?
*   **Data Target**: Apakah ada file CSV tambahan yang berisi riwayat penanaman asli (Ground Truth)? Jika ada, model akan jauh lebih akurat dibanding menggunakan pelabelan otomatis.
