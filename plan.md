# Mengintegrasikan Model Machine Learning untuk Rekomendasi Komoditas

Saat ini, fungsi rekomendasi komoditas (`recommend_crops`) menggunakan aturan manual (berbasis skor atau logika if-else) untuk menentukan tingkat kecocokan komoditas. Karena Anda sudah menyediakan dataset historis rekomendasi tanaman (`dataset_lstm_tanaman.csv`), kita dapat mengganti aturan manual tersebut dengan sebuah **Model Neural Network / Klasifikasi** yang "belajar" langsung dari data yang Anda berikan.

## Open Questions

> [!WARNING]
> **Jenis Model Pembelajaran:**
> Karena data rekomendasi ini berbentuk data tabular statis (setiap baris adalah data terpisah, bukan data deret waktu seperti iklim), menggunakan algoritma *Neural Network* biasa (Dense/Feed Forward Network) atau model *Random Forest* akan jauh lebih akurat dan tepat dibanding LSTM.
> Apakah Anda setuju jika untuk prediksi **iklim** kita tetap menggunakan **LSTM**, sedangkan untuk **rekomendasi tanaman** kita buat **Model Jaringan Saraf Tiruan (Neural Network)** biasa agar hasil belajarnya optimal sesuai bentuk data yang baru?

## Proposed Changes

### 1. Pembacaan Data Profil Kecamatan yang Lebih Lengkap
Mengekstraksi fitur persentase komposisi tanah dari dataset untuk melengkapi parameter masukan bagi model rekomendasi:
- `tanah_liat_persen` (menggunakan kolom `ph_tanah` yang terbalik isinya di dataset `data_tanah_aceh_utara2.csv`).
- `tanah_pasir_persen` dan `tanah_debu_persen`.

#### [MODIFY] [main.py](file:///d:/SKRIPSII/PROJECT/lstm_komoditas/main.py)

### 2. Membangun Model Rekomendasi (Recommendation Model)
- Membuat fungsi baru `train_recommendation_model()` yang bertugas membaca `dataset_lstm_tanaman.csv`.
- Melakukan *encoding* terhadap data kategorikal (seperti jenis komoditas, tekstur tanah).
- Membangun dan melatih model Jaringan Saraf Tiruan / *Neural Network* yang memprediksi probabilitas label kelayakan (`Sangat Layak`, `Layak`, `Kurang Layak`).
- Menyimpan model (`recommender_model.keras`) beserta alat *preprocessing* (*scaler* & *encoder*) agar tidak perlu *training* berulang-ulang setiap kali program dijalankan.

#### [MODIFY] [main.py](file:///d:/SKRIPSII/PROJECT/lstm_komoditas/main.py)

### 3. Mengganti Logika Rekomendasi
Mengganti fungsi `recommend_crops()` agar memuat `recommender_model.keras`. Fungsi ini akan mensimulasikan input iklim dan tanah untuk ke-7 komoditas yang ada, lalu meminta model memprediksi probabilitas kelayakannya. Komoditas yang mendapat probabilitas tinggi untuk kelas "Sangat Layak" akan mendapatkan skor rekomendasi tertinggi.

#### [MODIFY] [main.py](file:///d:/SKRIPSII/PROJECT/lstm_komoditas/main.py)

## Verification Plan
1. Menjalankan skrip `main.py` menggunakan terminal.
2. Memastikan model iklim LSTM tetap berjalan dengan baik.
3. Memastikan model Neural Network rekomendasi sukses berlatih (*training*) dengan *loss* yang semakin mengecil.
4. Memverifikasi keluaran program memberikan urutan rekomendasi komoditas yang masuk akal berdasarkan hasil prediksi dari model AI yang baru (alih-alih fungsi *if/else* biasa).
