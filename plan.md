# Rencana Penyempurnaan Sistem Rekomendasi Komoditas (Tahap Lanjutan)

Sistem inti untuk prediksi iklim (LSTM) dan rekomendasi tanaman (Neural Network) secara in-memory telah berhasil diimplementasikan dan berjalan dengan baik. 

Untuk meningkatkan kualitas dari sistem ini (khususnya untuk kebutuhan penelitian/Skripsi), berikut adalah rencana tahap penyempurnaan yang dapat dilakukan selanjutnya:

## 1. Menampilkan Metrik Evaluasi Model (Akurasi & Error)
Membuktikan seberapa "pintar" model AI yang dibuat dengan menampilkan skor evaluasi di terminal:
*   **Akurasi (Accuracy):** Menampilkan tingkat akurasi dari model Neural Network rekomendasi tanaman setelah selesai dilatih.
*   **Tingkat Eror (MSE/MAE):** Menampilkan nilai *Mean Squared Error* (MSE) dari prediksi iklim LSTM sebagai bukti matematis validitas model.

## 2. Visualisasi Data dan Hasil (Plotting)
Mengganti output yang murni berbasis teks dengan visualisasi menggunakan pustaka `matplotlib` untuk menghasilkan grafik interaktif:
*   **Diagram Garis (Line Chart):** Visualisasi hasil prediksi iklim (suhu, curah hujan, dan kelembapan) untuk 7 hari ke depan.
*   **Diagram Batang (Bar Chart):** Visualisasi persentase tingkat kecocokan dari ke-7 komoditas untuk memudahkan pemahaman perbandingan hasil rekomendasi.

## 3. Merapikan Struktur Kode (Modularisasi)
Memecah file `main.py` yang sudah sangat panjang (300+ baris) ke dalam arsitektur yang lebih rapi sesuai standar *software engineering*:
*   `data_loader.py`: Modul khusus untuk membaca dataset dan memproses input pengguna.
*   `models.py`: Modul khusus untuk definisi arsitektur *Machine Learning* (LSTM & Neural Network) dan proses *training*.
*   `main.py`: Menjadi *entry point* yang bersih dan hanya berisi alur logika interaksi utama.

## 4. Integrasi ke Dashboard Web / API
*(Opsional / Jika Diperlukan)* 
Mengubah skrip Python ini menjadi layanan *backend* mandiri (menggunakan Flask atau FastAPI). Hal ini diperlukan jika keluaran AI ini akan ditampilkan pada sebuah situs web cerdas atau aplikasi *dashboard* antarmuka (misalnya React/Next.js).

---
> **Catatan:** Pilih salah satu dari prioritas di atas untuk mulai diimplementasikan pada pengerjaan berikutnya.
