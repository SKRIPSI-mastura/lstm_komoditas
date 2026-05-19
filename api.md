# Dokumentasi API Sistem Rekomendasi Komoditas Pangan (Aceh Utara)

Dokumentasi ini menjelaskan penggunaan Web API berbasis Flask yang terintegrasi dengan model **LSTM (prediksi iklim)** dan **Neural Network (rekomendasi kelayakan komoditas)**.

## Informasi Umum
* **Base URL**: `http://localhost:5000` atau `http://127.0.0.1:5000`
* **Format Data**: JSON (`Content-Type: application/json`)
* **CORS**: Diaktifkan secara penuh (mengizinkan pemanggilan langsung dari frontend React/Vite di localhost)
* **Caching**: Menggunakan in-memory caching untuk endpoint `/api/recommend/<kecamatan_name>`.
  * **Kalkulasi Pertama**: ~10-15 detik (untuk melatih LSTM 20 epoch secara dinamis).
  * **Kalkulasi Berikutnya (Cache Hit)**: < 10 milidetik (respon instan).

---

## Ringkasan Endpoint

| No | Endpoint | Method | Deskripsi | Status |
|---|---|---|---|---|
| 1 | `/api/health` | `GET` | Cek status kesehatan server dan inisialisasi model | Aktif |
| 2 | `/api/kecamatan` | `GET` | Mengambil semua daftar kecamatan Aceh Utara | Aktif |
| 3 | `/api/kecamatan/<name>` | `GET` | Mengambil detail profil geografis/tanah kecamatan | Aktif |
| 4 | `/api/recommend/<name>` | `GET` | Menjalankan pipeline LSTM + NN untuk rekomendasi kecamatan | Aktif |
| 5 | `/api/recommend` | `POST` | Rekomendasi berdasarkan input manual parameter tanah & iklim | Aktif |

---

## Detail Endpoint

### 1. Cek Kesehatan Server (Health Check)
Memastikan API berjalan dengan baik dan status model rekomendasi telah dimuat ke dalam memori server.

* **URL**: `/api/health`
* **Method**: `GET`
* **Headers**: `None`

#### Contoh Respon (200 OK):
```json
{
  "status": "success",
  "message": "Sistem Rekomendasi Komoditas API berjalan normal.",
  "cache_entries": [],
  "model_loaded": true
}
```

#### Contoh Integrasi JavaScript:
```javascript
fetch('http://localhost:5000/api/health')
  .then(res => res.json())
  .then(data => console.log("Status API:", data));
```

#### Contoh cURL:
```bash
curl -X GET http://localhost:5000/api/health
```

---

### 2. Mengambil Semua Daftar Kecamatan
Mengambil daftar seluruh 27 nama kecamatan di Aceh Utara yang didukung oleh sistem.

* **URL**: `/api/kecamatan`
* **Method**: `GET`

#### Contoh Respon (200 OK):
```json
{
  "count": 27,
  "data": [
    "Baktiya",
    "Baktiya Barat",
    "Banda Baro",
    "Cot Girek",
    "Dewantara",
    "Kuta Makmur",
    "Langkahan",
    "Lapang",
    "Lhoksukon",
    "Matangkuli",
    "Meurah Mulia",
    "Muara Batu",
    "Nibong",
    "Nisam",
    "Nisam Antara",
    "Paya Bakong",
    "Samudera",
    "Sawang",
    "Seunuddon",
    "Syamtalira Aron",
    "Syamtalira Bayu",
    "Tanah Jambo Aye",
    "Tanah Luas",
    "Tanah Pasir",
    "Geureudong Pase",
    "Pirak Timu",
    "Simpang Keuramat",
  ],
  "status": "success"
}
```

#### Contoh Integrasi JavaScript (Dropdown Selector):
```javascript
fetch('http://localhost:5000/api/kecamatan')
  .then(res => res.json())
  .then(response => {
    if (response.status === 'success') {
      const kecamatanList = response.data;
      // Gunakan kecamatanList untuk dropdown menu di UI
    }
  });
```

---

### 3. Detail Profil Kecamatan
Mengambil data detail tanah (pH, persentase pasir/liat/debu), geografis (elevasi), curah hujan rata-rata tahunan, jenis tanah dominan, dan tingkat resiko bencana alam di kecamatan terkait.

* **URL**: `/api/kecamatan/<kecamatan_name>`
* **Method**: `GET`

#### Contoh Respon Sukses (200 OK - `/api/kecamatan/Lhoksukon`):
```json
{
  "data": {
    "elevasi": 12.10118031,
    "hujan_tahunan": 2283.52,
    "jenis_tanah": "Aluvial",
    "kecamatan": "Lhoksukon",
    "ph": 5.4,
    "resiko_bencana": "Tinggi",
    "tanah_debu_persen": 40.0,
    "tanah_liat_persen": 35.0,
    "tanah_pasir_persen": 25.0
  },
  "status": "success"
}
```

#### Contoh Respon Error (404 Not Found - Nama kecamatan salah):
```json
{
  "status": "error",
  "message": "Kecamatan 'Ngawur' tidak ditemukan."
}
```

---

### 4. Rekomendasi Berdasarkan Kecamatan (Pipeline LSTM + NN)
Endpoint paling utama. Endpoint ini melakukan proses berikut:
1. Membaca data iklim historis harian dari 2020-2025 untuk kecamatan terpilih.
2. Melatih model **LSTM dinamis** sebanyak 20 epoch khusus untuk kecamatan tersebut.
3. Memprediksi **suhu, kelembapan, dan kecepatan angin selama 7 hari ke depan**.
4. Melakukan rata-rata prediksi cuaca 7 hari tersebut.
5. Memasukkan parameter tanah asli daerah + rata-rata prediksi iklim ke model **Neural Network klasifikasi**.
6. Menghitung tingkat kecocokan (score %) beserta penjelasan logisnya untuk seluruh komoditas: **Padi, Jagung, Kedelai, Ubi Kayu, Ubi Jalar, Kacang Tanah, Kacang Hijau**.
7. Menyimpan hasilnya di memori cache server.

* **URL**: `/api/recommend/<kecamatan_name>`
* **Method**: `GET`

#### Contoh Respon Sukses (200 OK - `/api/recommend/Lhoksukon`):
```json
{
  "data": {
    "avg_climate_prediction": {
      "kecepatan_angin": 1.9397524653162275,
      "kelembapan": 87.59448436651913,
      "suhu": 26.46189575037786
    },
    "climate_prediction": [
      {
        "hari": 1,
        "kecepatan_angin": 1.9452862977981566,
        "kelembapan": 87.39821934461594,
        "suhu": 26.426506105661392
      },
      {
        "hari": 2,
        "kecepatan_angin": 1.9433988523483277,
        "kelembapan": 87.46049941658974,
        "suhu": 26.43746202111244
      },
      ...
      {
        "hari": 7,
        "kecepatan_angin": 1.9357272458076478,
        "kelembapan": 87.79801175355912,
        "suhu": 26.4940923422575
      }
    ],
    "kecamatan": "Lhoksukon",
    "profil_wilayah": {
      "curah_hujan_tahunan": 2283.52,
      "elevasi": 12.10118031,
      "jenis_tanah": "Aluvial",
      "ph": 5.4,
      "resiko_bencana": "Tinggi"
    },
    "recommendations": [
      {
        "kelayakan": "Layak",
        "komoditas": "Ubi Kayu",
        "score": 59.19
      },
      {
        "kelayakan": "Layak",
        "komoditas": "Ubi Jalar",
        "score": 58.76
      },
      {
        "kelayakan": "Layak",
        "komoditas": "Padi",
        "score": 57.32
      },
      {
        "kelayakan": "Layak",
        "komoditas": "Kedelai",
        "score": 55.31
      },
      {
        "kelayakan": "Layak",
        "komoditas": "Kacang Tanah",
        "score": 51.21
      },
      {
        "kelayakan": "Kurang Layak",
        "komoditas": "Kacang Hijau",
        "score": 47.34
      },
      {
        "kelayakan": "Kurang Layak",
        "komoditas": "Jagung",
        "score": 43.49
      }
    ],
    "top_recommendation": {
      "explanation": "Ubi Kayu merupakan komoditas yang paling optimal ditanam di Lhoksukon karena didukung oleh suhu hangat (26.5 °C), kelembapan udara tinggi (87.6%), curah hujan melimpah (2284 mm/tahun), toleransi yang baik terhadap tanah masam (pH 5.40), tekstur Aluvial yang mendukung perakaran, serta sesuai dengan ketinggian lahan 12.1 mdpl.",
      "kelayakan": "Layak",
      "komoditas": "Ubi Kayu",
      "score": 59.19
    }
  },
  "source": "model",
  "status": "success"
}
```
*Catatan: Pada pemanggilan kedua, field `"source"` akan bernilai `"cache"` dan respon akan dikembalikan dalam waktu < 10 md.*

#### Contoh Integrasi JavaScript:
```javascript
// Gunakan state loading saat memanggil ini di frontend Anda!
async function getRecommendations(kecamatanName) {
  try {
    const response = await fetch(`http://localhost:5000/api/recommend/${kecamatanName}`);
    const result = await response.json();
    if (result.status === 'success') {
      console.log("Rekomendasi Tanaman Teratas:", result.data.top_recommendation.komoditas);
      console.log("Rincian Kelayakan:", result.data.recommendations);
      console.log("Prediksi Cuaca 7 Hari:", result.data.climate_prediction);
    }
  } catch (error) {
    console.error("Gagal mendapatkan rekomendasi:", error);
  }
}
```

---

### 5. Evaluasi Rekomendasi Berdasarkan Parameter Kustom
Mengizinkan pengguna untuk memasukkan parameter tanah dan cuaca secara manual guna menganalisis kelayakan tanaman pada lahan khusus di luar database kecamatan resmi.

* **URL**: `/api/recommend`
* **Method**: `POST`
* **Headers**:
  * `Content-Type: application/json`

#### Format JSON Request Body:
```json
{
  "jenis_tanah": "Aluvial",
  "ph_tanah": 6.5,
  "tanah_liat_persen": 35.0,
  "tanah_pasir_persen": 25.0,
  "tanah_debu_persen": 40.0,
  "elevasi": 15.0,
  "hujan_tahunan": 1800.0,
  "suhu": 27.2,
  "kelembapan": 81.5,
  "kecamatan": "Lahan Pribadi A"  // Opsional
}
```

#### Contoh Respon Sukses (200 OK):
```json
{
  "data": {
    "inputs": {
      "elevasi": 15.0,
      "hujan_tahunan": 1800.0,
      "jenis_tanah": "Aluvial",
      "kecamatan": "Lahan Pribadi A",
      "kelembapan": 81.5,
      "ph_tanah": 6.5,
      "suhu": 27.2,
      "tanah_debu_persen": 40.0,
      "tanah_liat_persen": 35.0,
      "tanah_pasir_persen": 25.0
    },
    "recommendations": [
      {
        "kelayakan": "Layak",
        "komoditas": "Ubi Kayu",
        "score": 69.7
      },
      {
        "kelayakan": "Layak",
        "komoditas": "Ubi Jalar",
        "score": 69.4
      },
      {
        "kelayakan": "Layak",
        "komoditas": "Padi",
        "score": 68.9
      },
      ...
    ],
    "top_recommendation": {
      "explanation": "Ubi Kayu merupakan komoditas yang paling optimal ditanam di Lahan Pribadi A karena didukung oleh suhu hangat (27.2 °C), kelembapan udara tinggi (81.5%), curah hujan moderat (1800 mm/tahun), pH tanah ideal (6.50), tekstur Aluvial yang mendukung perakaran, serta sesuai dengan ketinggian lahan 15.0 mdpl.",
      "kelayakan": "Layak",
      "komoditas": "Ubi Kayu",
      "score": 69.7
    }
  },
  "status": "success"
}
```

#### Contoh Integrasi JavaScript:
```javascript
const payload = {
  jenis_tanah: "Aluvial",
  ph_tanah: 6.5,
  tanah_liat_persen: 35.0,
  tanah_pasir_persen: 25.0,
  tanah_debu_persen: 40.0,
  elevasi: 15.0,
  hujan_tahunan: 1800.0,
  suhu: 27.2,
  kelembapan: 81.5,
  kecamatan: "Lahan Percobaan B"
};

fetch('http://localhost:5000/api/recommend', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(payload)
})
  .then(res => res.json())
  .then(result => {
    if (result.status === 'success') {
      console.log("Rekomendasi Custom:", result.data.top_recommendation);
    }
  });
```

#### Contoh cURL:
```bash
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "jenis_tanah": "Aluvial",
    "ph_tanah": 6.5,
    "tanah_liat_persen": 35.0,
    "tanah_pasir_persen": 25.0,
    "tanah_debu_persen": 40.0,
    "elevasi": 15.0,
    "hujan_tahunan": 1800.0,
    "suhu": 27.2,
    "kelembapan": 81.5
  }'
```

---

## Pemetaan Label Kelayakan (Suitability Score Mapping)
Model Neural Network mengembalikan tingkat kecocokan tanaman dalam bentuk persentase (0 - 100%). Nilai tersebut kemudian dipetakan ke tingkat kelayakan berdasarkan standardisasi pertanian sebagai berikut:

* **Score $\ge$ 75.0%** : `Sangat Layak` (Tanaman sangat cocok dengan iklim & nutrisi tanah setempat)
* **Score $\ge$ 50.0%** : `Layak` (Tanaman dapat tumbuh subur dengan perawatan standar)
* **Score $\ge$ 25.0%** : `Kurang Layak` (Tanaman memerlukan perlakuan/nutrisi ekstra)
* **Score < 25.0%** : `Tidak Layak` (Kondisi alam setempat tidak bersahabat untuk komoditas ini)
