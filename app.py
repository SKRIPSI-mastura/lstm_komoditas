import os
import sys
import logging
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# Pastikan workspace root masuk dalam path pencarian python
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import fungsi dari main.py
import main

# --- KONFIGURASI FLASK & LOGGING ---
app = Flask(__name__)
CORS(app)  # Izinkan frontend mengakses API ini

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# --- GLOBAL DATA & CACHE ---
KEC_DATA = {}
prediction_cache = {}

def get_suitability_label(score):
    """Mengubah skor numerik menjadi label kelayakan."""
    if score >= 75.0:
        return "Sangat Layak"
    elif score >= 50.0:
        return "Layak"
    elif score >= 25.0:
        return "Kurang Layak"
    else:
        return "Tidak Layak"

def initialize_app():
    """Inisialisasi data kecamatan dan model rekomendasi di memori pada startup."""
    global KEC_DATA
    # Inisialisasi ini hanya dilakukan sekali
    if not KEC_DATA:
        logging.info("Memulai inisialisasi aplikasi backend...")
        try:
            # 1. Muat profil kecamatan
            KEC_DATA = main.load_kecamatan_data()
            logging.info(f"Berhasil memuat {len(KEC_DATA)} profil kecamatan.")
            
            # 2. Latih model rekomendasi (Neural Network) di memori
            main.train_recommendation_model()
            logging.info("Berhasil menginisialisasi model rekomendasi di memori.")
        except Exception as e:
            logging.error(f"Gagal melakukan inisialisasi startup: {str(e)}")

# Panggil inisialisasi saat aplikasi pertama kali dimuat
initialize_app()

@app.route('/', methods=['GET'])
def index():
    """Endpoint root untuk memastikan API berjalan dan tidak menampilkan 404 di browser."""
    return jsonify({
        "status": "success",
        "message": "Selamat datang di API Sistem Rekomendasi Komoditas Pertanian. Akses /api/health untuk status sistem."
    }), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint untuk mengecek status kesehatan API."""
    return jsonify({
        "status": "success",
        "message": "Sistem Rekomendasi Komoditas API berjalan normal.",
        "cache_entries": list(prediction_cache.keys()),
        "model_loaded": main.recommender_bundle is not None
    }), 200

@app.route('/api/kecamatan', methods=['GET'])
def get_all_kecamatan():
    """Mendapatkan daftar semua kecamatan yang tersedia."""
    if not KEC_DATA:
        return jsonify({"status": "error", "message": "Data kecamatan belum dimuat."}), 500
    
    list_kec = sorted(list(KEC_DATA.keys()))
    return jsonify({
        "status": "success",
        "count": len(list_kec),
        "data": list_kec
    }), 200

@app.route('/api/kecamatan/<string:kecamatan_name>', methods=['GET'])
def get_kecamatan_profile(kecamatan_name):
    """Mendapatkan profil tanah dan geografis untuk kecamatan tertentu."""
    # Cari dengan nama kecamatan yang di-strip (case-insensitive)
    search_name = kecamatan_name.strip()
    match_key = None
    
    for key in KEC_DATA.keys():
        if key.lower() == search_name.lower():
            match_key = key
            break
            
    if not match_key:
        return jsonify({
            "status": "error",
            "message": f"Kecamatan '{kecamatan_name}' tidak ditemukan."
        }), 404
        
    profile = KEC_DATA[match_key].copy()
    profile['kecamatan'] = match_key
    
    return jsonify({
        "status": "success",
        "data": profile
    }), 200

@app.route('/api/recommend/<string:kecamatan_name>', methods=['GET'])
def recommend_by_kecamatan(kecamatan_name):
    """
    Menjalankan pipeline LSTM + NN Rekomendasi untuk kecamatan tertentu.
    Menggunakan caching memori untuk memberikan respon instan setelah kalkulasi pertama.
    """
    search_name = kecamatan_name.strip()
    match_key = None
    
    for key in KEC_DATA.keys():
        if key.lower() == search_name.lower():
            match_key = key
            break
            
    if not match_key:
        return jsonify({
            "status": "error",
            "message": f"Kecamatan '{kecamatan_name}' tidak ditemukan."
        }), 404
        
    # Cek cache terlebih dahulu
    if match_key in prediction_cache:
        logging.info(f"[CACHE HIT] Mengembalikan hasil rekomendasi untuk {match_key} dari cache.")
        return jsonify({
            "status": "success",
            "source": "cache",
            "data": prediction_cache[match_key]
        }), 200

    logging.info(f"[CACHE MISS] Memproses prediksi LSTM dan rekomendasi untuk {match_key}...")
    
    try:
        kec_info = KEC_DATA[match_key]
        
        # 1. Load historical climate data
        df_climate = main.load_climate_data(match_key)
        if df_climate.empty:
            return jsonify({
                "status": "error",
                "message": f"Data iklim historis untuk {match_key} tidak ditemukan."
            }), 404
            
        # 2. Preprocess data untuk LSTM
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_climate.values)
        
        SEQ_LENGTH = 30
        if len(scaled_data) < SEQ_LENGTH:
            return jsonify({
                "status": "error",
                "message": f"Data iklim historis untuk {match_key} tidak cukup (min 30 hari)."
            }), 400
            
        X, y = main.create_sequences(scaled_data, SEQ_LENGTH)
        
        split = int(0.8 * len(X))
        # 3. Bangun dan Latih Model LSTM Climate (Live Training)
        logging.info(f"[LIVE TRAINING] Melatih LSTM baru untuk {match_key}...")
        X_train, y_train = X[:split], y[:split]
        
        lstm_model = Sequential([
            Input(shape=(SEQ_LENGTH, 3)),
            LSTM(64, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dense(3)
        ])
        lstm_model.compile(optimizer='adam', loss='mse')
        lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        
        # 4. Lakukan Prediksi Iklim 7 Hari ke Depan
        last_seq = scaled_data[-SEQ_LENGTH:]
        predictions = []
        curr_seq = last_seq.reshape(1, SEQ_LENGTH, 3)
        
        for _ in range(7):
            pred = lstm_model.predict(curr_seq, verbose=0)
            predictions.append(pred[0])
            curr_seq = np.append(curr_seq[:, 1:, :], pred.reshape(1, 1, 3), axis=1)
            
        predicted_climate = scaler.inverse_transform(predictions)
        avg_pred = np.mean(predicted_climate, axis=0)
        
        # Format prediksi cuaca per hari
        climate_list = []
        for i, pred_day in enumerate(predicted_climate, 1):
            climate_list.append({
                "hari": i,
                "suhu": float(pred_day[0]),
                "kelembapan": float(pred_day[1]),
                "kecepatan_angin": float(pred_day[2])
            })
            
        avg_climate = {
            "suhu": float(avg_pred[0]),
            "kelembapan": float(avg_pred[1]),
            "kecepatan_angin": float(avg_pred[2])
        }
        
        # 5. Siapkan Input untuk Model Rekomendasi
        user_inputs = {
            'kecamatan': match_key,
            'jenis_tanah': kec_info['jenis_tanah'],
            'ph_tanah': kec_info['ph'],
            'tanah_liat_persen': kec_info['tanah_liat_persen'],
            'tanah_pasir_persen': kec_info['tanah_pasir_persen'],
            'tanah_debu_persen': kec_info['tanah_debu_persen'],
            'elevasi': kec_info['elevasi'],
            'hujan_tahunan': kec_info['hujan_tahunan'],
            'resiko_bencana': kec_info['resiko_bencana']
        }
        
        # 6. Jalankan Evaluasi Kelayakan Komoditas dengan Neural Network
        raw_recommendations = main.recommend_crops(avg_pred, user_inputs)
        
        # Format daftar rekomendasi dengan label kelayakan
        recommendations_list = []
        for crop, score in raw_recommendations.items():
            recommendations_list.append({
                "komoditas": crop,
                "score": round(score, 2),
                "kelayakan": get_suitability_label(score)
            })
            
        # Dapatkan rekomendasi terbaik
        top_crop = recommendations_list[0]["komoditas"]
        top_score = recommendations_list[0]["score"]
        explanation = main.generate_explanation(top_crop, user_inputs, avg_pred)
        
        response_data = {
            "kecamatan": match_key,
            "profil_wilayah": {
                "elevasi": float(kec_info['elevasi']),
                "ph": float(kec_info['ph']),
                "jenis_tanah": kec_info['jenis_tanah'],
                "curah_hujan_tahunan": float(kec_info['hujan_tahunan']),
                "resiko_bencana": kec_info['resiko_bencana']
            },
            "climate_prediction": climate_list,
            "avg_climate_prediction": avg_climate,
            "recommendations": recommendations_list,
            "top_recommendation": {
                "komoditas": top_crop,
                "score": top_score,
                "kelayakan": get_suitability_label(top_score),
                "explanation": explanation
            }
        }
        
        # Simpan ke cache agar pemanggilan selanjutnya instan
        prediction_cache[match_key] = response_data
        logging.info(f"[SUCCESS] Berhasil memproses dan menyimpan hasil untuk {match_key} di cache.")
        
        return jsonify({
            "status": "success",
            "source": "model",
            "data": response_data
        }), 200
        
    except Exception as e:
        logging.error(f"Terjadi kesalahan saat memproses rekomendasi {match_key}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"Terjadi kesalahan internal: {str(e)}"
        }), 500

@app.route('/api/recommend', methods=['POST'])
def recommend_custom():
    """
    Endpoint kustom untuk menghitung kelayakan komoditas berdasarkan input manual pengguna.
    Menerima JSON parameter tanah & iklim, lalu mengevaluasi dengan model Neural Network.
    """
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "Payload JSON kosong."}), 400
        
    required_fields = [
        'jenis_tanah', 'ph_tanah', 'tanah_liat_persen', 
        'tanah_pasir_persen', 'tanah_debu_persen', 'elevasi', 
        'hujan_tahunan', 'suhu', 'kelembapan'
    ]
    
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({
            "status": "error",
            "message": f"Field berikut wajib diisi: {', '.join(missing)}"
        }), 400
        
    try:
        # Menjamin model NN sudah dilatih
        if main.recommender_bundle is None:
            main.train_recommendation_model()
            
        # Siapkan parameter input custom
        user_inputs = {
            'kecamatan': data.get('kecamatan', 'Kustom'),
            'jenis_tanah': data['jenis_tanah'],
            'ph_tanah': float(data['ph_tanah']),
            'tanah_liat_persen': float(data['tanah_liat_persen']),
            'tanah_pasir_persen': float(data['tanah_pasir_persen']),
            'tanah_debu_persen': float(data['tanah_debu_persen']),
            'elevasi': float(data['elevasi']),
            'hujan_tahunan': float(data['hujan_tahunan']),
            'resiko_bencana': data.get('resiko_bencana', "Tinggi" if float(data['elevasi']) < 15 else "Rendah")
        }
        
        # Susun prediksi iklim buatan
        avg_pred = [
            float(data['suhu']),
            float(data['kelembapan']),
            float(data.get('kecepatan_angin', 2.0))
        ]
        
        # Jalankan evaluasi
        raw_recommendations = main.recommend_crops(avg_pred, user_inputs)
        
        recommendations_list = []
        for crop, score in raw_recommendations.items():
            recommendations_list.append({
                "komoditas": crop,
                "score": round(score, 2),
                "kelayakan": get_suitability_label(score)
            })
            
        top_crop = recommendations_list[0]["komoditas"]
        top_score = recommendations_list[0]["score"]
        explanation = main.generate_explanation(top_crop, user_inputs, avg_pred)
        
        return jsonify({
            "status": "success",
            "data": {
                "inputs": {
                    "kecamatan": user_inputs['kecamatan'],
                    "jenis_tanah": user_inputs['jenis_tanah'],
                    "ph_tanah": user_inputs['ph_tanah'],
                    "tanah_liat_persen": user_inputs['tanah_liat_persen'],
                    "tanah_pasir_persen": user_inputs['tanah_pasir_persen'],
                    "tanah_debu_persen": user_inputs['tanah_debu_persen'],
                    "elevasi": user_inputs['elevasi'],
                    "hujan_tahunan": user_inputs['hujan_tahunan'],
                    "suhu": avg_pred[0],
                    "kelembapan": avg_pred[1]
                },
                "recommendations": recommendations_list,
                "top_recommendation": {
                    "komoditas": top_crop,
                    "score": top_score,
                    "kelayakan": get_suitability_label(top_score),
                    "explanation": explanation
                }
            }
        }), 200
        
    except Exception as e:
        logging.error(f"Terjadi kesalahan saat evaluasi kustom: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Terjadi kesalahan evaluasi: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Jalankan server Flask di semua interface (0.0.0.0) pada port 5000
    logging.info("Menjalankan API server...")
    app.run(host='0.0.0.0', port=5000, debug=False)
