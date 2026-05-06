import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from data_loader import load_kecamatan_profiles, load_climate_data

# --- KONFIGURASI ---
MODEL_PATH = "models/commodity_lstm_model.keras"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
SEQ_LENGTH = 30

def predict_commodity():
    print("\n" + "="*50)
    print("SISTEM PREDIKSI KOMODITAS (LSTM) - ACEH UTARA")
    print("="*50)
    
    # 1. Load Assets
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Model belum dilatih. Jalankan train.py terlebih dahulu.")
        return

    print("[INFO] Memuat model dan scaler...")
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(ENCODER_PATH)
    
    # 2. Load Kecamatan Data
    profiles = load_kecamatan_profiles()
    list_kecamatan = sorted(profiles['kecamatan'].unique().tolist())
    
    print("\nDaftar Kecamatan:")
    for i, kec in enumerate(list_kecamatan, 1):
        print(f"{i}. {kec:<20}", end="\t" if i % 3 != 0 else "\n")
    
    try:
        choice = int(input("\n\nPilih nomor kecamatan: "))
        selected_kec = list_kecamatan[choice - 1]
    except:
        print("Pilihan tidak valid, menggunakan Lhoksukon sebagai default.")
        selected_kec = "Lhoksukon"
        
    # 3. Get Data Terbaru (Last 30 days)
    print(f"\n[INFO] Mengambil data historis terbaru untuk {selected_kec}...")
    df_climate = load_climate_data()
    df_kec = df_climate[df_climate['kecamatan'] == selected_kec].sort_values('date')
    
    if len(df_kec) < SEQ_LENGTH:
        print("[ERROR] Data iklim tidak cukup untuk membentuk sequence.")
        return
    
    # Ambil profil kecamatan
    prof = profiles[profiles['kecamatan'] == selected_kec].iloc[0]
    
    # Gabungkan fitur
    df_recent = df_kec.tail(SEQ_LENGTH).copy()
    df_recent['elevasi_mdpl'] = prof['elevasi_mdpl']
    df_recent['ph_tanah_mean'] = prof['ph_tanah_mean']
    df_recent['curah_hujan_tahunan'] = prof['curah_hujan_tahunan']
    
    feature_cols = ['T2M', 'RH2M', 'WS2M', 'elevasi_mdpl', 'ph_tanah_mean', 'curah_hujan_tahunan']
    input_data = df_recent[feature_cols].values
    
    # 4. Preprocessing (Scaling)
    input_scaled = scaler.transform(input_data)
    input_reshaped = input_scaled.reshape(1, SEQ_LENGTH, len(feature_cols))
    
    # 5. Prediction
    print("[INFO] Melakukan prediksi dengan model LSTM...")
    prediction_probs = model.predict(input_reshaped, verbose=0)
    predicted_class_idx = np.argmax(prediction_probs)
    predicted_commodity = le.inverse_transform([predicted_class_idx])[0]
    confidence = prediction_probs[0][predicted_class_idx] * 100
    
    # 6. Output
    print(f"\n" + "-"*50)
    print(f"HASIL REKOMENDASI UNTUK {selected_kec}:")
    print(f"Komoditas: {predicted_commodity}")
    print(f"Tingkat Kepercayaan Model: {confidence:.2f}%")
    print("-"*50)
    
    # Tampilkan probabilitas lainnya
    print("\nDetail Probabilitas:")
    for idx, cls in enumerate(le.classes_):
        print(f"- {cls}: {prediction_probs[0][idx]*100:.2f}%")

if __name__ == "__main__":
    predict_commodity()
