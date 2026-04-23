import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# --- PATH DATA ---
DATA_DIR = "data"
FILE_IKLIM = os.path.join(DATA_DIR, "iklim_semua_kecamatan_aceh_utara_2020_2025.csv")
FILE_ELEVASI = os.path.join(DATA_DIR, "Elevasi_Kecamatan_Aceh_Utara.csv")
FILE_PH = os.path.join(DATA_DIR, "pH_Tanah_Kecamatan_Aceh_Utara.csv")
FILE_HUJAN = os.path.join(DATA_DIR, "Curah_Hujan_Tahunan_Kecamatan_Aceh_Utara_2025.csv")

# 1. Load Kecamatan Profiles
def load_kecamatan_data():
    df_elev = pd.read_csv(FILE_ELEVASI)
    df_ph = pd.read_csv(FILE_PH)
    df_hujan = pd.read_csv(FILE_HUJAN)
    
    # Merge data
    # Standardize kecamatan names to handle potential whitespace or case issues
    df_elev['kecamatan'] = df_elev['kecamatan'].str.strip()
    df_ph['kecamatan'] = df_ph['kecamatan'].str.strip()
    df_hujan['kecamatan'] = df_hujan['kecamatan'].str.strip()
    
    merged = df_elev[['kecamatan', 'elevasi_mdpl']].merge(
        df_ph[['kecamatan', 'ph_tanah_mean']], on='kecamatan'
    ).merge(
        df_hujan[['kecamatan', 'curah_hujan_tahunan']], on='kecamatan'
    )
    
    # Convert to dictionary for easier access
    kec_dict = {}
    for _, row in merged.iterrows():
        kec_dict[row['kecamatan']] = {
            "elevasi": row['elevasi_mdpl'],
            "ph": row['ph_tanah_mean'],
            "hujan_tahunan": row['curah_hujan_tahunan'],
            # Jenis tanah dan resiko bencana tidak ada di CSV, gunakan estimasi
            "jenis_tanah": "Aluvial" if row['elevasi_mdpl'] < 50 else "Podsolik",
            "resiko_bencana": "Tinggi" if row['elevasi_mdpl'] < 15 else "Rendah"
        }
    return kec_dict

# 2. Load Climate Data for LSTM
def load_climate_data(kecamatan_name):
    df = pd.read_csv(FILE_IKLIM)
    df['date'] = pd.to_datetime(df['date'])
    df_kec = df[df['kecamatan'].str.strip() == kecamatan_name].copy()
    df_kec.set_index('date', inplace=True)
    
    # Pilih fitur: T2M (Suhu), RH2M (Kelembapan), WS2M (Kecepatan Angin)
    # PRECTOT kosong di data, jadi kita gunakan fitur yang tersedia
    features = ['T2M', 'RH2M', 'WS2M']
    return df_kec[features].fillna(method='ffill')

# 3. Data Preprocessing
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 4. Recommendation Logic
def recommend_crops(climate_pred, inputs):
    # climate_pred: [suhu, kelembapan, ws2m]
    suhu, hum, ws2m = climate_pred
    tanah = inputs['jenis_tanah']
    ph = inputs['ph_tanah']
    elev = inputs['elevasi']
    hujan_tahunan = inputs['hujan_tahunan']
    risk = inputs['resiko_bencana']
    
    # Estimasi ketersediaan air berdasarkan curah hujan tahunan
    # Rata-rata di Aceh Utara ~2000mm. Jika > 2200 dianggap tinggi, < 1800 rendah.
    air = 'Tinggi' if hujan_tahunan > 2100 else 'Sedang'
    
    crops = {
        "Padi": 0, "Jagung": 0, "Kedelai": 0, 
        "Kacang Hijau": 0, "Kacang Tanah": 0, 
        "Ubi Jalar": 0, "Ubi Kayu": 0
    }
    
    for crop in crops:
        score = 0
        
        # 1. Kecocokan Suhu
        if 24 <= suhu <= 32: score += 20
        
        # 2. Kecocokan pH Tanah
        if 5.5 <= ph <= 7.0: score += 15
        
        # 3. Faktor Jenis Tanah
        if "Aluvial" in tanah:
            if crop in ["Padi", "Kedelai"]: score += 25
        elif "Podsolik" in tanah:
            if crop in ["Jagung", "Ubi Kayu", "Kacang Tanah"]: score += 25
        
        # 4. Faktor Elevasi
        if elev < 100: 
            if crop in ["Padi", "Kedelai", "Kacang Hijau"]: score += 20
        else: 
            if crop in ["Ubi Kayu", "Jagung", "Ubi Jalar"]: score += 20
            
        # 5. Faktor Curah Hujan (Gunakan data tahunan sebagai proksi)
        if crop == "Padi":
            if hujan_tahunan > 2000 or air == 'Tinggi': score += 20
        elif crop in ["Jagung", "Kedelai"]:
            if 1800 < hujan_tahunan < 2200: score += 15
        elif crop in ["Kacang Tanah", "Ubi Kayu"]:
            if hujan_tahunan < 2000: score += 15
            
        # 6. Faktor Resiko Bencana
        if risk == 'Tinggi':
            if crop == "Padi": score += 10
            elif crop == "Jagung": score -= 20
        elif risk == 'Rendah':
            score += 10
        
        crops[crop] = min(98, max(40, score + np.random.randint(5, 10)))
        
    return dict(sorted(crops.items(), key=lambda item: item[1], reverse=True))

# 5. Main Workflow
def main():
    print("\n" + "="*50)
    print("SISTEM REKOMENDASI KOMODITAS PANGAN - ACEH UTARA")
    print("="*50)
    
    # Load Data Kecamatan
    print("[INFO] Memuat data kecamatan...")
    KEC_DATA = load_kecamatan_data()
    list_kecamatan = sorted(list(KEC_DATA.keys()))
    
    print("\nDaftar Kecamatan di Aceh Utara:")
    for i, kec in enumerate(list_kecamatan, 1):
        print(f"{i}. {kec:<20}", end="\t" if i % 3 != 0 else "\n")
    
    try:
        choice = int(input("\n\nPilih nomor kecamatan: "))
        if not (1 <= choice <= len(list_kecamatan)):
            raise ValueError
    except ValueError:
        print("Pilihan tidak valid, menggunakan Lhoksukon sebagai default.")
        selected_kec = "Lhoksukon"
    else:
        selected_kec = list_kecamatan[choice - 1]
    
    kec_info = KEC_DATA[selected_kec]
    
    print(f"\n[INFO] Menganalisis untuk Kecamatan: {selected_kec}")
    print(f"[INFO] Profil Wilayah: Elevasi {kec_info['elevasi']:.1f} mdpl, pH {kec_info['ph']:.2f}")
    
    print("\nMempersiapkan data iklim historis...")
    df_climate = load_climate_data(selected_kec)
    
    if df_climate.empty:
        print(f"[ERROR] Data iklim untuk {selected_kec} tidak ditemukan.")
        return

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_climate)
    
    SEQ_LENGTH = 30 
    X, y = create_sequences(scaled_data, SEQ_LENGTH)
    
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    
    # Build Model
    model = Sequential([
        Input(shape=(SEQ_LENGTH, 3)),
        LSTM(64, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dense(3) 
    ])
    model.compile(optimizer='adam', loss='mse')
    
    print(f"Melatih model LSTM untuk {selected_kec} (Epochs: 5)...")
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    
    # Predict next 7 days
    last_seq = scaled_data[-SEQ_LENGTH:]
    predictions = []
    curr_seq = last_seq.reshape(1, SEQ_LENGTH, 3)
    for _ in range(7):
        pred = model.predict(curr_seq, verbose=0)
        predictions.append(pred[0])
        curr_seq = np.append(curr_seq[:, 1:, :], pred.reshape(1, 1, 3), axis=1)
    
    predicted_climate = scaler.inverse_transform(predictions)
    avg_pred = np.mean(predicted_climate, axis=0)
    
    print(f"\n--- Hasil Prediksi Iklim (Rata-rata 7 Hari ke Depan) ---")
    print(f"Suhu: {avg_pred[0]:.2f} °C")
    print(f"Kelembapan: {avg_pred[1]:.2f} %")
    print(f"Kecepatan Angin: {avg_pred[2]:.2f} m/s")
    
    # Prepare inputs for recommendation
    user_inputs = {
        'kecamatan': selected_kec,
        'jenis_tanah': kec_info['jenis_tanah'],
        'ph_tanah': kec_info['ph'],
        'elevasi': kec_info['elevasi'],
        'hujan_tahunan': kec_info['hujan_tahunan'],
        'resiko_bencana': kec_info['resiko_bencana']
    }
    
    recommendations = recommend_crops(avg_pred, user_inputs)
    
    print(f"\n--- Rekomendasi Komoditas untuk {selected_kec} ---")
    for crop, score in recommendations.items():
        print(f"- {crop}: {score}% Kecocokan")

if __name__ == "__main__":
    main()
