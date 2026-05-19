import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
# pyrefly: ignore [missing-import]
from tensorflow.keras.utils import to_categorical

# --- PATH DATA ---
DATA_DIR = "data"
FILE_IKLIM = os.path.join(DATA_DIR, "iklim_semua_kecamatan_aceh_utara_2020_2025.csv")
FILE_ELEVASI = os.path.join(DATA_DIR, "Elevasi_Kecamatan_Aceh_Utara.csv")
FILE_TANAH = os.path.join(DATA_DIR, "data_tanah_aceh_utara2.csv")
FILE_HUJAN = os.path.join(DATA_DIR, "data_curah_hujan_aceh_utara.csv")
FILE_DATASET_TANAMAN = os.path.join(DATA_DIR, "dataset_lstm_tanaman.csv")
# --- GLOBAL VARIABLES FOR IN-MEMORY MODELS ---
recommender_bundle = None

# 1. Load Kecamatan Profiles
def load_kecamatan_data():
    df_elev = pd.read_csv(FILE_ELEVASI)
    # File tanah dan hujan adalah Excel meskipun ekstensinya .csv
    df_tanah = pd.read_excel(FILE_TANAH, engine='openpyxl')
    df_hujan = pd.read_excel(FILE_HUJAN, engine='openpyxl')
    
    # Merge data
    # Standardize kecamatan names to handle potential whitespace or case issues
    df_elev['kecamatan'] = df_elev['kecamatan'].str.strip()
    df_tanah['kecamatan'] = df_tanah['kecamatan'].str.strip()
    df_hujan['kecamatan'] = df_hujan['kecamatan'].str.strip()
    
    # Hitung rata-rata curah hujan tahunan dari data harian
    df_hujan['date'] = pd.to_datetime(df_hujan['date'].astype(str))
    df_hujan['year'] = df_hujan['date'].dt.year
    hujan_tahunan = df_hujan.groupby(['kecamatan', 'year'])['curah_hujan'].sum().groupby('kecamatan').mean().reset_index()
    hujan_tahunan.rename(columns={'curah_hujan': 'curah_hujan_tahunan'}, inplace=True)
    
    merged = df_elev[['kecamatan', 'elevasi_mdpl']].merge(
        df_tanah[['kecamatan', 'ph_tanah', 'tanah_liat', 'tanah_pasir', 'tanah_debu', 'tekstur_tanah']], on='kecamatan'
    ).merge(
        hujan_tahunan[['kecamatan', 'curah_hujan_tahunan']], on='kecamatan'
    )
    
    # Convert to dictionary for easier access
    kec_dict = {}
    for _, row in merged.iterrows():
        kec_dict[row['kecamatan']] = {
            "elevasi": row['elevasi_mdpl'],
            # Nilai ph_tanah dan tanah_liat di dataset tertukar, jadi kita ambil kolom tanah_liat sebagai pH
            "ph": row['tanah_liat'],
            "tanah_liat_persen": row['ph_tanah'],
            "tanah_pasir_persen": row['tanah_pasir'],
            "tanah_debu_persen": row['tanah_debu'],
            "hujan_tahunan": row['curah_hujan_tahunan'],
            # Gunakan tekstur_tanah sebagai jenis tanah
            "jenis_tanah": row['tekstur_tanah'],
            "resiko_bencana": "Tinggi" if row['elevasi_mdpl'] < 15 else "Rendah"
        }
    return kec_dict

# 2. Load Climate Data for LSTM
def load_climate_data(kecamatan_name):
    df = pd.read_csv(FILE_IKLIM)
    df['date'] = pd.to_datetime(df['date'])
    df_kec = df[df['kecamatan'].str.strip() == kecamatan_name].copy()
    df_kec.set_index('date', inplace=True)
    
    # Pilih fitur: Menggunakan nama kolom bahasa Indonesia sesuai dataset baru
    features = ['Suhu rata-rata', 'Kelembapan udara', 'Kecepatan angin']
    # Gunakan ffill dari ffill() karena argumen 'method' deprecated di pandas terbaru
    return df_kec[features].ffill()

# 3. Data Preprocessing
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

import joblib

# 4. Recommendation Model Training & Inference
def train_recommendation_model():
    global recommender_bundle
    if recommender_bundle is not None:
        return

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    nn_model_path = os.path.join(model_dir, "nn_recommender.keras")
    bundle_path = os.path.join(model_dir, "recommender_bundle.pkl")

    if os.path.exists(nn_model_path) and os.path.exists(bundle_path):
        from tensorflow.keras.models import load_model
        print("\n[INFO] Memuat Model Neural Network Rekomendasi dari disk...")
        model = load_model(nn_model_path)
        bundle_data = joblib.load(bundle_path)
        
        recommender_bundle = {
            'model': model,
            'scaler': bundle_data['scaler'],
            'le_komoditas': bundle_data['le_komoditas'],
            'le_tekstur': bundle_data['le_tekstur'],
            'le_label': bundle_data['le_label']
        }
        print("[SUCCESS] Model rekomendasi berhasil dimuat.")
        return

    print("\n[INFO] Melatih Model Neural Network Rekomendasi Tanaman...")
    df = pd.read_csv(FILE_DATASET_TANAMAN)

    le_komoditas = LabelEncoder()
    komoditas_encoded = le_komoditas.fit_transform(df['komoditas'])
    komoditas_ohe = to_categorical(komoditas_encoded)
    
    le_tekstur = LabelEncoder()
    tekstur_encoded = le_tekstur.fit_transform(df['tekstur_tanah'])
    tekstur_ohe = to_categorical(tekstur_encoded)

    le_label = LabelEncoder()
    df['label_encoded'] = le_label.fit_transform(df['label_kelayakan'])
    y = to_categorical(df['label_encoded'])

    num_features = df[['suhu_c', 'curah_hujan_mm_tahun', 'kelembapan_persen', 
            'ph_tanah', 'tanah_liat_persen', 'tanah_pasir_persen', 'tanah_debu_persen', 
            'elevasi_mdpl']].values
            
    X = np.hstack((komoditas_ohe, num_features, tekstur_ohe))

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = Sequential([
        Input(shape=(X_scaled.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(y.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Latih model klasifikasi
    model.fit(X_train, y_train, epochs=150, batch_size=16, verbose=0)
    
    # Evaluasi model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Simpan ke disk
    model.save(nn_model_path)
    joblib.dump({
        'scaler': scaler,
        'le_komoditas': le_komoditas,
        'le_tekstur': le_tekstur,
        'le_label': le_label
    }, bundle_path)
    
    recommender_bundle = {
        'model': model,
        'scaler': scaler,
        'le_komoditas': le_komoditas,
        'le_tekstur': le_tekstur,
        'le_label': le_label
    }
    print("[SUCCESS] Model rekomendasi berhasil dilatih dan disimpan di memori dan disk.")
    print(f"[EVALUASI] Akurasi Neural Network Rekomendasi Tanaman: {accuracy*100:.2f}%")

def recommend_crops(climate_pred, inputs):
    # climate_pred: [suhu, kelembapan, ws2m]
    suhu, hum, _ = climate_pred
    
    if recommender_bundle is None:
        train_recommendation_model()
        
    model = recommender_bundle['model']
    scaler = recommender_bundle['scaler']
    le_komoditas = recommender_bundle['le_komoditas']
    le_tekstur = recommender_bundle['le_tekstur']
    le_label = recommender_bundle['le_label']

    crops_to_evaluate = le_komoditas.classes_
    scores = {}

    for crop in crops_to_evaluate:
        komoditas_encoded = le_komoditas.transform([crop])[0]
        komoditas_ohe = to_categorical(komoditas_encoded, num_classes=len(le_komoditas.classes_))
        
        tekstur_str = inputs['jenis_tanah']
        if tekstur_str in le_tekstur.classes_:
            tekstur_encoded = le_tekstur.transform([tekstur_str])[0]
        else:
            tekstur_encoded = 0
        tekstur_ohe = to_categorical(tekstur_encoded, num_classes=len(le_tekstur.classes_))
            
        num_features = np.array([
            suhu,
            inputs['hujan_tahunan'],
            hum,
            inputs['ph_tanah'],
            inputs['tanah_liat_persen'],
            inputs['tanah_pasir_persen'],
            inputs['tanah_debu_persen'],
            inputs['elevasi']
        ])
        
        X_input = np.hstack((komoditas_ohe, num_features, tekstur_ohe)).reshape(1, -1)
        
        X_scaled = scaler.transform(X_input)
        preds = model.predict(X_scaled, verbose=0)[0]
        
        score = 0
        for i, prob in enumerate(preds):
            label_name = le_label.classes_[i]
            if label_name == 'Sangat Layak':
                score += prob * 100
            elif label_name == 'Layak':
                score += prob * 60
            elif label_name == 'Kurang Layak':
                score += prob * 20
        
        scores[crop] = float(score)

    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

def generate_explanation(crop, inputs, climate_pred):
    suhu, hum, _ = climate_pred
    ph = inputs['ph_tanah']
    tekstur = inputs['jenis_tanah']
    elevasi = inputs['elevasi']
    hujan = inputs['hujan_tahunan']
    
    reasons = []
    
    # Suhu
    if suhu > 26:
        reasons.append(f"suhu hangat ({suhu:.1f} °C)")
    else:
        reasons.append(f"suhu sejuk ({suhu:.1f} °C)")
        
    # Kelembapan & Hujan
    if hum > 80:
        reasons.append(f"kelembapan udara tinggi ({hum:.1f}%)")
    else:
        reasons.append(f"kelembapan udara ideal ({hum:.1f}%)")
        
    if hujan > 2000:
        reasons.append(f"curah hujan melimpah ({hujan:.0f} mm/tahun)")
    else:
        reasons.append(f"curah hujan moderat ({hujan:.0f} mm/tahun)")
        
    # Tanah
    if 5.5 <= ph <= 7.0:
        reasons.append(f"pH tanah ideal ({ph:.2f})")
    elif ph < 5.5:
        reasons.append(f"toleransi yang baik terhadap tanah masam (pH {ph:.2f})")
    else:
        reasons.append(f"kemampuan beradaptasi di pH {ph:.2f}")
        
    reasons.append(f"tekstur {tekstur} yang mendukung perakaran")
    reasons.append(f"ketinggian lahan {elevasi:.1f} mdpl")
    
    # Menggabungkan kalimat
    explanation = f"{crop} merupakan komoditas yang paling optimal ditanam di {inputs['kecamatan']} karena didukung oleh "
    explanation += ", ".join(reasons[:-1]) + f", serta sesuai dengan {reasons[-1]}."
    
    return explanation

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
    scaled_data = scaler.fit_transform(df_climate.values)
    
    SEQ_LENGTH = 30 
    X, y = create_sequences(scaled_data, SEQ_LENGTH)
    
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    # Build Model
    model = Sequential([
        Input(shape=(SEQ_LENGTH, 3)),
        LSTM(64, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dense(3) 
    ])
    model.compile(optimizer='adam', loss='mse')
    
    print(f"\n[INFO] Melatih model LSTM baru untuk {selected_kec} (Epochs: 20)...")
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    
    # Evaluasi LSTM
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[EVALUASI] Mean Squared Error (MSE) LSTM Iklim pada Test Set: {test_loss:.4f}")
    
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
        'tanah_liat_persen': kec_info['tanah_liat_persen'],
        'tanah_pasir_persen': kec_info['tanah_pasir_persen'],
        'tanah_debu_persen': kec_info['tanah_debu_persen'],
        'elevasi': kec_info['elevasi'],
        'hujan_tahunan': kec_info['hujan_tahunan'],
        'resiko_bencana': kec_info['resiko_bencana']
    }
    
    # Panggil fungsi training bila model rekomendasi belum dilatih
    train_recommendation_model()
    
    recommendations = recommend_crops(avg_pred, user_inputs)
    
    print(f"\n--- Rekomendasi Komoditas untuk {selected_kec} ---")
    for crop, score in recommendations.items():
        print(f"- {crop}: {score:.2f}% Kecocokan")
        
    # Tampilkan tanaman paling cocok dan penjelasannya
    top_crop = list(recommendations.keys())[0]
    top_score = list(recommendations.values())[0]
    
    print(f"\n==================================================")
    print(f"** TANAMAN PALING COCOK: {top_crop.upper()} ({top_score:.2f}%) **")
    print(f"==================================================")
    print("Alasan:")
    print(generate_explanation(top_crop, user_inputs, avg_pred))
    print("==================================================\n")

if __name__ == "__main__":
    main()
