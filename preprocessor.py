import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# --- KONFIGURASI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "labeled_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
SEQ_LENGTH = 30 # Menggunakan data 30 hari terakhir untuk prediksi

def prepare_data():
    """Membaca data terlabel, melakukan scaling, dan pembentukan sequence."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print(f"[INFO] Membaca data terlabel dari {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File data tidak ditemukan di: {DATA_PATH}")
        
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['kecamatan', 'date'])
    
    # 1. Encoding Label (Komoditas)
    print("[INFO] Melakukan Encoding Label...")
    le = LabelEncoder()
    df['target_encoded'] = le.fit_transform(df['target_commodity'])
    
    # Simpan LabelEncoder
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    
    # 2. Pemilihan Fitur
    # Fitur dinamis (iklim) + Fitur statis (wilayah)
    feature_cols = ['T2M', 'RH2M', 'WS2M', 'elevasi_mdpl', 'ph_tanah_mean', 'curah_hujan_tahunan']
    
    # 3. Scaling Fitur
    print("[INFO] Melakukan Scaling Fitur...")
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Simpan Scaler
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    
    # 4. Pembuatan Sequences (Windowing)
    # Kita harus melakukan ini per kecamatan agar tidak tercampur di batas antar kecamatan
    X, y = [], []
    
    print("[INFO] Membentuk sequences per kecamatan...")
    for kec in df['kecamatan'].unique():
        df_kec = df[df['kecamatan'] == kec]
        values = df_kec[feature_cols].values
        targets = df_kec['target_encoded'].values
        
        for i in range(len(values) - SEQ_LENGTH):
            X.append(values[i : i + SEQ_LENGTH])
            # Target adalah komoditas pada hari terakhir di window tersebut
            y.append(targets[i + SEQ_LENGTH])
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"[SUCCESS] Data siap: X shape {X.shape}, y shape {y.shape}")
    return X, y, le.classes_

if __name__ == "__main__":
    X, y, classes = prepare_data()
    print(f"Daftar Kelas: {classes}")
    print(f"Contoh X[0][0]: {X[0][0]}")
    print(f"Contoh y[0]: {y[0]}")
