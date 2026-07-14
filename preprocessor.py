import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


# --- KONFIGURASI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "data", "labeled_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
SEQ_LENGTH = 30 # Menggunakan data 30 hari terakhir untuk prediksi

def prepare_data():
    """Membaca data terlabel, mengisi missing values, melakukan scaling, dan pembentukan sequence."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print("[INFO] Membaca data terlabel...")
    df = None

    # 1. Coba ambil dari Supabase
    try:
        import urllib.request
        import json

        supabase_url = "https://hetclnzcfvchqoegdyil.supabase.co"
        supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhldGNsbnpjZnZjaHFvZWdkeWlsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODExNjAxNzcsImV4cCI6MjA5NjczNjE3N30.1oBnHVFQqaMinqaQ5IEF6jxOVh7TisTmT_FPlHbd0VY"
        
        url = f"{supabase_url}/rest/v1/dataset_pelatihan?select=*"
        req = urllib.request.Request(
            url,
            headers={
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}"
            }
        )
        with urllib.request.urlopen(req, timeout=5.0) as response:
            if response.status == 200:
                raw_data = json.loads(response.read().decode('utf-8'))
                if raw_data:
                    flat_data = []
                    for row in raw_data:
                        flat_data.append({
                            "target_commodity": row.get("komoditas", ""),
                            "T2M": float(row.get("suhu_c", 0.0)),
                            "RH2M": float(row.get("kelembapan_persen", 0.0)),
                            "WS2M": 1.5,  # Default kecepatan angin
                            "curah_hujan_tahunan": float(row.get("curah_hujan_mm_tahun", 0.0)),
                            "kecamatan": row.get("kecamatan", "Kustom"),
                            "elevasi_mdpl": float(row.get("elevasi_mdpl", 0.0)),
                            "ph_tanah_mean": float(row.get("ph_tanah", 7.0)),
                            "date": row.get("created_at", "2020-01-01")[:10],
                        })
                    df = pd.DataFrame(flat_data)
                    print(f"[SUCCESS] Berhasil memuat {len(df)} records data pelatihan dari Supabase.")
    except Exception as e:
        print(f"[WARNING] Gagal memuat data pelatihan dari Supabase: {str(e)}")

    # 2. Fallback CSV Lokal
    if df is None:
        print(f"[INFO] Menggunakan fallback file CSV lokal di {DATA_PATH}...")
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"File data tidak ditemukan di: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    
    # 0. Normalisasi kecamatan & Isi missing values (NaNs) akibat spelling mismatch di CSV asli
    from data_loader import load_kecamatan_profiles, normalize_kecamatan_name
    df['kecamatan'] = df['kecamatan'].apply(normalize_kecamatan_name)
    
    profiles = load_kecamatan_profiles()
    mapper_elev = dict(zip(profiles['kecamatan'], profiles['elevasi_mdpl']))
    mapper_hujan = dict(zip(profiles['kecamatan'], profiles['curah_hujan_tahunan']))
    mapper_ph = dict(zip(profiles['kecamatan'], profiles['ph_tanah']))
    
    df['elevasi_mdpl'] = df['elevasi_mdpl'].fillna(df['kecamatan'].map(mapper_elev))
    df['curah_hujan_tahunan'] = df['curah_hujan_tahunan'].fillna(df['kecamatan'].map(mapper_hujan))
    df['ph_tanah_mean'] = df['ph_tanah_mean'].fillna(df['kecamatan'].map(mapper_ph))
    
    # Cadangan: ffill jika ada data cuaca harian yang bolong
    for col in ['T2M', 'RH2M', 'WS2M']:
        df[col] = df[col].ffill()
        
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


# ============================================================
#  PREPARE DATA V2 — Dual Input: Sequential + Static Features
# ============================================================

SEQ_FEATURES  = ['T2M', 'RH2M', 'WS2M']           # 3 fitur iklim dinamis (per hari)
STATIC_FEATURES = [
    'elevasi_mdpl',
    'curah_hujan_tahunan',
    'ph_tanah_mean',
    'tanah_liat',
    'tanah_pasir',
    'tanah_debu'
]  # 6 fitur lahan statis (per kecamatan)

def prepare_data_v2():
    """
    Dual-input preprocessing untuk model LSTM v2.
    Mengembalikan:
      X_seq    — (N, SEQ_LENGTH, 3)  fitur iklim harian
      X_static — (N, 6)              fitur lahan statis
      y        — (N,)                label integer
      classes  — daftar nama kelas
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print("[INFO] [V2] Membaca data terlabel...")
    df = None

    # 1. Coba ambil dari Supabase
    try:
        import urllib.request
        import json

        supabase_url = "https://hetclnzcfvchqoegdyil.supabase.co"
        supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhldGNsbnpjZnZjaHFvZWdkeWlsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODExNjAxNzcsImV4cCI6MjA5NjczNjE3N30.1oBnHVFQqaMinqaQ5IEF6jxOVh7TisTmT_FPlHbd0VY"

        url = f"{supabase_url}/rest/v1/dataset_pelatihan?select=*"
        req = urllib.request.Request(
            url,
            headers={
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}"
            }
        )
        with urllib.request.urlopen(req, timeout=5.0) as response:
            if response.status == 200:
                raw_data = json.loads(response.read().decode('utf-8'))
                if raw_data:
                    flat_data = []
                    for row in raw_data:
                        flat_data.append({
                            "target_commodity": row.get("komoditas", ""),
                            "T2M": float(row.get("suhu_c", 0.0)),
                            "RH2M": float(row.get("kelembapan_persen", 0.0)),
                            "WS2M": 1.5,
                            "curah_hujan_tahunan": float(row.get("curah_hujan_mm_tahun", 0.0)),
                            "kecamatan": row.get("kecamatan", "Kustom"),
                            "elevasi_mdpl": float(row.get("elevasi_mdpl", 0.0)),
                            "ph_tanah_mean": float(row.get("ph_tanah", 7.0)),
                            "date": row.get("created_at", "2020-01-01")[:10],
                        })
                    df = pd.DataFrame(flat_data)
                    print(f"[SUCCESS] Berhasil memuat {len(df)} records dari Supabase.")
    except Exception as e:
        print(f"[WARNING] Gagal memuat dari Supabase: {str(e)}")

    # 2. Fallback CSV Lokal
    if df is None:
        print(f"[INFO] Menggunakan fallback CSV lokal di {DATA_PATH}...")
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"File data tidak ditemukan di: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)

    # 3. Normalisasi kecamatan & isi missing values dari profil wilayah
    from data_loader import load_kecamatan_profiles, normalize_kecamatan_name
    df['kecamatan'] = df['kecamatan'].apply(normalize_kecamatan_name)

    profiles = load_kecamatan_profiles()
    mapper_elev   = dict(zip(profiles['kecamatan'], profiles['elevasi_mdpl']))
    mapper_hujan  = dict(zip(profiles['kecamatan'], profiles['curah_hujan_tahunan']))
    mapper_ph     = dict(zip(profiles['kecamatan'], profiles['ph_tanah']))
    mapper_liat   = dict(zip(profiles['kecamatan'], profiles['tanah_liat']))
    mapper_pasir  = dict(zip(profiles['kecamatan'], profiles['tanah_pasir']))
    mapper_debu   = dict(zip(profiles['kecamatan'], profiles['tanah_debu']))

    df['elevasi_mdpl']       = df['elevasi_mdpl'].fillna(df['kecamatan'].map(mapper_elev))
    df['curah_hujan_tahunan']= df['curah_hujan_tahunan'].fillna(df['kecamatan'].map(mapper_hujan))
    df['ph_tanah_mean']      = df['ph_tanah_mean'].fillna(df['kecamatan'].map(mapper_ph))

    # Kolom tanah tekstur: dari CSV labeled_data atau dari profil
    for col, mapper in [('tanah_liat', mapper_liat), ('tanah_pasir', mapper_pasir), ('tanah_debu', mapper_debu)]:
        if col not in df.columns:
            df[col] = df['kecamatan'].map(mapper)
        else:
            df[col] = df[col].fillna(df['kecamatan'].map(mapper))

    # Forward fill fitur iklim harian
    for col in ['T2M', 'RH2M', 'WS2M']:
        df[col] = df[col].ffill()

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['kecamatan', 'date'])

    # 4. Encoding Label
    print("[INFO] [V2] Melakukan Encoding Label...")
    le = LabelEncoder()
    df['target_encoded'] = le.fit_transform(df['target_commodity'])
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    # 5. Scaling Terpisah
    print("[INFO] [V2] Melakukan Scaling Fitur (sequential & static terpisah)...")

    scaler_seq = MinMaxScaler()
    df[SEQ_FEATURES] = scaler_seq.fit_transform(df[SEQ_FEATURES])
    joblib.dump(scaler_seq, os.path.join(MODEL_DIR, "scaler_seq.pkl"))

    scaler_static = MinMaxScaler()
    df[STATIC_FEATURES] = scaler_static.fit_transform(df[STATIC_FEATURES])
    joblib.dump(scaler_static, os.path.join(MODEL_DIR, "scaler_static.pkl"))

    # 6. Pembuatan Sequences per kecamatan
    X_seq_list, X_static_list, y_list = [], [], []

    print("[INFO] [V2] Membentuk sequences per kecamatan...")
    for kec in df['kecamatan'].unique():
        df_kec = df[df['kecamatan'] == kec]
        seq_vals    = df_kec[SEQ_FEATURES].values
        static_vals = df_kec[STATIC_FEATURES].values   # statis, nilai sama per baris kecamatan
        targets     = df_kec['target_encoded'].values

        for i in range(len(seq_vals) - SEQ_LENGTH):
            X_seq_list.append(seq_vals[i : i + SEQ_LENGTH])
            # Ambil nilai statis pada hari terakhir window (konstan per kecamatan)
            X_static_list.append(static_vals[i + SEQ_LENGTH])
            y_list.append(targets[i + SEQ_LENGTH])

    X_seq    = np.array(X_seq_list,    dtype=np.float32)
    X_static = np.array(X_static_list, dtype=np.float32)
    y        = np.array(y_list)

    print(f"[SUCCESS] [V2] Data siap:")
    print(f"          X_seq    shape: {X_seq.shape}")
    print(f"          X_static shape: {X_static.shape}")
    print(f"          y        shape: {y.shape}")
    return X_seq, X_static, y, le.classes_


if __name__ == "__main__":
    X, y, classes = prepare_data()
    print(f"Daftar Kelas: {classes}")
    print(f"Contoh X[0][0]: {X[0][0]}")
    print(f"Contoh y[0]: {y[0]}")
    print("Jumlah NaNs di X:", np.isnan(X).sum())
