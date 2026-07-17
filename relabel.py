import pandas as pd
import numpy as np
import os
import urllib.request
import json

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "labeled_data.csv")

print(f"[INFO] Memuat dataset dari {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# Mengisi missing values untuk sementara agar penentuan label akurat
from data_loader import load_kecamatan_profiles, normalize_kecamatan_name
df['kecamatan'] = df['kecamatan'].apply(normalize_kecamatan_name)
profiles = load_kecamatan_profiles()

# Simpan profiles dalam dict agar lookup cepat
profiles_dict = {}
for _, row in profiles.iterrows():
    profiles_dict[row['kecamatan']] = {
        'elevasi_mdpl': row.get('elevasi_mdpl', 10.0),
        'ph_tanah': row.get('ph_tanah', 5.5),
        'tanah_liat': row.get('tanah_liat', 30.0),
        'tanah_pasir': row.get('tanah_pasir', 30.0),
        'tanah_debu': row.get('tanah_debu', 30.0),
        'curah_hujan_tahunan': row.get('curah_hujan_tahunan', 2000.0)
    }

# 1. Mengambil parameter komoditas dari Supabase (via Next.js API) atau fallback lokal
def get_komoditas_parameters():
    fallback = {
        "Padi": {
            "suhu_optimal": (25.0, 30.0),
            "ph_optimal": (5.5, 7.5),
            "elevasi_optimal": (0.0, 600.0),
            "kelembapan_optimal": (60.0, 90.0),
            "toleransi_liat": (30.0, 50.0),
            "toleransi_pasir": (20.0, 40.0),
            "toleransi_debu": (20.0, 40.0)
        },
        "Jagung": {
            "suhu_optimal": (21.0, 30.0),
            "ph_optimal": (5.5, 7.0),
            "elevasi_optimal": (0.0, 1000.0),
            "kelembapan_optimal": (50.0, 80.0),
            "toleransi_liat": (20.0, 40.0),
            "toleransi_pasir": (30.0, 50.0),
            "toleransi_debu": (20.0, 40.0)
        },
        "Kedelai": {
            "suhu_optimal": (25.0, 30.0),
            "ph_optimal": (5.8, 7.0),
            "elevasi_optimal": (0.0, 900.0),
            "kelembapan_optimal": (50.0, 80.0),
            "toleransi_liat": (20.0, 40.0),
            "toleransi_pasir": (30.0, 50.0),
            "toleransi_debu": (20.0, 40.0)
        },
        "Kacang Tanah": {
            "suhu_optimal": (25.0, 30.0),
            "ph_optimal": (5.5, 7.0),
            "elevasi_optimal": (0.0, 500.0),
            "kelembapan_optimal": (50.0, 70.0),
            "toleransi_liat": (10.0, 30.0),
            "toleransi_pasir": (40.0, 60.0),
            "toleransi_debu": (20.0, 30.0)
        },
        "Kacang Hijau": {
            "suhu_optimal": (25.0, 35.0),
            "ph_optimal": (5.5, 6.5),
            "elevasi_optimal": (0.0, 500.0),
            "kelembapan_optimal": (50.0, 80.0),
            "toleransi_liat": (10.0, 30.0),
            "toleransi_pasir": (40.0, 60.0),
            "toleransi_debu": (20.0, 40.0)
        },
        "Ubi Kayu": {
            "suhu_optimal": (25.0, 30.0),
            "ph_optimal": (5.5, 7.0),
            "elevasi_optimal": (0.0, 800.0),
            "kelembapan_optimal": (60.0, 85.0),
            "toleransi_liat": (10.0, 30.0),
            "toleransi_pasir": (40.0, 60.0),
            "toleransi_debu": (20.0, 40.0)
        },
        "Ubi Jalar": {
            "suhu_optimal": (21.0, 27.0),
            "ph_optimal": (5.5, 6.8),
            "elevasi_optimal": (0.0, 1000.0),
            "kelembapan_optimal": (50.0, 80.0),
            "toleransi_liat": (20.0, 40.0),
            "toleransi_pasir": (30.0, 50.0),
            "toleransi_debu": (20.0, 40.0)
        }
    }
    
    try:
        req = urllib.request.Request("http://localhost:3000/api/komoditas", headers={'User-Agent': 'FastAPI-Backend'})
        with urllib.request.urlopen(req, timeout=2.0) as response:
            if response.status == 200:
                res_data = json.loads(response.read().decode('utf-8'))
                if res_data.get("status") == "success" and res_data.get("data"):
                    db_data = res_data["data"]
                    print("[INFO] Berhasil mengambil parameters dari Supabase via Next.js API.")
                    
                    crops_data = {}
                    for item in db_data:
                        name = item["nama_komoditas"]
                        soil_tols = fallback.get(name, {
                            "toleransi_liat": (20.0, 40.0),
                            "toleransi_pasir": (30.0, 50.0),
                            "toleransi_debu": (20.0, 40.0)
                        })
                        crops_data[name] = {
                            "suhu_optimal": (float(item["suhu_min_c"]), float(item["suhu_max_c"])),
                            "ph_optimal": (float(item["ph_min"]), float(item["ph_max"])),
                            "elevasi_optimal": (float(item["elevasi_min_mdpl"]), float(item["elevasi_max_mdpl"])),
                            "kelembapan_optimal": (float(item["kelembapan_min_persen"]), float(item["kelembapan_max_persen"])),
                            "toleransi_liat": soil_tols["toleransi_liat"],
                            "toleransi_pasir": soil_tols["toleransi_pasir"],
                            "toleransi_debu": soil_tols["toleransi_debu"]
                        }
                    return crops_data
    except Exception as e:
        print(f"[WARNING] Gagal mengambil dari Next.js API: {str(e)}. Menggunakan fallback data parameter lokal.")
    return fallback

crops_parameters = get_komoditas_parameters()

# 2. Fungsi hitung skor kesesuaian parameter
def calculate_score(val, optimal_range, is_rainfall=False):
    min_opt, max_opt = optimal_range
    if min_opt <= val <= max_opt:
        return 100.0

    diff = min(abs(val - min_opt), abs(val - max_opt))
    if is_rainfall:
        score = 100.0 - (diff * 0.5)
    else:
        if max_opt <= 14.0:      # pH range
            score = 100.0 - (diff * 20.0)
        elif max_opt <= 100.0:   # Persentase tanah
            score = 100.0 - (diff * 2.0)
        else:
            score = 100.0 - diff

    return max(0.0, score)

# 3. Fungsi pelabelan baris berbasis evaluasi iklim harian + pembatasan zona elevasi + tie-breaking
def label_row_by_suitability(row):
    kec = row['kecamatan']
    prof = profiles_dict.get(kec, {})
    
    elev = row['elevasi_mdpl']
    if pd.isna(elev):
        elev = prof.get('elevasi_mdpl', 10.0)
        
    t2m = row['T2M_roll']
    if pd.isna(t2m):
        t2m = 25.0
        
    rh2m = row['RH2M_roll']
    if pd.isna(rh2m):
        rh2m = 70.0
        
    # Tentukan kandidat berdasarkan zona elevasi (zoning agronomis)
    if elev > 350.0:
        candidates = ["Ubi Jalar", "Ubi Kayu"]
    elif elev > 120.0:
        candidates = ["Kedelai", "Jagung"]
    elif elev > 40.0:
        candidates = ["Kacang Tanah", "Jagung"]
    else:
        candidates = ["Kacang Hijau", "Padi", "Kacang Tanah"]
        
    scores = {}
    for crop in candidates:
        kb = crops_parameters[crop]
        # Skor Iklim berbasis rolling average (Suhu & Kelembapan)
        suhu_s = calculate_score(t2m, kb["suhu_optimal"])
        kelembapan_s = calculate_score(rh2m, kb["kelembapan_optimal"])
        scores[crop] = (suhu_s + kelembapan_s) / 2.0
        
    # Urutkan berdasarkan skor tertinggi
    sorted_crops = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_crops) > 1:
        best_crop, best_score = sorted_crops[0]
        second_crop, second_score = sorted_crops[1]
        
        # Tie-breaking jika selisih skor sangat tipis (< 4.5 poin)
        if (best_score - second_score) < 4.5:
            if elev > 350.0:
                # Ubi Jalar vs Ubi Kayu
                if t2m < 24.5:
                    return "Ubi Jalar"
                else:
                    return "Ubi Kayu"
            elif elev > 120.0:
                # Kedelai vs Jagung
                if t2m < 25.5:
                    return "Kedelai"
                else:
                    return "Jagung"
            elif elev > 40.0:
                # Kacang Tanah vs Jagung
                if t2m < 25.0:
                    return "Kacang Tanah"
                else:
                    return "Jagung"
            else:
                # Kacang Hijau vs Padi vs Kacang Tanah
                if t2m > 26.2:
                    return "Kacang Hijau"
                elif t2m > 25.2:
                    return "Padi"
                else:
                    return "Kacang Tanah"
                    
    return sorted_crops[0][0]

# Pastikan data terurut berdasarkan kecamatan dan tanggal agar rolling average tepat
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['kecamatan', 'date'])

print("[INFO] Menghitung rolling average 30 hari untuk T2M dan RH2M per kecamatan...")
df['T2M_roll'] = df.groupby('kecamatan')['T2M'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
df['RH2M_roll'] = df.groupby('kecamatan')['RH2M'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())

print("[INFO] Melakukan relabeling kolom target_commodity secara agronomis (hybrid iklim 30-hari rolling + tie-breaking)...")
df['target_commodity'] = df.apply(label_row_by_suitability, axis=1)

# Bersihkan kolom temporary
df = df.drop(columns=['T2M_roll', 'RH2M_roll'])

print("\n--- Distribusi Kelas Baru ---")
dist = df['target_commodity'].value_counts()
print(dist)

# Simpan kembali ke file yang sama
print(f"\n[INFO] Menyimpan data terlabel baru ke {DATA_PATH}...")
df.to_csv(DATA_PATH, index=False)
print("[SUCCESS] Relabeling dataset selesai!")
