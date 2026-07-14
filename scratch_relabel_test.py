import pandas as pd
import numpy as np
import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "labeled_data.csv")

df = pd.read_csv(DATA_PATH)

from data_loader import load_kecamatan_profiles, normalize_kecamatan_name
df['kecamatan'] = df['kecamatan'].apply(normalize_kecamatan_name)
profiles = load_kecamatan_profiles()

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

crops_parameters = {
    "Padi": {
        "suhu_optimal": (25.0, 30.0),
        "kelembapan_optimal": (60.0, 90.0),
        "elevasi_optimal": (0.0, 600.0),
        "toleransi_liat": (30.0, 50.0),
        "toleransi_pasir": (20.0, 40.0),
        "toleransi_debu": (20.0, 40.0),
        "ph_optimal": (5.5, 7.5)
    },
    "Jagung": {
        "suhu_optimal": (21.0, 30.0),
        "kelembapan_optimal": (50.0, 80.0),
        "elevasi_optimal": (0.0, 1000.0),
        "toleransi_liat": (20.0, 40.0),
        "toleransi_pasir": (30.0, 50.0),
        "toleransi_debu": (20.0, 40.0),
        "ph_optimal": (5.5, 7.0)
    },
    "Kedelai": {
        "suhu_optimal": (25.0, 30.0),
        "kelembapan_optimal": (50.0, 80.0),
        "elevasi_optimal": (0.0, 900.0),
        "toleransi_liat": (20.0, 40.0),
        "toleransi_pasir": (30.0, 50.0),
        "toleransi_debu": (20.0, 40.0),
        "ph_optimal": (5.8, 7.0)
    },
    "Kacang Tanah": {
        "suhu_optimal": (25.0, 30.0),
        "kelembapan_optimal": (50.0, 70.0),
        "elevasi_optimal": (0.0, 500.0),
        "toleransi_liat": (10.0, 30.0),
        "toleransi_pasir": (40.0, 60.0),
        "toleransi_debu": (20.0, 30.0),
        "ph_optimal": (5.5, 7.0)
    },
    "Kacang Hijau": {
        "suhu_optimal": (25.0, 35.0),
        "kelembapan_optimal": (50.0, 80.0),
        "elevasi_optimal": (0.0, 500.0),
        "toleransi_liat": (10.0, 30.0),
        "toleransi_pasir": (40.0, 60.0),
        "toleransi_debu": (20.0, 40.0),
        "ph_optimal": (5.5, 6.5)
    },
    "Ubi Kayu": {
        "suhu_optimal": (25.0, 30.0),
        "kelembapan_optimal": (60.0, 85.0),
        "elevasi_optimal": (0.0, 800.0),
        "toleransi_liat": (10.0, 30.0),
        "toleransi_pasir": (40.0, 60.0),
        "toleransi_debu": (20.0, 40.0),
        "ph_optimal": (5.5, 7.0)
    },
    "Ubi Jalar": {
        "suhu_optimal": (21.0, 27.0),
        "kelembapan_optimal": (50.0, 80.0),
        "elevasi_optimal": (0.0, 1000.0),
        "toleransi_liat": (20.0, 40.0),
        "toleransi_pasir": (30.0, 50.0),
        "toleransi_debu": (20.0, 40.0),
        "ph_optimal": (5.5, 6.8)
    }
}

def calculate_score(val, optimal_range, is_rainfall=False):
    min_opt, max_opt = optimal_range
    if min_opt <= val <= max_opt:
        return 100.0

    diff = min(abs(val - min_opt), abs(val - max_opt))
    if is_rainfall:
        score = 100.0 - (diff * 0.5)
    else:
        if max_opt <= 14.0:
            score = 100.0 - (diff * 20.0)
        elif max_opt <= 100.0:
            score = 100.0 - (diff * 2.0)
        else:
            score = 100.0 - diff

    return max(0.0, score)

def label_row_by_suitability_v2(row):
    kec = row['kecamatan']
    prof = profiles_dict.get(kec, {})
    
    elev = row['elevasi_mdpl']
    if pd.isna(elev):
        elev = prof.get('elevasi_mdpl', 10.0)
        
    t2m = row['T2M']
    if pd.isna(t2m):
        t2m = 25.0
        
    rh2m = row['RH2M']
    if pd.isna(rh2m):
        rh2m = 70.0
        
    # Zoning
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
        suhu_s = calculate_score(t2m, kb["suhu_optimal"])
        kelembapan_s = calculate_score(rh2m, kb["kelembapan_optimal"])
        scores[crop] = (suhu_s + kelembapan_s) / 2.0
        
    # Sort candidates by score descending
    sorted_crops = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Tie-breaking logic if score difference is very small (< 4.5 points)
    if len(sorted_crops) > 1:
        best_crop, best_score = sorted_crops[0]
        second_crop, second_score = sorted_crops[1]
        
        if (best_score - second_score) < 4.5:
            # Saling berkompetisi dekat, lakukan tie-break berbasis aturan fisik/suhu yang lebih khas
            if elev > 350.0:
                # Ubi Jalar vs Ubi Kayu
                # Ubi Jalar lebih suka dingin (< 24.5), Ubi Kayu lebih hangat
                if t2m < 24.5:
                    return "Ubi Jalar"
                else:
                    return "Ubi Kayu"
            elif elev > 120.0:
                # Kedelai vs Jagung
                # Kedelai lebih dingin (< 25.5), Jagung lebih hangat
                if t2m < 25.5:
                    return "Kedelai"
                else:
                    return "Jagung"
            elif elev > 40.0:
                # Kacang Tanah vs Jagung
                # Kacang Tanah lebih dingin (< 25.0), Jagung lebih hangat
                if t2m < 25.0:
                    return "Kacang Tanah"
                else:
                    return "Jagung"
            else:
                # Kacang Hijau vs Padi vs Kacang Tanah
                # Kacang Hijau (suhu > 26.2), Padi (25.2 < suhu <= 26.2), Kacang Tanah (suhu <= 25.2)
                if t2m > 26.2:
                    return "Kacang Hijau"
                elif t2m > 25.2:
                    return "Padi"
                else:
                    return "Kacang Tanah"
                    
    return sorted_crops[0][0]

df['target_commodity'] = df.apply(label_row_by_suitability_v2, axis=1)
print(df['target_commodity'].value_counts())
