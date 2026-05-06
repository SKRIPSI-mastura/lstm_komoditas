import pandas as pd
import numpy as np
from data_loader import merge_all_data

def calculate_suitability(row):
    """
    Menghitung skor kesesuaian untuk setiap komoditas berdasarkan baris data.
    Logika diadaptasi dari kriteria agronomis di main.py.
    """
    suhu = row['T2M']
    ph = row['ph_tanah_mean']
    elev = row['elevasi_mdpl']
    hujan_tahunan = row['curah_hujan_tahunan']
    tanah = row['jenis_tanah']
    risk = row['resiko_bencana']
    
    crops = {
        "Padi": 0, "Jagung": 0, "Kedelai": 0, 
        "Kacang Hijau": 0, "Kacang Tanah": 0, 
        "Ubi Jalar": 0, "Ubi Kayu": 0
    }
    
    for crop in crops:
        score = 0
        
        # 1. Kecocokan Suhu (Ideal 24-32 C)
        if 24 <= suhu <= 32: score += 20
        elif 20 <= suhu < 24 or 32 < suhu <= 35: score += 10
        
        # 2. Kecocokan pH Tanah (Ideal 5.5-7.0)
        if 5.5 <= ph <= 7.0: score += 20
        elif 5.0 <= ph < 5.5 or 7.0 < ph <= 7.5: score += 10
        
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
            
        # 5. Faktor Curah Hujan
        if crop == "Padi":
            if hujan_tahunan > 2000: score += 20
        elif crop in ["Jagung", "Kedelai"]:
            if 1800 < hujan_tahunan < 2200: score += 15
        elif crop in ["Kacang Tanah", "Ubi Kayu"]:
            if hujan_tahunan < 2000: score += 15
            
        # 6. Faktor Resiko Bencana
        if risk == 'Tinggi':
            if crop == "Padi": score += 10 # Padi lebih tahan genangan dibanding jagung
            elif crop == "Jagung": score -= 10
        else:
            score += 10
            
        crops[crop] = score

    # Mengembalikan komoditas dengan skor tertinggi
    best_crop = max(crops, key=crops.get)
    return best_crop

def generate_labeled_dataset():
    """Memproses data gabungan dan menambahkan label target."""
    df = merge_all_data()
    
    print("[INFO] Melakukan pelabelan data (ini mungkin memakan waktu)...")
    # Menggunakan apply untuk menentukan label terbaik tiap baris
    df['target_commodity'] = df.apply(calculate_suitability, axis=1)
    
    # Simpan dataset terlabel
    output_path = "data/labeled_data.csv"
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Dataset terlabel disimpan di: {output_path}")
    
    # Print distribusi label
    print("\n--- Distribusi Label Komoditas ---")
    print(df['target_commodity'].value_counts())
    
    return df

if __name__ == "__main__":
    generate_labeled_dataset()
