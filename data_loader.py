import pandas as pd
import os

# --- KONFIGURASI PATH ---
DATA_DIR = "data"
FILE_IKLIM = os.path.join(DATA_DIR, "iklim_semua_kecamatan_aceh_utara_2020_2025.csv")
FILE_ELEVASI = os.path.join(DATA_DIR, "Elevasi_Kecamatan_Aceh_Utara.csv")
FILE_PH = os.path.join(DATA_DIR, "pH_Tanah_Kecamatan_Aceh_Utara.csv")
FILE_HUJAN = os.path.join(DATA_DIR, "Curah_Hujan_Tahunan_Kecamatan_Aceh_Utara_2025.csv")

def load_kecamatan_profiles():
    """Memuat data profil tanah dan wilayah untuk setiap kecamatan."""
    print("[INFO] Memuat profil wilayah...")
    
    df_elev = pd.read_csv(FILE_ELEVASI)
    df_ph = pd.read_csv(FILE_PH)
    df_hujan = pd.read_csv(FILE_HUJAN)
    
    # Standardisasi nama kecamatan
    mapping = {
        "Simpang Keramat": "Simpang Keuramat",
        "Geureudong Pase": "Geuredong Pase",
        "Lapang": "Lapangan"
    }
    
    for df in [df_elev, df_ph, df_hujan]:
        df['kecamatan'] = df['kecamatan'].str.strip().replace(mapping)
    
    # Merging profile data
    profiles = df_elev[['kecamatan', 'elevasi_mdpl']].merge(
        df_ph[['kecamatan', 'ph_tanah_mean']], on='kecamatan', how='left'
    ).merge(
        df_hujan[['kecamatan', 'curah_hujan_tahunan']], on='kecamatan', how='left'
    )
    
    # Estimasi fitur tambahan yang tidak ada di CSV
    # Ini bisa disesuaikan jika ada data baru
    profiles['jenis_tanah'] = profiles['elevasi_mdpl'].apply(lambda x: "Aluvial" if x < 50 else "Podsolik")
    profiles['resiko_bencana'] = profiles['elevasi_mdpl'].apply(lambda x: "Tinggi" if x < 15 else "Rendah")
    
    return profiles

def load_climate_data():
    """Memuat data iklim harian historis."""
    print("[INFO] Memuat data iklim harian...")
    df = pd.read_csv(FILE_IKLIM)
    df['date'] = pd.to_datetime(df['date'])
    df['kecamatan'] = df['kecamatan'].str.strip()
    
    # PRECTOT biasanya kosong atau tidak konsisten, kita isi dengan 0 jika NaN
    # atau drop jika tidak diperlukan untuk MVP
    if 'PRECTOT' in df.columns:
        df['PRECTOT'] = df['PRECTOT'].fillna(0)
    
    # Forward fill untuk fitur iklim lainnya jika ada yang bolong
    features = ['T2M', 'RH2M', 'WS2M']
    for f in features:
        df[f] = df.groupby('kecamatan')[f].transform(lambda x: x.fillna(method='ffill'))
        
    return df

def merge_all_data():
    """Menggabungkan data iklim dengan profil kecamatan."""
    df_climate = load_climate_data()
    df_profiles = load_kecamatan_profiles()
    
    print("[INFO] Melakukan merging dataset...")
    merged_df = df_climate.merge(df_profiles, on='kecamatan', how='left')
    
    return merged_df

if __name__ == "__main__":
    # Test loading
    data = merge_all_data()
    print("\n--- Ringkasan Data Gabungan ---")
    print(data.info())
    print("\nContoh 5 Baris Teratas:")
    print(data.head())
    
    # Validasi kecamatan yang mungkin tidak ter-merge
    missing_kec = data[data['elevasi_mdpl'].isna()]['kecamatan'].unique()
    if len(missing_kec) > 0:
        print(f"\n[WARNING] Kecamatan berikut tidak memiliki data profil: {missing_kec}")
    else:
        print("\n[SUCCESS] Semua kecamatan berhasil dipasangkan dengan profilnya.")
