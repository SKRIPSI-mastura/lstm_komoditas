import pandas as pd
import os

# --- KONFIGURASI PATH ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FILE_IKLIM = os.path.join(DATA_DIR, "iklim_semua_kecamatan_aceh_utara_2020_2025.csv")
FILE_ELEVASI = os.path.join(DATA_DIR, "Elevasi_Kecamatan_Aceh_Utara.csv")
FILE_TANAH = os.path.join(DATA_DIR, "data_tanah_aceh_utara2.csv")
FILE_HUJAN = os.path.join(DATA_DIR, "data_curah_hujan_aceh_utara.csv")

def normalize_kecamatan_name(name):
    """Normalisasi nama kecamatan ke standar database."""
    n = name.strip()
    n_lower = n.lower()
    if n_lower in ["pirak timu", "pirak timur"]:
        return "Pirak Timur"
    if n_lower in ["simpang keramat", "simpang kramat", "simpang keuramat"]:
        return "Simpang Kramat"
    if n_lower in ["geureudong pase", "geuredong pase"]:
        return "Geuredong Pase"
    if n_lower in ["lapang", "lapangan"]:
        return "Lapang"
    return n

def load_kecamatan_profiles():
    """
    Memuat data profil tanah dan wilayah untuk setiap kecamatan.
    Kolom tanah yang digunakan: ph_tanah, tanah_liat, tanah_pasir, tanah_debu.
    Kolom tekstur_tanah tidak digunakan karena sudah direpresentasikan oleh
    komposisi tanah_liat, tanah_pasir, dan tanah_debu.
    """
    print("[INFO] Memuat profil wilayah...")

    df_elev = pd.read_csv(FILE_ELEVASI)
    df_tanah = pd.read_csv(FILE_TANAH)
    df_hujan = pd.read_csv(FILE_HUJAN)

    for df in [df_elev, df_tanah, df_hujan]:
        df['kecamatan'] = df['kecamatan'].apply(normalize_kecamatan_name)
    
    # Hitung rata-rata curah hujan tahunan dari data harian
    df_hujan['date'] = pd.to_datetime(df_hujan['date'].astype(str))
    df_hujan['year'] = df_hujan['date'].dt.year
    hujan_tahunan = (
        df_hujan.groupby(['kecamatan', 'year'])['curah_hujan']
        .sum()
        .groupby('kecamatan')
        .mean()
        .reset_index()
    )
    hujan_tahunan.rename(columns={'curah_hujan': 'curah_hujan_tahunan'}, inplace=True)

    # Merging profile data
    # Kolom tanah yang digunakan: ph_tanah, tanah_liat, tanah_pasir, tanah_debu
    profiles = df_elev[['kecamatan', 'elevasi_mdpl']].merge(
        df_tanah[['kecamatan', 'ph_tanah', 'tanah_liat', 'tanah_pasir', 'tanah_debu']],
        on='kecamatan', how='left'
    ).merge(
        hujan_tahunan[['kecamatan', 'curah_hujan_tahunan']], on='kecamatan', how='left'
    )

    # Estimasi risiko bencana berdasarkan elevasi
    profiles['resiko_bencana'] = profiles['elevasi_mdpl'].apply(
        lambda x: "Tinggi" if x < 15 else "Rendah"
    )

    return profiles


def load_climate_data():
    """Memuat data iklim harian historis."""
    print("[INFO] Memuat data iklim harian...")
    df = pd.read_csv(FILE_IKLIM)
    df['date'] = pd.to_datetime(df['date'])
    df['kecamatan'] = df['kecamatan'].apply(normalize_kecamatan_name)

    # Rename kolom agar konsisten
    rename_dict = {
        'Suhu rata-rata': 'T2M',
        'Kelembapan udara': 'RH2M',
        'Kecepatan angin': 'WS2M'
    }
    df.rename(columns=rename_dict, inplace=True)

    # Forward fill untuk fitur iklim jika ada yang kosong
    features = ['T2M', 'RH2M', 'WS2M']
    for f in features:
        df[f] = df.groupby('kecamatan')[f].transform(lambda x: x.ffill())

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
