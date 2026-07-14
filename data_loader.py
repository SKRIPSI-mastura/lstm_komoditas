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
    Mencoba mengambil data terbaru dari database Supabase terlebih dahulu.
    Jika gagal/offline, menggunakan data dari berkas CSV lokal sebagai fallback.
    """
    print("[INFO] Memuat profil wilayah...")

    # 1. Coba ambil dari Supabase
    try:
        import urllib.request
        import json

        supabase_url = "https://hetclnzcfvchqoegdyil.supabase.co"
        supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhldGNsbnpjZnZjaHFvZWdkeWlsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODExNjAxNzcsImV4cCI6MjA5NjczNjE3N30.1oBnHVFQqaMinqaQ5IEF6jxOVh7TisTmT_FPlHbd0VY"
        
        url = f"{supabase_url}/rest/v1/kecamatan?select=*"
        req = urllib.request.Request(
            url,
            headers={
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}"
            }
        )
        with urllib.request.urlopen(req, timeout=4.0) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                if data:
                    df_profiles = pd.DataFrame(data)
                    df_profiles.rename(columns={"nama_kecamatan": "kecamatan"}, inplace=True)
                    df_profiles["kecamatan"] = df_profiles["kecamatan"].apply(normalize_kecamatan_name)
                    
                    # Konversi tipe data ke numeric
                    numeric_cols = ["elevasi_mdpl", "ph_tanah", "tanah_liat", "tanah_pasir", "tanah_debu", "curah_hujan_tahunan"]
                    for col in numeric_cols:
                        if col in df_profiles.columns:
                            df_profiles[col] = pd.to_numeric(df_profiles[col], errors='coerce')
                    
                    df_profiles["curah_hujan_tahunan"] = df_profiles["curah_hujan_tahunan"].fillna(2000.0)
                    
                    if "resiko_bencana" not in df_profiles.columns:
                        df_profiles["resiko_bencana"] = df_profiles["elevasi_mdpl"].apply(
                            lambda x: "Tinggi" if x < 15 else "Rendah"
                        )
                    print(f"[SUCCESS] Berhasil memuat {len(df_profiles)} profil kecamatan dari Supabase.")
                    return df_profiles
    except Exception as e:
        print(f"[WARNING] Gagal memuat profil kecamatan dari Supabase: {str(e)}")
        print("[INFO] Menggunakan fallback file CSV lokal...")

    # 2. Fallback CSV Lokal
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
    profiles = df_elev[['kecamatan', 'elevasi_mdpl']].merge(
        df_tanah[['kecamatan', 'ph_tanah', 'tanah_liat', 'tanah_pasir', 'tanah_debu']],
        on='kecamatan', how='left'
    ).merge(
        hujan_tahunan[['kecamatan', 'curah_hujan_tahunan']], on='kecamatan', how='left'
    )

    profiles['resiko_bencana'] = profiles['elevasi_mdpl'].apply(
        lambda x: "Tinggi" if x < 15 else "Rendah"
    )

    return profiles


def load_climate_data():
    """
    Memuat data iklim harian historis.
    Mencoba mengambil dari Supabase terlebih dahulu.
    Jika gagal/kosong, menggunakan data dari berkas CSV lokal.
    """
    print("[INFO] Memuat data iklim harian...")

    # 1. Coba ambil dari Supabase
    try:
        import urllib.request
        import json

        supabase_url = "https://hetclnzcfvchqoegdyil.supabase.co"
        supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhldGNsbnpjZnZjaHFvZWdkeWlsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODExNjAxNzcsImV4cCI6MjA5NjczNjE3N30.1oBnHVFQqaMinqaQ5IEF6jxOVh7TisTmT_FPlHbd0VY"
        
        # Query join menggunakan PostgREST
        url = f"{supabase_url}/rest/v1/data_iklim_historis?select=*,kecamatan:kecamatan_id(nama_kecamatan)"
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
                            "date": row.get("tanggal", ""),
                            "T2M": float(row.get("suhu_c", 0.0)),
                            "RH2M": float(row.get("kelembapan_persen", 0.0)),
                            "WS2M": float(row.get("kecepatan_angin_ms", 0.0)),
                            "kecamatan": row.get("kecamatan", {}).get("nama_kecamatan", "")
                        })
                    df = pd.DataFrame(flat_data)
                    df['date'] = pd.to_datetime(df['date'])
                    df['kecamatan'] = df['kecamatan'].apply(normalize_kecamatan_name)
                    print(f"[SUCCESS] Berhasil memuat {len(df)} records iklim dari Supabase.")
                    return df
    except Exception as e:
        print(f"[WARNING] Gagal memuat data iklim dari Supabase: {str(e)}")
        print("[INFO] Menggunakan fallback file CSV lokal...")

    # 2. Fallback CSV Lokal
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
