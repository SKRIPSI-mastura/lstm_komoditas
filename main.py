import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FILE_IKLIM = os.path.join(DATA_DIR, "iklim_semua_kecamatan_aceh_utara_2020_2025.csv")
FILE_ELEVASI = os.path.join(DATA_DIR, "Elevasi_Kecamatan_Aceh_Utara.csv")
FILE_TANAH = os.path.join(DATA_DIR, "data_tanah_aceh_utara2.csv")
FILE_HUJAN = os.path.join(DATA_DIR, "data_curah_hujan_aceh_utara.csv")

# KNOWLEDGE BASE TANAMAN
# Variabel tanah yang digunakan: ph_tanah, tanah_liat, tanah_pasir, tanah_debu
CROP_KB = {
    "Padi": {
        "umur_tanam_bulan": 4,
        "kebutuhan_hujan_bulanan": [150, 200, 200, 100],
        "suhu_optimal": (25.0, 30.0),
        "ph_optimal": (5.5, 7.5),
        "elevasi_optimal": (0.0, 600.0),
        "kelembapan_optimal": (70.0, 90.0),
        "toleransi_liat": (30.0, 50.0),
        "toleransi_pasir": (20.0, 40.0),
        "toleransi_debu": (20.0, 40.0)
    },
    "Jagung": {
        "umur_tanam_bulan": 3,
        "kebutuhan_hujan_bulanan": [100, 150, 80],
        "suhu_optimal": (21.0, 30.0),
        "ph_optimal": (5.5, 7.0),
        "elevasi_optimal": (0.0, 1500.0),
        "kelembapan_optimal": (60.0, 80.0),
        "toleransi_liat": (20.0, 40.0),
        "toleransi_pasir": (30.0, 50.0),
        "toleransi_debu": (20.0, 40.0)
    },
    "Kedelai": {
        "umur_tanam_bulan": 3,
        "kebutuhan_hujan_bulanan": [100, 120, 50],
        "suhu_optimal": (25.0, 30.0),
        "ph_optimal": (5.8, 7.0),
        "elevasi_optimal": (0.0, 900.0),
        "kelembapan_optimal": (60.0, 80.0),
        "toleransi_liat": (20.0, 40.0),
        "toleransi_pasir": (30.0, 50.0),
        "toleransi_debu": (20.0, 40.0)
    },
    "Kacang Tanah": {
        "umur_tanam_bulan": 3,
        "kebutuhan_hujan_bulanan": [80, 100, 50],
        "suhu_optimal": (25.0, 30.0),
        "ph_optimal": (5.5, 7.0),
        "elevasi_optimal": (0.0, 800.0),
        "kelembapan_optimal": (55.0, 75.0),
        "toleransi_liat": (10.0, 30.0),
        "toleransi_pasir": (40.0, 60.0),
        "toleransi_debu": (20.0, 30.0)
    },
    "Kacang Hijau": {
        "umur_tanam_bulan": 2,
        "kebutuhan_hujan_bulanan": [50, 80],
        "suhu_optimal": (25.0, 35.0),
        "ph_optimal": (5.5, 6.5),
        "elevasi_optimal": (0.0, 800.0),
        "kelembapan_optimal": (50.0, 75.0),
        "toleransi_liat": (10.0, 30.0),
        "toleransi_pasir": (40.0, 60.0),
        "toleransi_debu": (20.0, 40.0)
    },
    "Ubi Kayu": {
        "umur_tanam_bulan": 4,
        "kebutuhan_hujan_bulanan": [100, 150, 150, 100],
        "suhu_optimal": (25.0, 30.0),
        "ph_optimal": (5.5, 7.0),
        "elevasi_optimal": (0.0, 1500.0),
        "kelembapan_optimal": (50.0, 80.0),
        "toleransi_liat": (10.0, 30.0),
        "toleransi_pasir": (40.0, 60.0),
        "toleransi_debu": (20.0, 40.0)
    },
    "Ubi Jalar": {
        "umur_tanam_bulan": 4,
        "kebutuhan_hujan_bulanan": [80, 120, 120, 50],
        "suhu_optimal": (21.0, 27.0),
        "ph_optimal": (5.5, 6.8),
        "elevasi_optimal": (0.0, 1500.0),
        "kelembapan_optimal": (60.0, 80.0),
        "toleransi_liat": (20.0, 40.0),
        "toleransi_pasir": (30.0, 50.0),
        "toleransi_debu": (20.0, 40.0)
    }
}

def normalize_kecamatan_name(name):
    """Normalisasi nama kecamatan ke standar database untuk menghindari mismatch spelling di CSV."""
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

def load_kecamatan_data():
    """
    Memuat data profil kecamatan.
    Mencoba mengambil data terbaru dari Supabase terlebih dahulu.
    Jika gagal/offline, menggunakan data dari file CSV lokal.
    """
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
        with urllib.request.urlopen(req, timeout=5.0) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                if data:
                    kec_dict = {}
                    for row in data:
                        nama = normalize_kecamatan_name(row.get("nama_kecamatan", ""))
                        elev = float(row.get("elevasi_mdpl") or 0.0)
                        kec_dict[nama] = {
                            "elevasi": elev,
                            "ph": float(row.get("ph_tanah") or 6.5),
                            "tanah_liat_persen": float(row.get("tanah_liat") or 30.0),
                            "tanah_pasir_persen": float(row.get("tanah_pasir") or 35.0),
                            "tanah_debu_persen": float(row.get("tanah_debu") or 35.0),
                            "hujan_tahunan": float(row.get("curah_hujan_tahunan") or 2000.0),
                            "resiko_bencana": "Tinggi" if elev < 15 else "Rendah"
                        }
                    print(f"[SUCCESS] Berhasil memuat {len(kec_dict)} profil kecamatan dari Supabase.")
                    return kec_dict
    except Exception as e:
        print(f"[WARNING] Gagal memuat profil kecamatan dari Supabase: {e}")
        print("[INFO] Menggunakan fallback file CSV lokal...")

    # 2. Fallback CSV Lokal
    df_elev = pd.read_csv(FILE_ELEVASI)
    df_tanah = pd.read_csv(FILE_TANAH)
    df_hujan = pd.read_csv(FILE_HUJAN)

    df_elev["kecamatan"] = df_elev["kecamatan"].apply(normalize_kecamatan_name)
    df_tanah["kecamatan"] = df_tanah["kecamatan"].apply(normalize_kecamatan_name)
    df_hujan["kecamatan"] = df_hujan["kecamatan"].apply(normalize_kecamatan_name)

    df_hujan["date"] = pd.to_datetime(df_hujan["date"].astype(str))
    df_hujan["year"] = df_hujan["date"].dt.year
    hujan_tahunan = (
        df_hujan.groupby(["kecamatan", "year"])["curah_hujan"]
        .sum()
        .groupby("kecamatan")
        .mean()
        .reset_index()
    )
    hujan_tahunan.rename(columns={"curah_hujan": "curah_hujan_tahunan"}, inplace=True)

    merged = df_elev[["kecamatan", "elevasi_mdpl"]].merge(
        df_tanah[["kecamatan", "ph_tanah", "tanah_liat", "tanah_pasir", "tanah_debu"]],
        on="kecamatan"
    ).merge(
        hujan_tahunan[["kecamatan", "curah_hujan_tahunan"]], on="kecamatan"
    )

    kec_dict = {}
    for _, row in merged.iterrows():
        kec_dict[row["kecamatan"]] = {
            "elevasi": row["elevasi_mdpl"],
            "ph": row["ph_tanah"],
            "tanah_liat_persen": row["tanah_liat"],
            "tanah_pasir_persen": row["tanah_pasir"],
            "tanah_debu_persen": row["tanah_debu"],
            "hujan_tahunan": row["curah_hujan_tahunan"],
            "resiko_bencana": "Tinggi" if row["elevasi_mdpl"] < 15 else "Rendah"
        }
    return kec_dict

def load_climate_data(kecamatan_name):
    """
    Memuat data iklim bulanan untuk kecamatan tertentu.
    Mencoba mengambil dari Supabase (data_iklim_historis) terlebih dahulu.
    Jika gagal/kosong, menggunakan data dari file CSV lokal.
    """
    target_name = normalize_kecamatan_name(kecamatan_name)

    # 1. Coba ambil dari Supabase
    try:
        import urllib.request
        import json
        from urllib.parse import quote

        supabase_url = "https://hetclnzcfvchqoegdyil.supabase.co"
        supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhldGNsbnpjZnZjaHFvZWdkeWlsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODExNjAxNzcsImV4cCI6MjA5NjczNjE3N30.1oBnHVFQqaMinqaQ5IEF6jxOVh7TisTmT_FPlHbd0VY"

        url = (
            f"{supabase_url}/rest/v1/data_iklim_historis"
            f"?select=*,kecamatan:kecamatan_id(nama_kecamatan)"
        )
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
                    rows = [
                        {
                            "date": r.get("tanggal", ""),
                            "Suhu rata-rata": float(r.get("suhu_c") or 0.0),
                            "Kelembapan udara": float(r.get("kelembapan_persen") or 0.0),
                            "curah_hujan": float(r.get("curah_hujan_mm") or 0.0),
                            "kecamatan": normalize_kecamatan_name(
                                r.get("kecamatan", {}).get("nama_kecamatan", "")
                            ),
                        }
                        for r in raw_data
                    ]
                    df_sb = pd.DataFrame(rows)
                    df_sb["date"] = pd.to_datetime(df_sb["date"])
                    df_sb = df_sb[df_sb["kecamatan"] == target_name]
                    if not df_sb.empty:
                        df_sb.set_index("date", inplace=True)
                        monthly_data = df_sb.resample("ME").agg({
                            "Suhu rata-rata": "mean",
                            "Kelembapan udara": "mean",
                            "curah_hujan": "sum"
                        })
                        monthly_data.dropna(inplace=True)
                        print(f"[SUCCESS] Data iklim {target_name} dari Supabase: {len(monthly_data)} bulan.")
                        return monthly_data[["Suhu rata-rata", "Kelembapan udara", "curah_hujan"]]
    except Exception as e:
        print(f"[WARNING] Gagal memuat data iklim dari Supabase: {e}")
        print("[INFO] Menggunakan fallback file CSV lokal...")

    # 2. Fallback CSV Lokal
    df_iklim = pd.read_csv(FILE_IKLIM)
    df_iklim["date"] = pd.to_datetime(df_iklim["date"])
    df_iklim["kecamatan"] = df_iklim["kecamatan"].apply(normalize_kecamatan_name)

    df_hujan = pd.read_csv(FILE_HUJAN)
    df_hujan["date"] = pd.to_datetime(df_hujan["date"].astype(str))
    df_hujan["kecamatan"] = df_hujan["kecamatan"].apply(normalize_kecamatan_name)

    df_iklim = df_iklim[df_iklim["kecamatan"] == target_name]
    df_hujan = df_hujan[df_hujan["kecamatan"] == target_name]

    if df_iklim.empty or df_hujan.empty:
        return pd.DataFrame()

    df_merged = df_iklim.merge(df_hujan[["date", "curah_hujan"]], on="date", how="left")

    df_merged["curah_hujan"] = df_merged["curah_hujan"].fillna(0)
    df_merged["Suhu rata-rata"] = df_merged["Suhu rata-rata"].ffill()
    df_merged["Kelembapan udara"] = df_merged["Kelembapan udara"].ffill()

    df_merged.set_index("date", inplace=True)
    monthly_data = df_merged.resample("ME").agg({
        "Suhu rata-rata": "mean",
        "Kelembapan udara": "mean",
        "curah_hujan": "sum"
    })

    monthly_data.dropna(inplace=True)
    return monthly_data[["Suhu rata-rata", "Kelembapan udara", "curah_hujan"]]

def create_sequences(data, seq_length):
    """Membuat pasangan input-output untuk pelatihan LSTM."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

def train_lstm_climate_model(df_climate):
    """Melatih model LSTM untuk prediksi iklim bulanan."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_climate.values)

    SEQ_LENGTH = 12
    if len(scaled_data) <= SEQ_LENGTH:
        SEQ_LENGTH = max(1, len(scaled_data) - 4)

    X, y = create_sequences(scaled_data, SEQ_LENGTH)

    model = Sequential([
        Input(shape=(SEQ_LENGTH, 3)),
        LSTM(64, activation="tanh", return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation="tanh"),
        Dense(3)
    ])
    model.compile(optimizer="adam", loss="mse")

    if len(X) > 0:
        model.fit(X, y, epochs=50, batch_size=8, verbose=0)

    return model, scaler, scaled_data[-SEQ_LENGTH:]

def predict_future_climate(model, scaler, last_seq, months=4, df_historical=None):
    """
    Melakukan prediksi iklim autoregresif untuk beberapa bulan ke depan.
    Jika df_historical diberikan, model menggunakan Climatological Blending untuk
    mengurangi akumulasi error (error propagation) dengan mencampurkan prediksi LSTM
    dengan rata-rata iklim historis (climatology) bulanan terkait.
    """
    predictions = []
    curr_seq = last_seq.reshape(1, len(last_seq), 3)

    if df_historical is not None:
        last_date = df_historical.index[-1]
        climatology = df_historical.groupby(df_historical.index.month).mean()
        future_months = []
        current_date = last_date
        for _ in range(months):
            current_date = current_date + pd.DateOffset(months=1)
            future_months.append(current_date.month)
    else:
        future_months = []

    for i in range(months):
        pred = model.predict(curr_seq, verbose=0)

        if df_historical is not None:
            month_val = future_months[i]
            clim_val = climatology.loc[month_val].values
            clim_scaled = scaler.transform([clim_val])[0]

            # Bobot blending (Alpha): menurun seiring bertambahnya bulan (error propagation meningkat)
            # i=0: 80% LSTM, i=1: 60% LSTM, i=2: 40% LSTM, i=3: 20% LSTM
            alpha = 0.8 - (i * 0.2)
            alpha = max(0.1, alpha)

            blended_pred = alpha * pred[0] + (1 - alpha) * clim_scaled
            predictions.append(blended_pred)
            curr_seq = np.append(curr_seq[:, 1:, :], blended_pred.reshape(1, 1, 3), axis=1)
        else:
            predictions.append(pred[0])
            curr_seq = np.append(curr_seq[:, 1:, :], pred.reshape(1, 1, 3), axis=1)

    predicted_climate = scaler.inverse_transform(predictions)
    return predicted_climate


def calculate_score(val, optimal_range, is_rainfall=False):
    """Menghitung skor kesesuaian nilai terhadap rentang optimal."""
    min_opt, max_opt = optimal_range
    if min_opt <= val <= max_opt:
        return 100

    diff = min(abs(val - min_opt), abs(val - max_opt))
    if is_rainfall:
        score = 100 - (diff * 0.5)
    else:
        if max_opt <= 14:      # pH range
            score = 100 - (diff * 20)
        elif max_opt <= 100:   # Persentase tanah
            score = 100 - (diff * 2)
        else:
            score = 100 - diff

    return max(0, score)

def get_supabase_crop_kb():
    """
    Mengambil parameter komoditas terbaru dari Next.js API (Supabase) secara dinamis.
    Jika offline atau gagal, akan menggunakan fallback hardcoded CROP_KB.
    """
    import urllib.request
    import json
    import copy
    
    # Salin data hardcoded lokal sebagai basis
    dynamic_kb = copy.deepcopy(CROP_KB)
    
    try:
        url = "http://localhost:3000/api/komoditas"
        req = urllib.request.Request(url, headers={'User-Agent': 'FastAPI-Backend'})
        with urllib.request.urlopen(req, timeout=2.0) as response:
            if response.status == 200:
                res_body = response.read().decode('utf-8')
                res_data = json.loads(res_body)
                if res_data.get("status") == "success" and res_data.get("data"):
                    db_data = res_data["data"]
                    for item in db_data:
                        name = item.get("nama_komoditas")
                        if name in dynamic_kb:
                            # Update suhu dan pH optimal berdasarkan nilai di Supabase
                            dynamic_kb[name]["suhu_optimal"] = (
                                float(item.get("suhu_min_c", dynamic_kb[name]["suhu_optimal"][0])),
                                float(item.get("suhu_max_c", dynamic_kb[name]["suhu_optimal"][1]))
                            )
                            dynamic_kb[name]["ph_optimal"] = (
                                float(item.get("ph_min", dynamic_kb[name]["ph_optimal"][0])),
                                float(item.get("ph_max", dynamic_kb[name]["ph_optimal"][1]))
                            )
                            # Update elevasi dan kelembapan optimal berdasarkan nilai di Supabase
                            dynamic_kb[name]["elevasi_optimal"] = (
                                float(item.get("elevasi_min_mdpl", dynamic_kb[name]["elevasi_optimal"][0])),
                                float(item.get("elevasi_max_mdpl", dynamic_kb[name]["elevasi_optimal"][1]))
                            )
                            dynamic_kb[name]["kelembapan_optimal"] = (
                                float(item.get("kelembapan_min_persen", dynamic_kb[name]["kelembapan_optimal"][0])),
                                float(item.get("kelembapan_max_persen", dynamic_kb[name]["kelembapan_optimal"][1]))
                            )
    except Exception:
        pass
        
    return dynamic_kb

def recommend_crops(monthly_climate_pred, inputs):
    """
    Mengevaluasi kelayakan setiap komoditas berdasarkan prediksi iklim LSTM
    dan parameter tanah (ph_tanah, tanah_liat, tanah_pasir, tanah_debu, elevasi).
    Skor akhir = 40% skor tanah/lahan + 60% skor iklim.
    """
    results = {}

    # Menggunakan nilai ph_tanah dan tanah_liat langsung tanpa hacks karena database dan CSV sudah lurus
    true_ph = inputs["ph_tanah"]
    true_liat = inputs["tanah_liat_persen"]
    true_elev = inputs["elevasi"]

    # Ambil parameter CROP_KB yang sudah disesuaikan dengan Supabase
    active_crop_kb = get_supabase_crop_kb()

    for crop, kb in active_crop_kb.items():
        # Evaluasi skor parameter tanah/lahan (termasuk Elevasi)
        ph_score = calculate_score(true_ph, kb["ph_optimal"])
        liat_score = calculate_score(true_liat, kb["toleransi_liat"])
        pasir_score = calculate_score(inputs["tanah_pasir_persen"], kb["toleransi_pasir"])
        debu_score = calculate_score(inputs["tanah_debu_persen"], kb["toleransi_debu"])
        elev_score = calculate_score(true_elev, kb["elevasi_optimal"])

        soil_score = (ph_score + liat_score + pasir_score + debu_score + elev_score) / 5

        # Evaluasi skor iklim per bulan berdasarkan umur tanam (termasuk Kelembapan)
        umur = kb["umur_tanam_bulan"]
        eval_months = min(umur, len(monthly_climate_pred))

        climate_scores = []
        monthly_reasons = []

        for i in range(eval_months):
            suhu_pred = monthly_climate_pred[i][0]
            kelembapan_pred = monthly_climate_pred[i][1]
            hujan_pred = monthly_climate_pred[i][2]

            suhu_s = calculate_score(suhu_pred, kb["suhu_optimal"])
            kelembapan_s = calculate_score(kelembapan_pred, kb["kelembapan_optimal"])
            hujan_opt = kb["kebutuhan_hujan_bulanan"][i]
            hujan_s = calculate_score(hujan_pred, (hujan_opt - 30, hujan_opt + 30), is_rainfall=True)

            month_score = (suhu_s + kelembapan_s + hujan_s) / 3
            climate_scores.append(month_score)

            monthly_reasons.append(
                f"Bulan {i+1}: Hujan {hujan_pred:.0f}mm (Opt: {hujan_opt}mm), "
                f"Suhu {suhu_pred:.1f}°C, Kelembaban {kelembapan_pred:.1f}%."
            )

        avg_climate_score = np.mean(climate_scores) if climate_scores else 0
        final_score = (soil_score * 0.4) + (avg_climate_score * 0.6)

        results[crop] = {
            "score": final_score,
            "reasons": monthly_reasons
        }

    sorted_results = dict(sorted(results.items(), key=lambda item: item[1]["score"], reverse=True))
    return sorted_results

def generate_explanation(crop, inputs, monthly_climate_pred, crop_kb_result):
    """Menghasilkan narasi penjelasan untuk komoditas yang direkomendasikan."""
    reasons = crop_kb_result["reasons"]
    explanation = (
        f"{crop} direkomendasikan karena kecocokan masa fase pertumbuhan bulanannya "
        f"dengan prediksi iklim ke depan. "
    )
    explanation += " ".join(reasons)
    return explanation

def main():
    pass

if __name__ == "__main__":
    main()
