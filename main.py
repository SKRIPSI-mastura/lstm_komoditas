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
        "suhu_optimal": (25, 30),
        "ph_optimal": (5.5, 7.5),
        "toleransi_liat": (30, 50),
        "toleransi_pasir": (20, 40),
        "toleransi_debu": (20, 40)
    },
    "Jagung": {
        "umur_tanam_bulan": 3,
        "kebutuhan_hujan_bulanan": [100, 150, 80],
        "suhu_optimal": (21, 30),
        "ph_optimal": (5.5, 7.0),
        "toleransi_liat": (20, 40),
        "toleransi_pasir": (30, 50),
        "toleransi_debu": (20, 40)
    },
    "Kedelai": {
        "umur_tanam_bulan": 3,
        "kebutuhan_hujan_bulanan": [100, 120, 50],
        "suhu_optimal": (25, 30),
        "ph_optimal": (5.8, 7.0),
        "toleransi_liat": (20, 40),
        "toleransi_pasir": (30, 50),
        "toleransi_debu": (20, 40)
    },
    "Kacang Tanah": {
        "umur_tanam_bulan": 3,
        "kebutuhan_hujan_bulanan": [80, 100, 50],
        "suhu_optimal": (25, 30),
        "ph_optimal": (5.5, 7.0),
        "toleransi_liat": (10, 30),
        "toleransi_pasir": (40, 60),
        "toleransi_debu": (20, 30)
    },
    "Ubi Kayu": {
        "umur_tanam_bulan": 4,
        "kebutuhan_hujan_bulanan": [100, 150, 150, 100],
        "suhu_optimal": (25, 30),
        "ph_optimal": (5.5, 7.0),
        "toleransi_liat": (10, 30),
        "toleransi_pasir": (40, 60),
        "toleransi_debu": (20, 40)
    },
    "Ubi Jalar": {
        "umur_tanam_bulan": 4,
        "kebutuhan_hujan_bulanan": [80, 120, 120, 50],
        "suhu_optimal": (21, 27),
        "ph_optimal": (5.5, 6.8),
        "toleransi_liat": (20, 40),
        "toleransi_pasir": (30, 50),
        "toleransi_debu": (20, 40)
    }
}

def load_kecamatan_data():
    """
    Memuat dan menggabungkan data profil kecamatan dari CSV.
    Kolom tanah yang digunakan: ph_tanah, tanah_liat, tanah_pasir, tanah_debu.
    Kolom tekstur_tanah tidak digunakan karena sudah direpresentasikan
    oleh komposisi tanah_liat, tanah_pasir, dan tanah_debu.
    """
    df_elev = pd.read_csv(FILE_ELEVASI)
    df_tanah = pd.read_csv(FILE_TANAH)
    df_hujan = pd.read_csv(FILE_HUJAN)

    df_elev["kecamatan"] = df_elev["kecamatan"].str.strip()
    df_tanah["kecamatan"] = df_tanah["kecamatan"].str.strip()
    df_hujan["kecamatan"] = df_hujan["kecamatan"].str.strip()

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

    # Merge: gunakan hanya kolom yang diperlukan dari df_tanah
    # Dataset tanah: kecamatan, ph_tanah, tanah_liat, tanah_pasir, tanah_debu
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
            "ph": row["ph_tanah"],              # Nilai pH tanah yang benar
            "tanah_liat_persen": row["tanah_liat"],  # Persentase tanah liat
            "tanah_pasir_persen": row["tanah_pasir"],
            "tanah_debu_persen": row["tanah_debu"],
            "hujan_tahunan": row["curah_hujan_tahunan"],
            "resiko_bencana": "Tinggi" if row["elevasi_mdpl"] < 15 else "Rendah"
        }
    return kec_dict

def load_climate_data(kecamatan_name):
    """Memuat dan menggabungkan data iklim bulanan untuk kecamatan tertentu."""
    df_iklim = pd.read_csv(FILE_IKLIM)
    df_iklim["date"] = pd.to_datetime(df_iklim["date"])
    df_iklim["kecamatan"] = df_iklim["kecamatan"].str.strip()

    df_hujan = pd.read_csv(FILE_HUJAN)
    df_hujan["date"] = pd.to_datetime(df_hujan["date"].astype(str))
    df_hujan["kecamatan"] = df_hujan["kecamatan"].str.strip()

    df_iklim = df_iklim[df_iklim["kecamatan"] == kecamatan_name]
    df_hujan = df_hujan[df_hujan["kecamatan"] == kecamatan_name]

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
        LSTM(64, activation="relu", return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation="relu"),
        Dense(3)
    ])
    model.compile(optimizer="adam", loss="mse")

    if len(X) > 0:
        model.fit(X, y, epochs=50, batch_size=8, verbose=0)

    return model, scaler, scaled_data[-SEQ_LENGTH:]

def predict_future_climate(model, scaler, last_seq, months=4):
    """Melakukan prediksi iklim autoregresif untuk beberapa bulan ke depan."""
    predictions = []
    curr_seq = last_seq.reshape(1, len(last_seq), 3)
    for _ in range(months):
        pred = model.predict(curr_seq, verbose=0)
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

def recommend_crops(monthly_climate_pred, inputs):
    """
    Mengevaluasi kelayakan setiap komoditas berdasarkan prediksi iklim LSTM
    dan parameter tanah (ph_tanah, tanah_liat, tanah_pasir, tanah_debu).
    Skor akhir = 40% skor tanah + 60% skor iklim.
    """
    results = {}

    for crop, kb in CROP_KB.items():
        # Evaluasi skor parameter tanah
        ph_score = calculate_score(inputs["ph_tanah"], kb["ph_optimal"])
        liat_score = calculate_score(inputs["tanah_liat_persen"], kb["toleransi_liat"])
        pasir_score = calculate_score(inputs["tanah_pasir_persen"], kb["toleransi_pasir"])
        debu_score = calculate_score(inputs["tanah_debu_persen"], kb["toleransi_debu"])

        soil_score = (ph_score + liat_score + pasir_score + debu_score) / 4

        # Evaluasi skor iklim per bulan berdasarkan umur tanam
        umur = kb["umur_tanam_bulan"]
        eval_months = min(umur, len(monthly_climate_pred))

        climate_scores = []
        monthly_reasons = []

        for i in range(eval_months):
            suhu_pred = monthly_climate_pred[i][0]
            hujan_pred = monthly_climate_pred[i][2]

            suhu_s = calculate_score(suhu_pred, kb["suhu_optimal"])
            hujan_opt = kb["kebutuhan_hujan_bulanan"][i]
            hujan_s = calculate_score(hujan_pred, (hujan_opt - 30, hujan_opt + 30), is_rainfall=True)

            month_score = (suhu_s + hujan_s) / 2
            climate_scores.append(month_score)

            monthly_reasons.append(
                f"Bulan {i+1}: Prediksi hujan {hujan_pred:.0f}mm "
                f"(Optimal: {hujan_opt}mm), Suhu {suhu_pred:.1f}°C."
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
