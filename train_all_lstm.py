import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# Pastikan path main.py terbaca
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import main

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

def train_and_save_all():
    # 1. Buat folder models jika belum ada
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    logging.info(f"Folder models dipastikan aktif di: {models_dir}")

    # 2. Muat data kecamatan
    try:
        kec_data = main.load_kecamatan_data()
        kec_list = sorted(list(kec_data.keys()))
        logging.info(f"Berhasil memuat {len(kec_list)} kecamatan dari database.")
    except Exception as e:
        logging.error(f"Gagal memuat profil kecamatan: {str(e)}")
        return

    # 3. Latih LSTM untuk setiap kecamatan
    SEQ_LENGTH = 30
    for i, kec in enumerate(kec_list, 1):
        filename_kec = kec.lower().replace(" ", "_")
        model_path = os.path.join(models_dir, f"lstm_{filename_kec}.h5")
        
        logging.info(f"[{i}/{len(kec_list)}] Memulai pelatihan LSTM untuk {kec}...")
        
        try:
            # Load climate data
            df_climate = main.load_climate_data(kec)
            if df_climate.empty:
                logging.warning(f"Data iklim untuk {kec} kosong! Dilewati.")
                continue
                
            # Preprocess
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df_climate.values)
            
            if len(scaled_data) < SEQ_LENGTH:
                logging.warning(f"Data iklim untuk {kec} tidak cukup (min 30 hari)! Dilewati.")
                continue
                
            X, y = main.create_sequences(scaled_data, SEQ_LENGTH)
            split = int(0.8 * len(X))
            X_train, y_train = X[:split], y[:split]
            
            # Bangun model LSTM
            model = Sequential([
                Input(shape=(SEQ_LENGTH, 3)),
                LSTM(64, activation='relu', return_sequences=True),
                Dropout(0.2),
                LSTM(32, activation='relu'),
                Dense(3)
            ])
            model.compile(optimizer='adam', loss='mse')
            
            # Latih model (20 Epochs)
            model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
            
            # Simpan model
            model.save(model_path)
            logging.info(f"-> BERHASIL melatih dan menyimpan model untuk {kec} di: {model_path}")
            
        except Exception as e:
            logging.error(f"Terjadi kesalahan saat melatih model untuk {kec}: {str(e)}")

if __name__ == "__main__":
    train_and_save_all()
