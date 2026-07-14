import pandas as pd
import numpy as np
import os
from relabel import crops_parameters, profiles_dict, calculate_score

df = pd.read_csv("data/labeled_data.csv")

# Ambil satu baris dataran tinggi
row = df[df['elevasi_mdpl'] > 350].iloc[0]
print("=== DATARAN TINGGI ROW ===")
print("Kecamatan:", row['kecamatan'])
print("Elevasi:", row['elevasi_mdpl'])
print("T2M:", row['T2M'])
print("RH2M:", row['RH2M'])
print("ph_tanah_mean:", row['ph_tanah_mean'])

kec = row['kecamatan']
prof = profiles_dict.get(kec, {})
elev = row['elevasi_mdpl']
ph = row['ph_tanah_mean']
liat = prof.get('tanah_liat', 30.0)
pasir = prof.get('tanah_pasir', 30.0)
debu = prof.get('tanah_debu', 30.0)
t2m = row['T2M']
rh2m = row['RH2M']

print("\nDetail Skor per Komoditas:")
for crop, kb in crops_parameters.items():
    ph_s = calculate_score(ph, kb["ph_optimal"])
    liat_s = calculate_score(liat, kb["toleransi_liat"])
    pasir_s = calculate_score(pasir, kb["toleransi_pasir"])
    debu_s = calculate_score(debu, kb["toleransi_debu"])
    soil_score = (ph_s + liat_s + pasir_s + debu_s) / 4.0
    
    suhu_s = calculate_score(t2m, kb["suhu_optimal"])
    kelembapan_s = calculate_score(rh2m, kb["kelembapan_optimal"])
    climate_score = (suhu_s + kelembapan_s) / 2.0
    
    elev_score = calculate_score(elev, kb["elevasi_optimal"])
    
    final_score = (soil_score * 0.4) + (climate_score * 0.4) + (elev_score * 0.2)
    print(f"- {crop}:")
    print(f"  Soil: {soil_score:.2f} (pH: {ph_s:.1f}, Liat: {liat_s:.1f}, Pasir: {pasir_s:.1f}, Debu: {debu_s:.1f})")
    print(f"  Climate: {climate_score:.2f} (Suhu: {suhu_s:.1f}, Kelembapan: {kelembapan_s:.1f})")
    print(f"  Elevasi: {elev_score:.2f}")
    print(f"  FINAL SCORE: {final_score:.2f}")
