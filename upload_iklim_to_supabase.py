"""
Script untuk upload data iklim dari CSV ke Supabase (tabel data_iklim_historis).

PENTING: Sebelum menjalankan script ini, pastikan RLS sudah di-disable sementara
di Supabase Dashboard:
  Dashboard -> Table Editor -> data_iklim_historis -> RLS -> Disable RLS

Atau jika punya service_role key, ganti SUPABASE_KEY di bawah dengan service_role key.
"""

import csv
import json
import urllib.request
import urllib.error
from datetime import datetime
import os
import time

# === KONFIGURASI ===
SUPABASE_URL = "https://hetclnzcfvchqoegdyil.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhldGNsbnpjZnZjaHFvZWdkeWlsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODExNjAxNzcsImV4cCI6MjA5NjczNjE3N30.1oBnHVFQqaMinqaQ5IEF6jxOVh7TisTmT_FPlHbd0VY"

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "iklim_semua_kecamatan_aceh_utara_2020_2025.csv")
BATCH_SIZE = 500


def normalize_kecamatan_name(name: str) -> str:
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


def fetch_kecamatan_map() -> dict:
    url = f"{SUPABASE_URL}/rest/v1/kecamatan?select=id,nama_kecamatan"
    req = urllib.request.Request(url, headers={
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    })
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    mapping = {normalize_kecamatan_name(row["nama_kecamatan"]): row["id"] for row in data}
    print(f"[INFO] Berhasil memuat {len(mapping)} kecamatan dari Supabase.")
    return mapping


def parse_date(raw: str) -> str:
    for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"Format tanggal tidak dikenal: {raw!r}")


def upload_batch(batch: list) -> bool:
    url = f"{SUPABASE_URL}/rest/v1/data_iklim_historis"
    payload = json.dumps(batch).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST", headers={
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status in (200, 201)
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  [ERROR] HTTP {e.code}: {body[:200]}")
        return False


def main():
    print("=" * 60)
    print("  Upload Data Iklim ke Supabase (data_iklim_historis)")
    print("=" * 60)

    try:
        kec_map = fetch_kecamatan_map()
    except Exception as e:
        print(f"[FATAL] Tidak dapat terhubung ke Supabase: {e}")
        return

    print(f"[INFO] Membaca CSV: {CSV_PATH}")
    rows = []
    skipped = 0
    unknown_kec = set()

    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kec_raw = row.get("kecamatan", "").strip()
            kec_norm = normalize_kecamatan_name(kec_raw)
            kec_id = kec_map.get(kec_norm)

            if not kec_id:
                unknown_kec.add(kec_raw)
                skipped += 1
                continue

            try:
                tanggal = parse_date(row["date"])
                suhu = float(row["Suhu rata-rata"])
                kelembapan = float(row["Kelembapan udara"])
                angin = float(row["Kecepatan angin"])
            except (ValueError, KeyError):
                skipped += 1
                continue

            rows.append({
                "kecamatan_id": kec_id,
                "tanggal": tanggal,
                "suhu_c": round(suhu, 4),
                "kelembapan_persen": round(kelembapan, 4),
                "kecepatan_angin_ms": round(angin, 4),
            })

    print(f"[INFO] Total baris siap diupload : {len(rows):,}")
    print(f"[INFO] Baris yang dilewati       : {skipped:,}")
    if unknown_kec:
        print(f"[WARNING] Kecamatan tidak dikenali: {unknown_kec}")

    if not rows:
        print("[ERROR] Tidak ada data untuk diupload.")
        return

    total_batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE
    success_count = 0
    fail_count = 0

    print(f"\n[INFO] Mulai upload dalam {total_batches} batch (@ {BATCH_SIZE} baris)...\n")
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i: i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        ok = upload_batch(batch)
        if ok:
            success_count += len(batch)
            print(f"  [OK] Batch {batch_num:>3}/{total_batches} — {success_count:,} baris terupload")
        else:
            fail_count += len(batch)
            print(f"  [!!] Batch {batch_num:>3}/{total_batches} — GAGAL ({fail_count:,} baris gagal)")
        time.sleep(0.05)

    print("\n" + "=" * 60)
    print(f"  SELESAI: {success_count:,} baris berhasil, {fail_count:,} gagal")
    print("=" * 60)


if __name__ == "__main__":
    main()
