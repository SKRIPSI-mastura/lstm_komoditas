import urllib.request
import json
import os

supabase_url = "https://hetclnzcfvchqoegdyil.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhldGNsbnpjZnZjaHFvZWdkeWlsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODExNjAxNzcsImV4cCI6MjA5NjczNjE3N30.1oBnHVFQqaMinqaQ5IEF6jxOVh7TisTmT_FPlHbd0VY"

url = f"{supabase_url}/rest/v1/komoditas?select=*"
req = urllib.request.Request(
    url,
    headers={
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}"
    }
)

try:
    with urllib.request.urlopen(req, timeout=5.0) as response:
        if response.status == 200:
            data = json.loads(response.read().decode('utf-8'))
            print("=== KOMODITAS DARI SUPABASE ===")
            print(f"Jumlah komoditas: {len(data)}")
            for item in data:
                print(f"- {item['nama_komoditas']}:")
                print(f"  Suhu: {item['suhu_min_c']} - {item['suhu_max_c']} °C")
                print(f"  pH: {item['ph_min']} - {item['ph_max']}")
                print(f"  Elevasi: {item['elevasi_min_mdpl']} - {item['elevasi_max_mdpl']} mdpl")
                print(f"  Curah Hujan: {item['curah_hujan_min_mm']} - {item['curah_hujan_max_mm']} mm")
                print(f"  Kelembapan: {item['kelembapan_min_persen']} - {item['kelembapan_max_persen']} %")
except Exception as e:
    print("Error fetching from Supabase:", str(e))
