"""
evaluate_v2.py — Evaluasi Model LSTM v2 dengan laporan perbandingan sebelum/sesudah.

Menghasilkan:
  - evaluation_report_v2.md   : Laporan lengkap + tabel perbandingan v1 vs v2
  - confusion_matrix_v2.png
  - class_performance_metrics_v2.png
  - roc_curve_v2.png
  - evaluation_metrics_v2.json
"""

import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_fscore_support
)
from tensorflow.keras.models import load_model

from preprocessor import prepare_data_v2

# --- KONFIGURASI PATH ---
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_V2    = os.path.join(BASE_DIR, "models", "commodity_lstm_model_v2.keras")
SCALER_SEQ  = os.path.join(BASE_DIR, "models", "scaler_seq.pkl")
SCALER_STAT = os.path.join(BASE_DIR, "models", "scaler_static.pkl")
ENCODER     = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
EVAL_DIR    = os.path.join(BASE_DIR, "evaluation")
METRICS_V1  = os.path.join(EVAL_DIR, "evaluation_metrics.json")

os.makedirs(EVAL_DIR, exist_ok=True)


# ── Baseline v1 metrics ──────────────────────────────────────────────────────
BASELINE_V1 = {
    "accuracy": 0.7986,
    "loss": 0.4006,
    "classes": ['Jagung', 'Kacang Hijau', 'Kacang Tanah', 'Kedelai', 'Padi', 'Ubi Jalar', 'Ubi Kayu'],
    "precision": [0.99, 0.86, 0.27, 0.95, 0.81, 0.85, 0.76],
    "recall":    [0.98, 0.91, 0.92, 0.97, 0.49, 0.83, 0.78],
    "f1_score":  [0.98, 0.89, 0.42, 0.96, 0.61, 0.84, 0.77],
    "support":   [1770, 4126, 418, 807, 3245, 775, 534],
}


def evaluate_model_v2():
    print("\n" + "=" * 65)
    print("   EVALUASI MODEL LSTM v2 — DUAL-INPUT HYBRID (FITUR LENGKAP)")
    print("=" * 65)

    # 1. Cek keberadaan file model
    for path, name in [(MODEL_V2, "commodity_lstm_model_v2.keras"),
                       (SCALER_SEQ, "scaler_seq.pkl"),
                       (SCALER_STAT, "scaler_static.pkl"),
                       (ENCODER, "label_encoder.pkl")]:
        if not os.path.exists(path):
            print(f"[ERROR] File tidak ditemukan: {path}")
            print("Silakan jalankan train_v2.py terlebih dahulu.")
            return

    # 2. Load model & encoder
    print("[INFO] Memuat model v2, scaler, dan label encoder...")
    model  = load_model(MODEL_V2)
    le     = joblib.load(ENCODER)
    classes     = le.classes_
    num_classes = len(classes)
    print(f"[SUCCESS] Model v2 dimuat: {num_classes} kelas — {list(classes)}")

    # 3. Siapkan data (sama persis dengan split training)
    print("[INFO] Memuat dan mempersiapkan dataset evaluasi...")
    X_seq, X_static, y, _ = prepare_data_v2()

    _, X_seq_test, _, X_static_test, _, y_test = train_test_split(
        X_seq, X_static, y, test_size=0.2, random_state=42
    )

    # 4. Prediksi
    print("[INFO] Melakukan prediksi pada data test...")
    y_pred_probs = model.predict([X_seq_test, X_static_test], batch_size=128)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    # 5. Hitung metrik
    test_acc  = accuracy_score(y_test, y_pred)
    test_loss_list = model.evaluate([X_seq_test, X_static_test], y_test, verbose=0)
    test_loss = test_loss_list[0] if isinstance(test_loss_list, list) else test_loss_list

    print(f"\n[METRIK] Akurasi Global (v2) : {test_acc*100:.2f}%")
    print(f"[METRIK] Loss Uji      (v2) : {test_loss:.4f}")

    class_report_str = classification_report(y_test, y_pred, target_names=classes)
    print("\n--- Classification Report v2 ---")
    print(class_report_str)

    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)

    metrics_v2 = {
        "accuracy":  float(test_acc),
        "loss":      float(test_loss),
        "classes":   list(classes),
        "precision": [float(p) for p in precision],
        "recall":    [float(r) for r in recall],
        "f1_score":  [float(f) for f in f1],
        "support":   [int(s)   for s in support],
    }
    with open(os.path.join(EVAL_DIR, "evaluation_metrics_v2.json"), "w") as fp:
        json.dump(metrics_v2, fp, indent=4)

    cm = confusion_matrix(y_test, y_pred)

    # 6. Grafik evaluasi ─────────────────────────────────────────────────────
    print("\n[INFO] Menggambar grafik evaluasi v2...")

    # a. Confusion Matrix
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, classes, rotation=45, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.title('Confusion Matrix — LSTM v2 (Dual-Input Hybrid)', fontsize=13, pad=15)
    plt.ylabel('True Class',      fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(EVAL_DIR, "confusion_matrix_v2.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # b. Precision-Recall-F1 Bar Chart per Class (v2 vs v1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    x_idx = np.arange(num_classes)
    bw    = 0.25

    for ax, title, p_vals, r_vals, f_vals in [
        (axes[0], 'Model v1 — Baseline (6 fitur)',
         BASELINE_V1['precision'], BASELINE_V1['recall'], BASELINE_V1['f1_score']),
        (axes[1], 'Model v2 — Fitur Lengkap (9 fitur)',
         list(precision), list(recall), list(f1)),
    ]:
        ax.bar(x_idx - bw, p_vals, width=bw, label='Precision', color='#1f77b4')
        ax.bar(x_idx,      r_vals, width=bw, label='Recall',    color='#ff7f0e')
        ax.bar(x_idx + bw, f_vals, width=bw, label='F1-Score',  color='#2ca02c')
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.legend(loc='lower right', fontsize=9)

    plt.suptitle('Perbandingan Kinerja per Kelas: v1 vs v2', fontsize=14, fontweight='bold')
    plt.tight_layout()
    metrics_chart_path = os.path.join(EVAL_DIR, "class_performance_metrics_v2.png")
    plt.savefig(metrics_chart_path, dpi=300)
    plt.close()

    # c. ROC Curve v2
    plt.figure(figsize=(9, 7))
    roc_aucs = []
    for i in range(num_classes):
        y_bin = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_pred_probs[:, i])
        roc_auc_val = auc(fpr, tpr)
        roc_aucs.append(roc_auc_val)
        plt.plot(fpr, tpr, lw=2, label=f'{classes[i]} (AUC={roc_auc_val:.3f})')

    macro_auc = float(np.mean(roc_aucs))
    metrics_v2["roc_auc_per_class"] = [float(a) for a in roc_aucs]
    metrics_v2["roc_auc_macro"]     = macro_auc

    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate',  fontsize=12)
    plt.title(f'ROC Curve — LSTM v2 (Macro AUC = {macro_auc:.4f})', fontsize=13, pad=15)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    roc_path = os.path.join(EVAL_DIR, "roc_curve_v2.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # Update json dengan AUC
    with open(os.path.join(EVAL_DIR, "evaluation_metrics_v2.json"), "w") as fp:
        json.dump(metrics_v2, fp, indent=4)

    print(f"[INFO] Grafik disimpan di: {EVAL_DIR}")
    print(f"       - Confusion Matrix : confusion_matrix_v2.png")
    print(f"       - Bar Chart Metrik : class_performance_metrics_v2.png")
    print(f"       - ROC Curve        : roc_curve_v2.png")

    # 7. Generate Laporan Markdown
    generate_comparison_report(classes, metrics_v2, BASELINE_V1, class_report_str, cm)
    print(f"\n[SUCCESS] Laporan perbandingan berhasil dibuat!")


def generate_comparison_report(classes, v2, v1, report_str_v2, cm_v2):
    """Generate evaluation_report_v2.md dengan tabel perbandingan v1 vs v2."""

    report_path = os.path.join(EVAL_DIR, "evaluation_report_v2.md")
    root_path   = os.path.abspath(os.path.join(BASE_DIR, "..", "evaluation_report_v2.md"))

    correct_v2   = int(np.diag(cm_v2).sum())
    total_v2     = int(cm_v2.sum())
    incorrect_v2 = total_v2 - correct_v2

    # Delta akurasi
    delta_acc = (v2['accuracy'] - v1['accuracy']) * 100
    delta_sign = "+" if delta_acc >= 0 else ""

    # Baris tabel perbandingan per kelas
    class_rows = ""
    for i, cls in enumerate(classes):
        p1 = v1['precision'][i];  p2 = v2['precision'][i]
        r1 = v1['recall'][i];     r2 = v2['recall'][i]
        f1_v1 = v1['f1_score'][i]; f1_v2 = v2['f1_score'][i]

        dp = f"{delta_fmt(p2-p1, 'precision')}"
        dr = f"{delta_fmt(r2-r1, 'recall')}"
        df = f"{delta_fmt(f1_v2-f1_v1, 'f1')}"

        class_rows += (
            f"| **{cls}** "
            f"| {p1:.2f} | {p2:.2f} | {dp} "
            f"| {r1:.2f} | {r2:.2f} | {dr} "
            f"| {f1_v1:.2f} | {f1_v2:.2f} | {df} |\n"
        )

    # Baris confusion matrix v2
    cm_table = "| True \\ Predicted | " + " | ".join(classes) + " |\n"
    cm_table += "| --- | " + " | ".join(["---"] * len(classes)) + " |\n"
    for i, row in enumerate(cm_v2):
        cm_table += f"| **{classes[i]}** | " + " | ".join(map(str, row)) + " |\n"

    classes_str = ", ".join(classes)

    content = f"""# LAPORAN EVALUASI: PENGEMBANGAN MODEL LSTM v2 DENGAN FITUR LAHAN LENGKAP

Laporan ini membandingkan performa **Model LSTM v1** (baseline) dengan **Model LSTM v2** yang menggunakan
arsitektur *Dual-Input Hybrid* dengan penambahan fitur tekstur tanah (tanah liat, pasir, debu)
pada klasifikasi komoditas pertanian di Kabupaten Aceh Utara.

---

## 1. Perubahan Arsitektur Model

| Aspek | Model v1 (Baseline) | Model v2 (Fitur Lengkap) |
| :--- | :--- | :--- |
| **Tipe Arsitektur** | Sequential LSTM | Dual-Input Hybrid (Functional API) |
| **Jumlah Fitur Input** | 6 fitur | 9 fitur |
| **Fitur Sequential (LSTM)** | T2M, RH2M, WS2M, elevasi, pH, curah_hujan | T2M, RH2M, WS2M |
| **Fitur Statis (Dense Branch)** | *(tidak ada)* | elevasi, curah_hujan, pH, liat, pasir, debu |
| **Penanganan Imbalance** | Class Weighting | Class Weighting |
| **File Model** | `commodity_lstm_model.keras` | `commodity_lstm_model_v2.keras` |

### Arsitektur Dual-Input v2

```
Input_seq  (30, 3) ── LSTM(128) ── Dropout ── LSTM(64) ── Dropout ──┐
                                                                      ├── Concat ── Dense(64) ── Dropout ── Dense(32) ── Softmax(7)
Input_static  (6,) ── Dense(32) ── BatchNorm ── Dense(16) ───────────┘
```

> [!NOTE]
> Fitur statis (tanah & topografi) dipisahkan dari sequence LSTM karena nilainya **konstan per kecamatan**
> (tidak berubah harian). Memasukkannya sebagai time-step berulang akan menyebabkan redundansi.
> Arsitektur dual-input memungkinkan model belajar representasi temporal iklim dan representasi
> spasial lahan secara **independen**, lalu menggabungkannya untuk keputusan akhir.

---

## 2. Ringkasan Perbandingan Performa Utama

| Parameter | Model v1 (Baseline) | Model v2 (Fitur Lengkap) | Delta |
| :--- | :---: | :---: | :---: |
| **Akurasi Global** | 79.86% | **{v2['accuracy']*100:.2f}%** | **{delta_sign}{delta_acc:.2f}%** |
| **Loss Uji** | 0.4006 | **{v2['loss']:.4f}** | {v2['loss']-0.4006:+.4f} |
| **Macro ROC-AUC** | *(tidak dihitung)* | **{v2.get('roc_auc_macro', 0):.4f}** | — |
| **Total Sampel Uji** | 11,675 | **{total_v2:,}** | — |
| **Prediksi Benar** | 9,324 | **{correct_v2:,}** | {correct_v2-9324:+,} |
| **Prediksi Salah** | 2,351 | **{incorrect_v2:,}** | {incorrect_v2-2351:+,} |

---

## 3. Perbandingan Kinerja per Kelas Komoditas

| Kelas | Prec-v1 | Prec-v2 | ΔPrec | Rec-v1 | Rec-v2 | ΔRec | F1-v1 | F1-v2 | ΔF1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{class_rows}

> [!NOTE]
> **Fokus analisis** pada kelas bermasalah di v1:
> - **Kacang Tanah** — F1 baseline 0.42 (precision rendah 0.27, banyak false positive)
> - **Padi** — F1 baseline 0.61 (recall rendah 0.49, banyak prediksi salah ke Kacang Hijau/Tanah)

---

## 4. Classification Report Lengkap (Model v2)

```text
{report_str_v2}```

---

## 5. Matriks Kebingungan (Model v2)

{cm_table}

---

## 6. Fitur yang Ditambahkan dan Dampaknya

### 6.1 Fitur Baru: Tekstur Tanah (% Clay, Sand, Silt)

| Fitur | Relevansi Agronomis |
| :--- | :--- |
| **Tanah Liat (%)** | Padi membutuhkan liat tinggi (>35%) untuk menahan air sawah. Kedelai juga toleran liat. |
| **Tanah Pasir (%)** | Kacang Tanah tumbuh optimal di tanah berpasir (drainase cepat, aerasi akar baik). |
| **Tanah Debu (%)** | Ubi Jalar & Ubi Kayu preferensi tanah gembur dengan debu sedang. |

### 6.2 Fitur Statis yang Sudah Ada (Tetap Digunakan)

| Fitur | Keterangan |
| :--- | :--- |
| **Elevasi (m dpl)** | Pengaruh terhadap suhu dan jenis komoditas (dataran rendah vs tinggi) |
| **Curah Hujan Tahunan (mm)** | Kebutuhan air rata-rata tahunan per komoditas |
| **pH Tanah** | Kesesuaian kemasaman untuk masing-masing komoditas |

---

## 7. Visualisasi

Grafik berikut tersimpan di folder `lstm_komoditas/evaluation/`:

| File | Deskripsi |
| :--- | :--- |
| `confusion_matrix_v2.png` | Heatmap confusion matrix model v2 |
| `class_performance_metrics_v2.png` | Grafik perbandingan Precision/Recall/F1 v1 vs v2 per kelas |
| `roc_curve_v2.png` | ROC Curve multi-kelas model v2 (One-vs-Rest) |
| `training_history_v2.png` | Grafik training & validation loss/accuracy model v2 |

---

*Laporan digenerate otomatis oleh `evaluate_v2.py` — Model LSTM v2 (Dual-Input Hybrid)*
"""

    for path in [report_path, root_path]:
        with open(path, "w", encoding="utf-8") as fp:
            fp.write(content)

    print(f"[INFO] Laporan tersimpan di:")
    print(f"       - {report_path}")
    print(f"       - {root_path}")


def delta_fmt(delta, kind=''):
    """Format delta value dengan warna/tanda."""
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2f}"


if __name__ == "__main__":
    evaluate_model_v2()
