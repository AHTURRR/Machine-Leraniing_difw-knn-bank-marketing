# DIFW-KNN: Domain-Informed Feature-Weighted K-Nearest Neighbors for Bank Marketing Classification

Repositori penelitian kuantitatif eksperimental untuk pengembangan algoritma **Domain-Informed Feature-Weighted K-Nearest Neighbors (`DIFW-KNN`)** menggunakan *Mixed-Type Distance*, *Supervised Triplet Feature Weighting*, dan *Cost-Sensitive Decision Rule* pada dataset **UCI Bank Marketing (`bank-full.csv`, ID 222)**.

---

## Ringkasan Metodologi

- **Dataset:** `bank-full.csv` (`45.211` observasi, `16` prediktor asli, target biner `y`: `88,30%` `no` vs `11,70%` `yes`, *Imbalance Ratio* `7,55 : 1`).
- **Skenario Prediksi (*Pre-Call Targeting*):** Fitur `duration` dihapus sebelum pemodelan karena merupakan *post-call target leakage* (*single-feature ROC-AUC* = `0,8076`).
- **Protokol Anti-*Data Leakage*:**
  - *Stratified Train-Test Split* `80:20` (`random_state=42`) menghasilkan `36.168` data latih dan `9.043` data uji.
  - Seluruh parameter statistik (`M_pos_train` untuk rekayasa fitur `pdays`, `StandardScaler`, `OneHotEncoder`, serta kuantil `Q05` dan `Q95` pada `MixedTypePreprocessor`) dipelajari **hanya dari data latih**.
- **16 Fitur Domain:**
  - **6 Numerik:** `age`, `balance`, `day`, `campaign`, `previous`, `pdays_recency`
  - **10 Kategorikal/Biner:** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`, `previously_contacted` (label `"unknown"` dipertahankan sebagai kategori valid).

---

## Struktur Repositori

```text
.
├── bank-full.csv                                     # Dataset UCI Bank Marketing (ID 222)
├── Desain_Penelitian_DIFW_KNN.md                     # Dokumen desain & protokol penelitian
├── Laporan_EDA_DIFW_KNN.md                           # Laporan Data Understanding & EDA
├── Laporan_Preprocessing_DIFW_KNN.md                 # Laporan Preprocessing & Feature Engineering
├── requirements.txt                                  # Dependensi Python teruji (Python 3.13.7)
├── data/
│   └── processed/
│       ├── train_split.csv                           # Stratified train split (36.168 baris)
│       └── test_split.csv                            # Stratified test split (9.043 baris)
├── notebooks/
│   ├── 01_EDA.ipynb                                  # Notebook interaktif Data Understanding & EDA
│   └── 02_Preprocessing_FeatureEngineering.ipynb     # Notebook interaktif Preprocessing & Unit Checks
├── outputs/
│   ├── eda/                                          # Tabel statistik & 5 grafik EDA
│   └── preprocessing/                                # Tabel parameter kuantil, unit checks, & 3 grafik
└── src/
    ├── __init__.py
    ├── 01_eda.py                                     # Skrip CLI untuk Data Understanding & EDA
    ├── preprocessing.py                              # Modul reusable (DomainFeatureEngineer, Preprocessors)
    └── 02_preprocessing.py                           # Skrip CLI untuk Preprocessing & verifikasi metrik jarak
```

---

## Cara Menjalankan

### 1. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 2. Menjalankan Skrip Python (`.py`)
```bash
# Tahap 1: Data Understanding & EDA
python src/01_eda.py

# Tahap 2: Preprocessing, Feature Engineering, & Distance Unit Checks
python src/02_preprocessing.py
```

### 3. Membuka Notebook Interaktif (`.ipynb`)
Buka file berikut menggunakan VS Code atau JupyterLab (seluruh sel sudah dieksekusi dan menyertakan output tabel serta grafik *inline*):
- `notebooks/01_EDA.ipynb`
- `notebooks/02_Preprocessing_FeatureEngineering.ipynb`

---

## Status Tahapan Penelitian

- [x] **Tahap 1:** Data Understanding & Exploratory Data Analysis (`src/01_eda.py`, `notebooks/01_EDA.ipynb`)
- [x] **Tahap 2:** Leak-Free Preprocessing & Feature Engineering (`src/preprocessing.py`, `src/02_preprocessing.py`, `notebooks/02_Preprocessing_FeatureEngineering.ipynb`)
- [ ] **Tahap 3:** Baseline KNN (`E0` Standard KNN & `E1` Optimized KNN via Stratified CV)
- [ ] **Tahap 4:** Proposed DIFW-KNN (`E2`–`E5`: Mixed-Type Distance, Balanced Triplet Margin Loss, & Cost-Sensitive Decision Rule)
- [ ] **Tahap 5:** Evaluasi Hold-Out Test, Ablation Study, & Interpretasi LIME
