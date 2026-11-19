# DIFW-KNN: Domain-Informed Feature-Weighted K-Nearest Neighbors for Bank Marketing Classification

Repositori penelitian kuantitatif eksperimental untuk pengembangan algoritma **Domain-Informed Feature-Weighted K-Nearest Neighbors (`DIFW-KNN`)** menggunakan *Mixed-Type Distance*, *Supervised Triplet Feature Weighting*, dan *Cost-Sensitive Decision Rule* pada dataset **UCI Bank Marketing (`data/bank-full.csv`, ID 222)**.

---

## Ringkasan Metodologi

- **Dataset:** `data/bank-full.csv` (`45.211` observasi, `16` prediktor asli, target biner `y`: `88,30%` `no` vs `11,70%` `yes`, *Imbalance Ratio* `7,55 : 1`).
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
DIFW-KNN/
│
├── data/
│   ├── bank-full.csv                                 # Dataset UCI Bank Marketing (ID 222)
│   └── processed/
│       ├── train_split.csv                           # Stratified train split (36.168 baris)
│       ├── test_split.csv                            # Stratified test split (9.043 baris)
│       └── train_cv_folds.csv                        # Indeks kanonik Stratified 5-Fold CV
│
├── src/
│   ├── __init__.py
│   ├── mixed_type_distance.py                        # Developer 1: Mixed-Type Distance (Q05-Q95 + mismatch)
│   ├── triplet_learning.py                           # Developer 2: Balanced Triplet Sampling & Margin Loss
│   ├── feature_weight_optimizer.py                   # Developer 2: Simplex Projected GD Feature Weight Optimizer
│   ├── difw_knn.py                                   # Developer 3: DIFW-KNN Classifier & Ablation Suite (E2-E5)
│   ├── cost_sensitive_decision.py                    # Developer 3: Cost-Sensitive Bayes Rule & Trade-off Grid
│   ├── preprocessing.py                              # Modul DomainFeatureEngineer & BaselineEuclideanPreprocessor
│   ├── baseline_knn.py                               # Modul reusable evaluasi & CV Baseline KNN (E0 & E1)
│   ├── 01_eda.py                                     # Skrip CLI untuk Data Understanding & EDA
│   ├── 02_preprocessing.py                           # Skrip CLI untuk Preprocessing & verifikasi metrik jarak
│   ├── 03_stratified_split.py                        # Skrip CLI untuk Stratified Split & 5-Fold CV Setup
│   └── 04_baseline_knn.py                            # Skrip CLI untuk Baseline KNN (E0 & E1)
│
├── tests/
│   ├── test_mixed_distance.py                        # Unit test metrik jarak Mixed-Type & rekayasa pdays
│   ├── test_triplet_learning.py                      # Unit test BalancedTripletSampler, grad check, & optimizer
│   └── test_difw_knn.py                              # Unit & integration test Cost-Sensitive Rule & DIFW-KNN
│
├── notebooks/
│   ├── 01_DIFW_KNN_Mixed_Type_Distance_Triplet_Loss.ipynb  # Notebook integrasi DIFW-KNN (Dev 1, 2, 3)
│   ├── 01_EDA.ipynb                                  # Notebook interaktif Data Understanding & EDA
│   ├── 02_Preprocessing_FeatureEngineering.ipynb     # Notebook interaktif Preprocessing & Unit Checks
│   ├── 03_Stratified_Train_Test_Split.ipynb          # Notebook interaktif Stratified Split & Balance Check
│   └── 04_Baseline_KNN.ipynb                         # Notebook interaktif Baseline KNN (E0 & E1)
│
├── outputs/
│   ├── eda/                                          # Tabel statistik & 5 grafik EDA
│   ├── preprocessing/                                # Tabel parameter kuantil, unit checks, & 3 grafik
│   ├── split/                                        # Uji KS/Chi2 covariate balance, ringkasan CV, & 2 grafik
│   └── baseline_knn/                                 # Hasil 5-Fold CV, Hold-out Test + 95% CI, & 3 grafik
│
├── Desain_Penelitian_DIFW_KNN.md                     # Dokumen desain & protokol penelitian
├── Laporan_EDA_DIFW_KNN.md                           # Laporan Data Understanding & EDA
├── Laporan_Preprocessing_DIFW_KNN.md                 # Laporan Preprocessing & Feature Engineering
├── Laporan_Stratified_Split_DIFW_KNN.md              # Laporan Stratified Split, Balance Check, & 5-Fold CV
├── Laporan_Baseline_KNN_DIFW_KNN.md                  # Laporan Baseline KNN (E0 & E1)
├── README.md
└── requirements.txt                                  # Dependensi Python teruji (Python 3.13.7)
```

---

## Cara Menjalankan

### 1. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 2. Menjalankan Unit & Integration Tests (`tests/`)
```bash
pytest -v
```

### 3. Menjalankan Skrip Python (`.py`)
```bash
# Tahap 1: Data Understanding & EDA
python src/01_eda.py

# Tahap 2: Preprocessing, Feature Engineering, & Distance Unit Checks
python src/02_preprocessing.py

# Tahap 3: Stratified Train-Test Split, Covariate Balance, & 5-Fold CV Setup
python src/03_stratified_split.py

# Tahap 4: Baseline KNN (E0 Standard & E1 Optimized via 5-Fold CV + Hold-out Test)
python src/04_baseline_knn.py
```

### 4. Membuka Notebook Interaktif (`.ipynb`)
Buka file berikut menggunakan VS Code atau JupyterLab (seluruh sel sudah dieksekusi dan menyertakan output tabel serta grafik *inline*):
- `notebooks/01_DIFW_KNN_Mixed_Type_Distance_Triplet_Loss.ipynb`
- `notebooks/01_EDA.ipynb`
- `notebooks/02_Preprocessing_FeatureEngineering.ipynb`
- `notebooks/03_Stratified_Train_Test_Split.ipynb`
- `notebooks/04_Baseline_KNN.ipynb`

---

## Status Tahapan Penelitian

- [x] **Tahap 1:** Data Understanding & Exploratory Data Analysis (`src/01_eda.py`, `notebooks/01_EDA.ipynb`)
- [x] **Tahap 2:** Leak-Free Preprocessing & Feature Engineering (`src/preprocessing.py`, `src/mixed_type_distance.py`, `src/02_preprocessing.py`, `notebooks/02_Preprocessing_FeatureEngineering.ipynb`)
- [x] **Tahap 3:** Stratified Train-Test Split & 5-Fold CV Setup (`src/03_stratified_split.py`, `notebooks/03_Stratified_Train_Test_Split.ipynb`)
- [x] **Tahap 4:** Baseline KNN (`E0` Standard KNN & `E1` Optimized KNN via Stratified 5-Fold CV) (`src/baseline_knn.py`, `src/04_baseline_knn.py`, `notebooks/04_Baseline_KNN.ipynb`)
- [x] **Tahap 5:** Proposed DIFW-KNN (`E2`–`E5`: Mixed-Type Distance, Balanced Triplet Margin Loss, & Cost-Sensitive Decision Rule) (`src/mixed_type_distance.py`, `src/triplet_learning.py`, `src/feature_weight_optimizer.py`, `src/cost_sensitive_decision.py`, `src/difw_knn.py`, `tests/`, `notebooks/01_DIFW_KNN_Mixed_Type_Distance_Triplet_Loss.ipynb`)
- [ ] **Tahap 6:** Evaluasi Akhir Hold-Out Test Lanjutan (Tuning CV DIFW-KNN + Bootstrap CI) & Interpretasi LIME
