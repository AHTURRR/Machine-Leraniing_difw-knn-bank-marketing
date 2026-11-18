# LAPORAN PREPROCESSING & FEATURE ENGINEERING (DIFW-KNN)

**Sumber Data:** `bank-full.csv` (UCI Bank Marketing — 45.211 observasi)  
**Modul Kode:** [`src/preprocessing.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/preprocessing.py) dan [`src/02_preprocessing.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/02_preprocessing.py)

---

## 1. Pembersihan Deterministik & Penghapusan Target Leakage

- **Penghapusan `duration`:** Kolom `duration` dihapus sebelum *split* karena nilainya hanya diketahui setelah panggilan selesai (*post-call leakage*).
- **Pelestarian Kategori `"unknown"`:** Nilai `"unknown"` pada `job`, `education`, `contact`, dan `poutcome` dipertahankan sebagai kategori valid (tidak diimputasi dengan modus).
- **Encoding Target:** Target `y` dipetakan deterministik menjadi `yes -> 1` dan `no -> 0`.

---

## 2. Stratified Train-Test Split (80:20, `random_state=42`)

| subset | n_rows | n_cols_pre_engineering | n_no | n_yes | pct_yes |
| --- | --- | --- | --- | --- | --- |
| Full (Cleaned) | 45211 | 15 | 39922 | 5289 | 11.6985 |
| Train Split (80%) | 36168 | 15 | 31937 | 4231 | 11.6982 |
| Test Split (20%) | 9043 | 15 | 7985 | 1058 | 11.6997 |

Kedua file *split* bersih (sebelum transformasi `pdays` agar *Cross-Validation* per *fold* tetap bebas *leakage*) disimpan di:
- [`data/processed/train_split.csv`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/data/processed/train_split.csv) (`36.168` baris)
- [`data/processed/test_split.csv`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/data/processed/test_split.csv) (`9.043` baris)

---

## 3. Rekayasa Fitur `pdays` Berbasis Training-Only ($M_{\text{pos, train}}$)

Dari subset `pdays > -1` pada data training (`6,584` observasi), diperoleh median training **$M_{\text{pos, train}} = 195.0$ hari**. Fitur `pdays` mentah kemudian dipecah menjadi:
1. `previously_contacted`: `1` jika `pdays > -1`, dan `0` jika `pdays == -1`.
2. `pdays_recency`: $\frac{1}{1 + \text{pdays} / M_{\text{pos, train}}}$ jika `pdays > -1`, dan `0.0` jika `pdays == -1`.

| subset | n_contacted | pct_contacted | recency_min_pos | recency_median_pos | recency_max_pos |
| --- | --- | --- | --- | --- | --- |
| Train | 6584 | 18.2 | 0.1829 | 0.5 | 0.9949 |
| Test | 1673 | 18.5 | 0.1888 | 0.5052 | 0.9949 |

Setelah transformasi ini dan penghapusan `pdays` mentah, terbentuk **16 fitur domain**:
- **6 Fitur Numerik:** `age`, `balance`, `day`, `campaign`, `previous`, `pdays_recency`
- **10 Fitur Kategorikal/Biner:** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`, `previously_contacted`

---

## 4. Parameter Dua Jalur Preprocessor (Training-Only Fit)

### a. Jalur Baseline Euclidean (`E0`, `E1`) — `BaselineEuclideanPreprocessor`
- `StandardScaler` pada 6 fitur numerik + `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` pada 10 fitur kategorikal/biner.
- **Dimensi Matriks Output:** `X_train_euc = (36168, 52)`, `X_test_euc = (9043, 52)` (`52` fitur hasil encoding).

### b. Jalur Proposed Mixed-Type (`E2`–`E5`) — `MixedTypePreprocessor`
- Parameter kuantil $(Q_{05,j}, Q_{95,j})$ dan rentang $R_j = Q_{95,j} - Q_{05,j}$ dihitung **hanya dari training set**:

| feature | q05_train | q95_train | range_q95_q05 |
| --- | --- | --- | --- |
| age | 27.0 | 59.0 | 32.0 |
| balance | -171.0 | 5779.65 | 5950.65 |
| day | 3.0 | 29.0 | 26.0 |
| campaign | 1.0 | 8.0 | 7.0 |
| previous | 0.0 | 3.0 | 3.0 |
| pdays_recency | 0.0 | 0.568513 | 0.568513 |

### Audit Kategori Unseen pada Test Set

| feature | n_train_categories | n_test_categories | n_unseen_in_test | unseen_labels |
| --- | --- | --- | --- | --- |
| job | 12 | 12 | 0 | - |
| marital | 3 | 3 | 0 | - |
| education | 4 | 4 | 0 | - |
| default | 2 | 2 | 0 | - |
| housing | 2 | 2 | 0 | - |
| loan | 2 | 2 | 0 | - |
| contact | 3 | 3 | 0 | - |
| month | 12 | 12 | 0 | - |
| poutcome | 4 | 4 | 0 | - |
| previously_contacted | 2 | 2 | 0 | - |

---

## 5. Verifikasi Kualitas Data & Properti Metrik Jarak

| check_name | status | measured_value |
| --- | --- | --- |
| Finite check (tanpa NaN/Inf pada Euclidean & Mixed) | PASS | euc=True, mixed=True |
| Self-distance d_j(x, x) = 0 | PASS | max_diag=0.0e+00 |
| Simetri d_j(x, z) = d_j(z, x) | PASS | max_diff=0.0e+00 |
| Rentang jarak fitur d_j(x, z) dalam [0, 1] | PASS | [0.0000, 1.0000] |
| Konsistensi batched weighted distance vs exact pairwise | PASS | max_diff=1.1e-16 |

---

## 6. Artefak Visual

- [01_stratified_split_verification.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/preprocessing/figures/01_stratified_split_verification.png) (`outputs/preprocessing/figures/01_stratified_split_verification.png`)
- [02_pdays_engineering_comparison.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/preprocessing/figures/02_pdays_engineering_comparison.png) (`outputs/preprocessing/figures/02_pdays_engineering_comparison.png`)
- [03_quantile_scaling_vs_standard_scaling.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/preprocessing/figures/03_quantile_scaling_vs_standard_scaling.png) (`outputs/preprocessing/figures/03_quantile_scaling_vs_standard_scaling.png`)