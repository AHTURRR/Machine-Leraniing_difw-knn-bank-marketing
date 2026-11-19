# LAPORAN TAHAP 03 — STRATIFIED TRAIN-TEST SPLIT & 5-FOLD CV SETUP

**Skrip:** [`src/03_stratified_split.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/03_stratified_split.py)  
**Notebook:** [`notebooks/03_Stratified_Train_Test_Split.ipynb`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/notebooks/03_Stratified_Train_Test_Split.ipynb)

---

## 1. Pembagian Stratified Train-Test Split (80:20, `random_state=42`)

| subset | n_rows | n_no | n_yes | pct_yes |
| --- | --- | --- | --- | --- |
| Full (Cleaned, Pre-Split) | 45211 | 39922 | 5289 | 11.6985 |
| Training / Development (80%) | 36168 | 31937 | 4231 | 11.6982 |
| Hold-out Test (20%) | 9043 | 7985 | 1058 | 11.6997 |

Artefak split kanonik tersimpan di:
- [`data/processed/train_split.csv`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/data/processed/train_split.csv) (`36.168` observasi — Training / Development Data)
- [`data/processed/test_split.csv`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/data/processed/test_split.csv) (`9.043` observasi — Hold-out Test Data, dikunci hingga evaluasi akhir)

---

## 2. Uji Keseimbangan Distribusi Fitur (*Covariate Balance Check*)

Dilakukan uji **Kolmogorov-Smirnov (KS)** pada `6` fitur numerik dan **Chi-Square Homogeneity** pada `9` fitur kategorikal untuk memastikan distribusi kovariat antara Training dan Hold-out Test setara:
- **Fitur Numerik seimbang ($p \ge 0,05$):** `6/6` (p-value minimum = `0.2824`)
- **Fitur Kategorikal seimbang ($p \ge 0,05$):** `9/9` (p-value minimum = `0.0651`)

### a. Uji Kolmogorov-Smirnov (Fitur Numerik)

| feature | train_mean | test_mean | train_median | test_median | ks_stat | p_value | balanced_at_alpha_0_05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| age | 40.893 | 41.109 | 39.0 | 39.0 | 0.0116 | 0.2824 | True |
| balance | 1365.4934 | 1349.388 | 451.0 | 437.0 | 0.0111 | 0.3328 | True |
| day | 15.818 | 15.7603 | 16.0 | 16.0 | 0.00734 | 0.8281 | True |
| campaign | 2.7639 | 2.7635 | 2.0 | 2.0 | 0.00251 | 1.0 | True |
| pdays | 40.1572 | 40.3602 | -1.0 | -1.0 | 0.00499 | 0.9934 | True |
| previous | 0.5817 | 0.5747 | 0.0 | 0.0 | 0.00496 | 0.9938 | True |

### b. Uji Chi-Square Homogeneity (Fitur Kategorikal)

| feature | n_categories | chi2_stat | dof | p_value | balanced_at_alpha_0_05 |
| --- | --- | --- | --- | --- | --- |
| job | 12 | 10.5001 | 11 | 0.486 | True |
| marital | 3 | 0.0354 | 2 | 0.9825 | True |
| education | 4 | 3.3715 | 3 | 0.3378 | True |
| default | 2 | 0.1571 | 1 | 0.6918 | True |
| housing | 2 | 3.4017 | 1 | 0.0651 | True |
| loan | 2 | 0.312 | 1 | 0.5764 | True |
| contact | 3 | 0.8549 | 2 | 0.6522 | True |
| month | 12 | 18.0328 | 11 | 0.0808 | True |
| poutcome | 4 | 2.0035 | 3 | 0.5717 | True |

---

## 3. Penyiapan Stratified 5-Fold Cross-Validation (`Training / Development`)

Untuk menjaga perbandingan yang **100% adil** antara seluruh model (`E0`–`E5`), indeks lipatan *Stratified 5-Fold CV* (`shuffle=True`, `random_state=42`) disimpan secara eksplisit pada [`data/processed/train_cv_folds.csv`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/data/processed/train_cv_folds.csv).

| fold_id | n_train_fold | n_val_fold | n_train_yes | pct_train_yes | n_val_yes | pct_val_yes |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 28934.0 | 7234.0 | 3385.0 | 11.699 | 846.0 | 11.6948 |
| 1.0 | 28934.0 | 7234.0 | 3385.0 | 11.699 | 846.0 | 11.6948 |
| 2.0 | 28934.0 | 7234.0 | 3384.0 | 11.6956 | 847.0 | 11.7086 |
| 3.0 | 28935.0 | 7233.0 | 3385.0 | 11.6986 | 846.0 | 11.6964 |
| 4.0 | 28935.0 | 7233.0 | 3385.0 | 11.6986 | 846.0 | 11.6964 |

---

## 4. Visualisasi Validasi Split

- [01_train_test_and_5fold_balance.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/split/figures/01_train_test_and_5fold_balance.png) (`outputs/split/figures/01_train_test_and_5fold_balance.png`)
- [02_covariate_distribution_alignment.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/split/figures/02_covariate_distribution_alignment.png) (`outputs/split/figures/02_covariate_distribution_alignment.png`)