# LAPORAN DATA UNDERSTANDING & EXPLORATORY DATA ANALYSIS (EDA)

**Dataset:** `bank-full.csv` (UCI Bank Marketing Dataset — ID 222)  
**Tujuan:** Mendukung desain algoritma *Domain-Informed Feature-Weighted KNN* (`DIFW-KNN`) pada skenario prediksi pra-panggilan (*pre-call telemarketing prediction*).

---

## 1. Profil Umum & Kualitas Data (*Data Understanding*)

- **Dimensi Dataset:** `45,211` observasi × `17` kolom (`16` prediktor + 1 target `y`).
- **Komposisi Tipe Fitur:** `7` fitur numerik (`age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`) dan `9` fitur kategorikal (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`).
- **Missing Values Eksplisit (`NaN`):** `0` sel kosong.
- **Baris Duplikat Eksak:** `0` baris.
- **Distribusi Kelas Target (`y`):**
  - Kelas negatif (`no`): `39,922` (`88.30%`)
  - Kelas positif (`yes`): `5,289` (`11.70%`)
  - **Imbalance Ratio:** `7.55 : 1` (kelas minoritas `yes` sebesar `11.70%`).

### Ringkasan Kategori `unknown` (Implicit Missing Values)

Tidak terdapat nilai `NaN`, namun terdapat nilai string `"unknown"` pada 4 fitur kategorikal:

| feature | n_unique | n_unknown | pct_unknown | mode_category | mode_pct |
| --- | --- | --- | --- | --- | --- |
| job | 12 | 288 | 0.64 | blue-collar | 21.53 |
| education | 4 | 1857 | 4.11 | secondary | 51.32 |
| contact | 3 | 13020 | 28.8 | cellular | 64.77 |
| poutcome | 4 | 36959 | 81.75 | unknown | 81.75 |

> **Catatan Peneliti:** Pada `poutcome` (`81.75%` `unknown`) dan `contact` (`28.80%` `unknown`), label `"unknown"` membawa sinyal informatif yang kuat (misalnya ketiadaan riwayat kampanye sebelumnya atau tipe kanal lama), sehingga **tidak boleh diimputasi dengan modus atau dihapus**. Label `"unknown"` wajib dipertahankan sebagai kategori valid.

---

## 2. Profil Statistik Fitur Numerik & Implikasi *Mixed-Type Distance*

| feature | mean | std | min | q05 | median | q95 | max | q95_minus_q05 | skewness | pct_outliers_iqr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| age | 40.9362 | 10.6188 | 18.0 | 27.0 | 39.0 | 59.0 | 95.0 | 32.0 | 0.6848 | 1.08 |
| balance | 1362.2721 | 3044.7658 | -8019.0 | -172.0 | 448.0 | 5768.0 | 102127.0 | 5940.0 | 8.3603 | 10.46 |
| day | 15.8064 | 8.3225 | 1.0 | 3.0 | 16.0 | 29.0 | 31.0 | 26.0 | 0.0931 | 0.0 |
| duration | 258.1631 | 257.5278 | 0.0 | 35.0 | 180.0 | 751.0 | 4918.0 | 716.0 | 3.1443 | 7.16 |
| campaign | 2.7638 | 3.098 | 1.0 | 1.0 | 2.0 | 8.0 | 63.0 | 7.0 | 4.8987 | 6.78 |
| pdays | 40.1978 | 100.1287 | -1.0 | -1.0 | -1.0 | 317.0 | 871.0 | 318.0 | 2.6157 | 18.26 |
| previous | 0.5803 | 2.3034 | 0.0 | 0.0 | 0.0 | 3.0 | 275.0 | 3.0 | 41.8465 | 18.26 |

### Perbandingan Fitur Numerik Berdasarkan Kelas Target (`y=yes` vs `y=no`)

| feature | mean_no | median_no | mean_yes | median_yes | point_biserial_r | rank_biserial_r | single_feature_auc | mw_pvalue |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| age | 40.839 | 39.0 | 41.6701 | 38.0 | 0.0252 | -0.0157 | 0.5079 | 0.06281791018697587 |
| balance | 1303.715 | 417.0 | 1804.2679 | 733.0 | 0.0528 | 0.1801 | 0.5901 | 6.593845808689515e-101 |
| day | 15.8923 | 16.0 | 15.1583 | 15.0 | -0.0283 | -0.053 | 0.5265 | 3.326078633588937e-10 |
| duration | 221.1828 | 164.0 | 537.2946 | 426.0 | 0.3945 | 0.6152 | 0.8076 | 0.0 |
| campaign | 2.8464 | 2.0 | 2.141 | 2.0 | -0.0732 | -0.1447 | 0.5723 | 1.9484904873905108e-71 |
| pdays | 36.4214 | -1.0 | 68.703 | -1.0 | 0.1036 | 0.1864 | 0.5932 | 2.4841176503776935e-235 |
| previous | 0.5022 | 0.0 | 1.1704 | 0.0 | 0.0932 | 0.2046 | 0.6023 | 3.491824260414992e-283 |

> **Temuan Kritis untuk Formulasi Jarak (`DIFW-KNN`):**
> 1. Fitur `balance` (skewness `8.3603`), `campaign`, dan `previous` sangat *right-skewed* serta memiliki outlier ekstrem. Pengukuran jarak Euclidean/Min-Max standar akan terkompresi oleh nilai ekstrem (`balance` max `102,127` vs `Q95-Q05` sebesar `5,940.0`). Oleh karena itu, penggunaan **normalisasi interkuantil terklip $\min(1, |x_j - z_j| / (Q_{95,j} - Q_{05,j}))$** pada *Mixed-Type Distance* terbukti sangat tepat secara empiris.
> 2. Perhatikan `pdays` pada tabel profil global: karena `81.74%` bernilai `-1`, nilai `Q05` dan `median` bernilai `-1`, sehingga **`Q95 - Q05` global terdistorsi oleh mayoritas nilai sentinel `-1`**, seperti dijelaskan lebih lanjut pada Bagian 3.

---

## 3. Analisis Khusus `pdays` (Bukti Empiris Perlunya Rekayasa Fitur)

- **Proporsi `pdays = -1` (belum pernah dihubungi):** `36,954` observasi (**`81.74%`**).
- **Proporsi `pdays > -1` (pernah dihubungi sebelumnya):** `8,257` observasi (**`18.26%`**), dengan `pdays = 0` sebanyak `0` observasi (minimum pada subset pernah dihubungi adalah `1` hari).
- **Konsistensi Lintas Fitur:** Dari seluruh baris dengan `pdays = -1`, `100.0%` memiliki `previous = 0` dan `100.0%` memiliki `poutcome = 'unknown'`. Sebaliknya, pada baris `pdays > -1`, `100.0%` memiliki `previous > 0`.
- **Distribusi `pdays` KHUSUS Subset Pernah Dihubungi (`pdays > -1`):**
  - Mean = `224.58` hari, Std = `115.34` hari
  - `Q05` = `79.0`, Median = `194.0`, `Q95` = `370.0`, Max = `871`
  - Rentang `Q95 - Q05` pada subset pernah dihubungi = **`291.0`** hari (berbeda jauh dari `Q95 - Q05` global = `318.0`).
- **Dampak terhadap Tingkat Konversi (`y=yes`):**
  - Belum pernah dihubungi (`pdays = -1`): **`9.16%`**
  - Pernah dihubungi (`pdays > -1`): **`23.07%`** (`2.52x` lebih tinggi)

| pdays_bucket | count | yes_count | yes_rate_pct | mean_pdays | pct_total |
| --- | --- | --- | --- | --- | --- |
| Belum pernah (-1) | 36954 | 3384 | 9.16 | -1.0 | 81.74 |
| 1-90 hari | 718 | 251 | 34.96 | 57.79 | 1.59 |
| 91-180 hari | 2480 | 680 | 27.42 | 130.53 | 5.49 |
| 181-365 hari | 4416 | 786 | 17.8 | 274.38 | 9.77 |
| > 365 hari | 643 | 188 | 29.24 | 431.54 | 1.42 |

> **Keputusan Metodologis (`Preprocessing`):**
> Jika `pdays = -1` langsung diganti menjadi `pdays_recency = 0` seperti draf awal, nasabah yang baru dihubungi `1` hari lalu menjadi bertetangga dekat dengan nasabah yang belum pernah dihubungi (`0`). Sebaliknya, pada desain perbaikan: (1) buat fitur biner `previously_contacted` (`1` jika `pdays > -1`, `0` jika `-1`), dan (2) definisikan `pdays_recency = 1 / (1 + pdays)` jika pernah dihubungi dan `0` jika belum pernah dihubungi (sehingga `0` = tidak pernah dihubungi, mendekati `1` = baru saja dihubungi, sementara efek non-monoton pada kontak `> 365` hari tetap dapat tertangkap oleh ketetanggaan lokal KNN). Rentang `Q95 - Q05` untuk `pdays_recency` harus diperiksa dari fold training.

---

## 4. Analisis Fitur `duration` (Bukti Empiris *Target Leakage*)

- **Korelasi & Daya Pisah Tunggal:** Fitur `duration` sendiri menghasilkan **ROC-AUC = `0.8076`** dan Point-Biserial $r$ = `0.3945`.
- **Median Durasi:** Kelas `no` = `164.0` detik (mean `221.18` dtk) vs Kelas `yes` = `426.0` detik (mean `537.29` dtk).
- **Durasi = 0 detik:** Terdapat `3` observasi, dengan `0` berlabel `yes`.

| duration_bucket | count | yes_count | yes_rate_pct | pct_total |
| --- | --- | --- | --- | --- |
| 0-60 dtk | 4766 | 9 | 0.19 | 10.54 |
| 61-120 dtk | 9278 | 202 | 2.18 | 20.52 |
| 121-180 dtk | 8616 | 498 | 5.78 | 19.06 |
| 181-300 dtk | 10277 | 1122 | 10.92 | 22.73 |
| 301-600 dtk | 8484 | 1625 | 19.15 | 18.77 |
| 601-1200 dtk | 3230 | 1487 | 46.04 | 7.14 |
| > 1200 dtk | 560 | 346 | 61.79 | 1.24 |

> **Validasi Keputusan Desain:** Tingkat konversi melonjak dari `0.19%` (`0-60 dtk`) menjadi `61.79%` (`> 1200 dtk`) karena durasi panggilan adalah akibat langsung dari ketertarikan nasabah selama percakapan berlangsung. Untuk tujuan *pre-call targeting* (memilih siapa yang harus ditelepon sebelum panggilan dilakukan), **fitur `duration` wajib dihapus sepenuhnya sebelum pemodelan**.

---

## 5. Kekuatan Diskriminatif Fitur & Tingkat Konversi Kategorikal

### Peringkat Mutual Information (Semua Prediktor Asli)

| feature | feature_type | is_leakage | mutual_information |
| --- | --- | --- | --- |
| duration | Numerik | True | 0.0709 |
| poutcome | Kategorikal | False | 0.0294 |
| pdays | Numerik | False | 0.02748 |
| month | Kategorikal | False | 0.02435 |
| balance | Numerik | False | 0.02234 |
| contact | Kategorikal | False | 0.01363 |
| age | Numerik | False | 0.01332 |
| previous | Numerik | False | 0.01106 |
| housing | Kategorikal | False | 0.00965 |
| job | Kategorikal | False | 0.00826 |
| day | Numerik | False | 0.00693 |
| campaign | Numerik | False | 0.00656 |
| loan | Kategorikal | False | 0.00263 |
| education | Kategorikal | False | 0.0026 |
| marital | Kategorikal | False | 0.0021 |
| default | Kategorikal | False | 0.00029 |

### Uji Asosiasi Chi-Square & Cramér's V Fitur Kategorikal

| feature | n_categories | chi2 | dof | p_value | cramers_v |
| --- | --- | --- | --- | --- | --- |
| poutcome | 4 | 4391.51 | 3 | 0.0 | 0.3117 |
| month | 12 | 3061.84 | 11 | 0.0 | 0.2602 |
| contact | 3 | 1035.71 | 2 | 1.251738325340638e-225 | 0.1514 |
| housing | 2 | 874.82 | 1 | 2.918797605076633e-192 | 0.1391 |
| job | 12 | 836.11 | 11 | 3.337121944935502e-172 | 0.136 |
| education | 4 | 238.92 | 3 | 1.6266562124072994e-51 | 0.0727 |
| loan | 2 | 209.62 | 1 | 1.665061163492756e-47 | 0.0681 |
| marital | 3 | 196.5 | 2 | 2.1450999986791792e-43 | 0.0659 |
| default | 2 | 22.2 | 1 | 2.4538606753508344e-06 | 0.0222 |

### Rincian Konversi pada Fitur Kategorikal Berpengaruh

**a. `poutcome` (Hasil Kampanye Sebelumnya)**

| category | count | pct_total | yes_count | yes_rate_pct | lift_vs_baseline |
| --- | --- | --- | --- | --- | --- |
| success | 1511 | 3.34 | 978 | 64.73 | 5.533 |
| other | 1840 | 4.07 | 307 | 16.68 | 1.426 |
| failure | 4901 | 10.84 | 618 | 12.61 | 1.078 |
| unknown | 36959 | 81.75 | 3386 | 9.16 | 0.783 |

**b. `contact` (Kanal Komunikasi)**

| category | count | pct_total | yes_count | yes_rate_pct | lift_vs_baseline |
| --- | --- | --- | --- | --- | --- |
| cellular | 29285 | 64.77 | 4369 | 14.92 | 1.275 |
| telephone | 2906 | 6.43 | 390 | 13.42 | 1.147 |
| unknown | 13020 | 28.8 | 530 | 4.07 | 0.348 |

**c. `month` (Bulan Kontak Terakhir)**

| category | count | pct_total | yes_count | yes_rate_pct | lift_vs_baseline |
| --- | --- | --- | --- | --- | --- |
| mar | 477 | 1.06 | 248 | 51.99 | 4.444 |
| dec | 214 | 0.47 | 100 | 46.73 | 3.994 |
| sep | 579 | 1.28 | 269 | 46.46 | 3.971 |
| oct | 738 | 1.63 | 323 | 43.77 | 3.741 |
| apr | 2932 | 6.49 | 577 | 19.68 | 1.682 |
| feb | 2649 | 5.86 | 441 | 16.65 | 1.423 |
| aug | 6247 | 13.82 | 688 | 11.01 | 0.941 |
| jun | 5341 | 11.81 | 546 | 10.22 | 0.874 |
| nov | 3970 | 8.78 | 403 | 10.15 | 0.868 |
| jan | 1403 | 3.1 | 142 | 10.12 | 0.865 |
| jul | 6895 | 15.25 | 627 | 9.09 | 0.777 |
| may | 13766 | 30.45 | 925 | 6.72 | 0.574 |

**d. `job` (Jenis Pekerjaan)**

| category | count | pct_total | yes_count | yes_rate_pct | lift_vs_baseline |
| --- | --- | --- | --- | --- | --- |
| student | 938 | 2.07 | 269 | 28.68 | 2.451 |
| retired | 2264 | 5.01 | 516 | 22.79 | 1.948 |
| unemployed | 1303 | 2.88 | 202 | 15.5 | 1.325 |
| management | 9458 | 20.92 | 1301 | 13.76 | 1.176 |
| admin. | 5171 | 11.44 | 631 | 12.2 | 1.043 |
| self-employed | 1579 | 3.49 | 187 | 11.84 | 1.012 |
| unknown | 288 | 0.64 | 34 | 11.81 | 1.009 |
| technician | 7597 | 16.8 | 840 | 11.06 | 0.945 |
| services | 4154 | 9.19 | 369 | 8.88 | 0.759 |
| housemaid | 1240 | 2.74 | 109 | 8.79 | 0.751 |
| entrepreneur | 1487 | 3.29 | 123 | 8.27 | 0.707 |
| blue-collar | 9732 | 21.53 | 708 | 7.27 | 0.622 |

> **Implikasi untuk *Feature Weighting* (`DIFW-KNN`):**
> Di luar `duration` (yang dihapus), prediktor pra-panggilan terkuat menurut Mutual Information adalah `poutcome`, `pdays`, `month`, `balance`, `contact`. Sementara fitur dengan informasi marginal kecil akan menambah noise pada jarak seragam. Hal ini memvalidasi hipotesis bahwa **KNN dengan bobot seragam (`E0`–`E2`) rentan terdilusi oleh fitur berdaya pisah rendah**, sehingga *Supervised Triplet Feature Weighting* (`E3`, `E5`) diharapkan memberikan bobot lebih besar pada fitur relevan dan menekan fitur *noise*.

---

## 6. Ringkasan Artefak Visual & Kesiapan Tahap Berikutnya

Seluruh grafik dan tabel mentah telah disimpan di folder `outputs/eda/`:
- [01_target_and_unknown_distribution.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/eda/figures/01_target_and_unknown_distribution.png) (`outputs/eda/figures/01_target_and_unknown_distribution.png`)
- [02_numeric_distributions_and_outliers.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/eda/figures/02_numeric_distributions_and_outliers.png) (`outputs/eda/figures/02_numeric_distributions_and_outliers.png`)
- [03_pdays_and_duration_leakage.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/eda/figures/03_pdays_and_duration_leakage.png) (`outputs/eda/figures/03_pdays_and_duration_leakage.png`)
- [04_feature_discriminative_power.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/eda/figures/04_feature_discriminative_power.png) (`outputs/eda/figures/04_feature_discriminative_power.png`)
- [05_categorical_conversion_rates.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/eda/figures/05_categorical_conversion_rates.png) (`outputs/eda/figures/05_categorical_conversion_rates.png`)

### Kontrak *Preprocessing* Menuju Tahap Eksperimen (`02_preprocessing.py`)
1. **Deterministic Cleaning (Pre-Split Safe):** Hapus kolom `duration`; pertahankan kategori `"unknown"`; buat fitur `previously_contacted` (`1` jika `pdays > -1`, `0` jika `-1`) dan `pdays_recency = 1 / (1 + pdays)` untuk `pdays > -1` (`0` jika `-1`), lalu hapus `pdays` mentah.
2. **Stratified Split (80:20):** Bagi data `45,211` baris menjadi `36,168` data latih dan `9,043` data uji (`random_state=42`) dengan proporsi `yes = 11.70%` identik di kedua subset.
3. **Training-Only Parameter Fitting:** Seluruh parameter `StandardScaler`/`OneHotEncoder` (jalur Euclidean `E0`–`E1`) dan kuantil `(Q05, Q95)` (jalur Mixed-Type `E2`–`E5`) hanya dihitung dari data latih (`36,168` baris atau per *fold* CV), tidak pernah menyentuh *test set*.