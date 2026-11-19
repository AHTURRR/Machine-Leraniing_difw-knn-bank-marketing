# LAPORAN TAHAP 04 — BASELINE KNN (`E0` STANDARD & `E1` OPTIMIZED)

**Modul Kode:** [`src/baseline_knn.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/baseline_knn.py), [`src/04_baseline_knn.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/04_baseline_knn.py)  
**Notebook:** [`notebooks/04_Baseline_KNN.ipynb`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/notebooks/04_Baseline_KNN.ipynb)

---

## 1. Unit Test Ekuivalensi Pencarian Tetangga

Memastikan implementasi cepat berbasis `NearestNeighbors` menghasilkan probabilitas prediksi yang **identik secara matematis** dengan `sklearn.neighbors.KNeighborsClassifier` (toleransi $< 10^{-9}$) pada Fold 0:

| k | weights | max_prob_abs_diff_k | max_dist_trunc_diff | n_boundary_tie_rows_maxk | status |
| --- | --- | --- | --- | --- | --- |
| 5 | uniform | 0.0 | 0.0 | 2 | PASS |
| 5 | distance | 2.220446049250313e-16 | 0.0 | 2 | PASS |
| 21 | uniform | 0.0 | 0.0 | 0 | PASS |
| 21 | distance | 3.3306690738754696e-16 | 0.0 | 0 | PASS |

---

## 2. Hasil Tuning Hyperparameter pada Stratified 5-Fold CV (`36.168` Data Training)

- **Kandidat Terbaik berdasarkan `F1 (yes)` & `MCC` (dipilih sebagai `E1` utama):** `k = 3`, `weights = 'uniform'` -> CV `F1 = 0.3131`, `MCC = 0.2746`, `PR-AUC = 0.2474`, `ROC-AUC = 0.6664`.
- **Kandidat Terbaik berdasarkan `PR-AUC`:** `k = 51`, `weights = 'distance'` -> CV `PR-AUC = 0.3912`, `ROC-AUC = 0.7605`, `F1 = 0.2247`, `MCC = 0.2694`.

| weights | k | accuracy_mean | precision_yes_mean | recall_yes_mean | f1_yes_mean | pr_auc_mean | roc_auc_mean | mcc_mean | g_mean_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| distance | 3 | 0.8753 | 0.4394 | 0.2401 | 0.3104 | 0.2709 | 0.6679 | 0.2622 | 0.4796 |
| distance | 5 | 0.8838 | 0.5078 | 0.2113 | 0.2982 | 0.3105 | 0.6925 | 0.275 | 0.4531 |
| distance | 7 | 0.8878 | 0.5563 | 0.1997 | 0.2938 | 0.3315 | 0.7092 | 0.2862 | 0.4419 |
| distance | 11 | 0.8897 | 0.5944 | 0.1799 | 0.276 | 0.3562 | 0.7271 | 0.2845 | 0.4202 |
| distance | 15 | 0.8902 | 0.6121 | 0.1676 | 0.2631 | 0.3683 | 0.7349 | 0.2802 | 0.4061 |
| distance | 21 | 0.8916 | 0.6453 | 0.1636 | 0.2609 | 0.3787 | 0.7433 | 0.2873 | 0.4019 |
| distance | 31 | 0.8919 | 0.667 | 0.1517 | 0.2471 | 0.3874 | 0.7519 | 0.2829 | 0.3875 |
| distance | 51 | 0.8913 | 0.6788 | 0.1347 | 0.2247 | 0.3912 | 0.7605 | 0.2694 | 0.3654 |
| distance | 71 | 0.8908 | 0.6864 | 0.1229 | 0.2084 | 0.3893 | 0.7642 | 0.259 | 0.349 |
| distance | 101 | 0.8906 | 0.7004 | 0.113 | 0.1945 | 0.3889 | 0.7663 | 0.2517 | 0.3349 |
| uniform | 3 | 0.8799 | 0.4729 | 0.2342 | 0.3131 | 0.2474 | 0.6664 | 0.2746 | 0.4751 |
| uniform | 5 | 0.8863 | 0.5364 | 0.2026 | 0.2938 | 0.2823 | 0.6896 | 0.2804 | 0.4444 |
| uniform | 7 | 0.8893 | 0.5803 | 0.1945 | 0.2913 | 0.3042 | 0.7062 | 0.2912 | 0.4366 |
| uniform | 11 | 0.8902 | 0.6063 | 0.1737 | 0.2699 | 0.3324 | 0.7249 | 0.2835 | 0.4132 |
| uniform | 15 | 0.8905 | 0.6229 | 0.1626 | 0.2579 | 0.3476 | 0.7331 | 0.2794 | 0.4004 |
| uniform | 21 | 0.8918 | 0.6567 | 0.1572 | 0.2536 | 0.3622 | 0.7418 | 0.285 | 0.3942 |
| uniform | 31 | 0.8919 | 0.6756 | 0.1472 | 0.2417 | 0.374 | 0.7512 | 0.281 | 0.3818 |
| uniform | 51 | 0.8913 | 0.6857 | 0.1312 | 0.2201 | 0.3813 | 0.7597 | 0.2676 | 0.3606 |
| uniform | 71 | 0.891 | 0.6995 | 0.1196 | 0.2042 | 0.3816 | 0.7635 | 0.2588 | 0.3444 |
| uniform | 101 | 0.8907 | 0.7121 | 0.1097 | 0.19 | 0.3824 | 0.765 | 0.2508 | 0.33 |

> **Observasi Peneliti (Fenomena Klasik Imbalanced KNN pada Euclidean):**
> Semakin besar nilai $k$, probabilitas lokal $\hat{P}(\text{yes}|x)$ menjadi lebih halus sehingga metrik berbasis ranking (`PR-AUC` dan `ROC-AUC`) meningkat hingga sekitar $k=51$–$101$. Namun pada *decision threshold* standar `0.50`, nilai $k$ besar menyebabkan kelas minoritas (`11,70%`) kalah suara sehingga `Recall (yes)` turun. Ini menjadi bukti empiris awal mengapa penelitian ini membutuhkan **Mixed-Type Distance + Supervised Feature Weighting (`E3`) + Cost-Sensitive Decision Rule (`E4`/`E5`)**.

---

## 3. Hasil Evaluasi pada Hold-out Test Set (`9.043` Observasi)

| model_id | model_name | k | weights | accuracy | precision_yes | recall_yes | f1_yes | pr_auc | roc_auc | mcc | g_mean | fit_sec | predict_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | Standard KNN | 5 | uniform | 0.8884 | 0.565 | 0.2013 | 0.2969 | 0.2921 | 0.7014 | 0.2907 | 0.4441 | 0.3311 | 0.9565 |
| E1 | Optimized KNN | 3 | uniform | 0.8816 | 0.4876 | 0.242 | 0.3234 | 0.2619 | 0.6779 | 0.2863 | 0.4835 | 0.2901 | 0.6834 |
| E1-PR | Optimized KNN (Best CV PR-AUC) | 51 | distance | 0.8924 | 0.7094 | 0.1361 | 0.2284 | 0.4042 | 0.7656 | 0.2793 | 0.3676 | 0.3148 | 0.7326 |

### Interval Kepercayaan Bootstrap 95% (`n_boot = 1.000`)

| model_id | config | accuracy | precision_yes | recall_yes | f1_yes | pr_auc | roc_auc | mcc | g_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | k=5, uniform | 0.8884 [0.8823, 0.8946] | 0.5650 [0.5153, 0.6121] | 0.2013 [0.1775, 0.2247] | 0.2969 [0.2658, 0.3257] | 0.2921 [0.2651, 0.3184] | 0.7014 [0.6858, 0.7186] | 0.2907 [0.2584, 0.3204] | 0.4441 [0.4170, 0.4692] |
| E1 | k=3, uniform | 0.8816 [0.8750, 0.8875] | 0.4876 [0.4465, 0.5269] | 0.2420 [0.2167, 0.2671] | 0.3234 [0.2934, 0.3502] | 0.2619 [0.2398, 0.2866] | 0.6779 [0.6626, 0.6941] | 0.2863 [0.2550, 0.3145] | 0.4835 [0.4575, 0.5079] |
| E1-PR | k=51, distance | 0.8924 [0.8862, 0.8983] | 0.7094 [0.6473, 0.7747] | 0.1361 [0.1151, 0.1565] | 0.2284 [0.1970, 0.2591] | 0.4042 [0.3729, 0.4383] | 0.7656 [0.7478, 0.7827] | 0.2793 [0.2460, 0.3135] | 0.3676 [0.3380, 0.3941] |

### Confusion Matrix Hold-out Test

| model_id | model_name | k | weights | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | Standard KNN | 5 | uniform | 7821 | 164 | 845 | 213 |
| E1 | Optimized KNN | 3 | uniform | 7716 | 269 | 802 | 256 |
| E1-PR | Optimized KNN (Best CV PR-AUC) | 51 | distance | 7926 | 59 | 914 | 144 |

> **Ringkasan Kinerja Baseline (`E0` vs `E1`):**
> - **`E0 (Standard KNN, k=5, uniform)`:** `Accuracy = 0.8884`, `Precision = 0.5650`, `Recall = 0.2013`, `F1 = 0.2969`, `PR-AUC = 0.2921`, `ROC-AUC = 0.7014`, `MCC = 0.2907`, `G-Mean = 0.4441`.
> - **`E1 (Optimized KNN, k=3, uniform)`:** `Accuracy = 0.8816`, `Precision = 0.4876`, `Recall = 0.2420`, `F1 = 0.3234`, `PR-AUC = 0.2619`, `ROC-AUC = 0.6779`, `MCC = 0.2863`, `G-Mean = 0.4835`.

---

## 4. Artefak Visual

- [01_cv_hyperparameter_tuning.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/baseline_knn/figures/01_cv_hyperparameter_tuning.png) (`outputs/baseline_knn/figures/01_cv_hyperparameter_tuning.png`)
- [02_e0_vs_e1_roc_pr_curves.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/baseline_knn/figures/02_e0_vs_e1_roc_pr_curves.png) (`outputs/baseline_knn/figures/02_e0_vs_e1_roc_pr_curves.png`)
- [03_e0_vs_e1_confusion_and_metrics.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/baseline_knn/figures/03_e0_vs_e1_confusion_and_metrics.png) (`outputs/baseline_knn/figures/03_e0_vs_e1_confusion_and_metrics.png`)