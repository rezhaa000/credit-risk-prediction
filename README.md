# 💳 Credit Risk Prediction

Sistem prediksi risiko kredit untuk memprediksi probabilitas seseorang mengalami financial distress dalam 2 tahun ke depan, menggunakan LightGBM dengan SHAP analysis dan KS-Statistic untuk evaluasi model.

> **Catatan:** Project ini merupakan bagian dari group final project. Notebook ini dikerjakan sebagai **Data Scientist** dan menjadi foundation utama dari notebook final kelompok, termasuk pipeline feature engineering, SHAP analysis, dan KS-Statistic yang diadopsi ke notebook final.
>
> Kolaborator: [mnabilp](https://github.com/mnabilp) *(Data Scientist)*

---

## 📋 Daftar Isi
- [Latar Belakang](#latar-belakang)
- [Dataset](#dataset)
- [Struktur Repository](#struktur-repository)
- [Alur Project](#alur-project)
- [Hasil Model](#hasil-model)
- [SHAP Analysis](#shap-analysis)
- [KS-Statistic](#ks-statistic)
- [Tools & Libraries](#tools--libraries)

---

## 📌 Latar Belakang

Lembaga keuangan menghadapi risiko kerugian besar akibat pemberian kredit kepada nasabah yang berpotensi gagal bayar. Sistem prediksi risiko kredit yang akurat memungkinkan lembaga keuangan untuk mengidentifikasi peminjam berisiko tinggi sejak awal, mengurangi kerugian akibat default, dan mengambil keputusan pemberian kredit yang lebih tepat.

---

## 🗃️ Dataset

- **Sumber:** [Give Me Some Credit — Kaggle Competition](https://www.kaggle.com/competitions/GiveMeSomeCredit)
- **Total baris:** 150.000 (training)
- **Total fitur:** 11
- **Target:** `SeriousDlqin2yrs` — apakah seseorang akan mengalami financial distress dalam 2 tahun

**Kondisi dataset:**
- Class imbalance: hanya **6.7%** data positif (default)
- Missing values: `MonthlyIncome` (29.731) dan `NumberOfDependents` (3.924)
- Outlier ekstrem: `RevolvingUtilization` max 50.708, `DebtRatio` max 329.664

---

## 📁 Struktur Repository

```
├── credit_risk_training.ipynb      # Notebook training utama
├── credit_risk_inference.ipynb     # Notebook inference & risk categorization
├── inference_results.csv           # Hasil prediksi pada data test
└── cs-training.csv                 # Dataset training
```

---

## 🔄 Alur Project

### 1. Exploratory Data Analysis
- Analisis distribusi target → class imbalance 93.3% vs 6.7%
- Deteksi sentinel values (96, 98) pada kolom late payment
- Analisis korelasi fitur terhadap target → WeightedLateScore tertinggi (0.3369)
- Perbandingan distribusi default vs non-default per fitur

### 2. Feature Engineering

**Handling Missing Values:**
- `MonthlyIncome` → imputasi median
- `NumberOfDependents` → imputasi median

**Handling Outlier:**
- Winsorization pada `RevolvingUtilization`, `DebtRatio`, `MonthlyIncome` (persentil 1%–99%)
- Log transform pada `RevolvingUtilization`
- Capping fitur count pada rentang 0–10 (menghilangkan sentinel 96 & 98)
- Filter usia tidak valid (hanya 18–100 tahun)

**Feature Creation:**

| Fitur Baru | Formula | Alasan |
|---|---|---|
| `WeightedLateScore` | (30-59d × 1) + (60-89d × 2) + (90+d × 4) | Menimbang keparahan keterlambatan |
| `LogIncome` | log1p(MonthlyIncome) | Mengurangi skewness ekstrem |
| `DisposableIncome` | MonthlyIncome × (1 − DebtRatio) | Estimasi pendapatan bersih |
| `LogDebtRatio` | log1p(DebtRatio) | Mengurangi skewness ekstrem |

### 3. Model Training

5 model dilatih dan dibandingkan:

| Model | AUC Train | AUC Test | Recall Default |
|---|---|---|---|
| Logistic Regression | - | 0.8565 | - |
| Random Forest | 0.9996 | 0.8353 | 15% ❌ |
| XGBoost | - | 0.8466 | 70% |
| **LightGBM** | - | **0.8645** | **77%** ✅ |
| ANN (MLP) | - | 0.8652 | 81% |

### 4. Hyperparameter Tuning

Model terpilih **LightGBM** karena keseimbangan terbaik antara AUC dan Recall. Setelah tuning dengan threshold 0.3:
- **Recall Default: ~89%** — 89% orang yang akan default berhasil terdeteksi
- Prioritas recall karena di industri kredit lebih baik menolak beberapa orang aman daripada meloloskan satu orang yang akan default

---

## 📊 Hasil Model

| Metrik | Score |
|---|---|
| ROC-AUC | **0.865** |
| Recall Default | **89%** (threshold 0.3) |
| KS-Statistic | **0.5779** |

---

## 🔍 SHAP Analysis

SHAP (SHapley Additive exPlanations) digunakan untuk menginterpretasi keputusan model secara global maupun per individu.

**Fitur paling berpengaruh (Summary Plot):**
1. **DebtRatio** — DebtRatio tinggi mendorong risiko default meningkat signifikan
2. **WeightedLateScore** — Pengaruh paling ekstrem, SHAP value bisa mencapai +4.0
3. **age** — Usia lebih tua justru menurunkan risiko (SHAP negatif)
4. **LogIncome** — Hampir tidak berpengaruh, mengkonfirmasi keputusan untuk tidak memprioritaskan fitur ini

**Contoh Waterfall Plot (individu berisiko tinggi):**
- `WeightedLateScore = 16` → kontribusi **+3.05** (paling dominan)
- `DebtRatio = 1.719` → kontribusi **+1.44**
- Base value: -0.965 → Final f(x): **3.753** → model sangat yakin **Default**

### 5. KS-Statistic

KS-Statistic mengukur kemampuan model memisahkan antara nasabah default dan non-default.

**Hasil:**
- **KS Statistic: 0.5779** — separasi model sangat kuat untuk credit risk
- Puncak KS pada **desil ke-2** — model paling tajam memisahkan default vs non-default pada 20–30% populasi berisiko tertinggi

**Kategorisasi risiko berdasarkan KS threshold:**

| Kategori | Keterangan |
|---|---|
| 🔴 High Risk | Probabilitas default di atas threshold KS |
| 🟡 Medium Risk | Probabilitas default di area transisi |
| 🟢 Low Risk | Probabilitas default di bawah threshold |

---

## 🛠️ Tools & Libraries

| Kategori | Tools |
|---|---|
| Data Processing | Pandas, NumPy |
| Visualisasi | Matplotlib, Seaborn, Missingno |
| Machine Learning | Scikit-Learn, XGBoost, LightGBM, feature-engine |
| Deep Learning | TensorFlow, Keras |
| Explainability | SHAP |
| Statistik | SciPy |
| Model Saving | Joblib, Pickle |

---

## 👤 Author

**Rezha Aulia**
Hacktiv8 Data Science Bootcamp — Batch FTDS-037-HCK
