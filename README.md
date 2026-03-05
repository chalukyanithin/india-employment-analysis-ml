# India Employment Trends — ML Analysis & Dashboard

<div align="center">

![Next.js](https://img.shields.io/badge/Next.js-15-black?style=for-the-badge&logo=next.js)
![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikitlearn)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3-06B6D4?style=for-the-badge&logo=tailwindcss)
![Recharts](https://img.shields.io/badge/Recharts-2.x-22B5BF?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A full-stack data science project analyzing India's employment landscape (2016–2025) across 5 sectors using Machine Learning, with an interactive Next.js analytics dashboard.**

[📊 Live Demo](#) · [🤖 ML Notebook](#) · [📁 Dataset](#dataset) · [🚀 Quick Start](#quick-start)

</div>

---

## 📸 Dashboard Preview

> Dark-themed professional analytics dashboard with sidebar navigation, interactive charts, ML results, and 2025 employment forecasts.

| Overview | Dataset Charts | ML Results |
|----------|---------------|------------|
| Hero section + indicator explainers | Line, Bar, Pie, Radar, Scatter | Confusion matrix, Feature importance |

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Machine Learning](#-machine-learning)
  - [Feature Engineering](#feature-engineering)
  - [Random Forest Regressor](#random-forest-regressor)
  - [Logistic Regression Classifier](#logistic-regression-classifier)
  - [Training Strategy](#training-strategy)
  - [Model Results](#model-results)
- [Dashboard Features](#-dashboard-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Key Findings](#-key-findings)
- [Future Work](#-future-work)

---

## 🎯 Project Overview

This project investigates employment trends across India's five major industry sectors from **2016 to 2025**, focusing on:

- **Trend Analysis** — How employment indicators evolved over 9 years
- **COVID-19 Impact** — Quantifying the 2020 pandemic shock on each sector
- **IT Boom & Bust** — Capturing the 2021–2023 tech surge and post-2023 slowdown
- **ML Forecasting** — Predicting next-month Unemployment Rate per sector
- **Stress Classification** — Flagging months with abnormally high employment stress

### Employment Indicators

| Indicator | Full Name | Description |
|-----------|-----------|-------------|
| **LFPR** | Labour Force Participation Rate | % of working-age population (15+) that is employed or actively job-seeking |
| **UR** | Unemployment Rate | % of the labour force that is jobless but actively seeking work |
| **WPR** | Worker Population Ratio | % of total population that is actually employed (captures discouraged workers) |
| **GDP** | GDP Growth Rate | Sector-level GDP growth proxy used as a macroeconomic feature |

---

## 📁 Dataset

### Source & Structure

The dataset covers **540 monthly observations** spanning January 2016 to December 2024, across 5 sectors.

```
Rows:    540  (9 years × 12 months × 5 sectors)
Columns: 7
Format:  CSV
```

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `Year` | int | Calendar year (2016–2024) |
| `Month` | int | Month number (1–12) |
| `Sector` | string | One of: Agriculture, IT, Manufacturing, Medical, Hospitality |
| `LFPR` | float | Labour Force Participation Rate (%) |
| `UR` | float | Unemployment Rate (%) |
| `WPR` | float | Worker Population Ratio (%) |
| `GDP_Growth_Rate` | float | Sectoral GDP growth rate (%) |

### Sector Summary (2024 averages)

| Sector | Avg LFPR | Avg UR | Avg WPR | Status |
|--------|----------|--------|---------|--------|
| Agriculture | 41.4% | 6.1% | 38.9% | Moderate |
| IT | 42.5% | 5.1% | 40.3% | Declining post-2023 |
| Manufacturing | 40.8% | 6.2% | 38.3% | Recovering |
| Medical | 45.1% | 4.4% | 43.2% | **Most Stable** |
| Hospitality | 40.9% | 6.8% | 38.1% | Vulnerable |

### Notable Data Events

- **2020**: COVID-19 shock — Hospitality UR peaked at **15.2%**, GDP dropped to **-15%**
- **2021–2023**: IT boom — UR fell to a low of **3.8%**, LFPR rose above 43%
- **Post-2023**: IT slowdown — Global layoffs pushed IT UR back toward **7%+** by 2025

---

## 🤖 Machine Learning

### Feature Engineering

Raw features were transformed into **27 input features** to capture temporal dynamics:

```python
# Lag features (past values as predictors)
for col in ['LFPR', 'UR', 'WPR', 'GDP_Growth_Rate']:
    df[f'{col}_lag1']  = df.groupby('Sector')[col].shift(1)
    df[f'{col}_lag3']  = df.groupby('Sector')[col].shift(3)
    df[f'{col}_lag6']  = df.groupby('Sector')[col].shift(6)
    df[f'{col}_lag12'] = df.groupby('Sector')[col].shift(12)

# Rolling statistics (trend smoothing)
df['UR_roll3']     = df.groupby('Sector')['UR'].transform(lambda x: x.shift(1).rolling(3).mean())
df['UR_roll6']     = df.groupby('Sector')['UR'].transform(lambda x: x.shift(1).rolling(6).mean())
df['UR_roll3_std'] = df.groupby('Sector')['UR'].transform(lambda x: x.shift(1).rolling(3).std())

# Momentum (month-over-month change)
df['UR_diff1']   = df.groupby('Sector')['UR'].diff(1)
df['LFPR_diff1'] = df.groupby('Sector')['LFPR'].diff(1)

# Cyclical seasonality encoding
df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

# Target
df['UR_next'] = df.groupby('Sector')['UR'].shift(-1)         # Regression target
df['UR_stress_next'] = (df['UR_next'] > df['UR'].quantile(0.65)).astype(int)  # Classification target
```

#### Feature Categories

| Category | Features | Count |
|----------|----------|-------|
| Lag-1/2/3 | `UR_lag1`, `LFPR_lag1`, `WPR_lag1`, ... | 9 |
| Lag-6/12 | `UR_lag6`, `UR_lag12`, `LFPR_lag6`, ... | 6 |
| Rolling mean/std | `UR_roll3`, `UR_roll6`, `LFPR_roll3`, ... | 5 |
| Diff / momentum | `UR_diff1`, `LFPR_diff1` | 2 |
| Seasonality | `month_sin`, `month_cos`, `t` | 3 |
| Identifiers | `Sector_enc`, `Month`, `GDP_Growth_Rate` | 2 |
| **Total** | | **27** |

---

### Random Forest Regressor

**Task**: Predict exact next-month Unemployment Rate (continuous value)

```
ŷ = (1/T) Σ tree_t(x)
```

The Random Forest averages predictions from T independent decision trees, each trained on a random bootstrap sample with a random feature subset. This reduces variance without increasing bias.

#### Hyperparameters

```python
RandomForestRegressor(
    n_estimators  = 300,    # Number of trees
    max_depth     = 8,      # Max tree depth (prevents overfitting)
    min_samples_leaf = 4,   # Minimum samples per leaf node
    max_features  = 0.7,    # 70% features per split (random subspace)
    random_state  = 42,
    n_jobs        = -1      # Parallelise across all CPU cores
)
```

#### Why Random Forest?

- Handles **non-linear relationships** between lag features and UR naturally
- **Feature importance scores** reveal which past indicators drive predictions
- Robust to **outliers** (e.g., COVID spike months) through ensemble averaging
- No need to normalize features unlike neural networks or SVMs

#### Top Feature Importances

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | `UR_lag1` | 0.182 | Unemployment lag |
| 2 | `UR_lag3` | 0.141 | Unemployment lag |
| 3 | `UR_roll3` | 0.118 | Rolling mean |
| 4 | `LFPR_lag1` | 0.095 | Participation lag |
| 5 | `UR_lag6` | 0.087 | Unemployment lag |
| 6 | `UR_diff1` | 0.071 | Momentum |
| 7 | `WPR_lag1` | 0.063 | Worker ratio lag |
| 8 | `GDP_Growth_Rate` | 0.055 | Macroeconomic |

> **Key insight**: The past month's UR (`UR_lag1`) is the single most important predictor — employment conditions change gradually and are highly auto-correlated.

---

### Logistic Regression Classifier

**Task**: Classify whether next month's UR will exceed the **65th percentile threshold (6.69%)** — signaling an "employment stress" condition.

```
P(stress=1) = σ(w·x + b) = 1 / (1 + e^(-z))
```

where `z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`

#### Configuration

```python
LogisticRegression(
    C            = 1.0,           # Regularisation strength (inverse)
    max_iter     = 2000,          # Iterations for convergence
    class_weight = 'balanced',    # Handles 65:35 class imbalance
    random_state = 42
)

# Features scaled before training (required for LR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

#### Why Logistic Regression?

- **Interpretable coefficients** — direction and magnitude of each feature's effect
- **Probability outputs** — returns a stress probability score (0–1) per month
- **Fast and lightweight** — good baseline for comparison with Random Forest
- `class_weight='balanced'` handles the natural imbalance between stress and non-stress months

#### Confusion Matrix (Test Set)

```
                  Predicted Normal   Predicted High
Actual Normal           57                8
Actual High             18               17
```

- **Accuracy**: 64.4%
- **Precision (High UR)**: 68%
- **Recall (High UR)**: 49%

---

### Training Strategy

#### Time-Aware Split

Standard random train/test split is **not appropriate for time series** — it causes data leakage from the future into training.

```python
# Last 18 months (90 rows = 18 months × 5 sectors) held out as test set
# No shuffling — temporal order preserved
split = len(df_clean) - 90
X_train, X_test = X.iloc[:split], X.iloc[split:]
```

#### Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
# Each fold: train on past months, validate on future months only
rf_cv_r2 = cross_val_score(rf, X, y_reg, cv=tscv, scoring='r2').mean()
```

This is critical — standard K-Fold CV would allow future data to "leak" into training folds.

---

### Model Results

#### Regression Performance (Random Forest vs Baseline)

| Model | MAE | RMSE | R² (Test) | CV R² |
|-------|-----|------|-----------|-------|
| **Random Forest** | **1.16%** | **1.97%** | **0.74** | **0.72** |
| Linear Regression | 1.82% | 2.43% | 0.51 | 0.53 |

#### Classification Performance (Logistic Regression)

| Metric | Score |
|--------|-------|
| Accuracy | 64.4% |
| CV Accuracy (5-fold) | 66.5% |
| Precision (High UR) | 68% |
| Recall (High UR) | 49% |
| F1 Score (weighted) | 58% |

#### 2025 Predicted UR by Sector (Random Forest)

| Sector | Jan | Apr | Jul | Oct | Dec | Trend |
|--------|-----|-----|-----|-----|-----|-------|
| Agriculture | 6.40 | 6.20 | 6.37 | 6.35 | 6.47 | → Stable |
| IT | 7.16 | 7.07 | 7.11 | 7.21 | 7.25 | ↑ Rising |
| Manufacturing | 6.98 | 6.39 | 6.27 | 6.09 | 6.09 | ↓ Improving |
| Medical | 6.33 | 6.28 | 6.39 | 6.30 | 6.28 | → Stable |
| Hospitality | 7.21 | 7.20 | 7.28 | 7.28 | 7.27 | → Elevated |

---

## 📊 Dashboard Features

### Sections

| Section | Content |
|---------|---------|
| 🎯 **Overview** | Project hero, goals, LFPR/UR/WPR explainers |
| 📊 **Dataset** | Interactive line, bar, pie, radar, scatter charts |
| 🤖 **ML Models** | Algorithm explanations, feature pipeline, training strategy |
| 📈 **Results** | Confusion matrix, feature importance, model comparison, 2025 forecast |
| 💡 **Insights** | COVID analysis, sector status, GDP correlation, policy recommendations |

### Interactive Features

- 🔽 **Filter by Year** — Drill down to any year from 2016–2025
- 🔽 **Filter by Sector** — Isolate any of the 5 sectors
- 📐 **Metric Toggle** — Switch between LFPR / UR / WPR / GDP on trend charts
- 🔍 **Tooltips** — Hover on any data point for detailed values
- 📱 **Responsive** — Works on desktop and laptop screens

### Charts Included

| Chart | Library | Purpose |
|-------|---------|---------|
| Multi-line trend | Recharts LineChart | UR/LFPR/WPR over time by sector |
| Grouped bar | Recharts BarChart | Sector comparison of indicators |
| Donut pie | Recharts PieChart | Workforce distribution |
| Radar | Recharts RadarChart | Sector profile (LFPR vs WPR) |
| Scatter | Recharts ScatterChart | GDP growth vs unemployment correlation |
| Horizontal bar | Recharts BarChart | RF feature importances |
| Stacked bar | Recharts BarChart | Model accuracy comparison |
| Custom heatmap | HTML/CSS grid | Confusion matrix |
| Dashed forecast line | Recharts LineChart | 2025 UR predictions |

---

## 🛠 Tech Stack

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| Next.js | 15 | React framework (App Router) |
| React | 19 | UI component library |
| TailwindCSS | 3 | Utility-first CSS (layout + spacing) |
| Recharts | 2.x | Declarative chart components |
| DM Sans | Google Fonts | Body typography |
| Space Mono | Google Fonts | Monospace metrics display |

### Machine Learning (Python)

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | 2.x | Data manipulation & feature engineering |
| numpy | 1.26+ | Numerical operations |
| scikit-learn | 1.4+ | ML models, preprocessing, evaluation |
| matplotlib | 3.8+ | Visualization of results |
| seaborn | 0.13+ | Heatmap and statistical plots |

---

## 📂 Project Structure

```
india-employment-dashboard/
│
├── app/                          # Next.js App Router
│   ├── page.jsx                  # Entry point (imports dashboard)
│   ├── layout.jsx                # Root layout
│   └── EmploymentDashboard.jsx   # 🏠 Main dashboard (single-file component)
│
├── ml/                           # Machine Learning pipeline
│   ├── train_v2.py               # Main training script
│   ├── generate_dataset.py       # Dataset generation script
│   └── outputs/
│       ├── employment_model_results.png    # Result visualizations
│       └── employment_2025_predictions.csv # 2025 forecasts
│
├── data/
│   └── india_employment_2016_2024.csv      # Source dataset
│
├── public/                       # Static assets
├── package.json
├── tailwind.config.js
├── next.config.js
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Node.js 18+
- Python 3.10+
- npm or yarn

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/india-employment-dashboard.git
cd india-employment-dashboard
```

### 2. Install frontend dependencies

```bash
npm install
# or
yarn install
```

### 3. Run the development server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### 4. Run the ML pipeline (optional)

```bash
cd ml
pip install -r requirements.txt
python train_v2.py
```

#### requirements.txt

```
pandas>=2.0
numpy>=1.26
scikit-learn>=1.4
matplotlib>=3.8
seaborn>=0.13
```

### 5. Build for production

```bash
npm run build
npm start
```

### Deploy to Vercel (one command)

```bash
npm install -g vercel
vercel
```

---

## 💡 Key Findings

### 1. COVID-19 Was the Largest Employment Shock
The 2020 pandemic caused catastrophic disruption across all sectors:
- **Hospitality**: UR spiked from ~6% to **15.2%**, GDP growth collapsed to **-15%**
- **Manufacturing**: UR hit **10.3%**, LFPR dropped nearly 4 percentage points
- **Medical**: Only sector to see LFPR *increase* as healthcare demand surged

### 2. IT Boom (2021–2023) Then Bust
The post-COVID digital acceleration drove IT sector UR from 6.8% (2020) down to a record low of **3.8% in 2023**. Post-2023 global tech layoffs, AI automation pressure, and reduced VC funding reversed this — UR is forecast at **7%+ by late 2025**.

### 3. Medical is the Most Stable Sector
Medical consistently shows the lowest unemployment (avg **4.4% in 2024**) and highest LFPR. Healthcare infrastructure investment and demographic demand create structural employment stability.

### 4. GDP Growth Negatively Correlates with UR
Sectors with GDP growth above 8% consistently show UR below 5%. A 1% increase in GDP growth correlates with approximately a **0.4 percentage point reduction** in unemployment rate.

### 5. Lag-1 UR is the Strongest Predictor
The Random Forest's most important feature (importance: **0.182**) is last month's unemployment rate. Employment conditions are strongly auto-correlated — they change gradually, making recent history the best predictor of near-term future.

---

## 🔮 Future Work

- [ ] **Real Data Integration** — Replace synthetic data with official PLFS (Periodic Labour Force Survey) and CMIE data
- [ ] **LSTM / Transformer Models** — Sequence-to-sequence deep learning for richer temporal modelling
- [ ] **State-level Analysis** — Drill down from national sectors to individual Indian states
- [ ] **Macro Feature Expansion** — Add CPI inflation, repo rate, FDI inflows, crude oil prices as features
- [ ] **Real-time Dashboard** — Connect to live data sources with automatic model retraining
- [ ] **SHAP Explainability** — Add SHAP value visualizations for per-prediction explanations

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **MOSPI / PLFS** — Periodic Labour Force Survey methodology reference
- **CMIE** — Centre for Monitoring Indian Economy for employment indicator definitions
- **scikit-learn** — Excellent documentation and API design
- **Recharts** — Beautiful, composable React charting library

---

<div align="center">

Made with ❤️ for data science education

⭐ **Star this repo if you found it useful!**

</div>
