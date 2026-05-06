# Structural Break Lite — by CONDOR

> **Sovereign Intelligence** · [condor.qaibit.com](https://condor.qaibit.com)

A fast, efficient structural break detection system for financial time series. This is the **Lite** variant — it uses gradient boosting models (HGB, CatBoost, XGBoost) without neural network components. For the full ensemble with PINT-Seq neural transformers, see [structural-break-complete](https://github.com/qaibit/structural-break-complete).

---

## Table of Contents

1. [What is Structural Break Detection?](#what-is-structural-break-detection)
2. [Architecture Overview](#architecture-overview)
3. [Repository Structure](#repository-structure)
4. [Requirements](#requirements)
5. [Installation (Step by Step)](#installation-step-by-step)
6. [Data Format](#data-format)
7. [Quick Start](#quick-start)
8. [Training & Testing Guide](#training--testing-guide)
9. [Understanding the Output](#understanding-the-output)
10. [Configuration & Tuning](#configuration--tuning)
11. [Troubleshooting](#troubleshooting)
12. [Citation](#citation)
13. [License](#license)

---

## What is Structural Break Detection?

A **structural break** is a sudden, significant change in the statistical properties of a time series — for example, a shift in mean, variance, or autocorrelation structure. Detecting these breaks is critical in quantitative finance for:

- Identifying regime changes in asset returns
- Detecting anomalies in trading signals
- Risk management and portfolio rebalancing

This system was developed for the [CrunchDAO Structural Break Open Benchmark](https://www.crunchdao.com/), where it achieved a **Top 6 global ranking**.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  INPUT DATA                          │
│   DataFrame with MultiIndex (id, time)               │
│   Columns: value, period                             │
└───────────────┬─────────────────────────────────────┘
                │
    ┌───────────▼───────────┐
    │   FEATURE ENGINEERING  │
    │                        │
    │  ┌──────────────────┐  │
    │  │  Impl2 Features  │  │  Statistical, spectral, distributional
    │  │  (600+ features) │  │  KS tests, CUSUM, MMD, Wasserstein
    │  └──────────────────┘  │
    │  ┌──────────────────┐  │
    │  │  Impl3 Features  │  │  Quantile-based time-domain features
    │  │  (50+ features)  │  │  Delta features across periods
    │  └──────────────────┘  │
    │  ┌──────────────────┐  │
    │  │ Comprehensive    │  │  Robust CVs, IQR, IDR, MAD
    │  │ (80+ features)   │  │  Pre/post period comparisons
    │  └──────────────────┘  │
    └───────────┬────────────┘
                │
    ┌───────────▼───────────┐
    │   FEATURE SELECTION    │
    │                        │
    │  1. KS-Shift Filter    │  Removes features with distribution
    │     (stability check)  │  drift between folds
    │                        │
    │  2. Mutual Information │  Selects top-420 most informative
    │     (fold-wise MI)     │  features per fold
    └───────────┬────────────┘
                │
    ┌───────────▼───────────┐
    │     BASE MODELS        │
    │                        │
    │  • HGB (Histogram      │
    │    Gradient Boosting)   │
    │  • HGB Distance        │
    │  • CatBoost A          │
    │    (Bayesian bootstrap) │
    │  • CatBoost B          │
    │    (Bernoulli bootstrap)│
    │  • XGBoost (Impl3)     │
    │  • XGBoost (MI feats)  │
    └───────────┬────────────┘
                │
    ┌───────────▼───────────┐
    │   ENSEMBLE BLENDING    │
    │                        │
    │  1. Dirichlet Rank     │  4500 random trials + 2500 refinement
    │     Blend              │  
    │  2. SLSQP Optimization │  Constrained weight optimization
    │                        │
    │  Output: break_score   │  Probability of structural break
    └────────────────────────┘
```

---

## Repository Structure

```
structural-break-lite/
├── README.md              ← You are here
├── requirements.txt       ← Python dependencies
├── main.py                ← Entry point: run_lite_inference()
└── expertos_8642.py       ← Core engine: feature engineering,
                              models, blending (2300+ lines)
```

---

## Requirements

| Package         | Version  | Purpose                          |
|-----------------|----------|----------------------------------|
| Python          | ≥ 3.9    | Runtime                          |
| pandas          | ≥ 1.5    | Data manipulation                |
| numpy           | ≥ 1.23   | Numerical computing              |
| scikit-learn    | ≥ 1.2    | HGB, Logistic Regression, MI     |
| scipy           | ≥ 1.10   | Statistical tests, optimization  |
| catboost        | ≥ 1.2    | CatBoost models (optional)       |
| xgboost         | ≥ 1.7    | XGBoost models (optional)        |
| tqdm            | ≥ 4.64   | Progress bars                    |
| statsmodels     | ≥ 0.14   | Ljung-Box test (optional)        |

> **Note**: CatBoost and XGBoost are optional but strongly recommended. Without them, the ensemble will run with fewer models and lower accuracy.

---

## Installation (Step by Step)

### 1. Clone the repository

```bash
git clone https://github.com/qaibit/structural-break-lite.git
cd structural-break-lite
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify installation

```python
python -c "from main import run_lite_inference; print('✅ Ready')"
```

---

## Data Format

Your data must follow the **CrunchDAO Structural Break** format:

### X_train / X_test (Features)

A DataFrame with a **MultiIndex** of two levels: `(id, time)` and two columns:

| Column   | Type    | Description                              |
|----------|---------|------------------------------------------|
| `value`  | float64 | The observed time series value           |
| `period` | int     | 0 = pre-break period, 1 = post-break     |

```
                    value  period
id       time                    
series_0 0      0.234521       0
         1      0.198432       0
         ...
         870    0.543210       1
         871    0.567890       1
series_1 0     -0.112345       0
         ...
```

### y_train / y_test (Labels)

A Series or single-column DataFrame indexed by `id`:

| id       | target |
|----------|--------|
| series_0 | 1      |
| series_1 | 0      |
| series_2 | 1      |

Where:
- `1` = structural break detected
- `0` = no structural break

### Loading from Parquet files

```python
import pandas as pd

X_train = pd.read_parquet("data/X_train.parquet")
y_train = pd.read_parquet("data/y_train.parquet")
X_test  = pd.read_parquet("data/X_test.parquet")
y_test  = pd.read_parquet("data/y_test.parquet")  # optional, for evaluation

print(f"Training: {X_train.index.get_level_values('id').nunique()} series")
print(f"Test:     {X_test.index.get_level_values('id').nunique()} series")
```

---

## Quick Start

```python
from main import run_lite_inference

results = run_lite_inference(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test      # pass None if you don't have test labels
)

print(f"OOF Blend AUC: {results['oof_blend']:.4f}")
print(f"Test AUC:      {results['test_auc']:.4f}")
```

The function automatically saves predictions to `submission_stack_impl2_impl3.csv`.

---

## Training & Testing Guide

### Step 1: Prepare your data

Place your Parquet files in a `data/` directory:

```
structural-break-lite/
├── data/
│   ├── X_train.parquet
│   ├── y_train.parquet
│   ├── X_test.parquet
│   └── y_test.parquet       # optional
├── main.py
├── expertos_8642.py
└── ...
```

### Step 2: Run the full pipeline

Create a script `run.py`:

```python
#!/usr/bin/env python3
"""Run the CONDOR Lite structural break detection pipeline."""

import pandas as pd
from main import run_lite_inference

# ── 1. Load data ──
print("Loading data...")
X_train = pd.read_parquet("data/X_train.parquet")
y_train = pd.read_parquet("data/y_train.parquet")
X_test  = pd.read_parquet("data/X_test.parquet")

# Optional: load test labels for evaluation
try:
    y_test = pd.read_parquet("data/y_test.parquet")
    print(f"✅ Test labels loaded ({len(y_test)} series)")
except FileNotFoundError:
    y_test = None
    print("⚠️  No test labels found — running inference only")

print(f"Training series: {X_train.index.get_level_values('id').nunique()}")
print(f"Test series:     {X_test.index.get_level_values('id').nunique()}")

# ── 2. Run inference ──
results = run_lite_inference(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

# ── 3. Print results ──
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

print("\n📊 Per-model OOF AUC:")
for model, auc in sorted(results['oof'].items(), key=lambda x: x[1], reverse=True):
    print(f"   {model:12s}  →  {auc:.4f}")

print(f"\n🎯 Ensemble Blend AUC: {results['oof_blend']:.4f}")

if results.get('test_auc') is not None:
    print(f"📈 Test AUC:           {results['test_auc']:.4f}")

print("\n⚖️  Blend Weights:")
for model, weight in sorted(results['weights'].items(), key=lambda x: x[1], reverse=True):
    bar = "█" * int(weight * 50)
    print(f"   {model:12s}  {weight:.3f}  {bar}")

print(f"\n💾 Predictions saved to: submission_stack_impl2_impl3.csv")
```

Run it:

```bash
python run.py
```

### Step 3: Read the predictions

```python
import pandas as pd

predictions = pd.read_csv("submission_stack_impl2_impl3.csv", index_col="id")
print(predictions.head())
#              break_score
# id                      
# series_0       0.823456
# series_1       0.124567
# series_2       0.956789
```

Higher `break_score` = higher probability of a structural break.

### Step 4: Restrict to specific models (optional)

If you want to run only a subset of models:

```python
results = run_lite_inference(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    allowed_models=["hgb", "cbA", "xgb_raw"]   # only these 3
)
```

Available model keys: `hgb`, `dist`, `cbA`, `cbB`, `xgb_raw`, `xgb_xmi`

---

## Understanding the Output

The `run_lite_inference()` function returns a dictionary with:

| Key          | Type              | Description                                    |
|--------------|-------------------|------------------------------------------------|
| `oof`        | dict[str, float]  | Out-of-fold AUC for each base model            |
| `oof_blend`  | float             | Blended ensemble AUC (OOF)                     |
| `weights`    | dict[str, float]  | Optimal blend weight for each model             |
| `test_auc`   | float or None     | Test AUC (only if `y_test` was provided)        |

It also writes `submission_stack_impl2_impl3.csv` to disk with the final predictions.

---

## Configuration & Tuning

Key parameters in `expertos_8642.py` that you can adjust:

| Parameter              | Default | Description                                    |
|------------------------|---------|------------------------------------------------|
| `SEED`                 | 42      | Random seed for reproducibility                |
| `FOLDS`                | 5       | Number of cross-validation folds               |
| `FAST_MODE`            | False   | Set to True for faster (but less accurate) runs|
| `TOPK_FULL`            | 420     | Number of features to keep after MI selection  |
| `BLEND_RANDOM_TRIALS`  | 4500    | Dirichlet sampling trials for blending         |
| `BLEND_REFINE_TRIALS`  | 2500    | Refinement trials around best blend            |
| `SHIFT_FILTER_FRAC`    | 0.05    | KS-shift filter significance threshold         |

### Fast Mode

For quick experimentation, enable `FAST_MODE` in `expertos_8642.py`:

```python
FAST_MODE = True  # Reduces feature engineering and model training time
```

This reduces spectral scales, bag counts, and iterations — useful for prototyping.

---

## Troubleshooting

### "CatBoost no disponible"
```bash
pip install catboost
```

### "XGBoost no disponible"
```bash
pip install xgboost
```

### "statsmodels no disponible"
```bash
pip install statsmodels
```

### Memory issues with large datasets
- Enable `FAST_MODE = True`
- Reduce `TOPK_FULL` from 420 to 200
- Reduce `BLEND_RANDOM_TRIALS` from 4500 to 1000

### Slow execution
- Enable `FAST_MODE = True`
- Use `allowed_models=["hgb", "dist"]` for minimal ensemble
- Reduce `FOLDS` from 5 to 3

---

## Citation

If you use this code in your research or projects, please cite:

```
@software{condor_structural_break_lite,
  author = {CONDOR},
  title = {Structural Break Detection (Lite) — Sovereign Intelligence},
  year = {2026},
  publisher = {Qaibit},
  url = {https://github.com/qaibit/structural-break-lite}
}
```

---

## Authors

Developed by **CONDOR** — Sovereign Intelligence.

- 🌐 Platform: [condor.qaibit.com](https://condor.qaibit.com)
- 🏢 Organization: [Qaibit](https://qaibit.com)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
