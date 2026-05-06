# Structural Break Lite — by CONDOR

This repository contains the "Lite" implementation of the structural break detection algorithm developed by **CONDOR** for the CrunchDAO Structural Break Open Benchmark.

## Overview

The Lite version focuses on a fast, efficient, and reliable benchmark for detecting structural breaks in financial time series. It utilizes a combination of robust feature engineering and gradient boosting models to provide high-accuracy predictions with minimal computational overhead.

Unlike the [Complete version](https://github.com/qaibit/structural-break-complete), the Lite variant does **not** include the PINT-Seq neural network components, making it simpler to deploy and faster to run.

## Architecture

```
Raw Data (MultiIndex: id, time)
  │
  ├── Feature Engineering
  │     ├── Impl2 (statistical + signal features)
  │     ├── Impl3 (time-domain features)
  │     └── Comprehensive (domain-specific features)
  │
  ├── Feature Selection
  │     ├── KS-Shift Filter (distribution stability)
  │     └── Fold-wise Mutual Information (top-k selection)
  │
  ├── Base Models (Teacher)
  │     ├── HistGradientBoosting (HGB)
  │     ├── HGB Distance
  │     ├── CatBoost A (Bayesian bootstrap)
  │     ├── CatBoost B (Bernoulli bootstrap)
  │     ├── XGBoost (Impl3 features)
  │     └── XGBoost (MI-selected features)
  │
  └── Ensemble Blending
        ├── Dirichlet Rank-Blend
        ├── SLSQP Optimization
        └── Final Structural Break Score
```

## Repository Structure

```
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── main.py                # Entry point: run_lite_inference()
└── expertos_8642.py       # Core engine: feature engineering, models, blending
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import pandas as pd
from main import run_lite_inference

# Load your data (MultiIndex: id, time) with columns [value, period]
X_train = pd.read_parquet("data/X_train.parquet")
y_train = pd.read_parquet("data/y_train.parquet")
X_test = pd.read_parquet("data/X_test.parquet")
y_test = pd.read_parquet("data/y_test.parquet")  # optional

# Run inference
results = run_lite_inference(X_train, y_train, X_test, y_test)

print(f"Ensemble AUC: {results['oof_blend']:.4f}")
print(f"Test AUC: {results['test_auc']:.4f}")
```

## Key Features

- **Efficient Feature Engineering**: Optimized transformations for financial time series.
- **KS-Shift Filtering**: Automatic detection and removal of drifting features to ensure model stability.
- **Mutual Information Selection**: Fold-wise feature selection for robust generalization.
- **Multi-Model Ensemble**: Combines HGB, CatBoost, and XGBoost via rank-blending.
- **Dirichlet + SLSQP Optimization**: Optimal weight allocation across models.

## Authors

Developed by **CONDOR** — Sovereign Intelligence.
Visit us at [condor.qaibit.com](https://condor.qaibit.com)

## License

MIT License
