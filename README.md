# Structural Break Lite — by CONDOR

This repository contains the "Lite" implementation of the structural break detection algorithm developed by **CONDOR** for the CrunchDAO Structural Break Open Benchmark.

## Overview

The Lite version focuses on a fast, efficient, and reliable benchmark for detecting structural breaks in financial time series. It utilizes a combination of robust feature engineering and gradient boosting models (HGB and CatBoost) to provide high-accuracy predictions with minimal computational overhead.

## Key Features

- **Efficient Feature Engineering**: Optimized transformations for financial time series.
- **KS-Shift Filtering**: Automatic detection and removal of drifting features to ensure model stability.
- **Ensemble Approach**: Combines multiple boosting models for robust performance.
- **Top 6 Global Performance**: This architecture contributed to our Top 6 ranking in the global leaderboard.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import pandas as pd
from main import run_lite_inference

# Load your data (MultiIndex: id, time)
X_test = pd.read_parquet("data/X_test.parquet")

# Run inference
predictions = run_lite_inference(X_test)
print(predictions.head())
```

## Authors

Developed by **CONDOR** — Sovereign Intelligence.
Visit us at [condor.qaibit.com](https://condor.qaibit.com)

## License

MIT License
