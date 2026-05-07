# ============================================================
# Impl2 (+MMD + KS-shift + MI)  +  Impl3 (XGB fold-wise OOF)
#  -> Rank-blend (Dirichlet + SLSQP) + Meta-stacker LR
#  -> FAST_MODE + Curriculum Weights (sin flip duro)
#  -> PINT (Physics-Informed LSTM, SHO prior) AR one-step target (FIX)
#  -> Anti-shift KS usando SOLO TRAIN (sin fuga) + opción train_test
#  -> Impl3: cuantiles completos y deltas
# ============================================================

import os, gc, math, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy import signal, stats
from numpy.linalg import lstsq
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import HistGradientBoostingClassifier

# ==== NEW: optimización SLSQP para el blend (con caja y simplex) ====
try:
    from scipy.optimize import minimize
    HAS_SLSQP = True
except Exception:
    HAS_SLSQP = False

# ==== NEW: PyTorch para PINT ====
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    print(">> [PINT] PyTorch no disponible: instala torch para activar PINT.")

# ===== Config =====
# NOTE: These are starter/default hyperparameters provided for reference.
# The production CONDOR system uses proprietary tuned values.
# Tune these for your specific dataset and compute budget.
# See: https://condor.qaibit.com for the optimized inference engine.
SEED  = 42
FOLDS = 5
FAST_MODE = False  # acelera espectral, jitter, bags, etc.

# Salidas
SUBMISSION_NAME = "submission_stack_impl2_impl3.csv"

# CatBoost
USE_CATBOOST   = True
CB_SEEDS_A     = (42,)              # Tune: add more seeds for multi-seed averaging
CB_SEEDS_B     = (123,)             # Tune: add more seeds for diversity
CB_FRACTION_A  = 0.75 if not FAST_MODE else 0.70
CB_FRACTION_B  = 0.65 if not FAST_MODE else 0.60

# Rank-blend (búsqueda + refine)
BLEND_RANDOM_TRIALS = 2000 if not FAST_MODE else 500
BLEND_REFINE_TRIALS = 1000 if not FAST_MODE else 300
BLEND_DIRICHLET_ALPHA_GLOBAL = 1.0  # Tune: controls exploration width
BLEND_DIRICHLET_ALPHA_LOCAL  = 50.0 # Tune: controls refinement concentration

# Impl2: parámetros de features
PERIOD0 = 1500                      # Tune: dominant period of your time series
H_H     = 3                         # Tune: number of harmonics
SAVGOL  = (31, 3)                   # Tune: (window, poly) for Savitzky-Golay
LAGS_PRED   = [5, 10, 20]           # Tune: lag orders for predictive divergence
ALPHA_RIDGE = 1e-2
W_LOCAL     = [64, 128]
W_DIST      = [100, 200] if not FAST_MODE else [100]
TOPK_FULL   = 300 if not FAST_MODE else 80  # Tune: top-K MI features

# Impl2: extras
ADD_MMD_FEATURES    = True
MMD_WINDOWS         = [128, 256] if not FAST_MODE else [256]
APPLY_SHIFT_FILTER  = True
SHIFT_FILTER_FRAC   = 0.05
SHIFT_FILTER_MODE   = "train_only"

# Impl3: XGB OOF
USE_IMPL3_XGB            = True
USE_SMOTE_IMPL3          = True
IMPL3_N_BAGS             = (5 if not FAST_MODE else 3)   # Tune: more bags = more stable
IMPL3_EARLY_STOP_ROUNDS  = 100 if not FAST_MODE else 50
IMPL3_MAX_ESTIMATORS     = 1500 if not FAST_MODE else 800
IMPL3_LEARNING_RATE      = 0.05                          # Tune: lower = better but slower
IMPL3_MAX_DEPTH          = 6                              # Tune: 5-9
IMPL3_REG_LAMBDA         = 1.0
IMPL3_SUBSAMPLE_GRID     = [0.75, 0.85]
IMPL3_COLSAMPLE_GRID     = [0.70, 0.85]

# Inyectar features de Impl3 en Impl2 (opcional)
INJECT_IMPL3_FEATURES_IN_IMPL2 = True

# ======== Curriculum + Pseudo-Labels (SIN flip duro) =========
USE_CURRICULUM = True
PL_POS_Q = 0.85                     # Tune: quantile threshold for positive pseudo-labels
PL_NEG_Q = 0.15                     # Tune: quantile threshold for negative pseudo-labels
TT_MIN_LOGP = 1.0                   # Tune: minimum -log10(p) for t-test signal
LB_DELTA_MIN_LOGP = 0.5             # Tune: minimum Ljung-Box delta
CURR_WEIGHT_BASE = 1.0
CURR_WEIGHT_POS  = 1.5              # Tune: upweight for confident positives
CURR_WEIGHT_NEG  = 1.2              # Tune: upweight for confident negatives

# FAST_MODE escalas espectrales
if FAST_MODE:
    SPEC_L_LIST = [20]
    SPEC_M_LIST = [3]
    SPEC_WPOST_LIST = [250]
    SPEC_BURN_LIST  = [80]
else:
    SPEC_L_LIST = [16, 24]           # Tune: lag-embedding dimensions
    SPEC_M_LIST = [2, 3]             # Tune: number of spectral components
    SPEC_WPOST_LIST = [200]          # Tune: post-break observation window
    SPEC_BURN_LIST  = [80]           # Tune: burn-in after break

# ======== PINT Config ========
USE_PINT = True
PINT_IN_LEN   = 64 if FAST_MODE else 96    # Tune: input sequence length
PINT_OUT_LEN  = 16 if FAST_MODE else 32    # Tune: output forecast horizon
PINT_H_LIST   = [16, 32] if FAST_MODE else [16, 32, 64]  # Tune: rollout horizons
PINT_HIDDEN   = 64 if FAST_MODE else 64    # Tune: LSTM hidden size
PINT_LAYERS   = 1 if FAST_MODE else 2      # Tune: LSTM depth
PINT_LR       = 2e-3 if FAST_MODE else 2e-3  # Tune: learning rate
PINT_EPOCHS   = (40 if FAST_MODE else 60)  # Tune: training epochs
PINT_FT_STEPS = 0 if FAST_MODE else 10
PINT_BS       = 128 if FAST_MODE else 128
PINT_LAMBDA_PHYS = 0.1              # Tune: physics loss weight (SHO regularization)
PINT_DROPOUT = 0.20                  # Tune: dropout rate
PINT_SEARCH_DELAY = 15               # Tune: break search window
PINT_DEVICE = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else ("mps" if (HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")

# ============================================================
random.seed(SEED); np.random.seed(SEED); os.environ["PYTHONHASHSEED"]=str(SEED)

# --------- CatBoost opcional ----------
try:
    from catboost import CatBoostClassifier, Pool
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False
    if USE_CATBOOST:
        print(">> CatBoost no disponible. Instálalo con: pip install catboost")

# --------- XGBoost (Impl3) ------------
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    if USE_IMPL3_XGB:
        print(">> XGBoost no disponible. Instálalo con: pip install xgboost")

# --------- Ljung–Box (statsmodels) ----
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    HAS_LB = True
except Exception:
    HAS_LB = False
    print(">> statsmodels no disponible; LB desactivado (pip install statsmodels)")

# =================== Utils IO ===================

# ================= Comprehensive Features (pandas) =================
import numpy as np
import pandas as pd

def create_comprehensive_features_pd(df: pd.DataFrame) -> pd.DataFrame:
    def safe_divide(a: pd.Series, b: pd.Series, fill_value: float = 0.0) -> pd.Series:
        out = a.copy().astype(float)
        denom = b.replace(0, np.nan).astype(float)
        out = out / denom
        return out.fillna(fill_value).replace([np.inf, -np.inf], fill_value)

    def calculate_mad_series(x: pd.Series, scaled: bool = True) -> float:
        med = x.median()
        mad = (x - med).abs().median()
        return float(mad * 1.4826) if scaled else float(mad)

    def trimmed_mean(x: pd.Series, lo: float = 0.05, hi: float = 0.95) -> float:
        qlo, qhi = x.quantile(lo), x.quantile(hi)
        mask = (x > qlo) & (x < qhi)
        if mask.any():
            return float(x[mask].mean())
        return float(x.mean())

    # Agregaciones principales por grupo id,period
    def agg_block(group: pd.DataFrame) -> pd.Series:
        v = group['value']
        mean_v = float(v.mean())
        std_v = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        var_v = float(v.var(ddof=1)) if len(v) > 1 else 0.0
        mad_v = calculate_mad_series(v)
        med_v = float(v.median())
        q25 = float(v.quantile(0.25))
        q75 = float(v.quantile(0.75))
        q90 = float(v.quantile(0.90))
        q99 = float(v.quantile(0.99))
        q10 = float(v.quantile(0.10))
        # pct_increasing
        diff = v.diff()
        pct_increasing = float((diff > 0).mean()) if len(v) > 1 else 0.0
        # cv/cv_var/cv_var_med/cv_mad/cv_mad_med
        cv = float(std_v / (abs(mean_v) + 1e-9))
        cv_var = float(var_v / (abs(mean_v) + 1e-9))
        cv_var_med = float(var_v / (abs(med_v) + 1e-9))
        cv_mad = float(mad_v / (abs(mean_v) + 1e-9))
        cv_mad_med = float(mad_v / (abs(med_v) + 1e-9))
        # pct_beyond_kstd
        dev_abs = (v - mean_v).abs()
        one_std = float((dev_abs > std_v).mean()) if std_v > 0 else 0.0
        two_std = float((dev_abs > 2*std_v).mean()) if std_v > 0 else 0.0
        three_std = float((dev_abs > 3*std_v).mean()) if std_v > 0 else 0.0
        # tail_ratio
        iqr = q75 - q25
        tail_ratio = float((q90 - q10) / (iqr + 1e-9))
        # nuevos CVs
        iqr_cv = float((iqr) / (abs(mean_v) + 1e-9))
        iqr_cv_med = float((iqr) / (abs(med_v) + 1e-9))
        idr_cv = float((q90 - q10) / (abs(mean_v) + 1e-9))
        full_range = float(v.max() - v.min()) if len(v) > 0 else 0.0
        range_cv = float(full_range / (abs(mean_v) + 1e-9))
        range_cv_med = float(full_range / (abs(med_v) + 1e-9))
        tmean = trimmed_mean(v)
        cv_trimmed_mean = float(std_v / (abs(tmean) + 1e-9))
        iqr_cv_trimmed_mean = float(iqr / (abs(tmean) + 1e-9))
        rms = float(np.sqrt(np.mean(np.square(v)))) if len(v) > 0 else 0.0
        std_div_by_rms = float(std_v / (rms + 1e-9))
        return pd.Series({
            'mean': mean_v, 'std': std_v, 'var': var_v, 'mad': mad_v, 'median': med_v,
            'q75': q75, 'q90': q90, 'q99': q99, 'q25': q25, 'pct_increasing': pct_increasing,
            'cv': cv, 'cv_var': cv_var, 'cv_var_med': cv_var_med,
            'cv_mad': cv_mad, 'cv_mad_med': cv_mad_med,
            'pct_beyond_1std': one_std, 'pct_beyond_2std': two_std, 'pct_beyond_3std': three_std,
            'tail_ratio': tail_ratio,
            'iqr_cv': iqr_cv, 'iqr_cv_med': iqr_cv_med,
            'idr_cv': idr_cv, 'range_cv': range_cv, 'range_cv_med': range_cv_med,
            'cv_trimmed_mean': cv_trimmed_mean, 'iqr_cv_trimmed_mean': iqr_cv_trimmed_mean,
            'std_div_by_rms': std_div_by_rms,
        })

    # Estadísticos globales por id (en toda la serie)
    global_stats = (
        df.groupby('id', sort=False)
          .apply(agg_block)
          .add_prefix('global_')
    )

    # Estadísticos por periodo (0/1)
    per_period_stats = (
        df.groupby(['id', 'period'], sort=False)
          .apply(agg_block)
          .reset_index()
    )

    stats_pre = (
        per_period_stats[per_period_stats['period'] == 0]
        .drop(columns=['period'])
        .rename(columns={c: f"{c}_pre" for c in per_period_stats.columns if c not in ['id', 'period']})
        .set_index('id')
    )

    stats_post = (
        per_period_stats[per_period_stats['period'] == 1]
        .drop(columns=['period'])
        .rename(columns={c: f"{c}_post" for c in per_period_stats.columns if c not in ['id', 'period']})
        .set_index('id')
    )

    features_df = stats_pre.join(stats_post, how='left').join(global_stats, how='left')

    # Generar diffs y ratios
    agg_names = [c for c in stats_pre.columns if c.endswith('_pre')]
    base_names = [c[:-4] for c in agg_names]  # remover _pre

    for col in base_names:
        pre, post = f"{col}_pre", f"{col}_post"
        features_df[f"{col}_diff"] = features_df.get(post, np.nan) - features_df.get(pre, np.nan)
        features_df[f"{col}_ratio"] = safe_divide(features_df.get(post, np.nan), features_df.get(pre, np.nan))

    # Contexto con globales
    def sv(name: str) -> pd.Series:
        return features_df[name] if name in features_df.columns else pd.Series(np.nan, index=features_df.index)

    features_df['norm_mean_diff'] = safe_divide(sv('mean_diff'), sv('global_std'))
    features_df['norm_mean_cv_diff'] = safe_divide(sv('mean_diff'), sv('global_cv'))
    features_df['norm_std_diff'] = safe_divide(sv('std_diff'), sv('global_std'))
    features_df['norm_cv_diff'] = safe_divide(sv('std_diff'), sv('global_cv'))
    features_df['pre_mean_deviation'] = safe_divide(sv('mean_pre') - sv('global_mean'), sv('global_std'))
    features_df['change_in_deviation_from_global'] = (
        safe_divide(sv('mean_post') - sv('global_mean'), sv('global_std')).abs()
        - safe_divide(sv('mean_pre') - sv('global_mean'), sv('global_std')).abs()
    )

    features_df = features_df.reset_index()
    return features_df


def _sanitize_for_model(X):
    X2 = X.copy().replace([np.inf, -np.inf], np.nan)
    all_nan = X2.columns[X2.isna().all()]
    if len(all_nan):
        X2 = X2.drop(columns=all_nan)
    med = X2.median(numeric_only=True)
    med = med.fillna(X2.mean(numeric_only=True)).fillna(0.0)
    X2 = X2.fillna(med).fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return X2

def ensure_data_in_memory():
    global X_train, y_train, X_test, y_test
    need = [nm for nm in ["X_train","y_train","X_test","y_test"] if nm not in globals()]
    if not need: return
    paths = {
        "X_train": "./data/X_train.parquet",
        "y_train": "./data/y_train.parquet",
        "X_test":  "./data/X_test.reduced.parquet",
        "y_test":  "./data/y_test.reduced.parquet",
    }
    for nm in need:
        try:
            globals()[nm] = pd.read_parquet(paths[nm])
            print(f"Cargado {nm} desde {paths[nm]}")
        except Exception as e:
            print(f"[WARN] No pude cargar {nm}: {e}")

def to_y_series(y, index):
    if isinstance(y, pd.DataFrame): s = y.iloc[:,0]
    elif isinstance(y, pd.Series): s = y
    else: s = pd.Series(y)
    return s.astype(int).reindex(index)

def rank01(a):
    s = pd.Series(a); r = s.rank(method='average'); r = r.fillna(r.size/2.0)
    return ((r-1)/(len(r)-1+1e-12)).to_numpy()

# =================== Blend ======================
def auc_of_weights(preds_dict, y, w_dict):
    keys = list(w_dict.keys()); w = np.array([w_dict[k] for k in keys], float)
    if w.sum() <= 0: return -np.inf
    w = w / w.sum()
    R = np.column_stack([rank01(preds_dict[k]) for k in keys])
    s = (R * w).sum(axis=1)
    return roc_auc_score(y, s)

def optimize_rank_blend_dirichlet(preds_dict, y, trials=BLEND_RANDOM_TRIALS, refine_trials=BLEND_REFINE_TRIALS,
                                  alpha_global=BLEND_DIRICHLET_ALPHA_GLOBAL, alpha_local=BLEND_DIRICHLET_ALPHA_LOCAL, seed=SEED):
    rng = np.random.default_rng(seed)
    keys = list(preds_dict.keys()); K=len(keys)
    best_auc=-1.0; best_w=None
    for _ in range(trials):
        w = rng.dirichlet(alpha=[alpha_global]*K)
        w_dict = {k: float(wi) for k,wi in zip(keys, w)}
        auc = auc_of_weights(preds_dict, y, w_dict)
        if auc > best_auc: best_auc, best_w = auc, w_dict
    alpha = np.array([max(alpha_local*best_w[k], 1e-3) for k in keys], float)
    for _ in range(refine_trials):
        w = rng.dirichlet(alpha=alpha)
        w_dict = {k: float(wi) for k,wi in zip(keys, w)}
        auc = auc_of_weights(preds_dict, y, w_dict)
        if auc > best_auc:
            best_auc, best_w = auc, w_dict
            alpha = np.array([max(alpha_local*best_w[k], 1e-3) for k in keys], float)
    return best_auc, best_w

def optimize_rank_blend_slsqp(preds_dict, y, w_init):
    if not HAS_SLSQP:
        return None, None
    keys = list(preds_dict.keys())
    R = np.column_stack([rank01(preds_dict[k]) for k in keys])
    yv = np.asarray(y, int)
    def loss(w):
        w = np.clip(w, 0, 1)
        if w.sum() <= 0: return 1.0
        w = w / w.sum()
        s = (R * w).sum(axis=1)
        return 1.0 - roc_auc_score(yv, s)
    cons = [{'type':'eq','fun':lambda w: np.sum(np.clip(w,0,1)) - 1.0}]
    bnds = [(0.0,1.0)] * len(keys)
    w0 = np.array([w_init.get(k, 1.0/len(keys)) for k in keys], float)
    res = minimize(loss, w0, method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter':200, 'ftol':1e-9, 'disp':False})
    if not res.success:
        return None, None
    w = np.clip(res.x, 0, 1); w = w / w.sum()
    w_dict = {k: float(wi) for k,wi in zip(keys, w)}
    auc_best = 1.0 - res.fun
    return auc_best, w_dict

# ================= Impl2 FE =====================
def zscore(x):
    x = np.asarray(x, float); mu = np.nanmean(x); sd = np.nanstd(x) + 1e-12
    return (x - mu) / sd

def _safe_savgol(x, win, poly):
    n = len(x)
    if n < 5: return x
    w = min(win, n if n%2==1 else n-1)
    if w < 5: w = 5
    if w % 2 == 0: w -= 1
    if w <= poly: w = poly + 3 if (poly+3)%2==1 else poly+4
    w = min(w, n if n%2==1 else n-1)
    try: return signal.savgol_filter(x, w, poly, mode='interp')
    except Exception: return x

def residuals_cusum(x):
    s = _safe_savgol(x, SAVGOL[0], SAVGOL[1]); r = x - s
    return np.cumsum(r - np.nanmean(r))

def boundary_index_from_period(period_values):
    pv = np.asarray(period_values, dtype=int)
    where1 = np.flatnonzero(pv == 1)
    return int(where1[0]) if len(where1) else None

def _winsorize_clip(seg, p=0.01):
    if len(seg) == 0: return seg
    lo, hi = np.quantile(seg, [p, 1-p]); return np.clip(seg, lo, hi)

def align_and_fill(train_df, test_df):
    train = train_df.copy().replace([np.inf, -np.inf], np.nan)
    test  = test_df.copy().replace([np.inf, -np.inf], np.nan)
    test = test.reindex(columns=train.columns)
    all_nan_cols = train.columns[train.isna().all()].tolist()
    if all_nan_cols:
        train = train.drop(columns=all_nan_cols)
        test  = test.drop(columns=all_nan_cols, errors='ignore')
    med = train.median(numeric_only=True)
    fallback_mean = train.mean(numeric_only=True)
    med = med.fillna(fallback_mean).fillna(0.0)
    train = train.fillna(med)
    test  = test.fillna(med)
    train = train.fillna(0.0)
    test  = test.fillna(0.0)
    train = train.replace([np.inf, -np.inf], 0.0)
    test  = test.replace([np.inf, -np.inf], 0.0)
    return train, test

def to_series_dict(X_mi: pd.DataFrame):
    assert isinstance(X_mi.index, pd.MultiIndex), "Se espera MultiIndex con nivel 'id' y 'time'"
    X_mi = X_mi.sort_index(level=[0,1])
    series, tbreak = {}, {}
    for gid, g in X_mi.groupby(level='id', sort=False):
        v = g['value'].to_numpy(); p = g['period'].to_numpy()
        tb = boundary_index_from_period(p)
        if tb is None: continue
        series[gid] = v; tbreak[gid] = int(tb)
    return series, tbreak

def base_features_for_id(x, t_break):
    x_raw = np.asarray(x, float)
    x = zscore(x_raw.copy()); n = len(x)
    a = max(0, t_break-200); b = min(n, t_break+200)
    pre = x[a:t_break]; post = x[t_break:b]
    # métricas base en zscore (compatibilidad con versión previa)
    base_dict = dict(
        var=float(np.var(x)),
        iqr=float(np.subtract(*np.percentile(x,[75,25]))),
        skew=float(stats.skew(x, bias=False)) if n>2 else np.nan,
        kurt=float(stats.kurtosis(x, bias=False)) if n>3 else np.nan,
    )
    if len(pre) >= 20 and len(post) >= 20:
        energy = float(stats.energy_distance(pre, post))
        ks = stats.ks_2samp(pre, post, method='asymp')
        base_dict.update(energy_dist=float(energy), ks_p=float(ks.pvalue))
    else:
        base_dict.update(energy_dist=0.0, ks_p=1.0)
    # ---- Golden-like sobre serie cruda (prefijo base_) ----
    try:
        med = float(np.median(x_raw)) if n>0 else 0.0
        mean = float(np.mean(x_raw)) if n>0 else 0.0
        std = float(np.std(x_raw)) if n>0 else 0.0
        q25 = float(np.quantile(x_raw, 0.25)) if n>0 else 0.0
        q75 = float(np.quantile(x_raw, 0.75)) if n>0 else 0.0
        q90 = float(np.quantile(x_raw, 0.90)) if n>0 else 0.0
        q99 = float(np.quantile(x_raw, 0.99)) if n>0 else 0.0
        q10 = float(np.quantile(x_raw, 0.10)) if n>0 else 0.0
        mad = float(np.median(np.abs(x_raw - med))) if n>0 else 0.0
        mad_scaled = mad * 1.4826
        diff = np.diff(x_raw) if n>1 else np.array([])
        pct_increasing = float(np.mean(diff > 0)) if diff.size>0 else 0.0
        cv = float(std / (abs(mean)+1e-9))
        var_raw = float(np.var(x_raw)) if n>1 else 0.0
        cv_var = float(var_raw / (abs(mean)+1e-9))
        cv_var_med = float(var_raw / (abs(med)+1e-9))
        cv_mad = float(mad_scaled / (abs(mean)+1e-9))
        cv_mad_med = float(mad_scaled / (abs(med)+1e-9))
        dev_abs = np.abs(x_raw - mean)
        pct1 = float(np.mean(dev_abs > std)) if std>0 else 0.0
        pct2 = float(np.mean(dev_abs > 2*std)) if std>0 else 0.0
        pct3 = float(np.mean(dev_abs > 3*std)) if std>0 else 0.0
        iqr_raw = q75 - q25
        tail_ratio = float((q90 - q10) / (iqr_raw + 1e-9))
        # variantes IQR/IDR/Range + normalizaciones robustas
        idr = q90 - q10
        rng = float(np.max(x_raw) - np.min(x_raw)) if n>0 else 0.0
        q05, q95 = (float(np.quantile(x_raw,0.05)), float(np.quantile(x_raw,0.95))) if n>0 else (0.0,0.0)
        tm = float(np.mean(x_raw[(x_raw>q05)&(x_raw<q95)])) if n>4 else mean
        rms = float(np.sqrt(np.mean(x_raw**2))) if n>0 else 0.0
        base_dict.update({
            'base_mean': mean,
            'base_std': std,
            'base_var': var_raw,
            'base_mad': mad_scaled,
            'base_median': med,
            'base_q25': q25,
            'base_q75': q75,
            'base_q90': q90,
            'base_q99': q99,
            'base_pct_increasing': pct_increasing,
            'base_cv': cv,
            'base_cv_var': cv_var,
            'base_cv_var_med': cv_var_med,
            'base_cv_mad': cv_mad,
            'base_cv_mad_med': cv_mad_med,
            'base_pct_beyond_1std': pct1,
            'base_pct_beyond_2std': pct2,
            'base_pct_beyond_3std': pct3,
            'base_tail_ratio': tail_ratio,
            'base_iqr_cv': float(iqr_raw/(abs(mean)+1e-9)),
            'base_iqr_cv_med': float(iqr_raw/(abs(med)+1e-9)),
            'base_idr_cv': float(idr/(abs(mean)+1e-9)),
            'base_range_cv': float(rng/(abs(mean)+1e-9)),
            'base_range_cv_med': float(rng/(abs(med)+1e-9)),
            'base_cv_trimmed_mean': float(std/(abs(tm)+1e-9)),
            'base_iqr_cv_trimmed_mean': float(iqr_raw/(abs(tm)+1e-9)),
            'base_std_div_by_rms': float(std/(rms+1e-9)),
        })
    except Exception:
        pass
    return base_dict

def _harmonic_design(n, period, H):
    t = np.arange(n, dtype=float); Xc = [np.ones(n)]
    period = max(1.0, float(period)); w = 2.0 * np.pi / period
    for h in range(1, H+1):
        Xc.append(np.sin(h*w*t)); Xc.append(np.cos(h*w*t))
    return np.column_stack(Xc)

def fit_harmonic(y, period, H):
    y = np.asarray(y, float); n = len(y)
    if n < 10: return np.zeros(n, float)
    X = _harmonic_design(n, period, H)
    try:
        beta, _, _, _ = lstsq(X, y, rcond=None); return X @ beta
    except Exception: return np.zeros(n, float)

def abc_features_for_id(x, t_break, period0=PERIOD0, H=H_H):
    x = zscore(np.asarray(x, float)); n = len(x)
    if t_break is None or t_break <= 10 or t_break >= n-10:
        return dict(ks_raw_p=1.0, ks_denoised_p=1.0, z_ks=0.0, log_ks_den=0.0, log_ks_raw=0.0)
    c = residuals_cusum(x); post = c[t_break:]
    if len(post) < 30:
        return dict(ks_raw_p=1.0, ks_denoised_p=1.0, z_ks=0.0, log_ks_den=0.0, log_ks_raw=0.0)
    yhat = fit_harmonic(post, period0, H); abs_y = np.abs(yhat)
    mx = float(np.nanmax(abs_y)) if np.isfinite(abs_y).any() else 0.0
    tol = max(1e-12, 1e-9 * mx)
    cand = np.flatnonzero(abs_y >= mx - tol)
    j = int(cand[-1]) if cand.size else int(np.nanargmax(abs_y))
    t_star = min(n-1, t_break + j)
    P = min(int(period0), n)
    pre  = x[max(0, t_star - P): t_break]; post_seg = x[t_break: min(n, t_star + 1)]
    if len(pre) < 20 or len(post_seg) < 20:
        pre  = x[max(0, t_break-200): t_break]; post_seg = x[t_break: min(n, t_break+200)]
    ks_raw = stats.ks_2samp(pre, post_seg, method='asymp'); p_raw = float(ks_raw.pvalue)
    xd = _safe_savgol(x, SAVGOL[0], SAVGOL[1])
    pre_d  = xd[max(0, t_break-200): t_break] if len(pre)>=20 else pre
    post_d = xd[t_break: min(n, t_break+200)] if len(post_seg)>=20 else post_seg
    ks_den = stats.ks_2samp(pre_d, post_d, method='asymp'); p_den = float(ks_den.pvalue)
    z_ks = -np.log10(max(min(p_den, 1.0), 1e-300))
    return dict(
        ks_raw_p=max(min(p_raw,1.0),1e-300),
        ks_denoised_p=max(min(p_den,1.0),1e-300),
        z_ks=float(z_ks),
        log_ks_den=-np.log10(max(min(p_den,1.0),1e-300)),
        log_ks_raw=-np.log10(max(min(p_raw,1.0),1e-300))
    )

def lag_matrix(x, L):
    x = np.asarray(x, float)
    if len(x) <= L: return None, None
    X = np.column_stack([x[i:len(x)-L+i] for i in range(L, 0, -1)])
    y = x[L:]; return X, y

def ridge_fit_pred(Xtr, ytr, Xte, alpha=ALPHA_RIDGE):
    Xtr = np.asarray(Xtr, float); ytr = np.asarray(ytr, float).ravel()
    XtX = Xtr.T @ Xtr; n = XtX.shape[0]; I = np.eye(n, dtype=float)
    try:    beta = np.linalg.solve(XtX + alpha * I, Xtr.T @ ytr)
    except np.linalg.LinAlgError: beta = np.linalg.pinv(XtX + alpha * I) @ (Xtr.T @ ytr)
    yhat_tr = Xtr @ beta; yhat_te = (np.asarray(Xte, float) @ beta) if Xte is not None and len(Xte)>0 else None
    return yhat_tr, yhat_te

def predictive_divergence_feats(x, tb, lags=LAGS_PRED, alpha=ALPHA_RIDGE):
    x = zscore(np.asarray(x, float)); pre = x[:tb]; post = x[tb:]; feats={}
    ratios_pre, ratios_post, deltas_pre, deltas_post = [], [], [], []
    for L in lags:
        Xtr,ytr = lag_matrix(pre, L);  Xte,yte = lag_matrix(post, L)
        if Xtr is not None and Xte is not None:
            yhat_tr, yhat_te = ridge_fit_pred(Xtr,ytr,Xte,alpha=alpha)
            mse_in, mse_out = float(np.mean((ytr - yhat_tr)**2)), float(np.mean((yte - yhat_te)**2))
            r_pre = mse_out/(mse_in+1e-9); d_pre = mse_out-mse_in
            feats[f"pred_ratio_pre_L{L}"]=r_pre; feats[f"pred_delta_pre_L{L}"]=d_pre
            ratios_pre.append(r_pre); deltas_pre.append(d_pre)
        else:
            feats[f"pred_ratio_pre_L{L}"]=np.nan; feats[f"pred_delta_pre_L{L}"]=np.nan
        Xtr,ytr = lag_matrix(post, L); Xte,yte = lag_matrix(pre, L)
        if Xtr is not None and Xte is not None:
            yhat_tr, yhat_te = ridge_fit_pred(Xtr,ytr,Xte,alpha=alpha)
            mse_in, mse_out = float(np.mean((ytr - yhat_tr)**2)), float(np.mean((yte - yhat_te)**2))
            r_post = mse_out/(mse_in+1e-9); d_post = mse_out-mse_in
            feats[f"pred_ratio_post_L{L}"]=r_post; feats[f"pred_delta_post_L{L}"]=d_post
            ratios_post.append(r_post); deltas_post.append(d_post)
        else:
            feats[f"pred_ratio_post_L{L}"]=np.nan; feats[f"pred_delta_post_L{L}"]=np.nan
    # agregados existentes
    feats["pred_ratio_pre_max"]  = np.nanmax([feats[f"pred_ratio_pre_L{L}"]  for L in lags])
    feats["pred_ratio_post_max"] = np.nanmax([feats[f"pred_ratio_post_L{L}"] for L in lags])
    feats["pred_delta_pre_max"]  = np.nanmax([feats[f"pred_delta_pre_L{L}"]  for L in lags])
    feats["pred_delta_post_max"] = np.nanmax([feats[f"pred_delta_post_L{L}"]  for L in lags])
    feats["pred_ratio_geom_mean"]= float(np.exp(np.nanmean(np.log(np.clip([feats[f"pred_ratio_pre_L{L}"] for L in lags] + [feats[f"pred_ratio_post_L{L}"] for L in lags], 1e-9, None)))))
    # robust extras tipo golden
    def robust_stats(arr):
        arr = np.array(arr, float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return dict(med=np.nan, iqr=np.nan, idr=np.nan, rng=np.nan)
        med = np.nanmedian(arr); q25 = np.nanpercentile(arr,25); q75 = np.nanpercentile(arr,75)
        q10 = np.nanpercentile(arr,10); q90 = np.nanpercentile(arr,90)
        return dict(med=med, iqr=q75-q25, idr=q90-q10, rng=np.nanmax(arr)-np.nanmin(arr))
    s_pre  = robust_stats(ratios_pre)
    s_post = robust_stats(ratios_post)
    feats["pred_ratio_pre_median"]  = s_pre["med"]
    feats["pred_ratio_post_median"] = s_post["med"]
    feats["pred_ratio_pre_iqr_cv"]  = s_pre["iqr"] / (abs(s_pre["med"])+1e-9)
    feats["pred_ratio_post_iqr_cv"] = s_post["iqr"] / (abs(s_post["med"])+1e-9)
    feats["pred_ratio_pre_idr_cv"]  = s_pre["idr"] / (abs(s_pre["med"])+1e-9)
    feats["pred_ratio_post_idr_cv"] = s_post["idr"] / (abs(s_post["med"])+1e-9)
    feats["pred_ratio_pre_range_cv"]  = s_pre["rng"] / (abs(s_pre["med"])+1e-9)
    feats["pred_ratio_post_range_cv"] = s_post["rng"] / (abs(s_post["med"])+1e-9)
    return feats

def dist_shift_feats(x, tb, wins=W_DIST):
    x = zscore(np.asarray(x,float)); n=len(x); feats={}
    qs = [0.10,0.25,0.50,0.75,0.90]
    for W in wins:
        a=max(0,tb-W); b=min(n,tb+W); pre=x[a:tb]; post=x[tb:b]
        if len(pre)<20 or len(post)<20:
            for nm in ["was","cvm","ad","iqr_ratio","var_ratio","med_delta"]:
                feats[f"{nm}_W{W}"]=np.nan
            for q in qs: feats[f"qdiff_{int(q*100)}_W{W}"]=np.nan
            continue
        pre_r=_winsorize_clip(pre,0.01); post_r=_winsorize_clip(post,0.01)
        feats[f"was_W{W}"]= float(stats.wasserstein_distance(pre_r, post_r))
        feats[f"cvm_W{W}"]= float(stats.cramervonmises_2samp(pre_r, post_r).statistic)
        try: feats[f"ad_W{W}"]= float(stats.anderson_ksamp([pre_r, post_r]).statistic)
        except Exception: feats[f"ad_W{W}"]= np.nan
        iqr_pre=np.subtract(*np.percentile(pre_r,[75,25])); iqr_post=np.subtract(*np.percentile(post_r,[75,25]))
        feats[f"iqr_ratio_W{W}"]= float((iqr_post+1e-9)/(iqr_pre+1e-9))
        feats[f"var_ratio_W{W}"]= float((np.var(post_r)+1e-9)/(np.var(pre_r)+1e-9))
        feats[f"med_delta_W{W}"]= float(np.median(post_r)-np.median(pre_r))
        for q in qs: feats[f"qdiff_{int(q*100)}_W{W}"]= float(np.quantile(post_r,q)-np.quantile(pre_r,q))
        # robust extras por ventana (feature dorada)
        mean_pre, mean_post = float(np.mean(pre_r)), float(np.mean(post_r))
        med_pre,  med_post  = float(np.median(pre_r)), float(np.median(post_r))
        std_pre,  std_post  = float(np.std(pre_r)), float(np.std(post_r))
        iqr_pre,  iqr_post  = float(np.subtract(*np.percentile(pre_r,[75,25]))), float(np.subtract(*np.percentile(post_r,[75,25])))
        q10_pre, q90_pre = float(np.quantile(pre_r,0.10)), float(np.quantile(pre_r,0.90))
        q10_post,q90_post= float(np.quantile(post_r,0.10)), float(np.quantile(post_r,0.90))
        idr_pre, idr_post = (q90_pre-q10_pre), (q90_post-q10_post)
        rng_pre, rng_post = float(np.max(pre_r)-np.min(pre_r)), float(np.max(post_r)-np.min(post_r))
        q05_pre,q95_pre = float(np.quantile(pre_r,0.05)), float(np.quantile(pre_r,0.95))
        q05_post,q95_post= float(np.quantile(post_r,0.05)), float(np.quantile(post_r,0.95))
        tm_pre = float(np.mean(pre_r[(pre_r>q05_pre)&(pre_r<q95_pre)])) if len(pre_r)>4 else mean_pre
        tm_post= float(np.mean(post_r[(post_r>q05_post)&(post_r<q95_post)])) if len(post_r)>4 else mean_post
        rms_pre = float(np.sqrt(np.mean(pre_r**2))) if len(pre_r)>0 else 0.0
        rms_post= float(np.sqrt(np.mean(post_r**2))) if len(post_r)>0 else 0.0
        # CVs robustos pre/post
        feats[f"iqr_cv_pre_W{W}"]  = float(iqr_pre /(abs(mean_pre)+1e-9))
        feats[f"iqr_cv_post_W{W}"] = float(iqr_post/(abs(mean_post)+1e-9))
        feats[f"iqr_cv_med_pre_W{W}"]  = float(iqr_pre /(abs(med_pre)+1e-9))
        feats[f"iqr_cv_med_post_W{W}"] = float(iqr_post/(abs(med_post)+1e-9))
        feats[f"idr_cv_pre_W{W}"]  = float(idr_pre /(abs(mean_pre)+1e-9))
        feats[f"idr_cv_post_W{W}"] = float(idr_post/(abs(mean_post)+1e-9))
        feats[f"range_cv_pre_W{W}"]  = float(rng_pre /(abs(mean_pre)+1e-9))
        feats[f"range_cv_post_W{W}"] = float(rng_post/(abs(mean_post)+1e-9))
        feats[f"range_cv_med_pre_W{W}"]  = float(rng_pre /(abs(med_pre)+1e-9))
        feats[f"range_cv_med_post_W{W}"] = float(rng_post/(abs(med_post)+1e-9))
        feats[f"cv_trimmed_mean_pre_W{W}"]  = float(std_pre /(abs(tm_pre)+1e-9))
        feats[f"cv_trimmed_mean_post_W{W}"] = float(std_post/(abs(tm_post)+1e-9))
        feats[f"iqr_cv_trimmed_mean_pre_W{W}"]  = float(iqr_pre /(abs(tm_pre)+1e-9))
        feats[f"iqr_cv_trimmed_mean_post_W{W}"] = float(iqr_post/(abs(tm_post)+1e-9))
        feats[f"std_div_by_rms_pre_W{W}"]  = float(std_pre /(rms_pre+1e-9))
        feats[f"std_div_by_rms_post_W{W}"] = float(std_post/(rms_post+1e-9))
    for base in ["was","cvm","ad","iqr_ratio","var_ratio","med_delta"]:
        vals=[feats.get(f"{base}_W{W}", np.nan) for W in wins]
        feats[f"{base}_max"]=np.nanmax(vals); feats[f"{base}_mean"]=np.nanmean(vals)
    # agregados de robust extras
    robust_keys = [
        "iqr_cv_pre_W{W}", "iqr_cv_post_W{W}", "iqr_cv_med_pre_W{W}", "iqr_cv_med_post_W{W}",
        "idr_cv_pre_W{W}", "idr_cv_post_W{W}", "range_cv_pre_W{W}", "range_cv_post_W{W}",
        "range_cv_med_pre_W{W}", "range_cv_med_post_W{W}", "cv_trimmed_mean_pre_W{W}", "cv_trimmed_mean_post_W{W}",
        "iqr_cv_trimmed_mean_pre_W{W}", "iqr_cv_trimmed_mean_post_W{W}", "std_div_by_rms_pre_W{W}", "std_div_by_rms_post_W{W}"
    ]
    for rk in robust_keys:
        vals=[feats.get(rk.format(W=W), np.nan) for W in wins]
        feats[rk.replace("{W}", "_max")]  = np.nanmax(vals)
        feats[rk.replace("{W}", "_mean")] = np.nanmean(vals)
    return feats

# ---- MMD RBF (Impl2) ----
def _median_heuristic_sigma(x, y):
    z = np.hstack([x, y]).astype(float).ravel()
    if z.size <= 1: return 1.0
    dz = np.abs(z[:, None] - z[None, :]).ravel()
    dz = dz[dz > 0]
    sig = np.median(dz) if dz.size else 1.0
    return max(float(sig), 1e-3)

def mmd_rbf_1d(x, y, sigma=None):
    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
    if x.size == 0 or y.size == 0: return 0.0
    if sigma is None: sigma = _median_heuristic_sigma(x, y)
    g = lambda a, b: np.exp(-((a[:,None]-b[None,:])**2)/(2*sigma**2))
    Kxx = g(x, x); Kyy = g(y, y); Kxy = g(x, y)
    np.fill_diagonal(Kxx, 0.0); np.fill_diagonal(Kyy, 0.0)
    m, n = len(x), len(y)
    return float(Kxx.sum()/(m*(m-1)+1e-9) + Kyy.sum()/(n*(n-1)+1e-9) - 2*Kxy.mean())

def mmd_block_feats(x, tb, wins=MMD_WINDOWS):
    x = zscore(np.asarray(x,float)); n=len(x); feats={}
    vals = []
    for W in wins:
        a=max(0, tb-W); b=min(n, tb+W); pre=x[a:tb]; post=x[tb:b]
        v = mmd_rbf_1d(pre, post) if (len(pre)>3 and len(post)>3) else 0.0
        feats[f"mmd_W{W}"]= v
        vals.append(v)
    arr = np.array(vals, float)
    if arr.size:
        med = float(np.nanmedian(arr))
        q25 = float(np.nanpercentile(arr,25))
        q75 = float(np.nanpercentile(arr,75))
        iqr = q75 - q25
        feats["mmd_max"]   = float(np.nanmax(arr))
        feats["mmd_mean"]  = float(np.nanmean(arr))
        feats["mmd_median"] = med
        feats["mmd_iqr"]    = iqr
        feats["mmd_iqr_cv"] = float(iqr / (abs(med)+1e-9))
    return feats

# ---- Espectral multiescala ( Impl2 ) ----
def lag_embed(x, L=20):
    x = np.asarray(x, float); n = len(x)
    if n < L: return None
    return np.column_stack([x[i:n-L+1+i] for i in range(L)][::-1])

def _spectral_basis_from_segment(seg, L, m):
    V = lag_embed(seg, L=L)
    if V is None or V.shape[0] < max(10, m+2): return None
    G = np.cov(V, rowvar=False); _, U = np.linalg.eigh(G)
    U_m = U[:, -min(m, U.shape[1]):]
    return U_m

def spectral_basis_pre_post(x, t_break, w_pre=200, spec_wpost=200, spec_burn=0, L=20, m=3):
    x = zscore(np.asarray(x, float)); n = len(x)
    a = max(0, t_break - w_pre); pre = x[a:t_break]
    start = min(n, t_break + spec_burn); stop = min(n, t_break + spec_wpost)
    post = x[start:stop]
    U_pre = _spectral_basis_from_segment(pre, L, m); U_post= _spectral_basis_from_segment(post, L, m)
    return U_pre, U_post

def spectral_projector_from_post(x, t_break, spec_wpost=200, spec_burn=0, L=20, m=3):
    x = zscore(np.asarray(x, float)); n = len(x)
    start = min(n, t_break + spec_burn); stop = min(n, t_break + spec_wpost)
    post = x[start:stop]; U_m = _spectral_basis_from_segment(post, L, m)
    return (U_m @ U_m.T) if U_m is not None else None

def spectral_cusum_percent(x, t_break, w_pre=200, spec_wpost=200, spec_burn=0, L=20, m=3):
    P = spectral_projector_from_post(x, t_break, spec_wpost, spec_burn, L, m)
    if P is None: return None
    x = zscore(np.asarray(x, float)); n = len(x)
    a = max(0, t_break - w_pre); start_post = min(n, t_break + spec_burn); b = min(n, t_break + spec_wpost)
    x_win = np.r_[x[a:t_break], x[start_post:b]]
    V = lag_embed(x_win, L=L)
    if V is None or V.shape[0] < 10: return None
    n_pre = max(0, (t_break - a) - L + 1); n_pre = min(n_pre, len(V))
    q = np.einsum('ij,jk,ik->i', V, P, V)
    d = float(np.median(q[:n_pre])) if n_pre > 5 else float(np.median(q)) if len(q)>0 else 0.0
    S = 0.0; S_vals=[]
    for qi in q:
        S = max(0.0, S + (-qi + d)); S_vals.append(S)
    S_pre_max  = float(np.max(S_vals[:n_pre])) if n_pre>0 else 0.0
    S_post_max = float(np.max(S_vals[n_pre:])) if n_pre<len(S_vals) else 0.0
    return S_post_max / (S_post_max + S_pre_max + 1e-12)

def subspace_similarity(x, t_break, w_pre=200, spec_wpost=200, spec_burn=0, L=20, m=3):
    U_pre, U_post = spectral_basis_pre_post(x, t_break, w_pre, spec_wpost, spec_burn, L, m)
    if U_pre is None or U_post is None: return None, None
    P_pre, P_post = U_pre @ U_pre.T, U_post @ U_post.T
    m_eff = min(U_pre.shape[1], U_post.shape[1])
    tr_norm = float(np.trace(P_pre @ P_post) / max(1, m_eff))
    try:
        s = np.linalg.svd(U_pre.T @ U_post, compute_uv=False); cos1 = float(np.max(s)) if s.size else np.nan
    except Exception: cos1 = np.nan
    return tr_norm, cos1

def build_multiscale_spectral_features(series_dict, tbreak_dict, desc="Spectral"):
    scales=[]
    for L in SPEC_L_LIST:
        for m in SPEC_M_LIST:
            for spec_wpost in SPEC_WPOST_LIST:
                for spec_burn in SPEC_BURN_LIST:
                    if spec_wpost - spec_burn > L + 12:
                        scales.append(dict(L=L, m=m, spec_wpost=spec_wpost, spec_burn=spec_burn))
    ids = list(series_dict.keys())
    data_percent, data_trace, data_cos = {}, {}, {}
    for sc in tqdm(scales, desc=desc):
        key_p = f"spP_L{sc['L']}_m{sc['m']}_b{sc['spec_burn']}_w{sc['spec_wpost']}"
        key_t = f"spT_L{sc['L']}_m{sc['m']}_b{sc['spec_burn']}_w{sc['spec_wpost']}"
        key_c = f"spC_L{sc['L']}_m{sc['m']}_b{sc['spec_burn']}_w{sc['spec_wpost']}"
        colP, colT, colC = [], [], []
        for i in ids:
            x = series_dict[i]; tb = tbreak_dict[i]
            vP = spectral_cusum_percent(x, tb, 200, sc['spec_wpost'], sc['spec_burn'], sc['L'], sc['m'])
            tr, cs = subspace_similarity(x, tb, 200, sc['spec_wpost'], sc['spec_burn'], sc['L'], sc['m'])
            colP.append(vP if vP is not None else np.nan)
            colT.append(tr if tr is not None else np.nan)
            colC.append(cs if cs is not None else np.nan)
        data_percent[key_p] = colP; data_trace[key_t] = colT; data_cos[key_c] = colC
    dfP = pd.DataFrame(data_percent, index=ids)
    dfT = pd.DataFrame(data_trace,   index=ids)
    dfC = pd.DataFrame(data_cos,     index=ids)
    agg = pd.DataFrame(index=ids)
    A = dfP.to_numpy(); agg["spec_percent_max"]  = np.nanmax(A, axis=1); agg["spec_percent_mean"] = np.nanmean(A, axis=1)
    B = dfT.to_numpy(); agg["spec_trace_max"]    = np.nanmax(B, axis=1); agg["spec_trace_mean"]   = np.nanmean(B, axis=1)
    C = dfC.to_numpy(); agg["spec_cos_max"]      = np.nanmax(C, axis=1); agg["spec_cos_mean"]     = np.nanmean(C, axis=1)
    # Robust summaries tipo golden
    agg["spec_percent_median"] = np.nanmedian(A, axis=1)
    agg["spec_trace_median"]   = np.nanmedian(B, axis=1)
    agg["spec_cos_median"]     = np.nanmedian(C, axis=1)
    def row_iqr(M):
        q75 = np.nanpercentile(M, 75, axis=1)
        q25 = np.nanpercentile(M, 25, axis=1)
        return q75 - q25
    iqrP = row_iqr(A); iqrT = row_iqr(B); iqrC = row_iqr(C)
    agg["spec_percent_iqr"] = iqrP
    agg["spec_trace_iqr"]   = iqrT
    agg["spec_cos_iqr"]     = iqrC
    agg["spec_percent_iqr_cv"] = iqrP / (np.abs(agg["spec_percent_median"]) + 1e-9)
    agg["spec_trace_iqr_cv"]   = iqrT / (np.abs(agg["spec_trace_median"]) + 1e-9)
    agg["spec_cos_iqr_cv"]     = iqrC / (np.abs(agg["spec_cos_median"]) + 1e-9)
    return agg.add_prefix("ag_")

# -------- Señales para Curriculum --------
def _welch_t_logp(pre, post):
    if len(pre) < 3 or len(post) < 3:
        return 0.0
    t, p = stats.ttest_ind(pre, post, equal_var=False, nan_policy='omit')
    p = float(0.5) if (p is None or not np.isfinite(p)) else float(p)
    return float(-np.log10(max(min(p,1.0),1e-300)))

def _lb_logp_delta(pre, post, lags=(10,20,30)):
    if not HAS_LB:
        return 0.0
    def logp(series):
        if len(series) < max(lags)+2:
            return 0.0
        out = acorr_ljungbox(series, lags=list(lags), return_df=True)
        pmin = np.clip(out['lb_pvalue'].values, 1e-300, 1.0)
        return float(np.nanmax(-np.log10(pmin)))
    return max(0.0, logp(post) - logp(pre))

def compute_signal_tests(series_dict, tbreak_dict, wins=(64,128)):
    ids = list(series_dict.keys())
    vals = []
    for i in tqdm(ids, desc="Signals (t-test & Ljung-Box)"):
        x = zscore(np.asarray(series_dict[i], float))
        tb = tbreak_dict[i]; n = len(x)
        tlogs, lbd = [], []
        for W in wins:
            a=max(0,tb-W); b=min(n,tb+W)
            pre = x[a:tb]; post = x[tb:b]
            if len(pre)<6 or len(post)<6:
                continue
            tlogs.append(_welch_t_logp(pre, post))
            lbd.append(_lb_logp_delta(pre, post))
        tt_logp_min = float(np.nanmax(tlogs)) if len(tlogs) else 0.0
        lb_delta    = float(np.nanmax(lbd))   if len(lbd)   else 0.0
        # robust extras tipo golden
        def robust(arr):
            arr = np.array(arr, float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return dict(med=np.nan, iqr=np.nan)
            med = float(np.nanmedian(arr))
            q25 = float(np.nanpercentile(arr,25))
            q75 = float(np.nanpercentile(arr,75))
            return dict(med=med, iqr=(q75-q25))
        s_t = robust(tlogs)
        s_l = robust(lbd)
        vals.append((i, tt_logp_min, lb_delta, s_t['med'], s_t['iqr'], s_l['med'], s_l['iqr']))
    df = pd.DataFrame(vals, columns=["id","tt_logp_min","lb_logp_delta_max","tt_logp_median","tt_logp_iqr","lb_logp_delta_median","lb_logp_delta_iqr"]).set_index("id")
    return df

# ===================== PINT (SHO) — FIX target =====================
model_pint = None

if HAS_TORCH:
    class PINTLSTM(nn.Module):
        def __init__(self, hidden=PINT_HIDDEN, layers=PINT_LAYERS, dropout=PINT_DROPOUT):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers, 
                               batch_first=True, dropout=dropout if layers > 1 else 0)
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden//2, 1)
            )
            self._w_raw = nn.Parameter(torch.tensor(0.0))
            self.softplus = nn.Softplus()
        def forward(self, x, h0=None):
            y, h = self.lstm(x, h0)
            out = self.head(y)
            return out, h
        @property
        def w2(self):
            return self.softplus(self._w_raw) + 1e-8

    def second_derivative_discrete(y):
        return y[:, 2:] - 2*y[:, 1:-1] + y[:, :-2]

    def sho_residual(y, w2):
        y2 = second_derivative_discrete(y)
        y_mid = y[:, 1:-1]
        return y2 + w2 * y_mid

    def pint_loss(pred, target, lambda_phys=PINT_LAMBDA_PHYS, w2=None):
        pred1 = pred.squeeze(-1); tgt1 = target.squeeze(-1)
        mse = torch.mean((pred1 - tgt1)**2)
        if pred1.size(1) >= 3 and (w2 is not None):
            res = sho_residual(pred1, w2=w2)
            phys = torch.mean(res**2)
        else:
            phys = torch.tensor(0.0, device=pred1.device)
        return mse + lambda_phys*phys, mse, phys

    def build_global_pint_dataset(series_dict, tbreak_dict, in_len=PINT_IN_LEN, out_len=PINT_OUT_LEN):
        """
        X: [in_len, 1] con el pre-break.
        Y: objetivo 'one-step-ahead' del mismo largo in_len:
           y[t] = x[t+1] para t<in_len-1, y[in_len-1] = primer valor post-break REAL.
        Busca el break real en ventana [t_break : t_break + search_delay].
        """
        X_list, Y_list = [], []
        for i, x in series_dict.items():
            tb = tbreak_dict[i]
            xz = zscore(np.asarray(x, float))
            if tb <= in_len + out_len + 4:
                continue
            
            # Buscar break real en ventana deslizante
            search_end = min(len(xz), tb + PINT_SEARCH_DELAY)
            best_error = -np.inf
            best_break = tb
            
            for candidate_break in range(tb, search_end):
                if candidate_break <= in_len + out_len + 4:
                    continue
                start = candidate_break - (in_len + out_len)
                if start < 0:
                    continue
                    
                xin = xz[start:start+in_len]
                yout = xz[start+in_len:start+in_len+out_len]
                if len(xin) == in_len and len(yout) >= 1:
                    # Calcular error de predicción simple (diferencias)
                    pred_error = np.mean(np.abs(np.diff(xin[-10:])))  # Variabilidad en últimos 10 puntos
                    if pred_error > best_error:
                        best_error = pred_error
                        best_break = candidate_break
            
            # Usar el break real encontrado
            start = best_break - (in_len + out_len)
            if start >= 0:
                xin = xz[start:start+in_len]
                yout = xz[start+in_len:start+in_len+out_len]
                if len(xin) == in_len and len(yout) >= 1:
                    # Objetivo del mismo largo que xin
                    y_target = np.empty_like(xin)
                    y_target[:-1] = xin[1:]            # next-step dentro de xin
                    y_target[-1] = yout[0]             # primer paso futuro (post-break real)
                    X_list.append(xin.reshape(-1,1))
                    Y_list.append(y_target.reshape(-1,1))
        
        if not X_list:
            return None, None
        X = np.stack(X_list, axis=0).astype(np.float32)
        Y = np.stack(Y_list, axis=0).astype(np.float32)
        return X, Y

    def train_pint_global(series_dict, tbreak_dict, desc=None):
        global model_pint
        X, Y = build_global_pint_dataset(series_dict, tbreak_dict)
        if X is None:
            print(">> [PINT] Dataset global vacío; se desactiva PINT.")
            return None
        ds_X = torch.from_numpy(X)
        target = torch.from_numpy(Y)  # <-- FIX CLAVE: usar objetivo correcto
        model = PINTLSTM().to(PINT_DEVICE)
        opt = optim.Adam(model.parameters(), lr=PINT_LR)
        model.train()
        n = ds_X.size(0)
        idx = np.arange(n)
        from tqdm import trange
        for ep in trange(PINT_EPOCHS, desc=(desc or "PINT"), leave=False):
            np.random.shuffle(idx)
            tot, mse_t, phys_t = 0.0, 0.0, 0.0
            for s in range(0, n, PINT_BS):
                batch = idx[s:s+PINT_BS]
                xb = ds_X[batch].to(PINT_DEVICE)
                yb = target[batch].to(PINT_DEVICE)
                pred, _ = model(xb)
                loss, l_mse, l_phys = pint_loss(pred, yb, lambda_phys=PINT_LAMBDA_PHYS, w2=model.w2)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.item()) * len(batch)
                mse_t += float(l_mse.item()) * len(batch)
                phys_t += float(l_phys.item()) * len(batch)
        model_pint = model
        return model

    
    @torch.no_grad()
    def pint_rollout_errors_for_id(x, tb, in_len=PINT_IN_LEN, H_list=PINT_H_LIST):
        """
        Recupera el flujo original con:
        - búsqueda de break real por ventana deslizante,
        - forward one-step sobre el input (pred_in),
        - rollout auto-regresivo para H en H_list (pred_ar),
        - métricas de error (MAE/MSE), backtests, ratios y slope,
        - restricciones físicas (residuo SHO) + estimación w^2,
        - similitud 'hidden_cos' y métricas robustas de error/contexto.

        Requiere: model_pint entrenado (global).
        """
        if model_pint is None:
            return None

        xz = zscore(np.asarray(x, float))
        n = len(xz)
        if tb <= in_len + 3 or tb >= n - 3:
            return None

        # --- 1) Buscar break "real" en [tb, tb + PINT_SEARCH_DELAY) (heurística original) ---
        search_end = min(n, tb + PINT_SEARCH_DELAY)
        best_error = -np.inf
        best_break = tb
        for candidate_break in range(tb, search_end):
            if candidate_break <= in_len + 3 or candidate_break >= n - 3:
                continue
            # misma heurística del código "muerto": media |diff| en últimos 10 puntos previos
            seg = xz[max(0, candidate_break - 10):candidate_break]
            if len(seg) >= 2:
                pred_error = float(np.mean(np.abs(np.diff(seg))))
                if pred_error > best_error:
                    best_error = pred_error
                    best_break = candidate_break

        # --- 2) Construir ventana pre y post en torno al break real ---
        xin = xz[best_break - in_len:best_break].astype(np.float32)
        post = xz[best_break:]
        if len(xin) < in_len or len(post) < 1:
            return None

        # --- 3) Forward sobre xin (predicción one-step para cada t del input) ---
        xb = torch.from_numpy(xin.reshape(1, -1, 1)).to(PINT_DEVICE)
        model_pint.eval()
        pred_in, (h, c) = model_pint(xb)              # pred_in: (1, in_len, 1)
        last = pred_in[:, -1:, :]                      # último paso predicho dentro del input

        # --- 4) Rollout auto-regresivo para horizonte máximo ---
        maxH = max(H_list)
        preds = []
        for _ in range(maxH):
            y_next, (h, c) = model_pint(last, (h, c))
            preds.append(y_next.squeeze(-1))          # (1, 1) -> (1,)
            last = y_next
        if preds:
            pred_ar = torch.cat(preds, dim=1).cpu().numpy().ravel()  # (1, maxH) -> (maxH,)
        else:
            pred_ar = np.array([], dtype=np.float32)

        feats = {}

        # --- 5) Errores sobre el tramo post (horizontes H_list) ---
        for H in H_list:
            if len(post) >= H and len(pred_ar) >= H:
                err = post[:H] - pred_ar[:H]
                feats[f"pint_mae_H{H}"] = float(np.mean(np.abs(err)))
                feats[f"pint_mse_H{H}"] = float(np.mean(err**2))
            else:
                feats[f"pint_mae_H{H}"] = np.nan
                feats[f"pint_mse_H{H}"] = np.nan

        # --- 6) Backtest dentro del input (comparar xin con pred_in) ---
        pred_in_np = pred_in.cpu().numpy().reshape(-1)  # (in_len,)
        for H in H_list:
            if in_len >= H:
                back_err = xin[-H:] - pred_in_np[-H:]
                feats[f"pint_back_mae_H{H}"] = float(np.mean(np.abs(back_err)))
                feats[f"pint_back_mse_H{H}"] = float(np.mean(back_err**2))
                denom = feats[f"pint_back_mse_H{H}"] + 1e-9
                feats[f"pint_err_ratio_H{H}"] = feats[f"pint_mse_H{H}"] / denom if np.isfinite(denom) else np.nan
            else:
                feats[f"pint_back_mae_H{H}"] = np.nan
                feats[f"pint_back_mse_H{H}"] = np.nan
                feats[f"pint_err_ratio_H{H}"] = np.nan

        # --- 7) Pendiente (slope) de MAE vs H (tendencia de error) ---
        Hs_valid = [h for h in H_list if np.isfinite(feats.get(f"pint_mae_H{h}", np.nan))]
        if len(Hs_valid) >= 2:
            ys = np.array([feats[f"pint_mae_H{h}"] for h in Hs_valid], float)
            xs = np.array(Hs_valid, float)
            A = np.vstack([xs, np.ones_like(xs)]).T
            slope, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
            feats["pint_mae_slope"] = float(slope)
        else:
            feats["pint_mae_slope"] = np.nan

        # --- 8) Penalización física (SHO) sobre predicciones (pre y post) ---
        w2 = float(model_pint.w2.item())
        def _phys_resid_mean(vec: np.ndarray) -> float:
            if vec.size < 3: 
                return np.nan
            y2 = vec[2:] - 2*vec[1:-1] + vec[:-2]
            ymid = vec[1:-1]
            r = y2 + w2 * ymid
            return float(np.mean(r**2))

        feats["pint_phys_resid_post"] = _phys_resid_mean(pred_ar)
        feats["pint_phys_resid_pre"]  = _phys_resid_mean(pred_in_np)
        denom_phys = feats["pint_phys_resid_pre"] + 1e-9 if np.isfinite(feats["pint_phys_resid_pre"]) else np.nan
        feats["pint_phys_resid_ratio"] = (feats["pint_phys_resid_post"] / denom_phys) if np.isfinite(denom_phys) else np.nan

        # --- 9) Estimación de ω^2 (w2) en datos reales pre/post y su delta ---
        def _est_omega_sq(y: np.ndarray) -> float:
            if len(y) < 5:
                return np.nan
            y = zscore(y.astype(float))
            y2 = y[2:] - 2*y[1:-1] + y[:-2]
            ymid = y[1:-1]
            num = -np.sum(y2 * ymid)
            den = np.sum(ymid**2) + 1e-12
            w2_hat = num / den
            return float(max(w2_hat, 0.0))

        pre_real  = xz[best_break - in_len:best_break]
        post_real = xz[best_break: min(n, best_break + maxH + in_len)]
        feats["pint_w2_pre_hat"]  = _est_omega_sq(pre_real)  if len(pre_real)  >= in_len else np.nan
        feats["pint_w2_post_hat"] = _est_omega_sq(post_real) if len(post_real) >= in_len else np.nan
        if np.isfinite(feats["pint_w2_pre_hat"]) and np.isfinite(feats["pint_w2_post_hat"]):
            feats["pint_abs_dw2"] = float(abs(feats["pint_w2_post_hat"] - feats["pint_w2_pre_hat"]))
        else:
            feats["pint_abs_dw2"] = np.nan

        # --- 10) Similitud entre colas de pred_in y cabeza de pred_ar ---
        if len(pred_in_np) >= 8 and len(pred_ar) >= 8:
            a = (pred_in_np[-8:] - np.mean(pred_in_np[-8:])) / (np.std(pred_in_np[-8:]) + 1e-9)
            b = (pred_ar[:8]       - np.mean(pred_ar[:8]))     / (np.std(pred_ar[:8]) + 1e-9)
            feats["pint_hidden_cos"] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        else:
            feats["pint_hidden_cos"] = np.nan

        # --- 11) Métricas robustas del error post + contexto global ---
        try:
            H_ref = max([h for h in H_list if h <= len(post)], default=None)
            if H_ref is not None and H_ref > 0 and len(pred_ar) >= H_ref:
                err_post = post[:H_ref] - pred_ar[:H_ref]

                def q(x, p):
                    return float(np.quantile(x, p)) if len(x) > 0 else np.nan

                mean_e = float(np.mean(err_post)) if len(err_post) > 0 else 0.0
                med_e  = float(np.median(err_post)) if len(err_post) > 0 else 0.0
                std_e  = float(np.std(err_post)) if len(err_post) > 0 else 0.0
                iqr_e  = q(err_post, 0.75) - q(err_post, 0.25)
                idr_e  = q(err_post, 0.90) - q(err_post, 0.10)
                rng_e  = (np.max(err_post) - np.min(err_post)) if len(err_post) > 0 else 0.0
                rms_e  = float(np.sqrt(np.mean(err_post**2))) if len(err_post) > 0 else 0.0
                q05_e, q95_e = q(err_post, 0.05), q(err_post, 0.95)
                tm_e = float(np.mean(err_post[(err_post > q05_e) & (err_post < q95_e)])) if len(err_post) > 4 else mean_e

                feats["pint_err_iqr_cv"] = float(iqr_e / (abs(mean_e) + 1e-9))
                feats["pint_err_iqr_cv_med"] = float(iqr_e / (abs(med_e) + 1e-9))
                feats["pint_err_idr_cv"] = float(idr_e / (abs(mean_e) + 1e-9))
                feats["pint_err_range_cv"] = float(rng_e / (abs(mean_e) + 1e-9))
                feats["pint_err_range_cv_med"] = float(rng_e / (abs(med_e) + 1e-9))
                feats["pint_err_cv_trimmed_mean"] = float(std_e / (abs(tm_e) + 1e-9))
                feats["pint_err_iqr_cv_trimmed_mean"] = float(iqr_e / (abs(tm_e) + 1e-9))
                feats["pint_err_std_div_by_rms"] = float(std_e / (rms_e + 1e-9))

                # Contexto global del id (sobre niveles, no errores)
                mu_g = float(np.mean(xz)) if len(xz) > 0 else 0.0
                sd_g = float(np.std(xz)) if len(xz) > 0 else 0.0
                feats["pint_norm_mae_Href"] = float(np.mean(np.abs(err_post)) / (sd_g + 1e-9))
                feats["pint_norm_mse_Href"] = float(np.mean(err_post**2) / ((sd_g**2) + 1e-9))

                pre_mu  = float(np.mean(xz[best_break - in_len:best_break])) if best_break - in_len >= 0 else float(np.mean(xz[:best_break]))
                post_mu = float(np.mean(post[:H_ref])) if H_ref > 0 else pre_mu
                feats["pint_pre_mean_deviation"]  = float((pre_mu  - mu_g) / (sd_g + 1e-9))
                feats["pint_post_mean_deviation"] = float((post_mu - mu_g) / (sd_g + 1e-9))
                feats["pint_change_in_deviation_from_global"] = float(
                    abs((post_mu - mu_g) / (sd_g + 1e-9)) - abs((pre_mu - mu_g) / (sd_g + 1e-9))
                )
        except Exception:
            # mantenemos robustez; si algo falla, seguimos con lo calculado
            pass

        return feats


    def pint_hybrid_rollout_for_id(x, tb_real, model, device=PINT_DEVICE):
        """Rollout híbrido: evalúa el LSTM en una ventana alineada con training
        y extrae features robustas alrededor del break real.
        """
        x = np.asarray(x, dtype=np.float32)
        n = len(x)
        left = max(0, tb_real - PINT_IN_LEN)
        right = min(n, tb_real + max(PINT_H_LIST))
        x_win = x[left:right]
        xt = torch.tensor(x_win, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)

        model.eval()
        with torch.no_grad():
            pred, _ = model(xt)
        pred = pred.squeeze().cpu().numpy()
        err = np.abs(x_win - pred)

        feats = {}
        mu, sd = np.nanmean(err), np.nanstd(err)
        feats["pint_h_err_mean"] = float(mu)
        feats["pint_h_err_std"] = float(sd)
        feats["pint_h_err_median"] = float(np.nanmedian(err))
        feats["pint_h_err_max"] = float(np.nanmax(err))
        feats["pint_h_err_q10"] = float(np.nanpercentile(err, 10))
        feats["pint_h_err_q25"] = float(np.nanpercentile(err, 25))
        feats["pint_h_err_q75"] = float(np.nanpercentile(err, 75))
        feats["pint_h_err_q90"] = float(np.nanpercentile(err, 90))
        feats["pint_h_err_q95"] = float(np.nanpercentile(err, 95))
        feats["pint_h_err_iqr"] = float(np.nanpercentile(err, 75) - np.nanpercentile(err, 25))
        feats["pint_h_err_cv"] = float(sd / (mu + 1e-9))
        feats["pint_h_err_mad"] = float(np.nanmedian(np.abs(err - np.nanmedian(err))))
        # Momentos si hay datos suficientes
        if len(err) > 3:
            from scipy.stats import skew, kurtosis
            feats["pint_h_err_skew"] = float(skew(err, nan_policy="omit"))
            feats["pint_h_err_kurt"] = float(kurtosis(err, nan_policy="omit"))
        else:
            feats["pint_h_err_skew"] = 0.0
            feats["pint_h_err_kurt"] = 0.0
        feats["pint_h_err_energy"] = float(np.nansum(err**2))
        feats["pint_h_err_peaks_gt2s"] = float(np.sum(err > (mu + 2*sd)))

        tb_rel = tb_real - left
        if 0 <= tb_rel < len(err):
            feats["pint_h_err_at_break"] = float(err[tb_rel])
            feats["pint_h_err_break_window"] = float(np.nanmean(err[max(0, tb_rel-5):min(len(err), tb_rel+5)]))
            for w in (10, 20, 30):
                lo = max(0, tb_rel - w)
                hi = min(len(err), tb_rel + w)
                seg = err[lo:hi]
                half = max(1, len(seg)//2)
                pre_err_w  = seg[:half]
                post_err_w = seg[-half:]
                if len(pre_err_w) and len(post_err_w):
                    feats[f"pint_h_err_pre_std_w{w}"]  = float(np.nanstd(pre_err_w))
                    feats[f"pint_h_err_post_std_w{w}"] = float(np.nanstd(post_err_w))
                    feats[f"pint_h_err_pre_post_ratio_w{w}"] = float(
                        (np.nanstd(post_err_w))/(np.nanstd(pre_err_w)+1e-9)
                    )
        else:
            feats["pint_h_err_at_break"] = 0.0
            feats["pint_h_err_break_window"] = 0.0
            for w in (10, 20, 30):
                feats[f"pint_h_err_pre_std_w{w}"] = 0.0
                feats[f"pint_h_err_post_std_w{w}"] = 0.0
                feats[f"pint_h_err_pre_post_ratio_w{w}"] = 0.0

        return feats


    def build_features_pint(series_dict, tbreak_dict, desc="PINT (features)"):
        if not (HAS_TORCH and USE_PINT):
            return pd.DataFrame(index=list(series_dict.keys()))
        if train_pint_global(series_dict, tbreak_dict) is None:
            return pd.DataFrame(index=list(series_dict.keys()))
        ids = list(series_dict.keys())
        rows = []
        for i in tqdm(ids, desc=desc):
            f = pint_rollout_errors_for_id(series_dict[i], tbreak_dict[i])
            rows.append(f if f is not None else {})
        DF = pd.DataFrame(rows, index=ids)
        # Evita 'pint_pint_...'
        if not all(c.startswith("pint_") for c in DF.columns):
            DF = DF.add_prefix("pint_")
        return DF

# ================= FE Impl2 (sin filtro KS aquí) =================
def build_features_impl2(X_train_mi: pd.DataFrame, X_test_mi: pd.DataFrame,
                         extra_impl3_feats=None, extra_pint_feats_tr=None, extra_pint_feats_te=None):
    tr_series, tr_tbreak = to_series_dict(X_train_mi)
    te_series, te_tbreak = to_series_dict(X_test_mi)

    ids_tr = list(tr_series.keys())
    base_tr = pd.DataFrame({i: base_features_for_id(tr_series[i], tr_tbreak[i]) for i in tqdm(ids_tr, desc="Base (train)")}).T
    ids_te = list(te_series.keys())
    base_te = pd.DataFrame({i: base_features_for_id(te_series[i], te_tbreak[i]) for i in tqdm(ids_te, desc="Base (test)")}).T

    sp_aggr_tr = build_multiscale_spectral_features(tr_series, tr_tbreak, desc="Spectral (train)")
    sp_aggr_te = build_multiscale_spectral_features(te_series, te_tbreak, desc="Spectral (test)")

    abc_tr = pd.DataFrame({i: abc_features_for_id(tr_series[i], tr_tbreak[i]) for i in tqdm(ids_tr, desc="ABC (train)")}).T
    abc_te = pd.DataFrame({i: abc_features_for_id(te_series[i], te_tbreak[i]) for i in tqdm(ids_te, desc="ABC (test)")}).T
    for df in (abc_tr, abc_te):
        df['log_ks_den'] = -np.log10(df['ks_denoised_p'].clip(1e-300,1.0))
        df['log_ks_raw'] = -np.log10(df['ks_raw_p'].clip(1e-300,1.0))
    abc_sel_tr = abc_tr[['ks_raw_p','ks_denoised_p','z_ks','log_ks_den','log_ks_raw']].copy()
    abc_sel_te = abc_te[['ks_raw_p','ks_denoised_p','z_ks','log_ks_den','log_ks_raw']].copy()

    pred_tr = pd.DataFrame({i: predictive_divergence_feats(tr_series[i], tr_tbreak[i]) for i in tqdm(ids_tr, desc="Predictive (train)")}).T
    pred_te = pd.DataFrame({i: predictive_divergence_feats(te_series[i], te_tbreak[i]) for i in tqdm(ids_te, desc="Predictive (test)")}).T
    dist_tr = pd.DataFrame({i: dist_shift_feats(tr_series[i], tr_tbreak[i]) for i in tqdm(ids_tr, desc="Distribution (train)")}).T
    dist_te = pd.DataFrame({i: dist_shift_feats(te_series[i], te_tbreak[i]) for i in tqdm(ids_te, desc="Distribution (test)")}).T

    if ADD_MMD_FEATURES:
        mmd_tr = pd.DataFrame({i: mmd_block_feats(tr_series[i], tr_tbreak[i], wins=MMD_WINDOWS) for i in tqdm(ids_tr, desc="MMD (train)")}).T
        mmd_te = pd.DataFrame({i: mmd_block_feats(te_series[i], te_tbreak[i], wins=MMD_WINDOWS) for i in tqdm(ids_te, desc="MMD (test)")}).T
    else:
        mmd_tr = pd.DataFrame(index=ids_tr); mmd_te = pd.DataFrame(index=ids_te)

    if extra_impl3_feats is None:
        extra_tr = extra_te = None
    else:
        extra_tr, extra_te = extra_impl3_feats

    if extra_pint_feats_tr is None or extra_pint_feats_te is None:
        pint_tr = pd.DataFrame(index=ids_tr)
        pint_te = pd.DataFrame(index=ids_te)
    else:
        pint_tr = extra_pint_feats_tr.reindex(ids_tr)
        pint_te = extra_pint_feats_te.reindex(ids_te)

    sig_tr = compute_signal_tests(tr_series, tr_tbreak, wins=(64,128))
    sig_te = compute_signal_tests(te_series, te_tbreak, wins=(64,128))

    Xb, Xb_t = base_tr.copy(), base_te.copy()
    Xfull_raw = (Xb.join(sp_aggr_tr, how='left')
                   .join(abc_sel_tr,   how='left')
                   .join(pred_tr,      how='left')
                   .join(dist_tr,      how='left')
                   .join(mmd_tr,       how='left')
                   .join(sig_tr,       how='left')
                   .join(pint_tr,      how='left'))
    if extra_tr is not None:
        Xfull_raw = Xfull_raw.join(extra_tr, how='left')

    Xfull_t_raw = (Xb_t.join(sp_aggr_te, how='left')
                     .join(abc_sel_te,   how='left')
                     .join(pred_te,      how='left')
                     .join(dist_te,      how='left')
                     .join(mmd_te,       how='left')
                     .join(sig_te,       how='left')
                     .join(pint_te,      how='left'))
    if extra_te is not None:
        Xfull_t_raw = Xfull_t_raw.join(extra_te, how='left')

    # Alineado y saneado (sin KS aquí)
    Xfull, Xfull_t = align_and_fill(Xfull_raw, Xfull_t_raw)
    Xfull    = _sanitize_for_model(Xfull)
    Xfull_t  = Xfull_t.reindex(columns=Xfull.columns)
    Xfull_t  = _sanitize_for_model(Xfull_t)

    # Matriz "dist"
    Xdist_raw   = (Xb.join(dist_tr,  how='left')).copy()
    Xdist_t_raw = (Xb_t.join(dist_te, how='left')).copy()
    Xdist, Xdist_t = align_and_fill(Xdist_raw, Xdist_t_raw)
    Xdist   = _sanitize_for_model(Xdist)
    Xdist_t = Xdist_t.reindex(columns=Xdist.columns)
    Xdist_t = _sanitize_for_model(Xdist_t)

    # removed_cols se calculará fuera con el filtro KS sin fuga
    return Xfull, Xfull_t, Xdist, Xdist_t, (tr_series, tr_tbreak), (te_series, te_tbreak), sig_tr

# ============= Impl3 (features + OOF XGB) =============
def extract_features_impl3_per_id(serie_df):
    from scipy import stats
    vals_0 = serie_df[serie_df['period']==0]['value'].to_numpy()
    vals_1 = serie_df[serie_df['period']==1]['value'].to_numpy()
    f = {}
    def safe(v, fn, default=np.nan):
        try:
            return fn(v) if len(v) else default
        except Exception:
            return default
    # moments
    f['mean_0'] = safe(vals_0, np.mean);   f['mean_1'] = safe(vals_1, np.mean)
    f['var_0']  = safe(vals_0, np.var);    f['var_1']  = safe(vals_1, np.var)
    f['skew_0'] = safe(vals_0, lambda z: stats.skew(z) if len(z)>2 else np.nan)
    f['skew_1'] = safe(vals_1, lambda z: stats.skew(z) if len(z)>2 else np.nan)
    f['kurt_0'] = safe(vals_0, lambda z: stats.kurtosis(z) if len(z)>3 else np.nan)
    f['kurt_1'] = safe(vals_1, lambda z: stats.kurtosis(z) if len(z)>3 else np.nan)
    # extrema
    f['max_0']  = safe(vals_0, np.max);    f['max_1']  = safe(vals_1, np.max)
    f['min_0']  = safe(vals_0, np.min);    f['min_1']  = safe(vals_1, np.min)
    # cuantiles completos y deltas
    f['q25_0']  = safe(vals_0, lambda z: np.quantile(z, 0.25))
    f['q50_0']  = safe(vals_0, lambda z: np.quantile(z, 0.50))
    f['q75_0']  = safe(vals_0, lambda z: np.quantile(z, 0.75))
    f['q25_1']  = safe(vals_1, lambda z: np.quantile(z, 0.25))
    f['q50_1']  = safe(vals_1, lambda z: np.quantile(z, 0.50))
    f['q75_1']  = safe(vals_1, lambda z: np.quantile(z, 0.75))
    f['dq25']   = f['q25_1'] - f['q25_0']   # <-- FIX
    f['dq50']   = f['q50_1'] - f['q50_0']
    f['dq75']   = f['q75_1'] - f['q75_0']
    # deltas de momentos
    f['diff_mean'] = f['mean_1'] - f['mean_0']
    f['diff_var']  = f['var_1']  - f['var_0']
    f['diff_skew'] = (f['skew_1'] if np.isfinite(f['skew_1']) else 0.0) - (f['skew_0'] if np.isfinite(f['skew_0']) else 0.0)
    f['diff_kurt'] = (f['kurt_1'] if np.isfinite(f['kurt_1']) else 0.0) - (f['kurt_0'] if np.isfinite(f['kurt_0']) else 0.0)
    # CUSUM helpers
    def cusum_vec(v):
        if len(v) <= 2: return None
        m = np.mean(v); return np.cumsum(v - m)
    def cusum_alarm(v, thr=5.0, drift=0.0):
        s_pos, s_neg = 0.0, 0.0; m = np.mean(v); alarm = 0
        for val in v:
            s_pos = max(0.0, s_pos + val - m - drift)
            s_neg = min(0.0, s_neg + val - m + drift)
            if s_pos > thr or abs(s_neg) > thr:
                alarm = 1; break
        return alarm
    for side, v in [('0', vals_0), ('1', vals_1)]:
        c = cusum_vec(v)
        if c is None:
            f[f'cusum_resid_{side}']=np.nan; f[f'cusum_mean_{side}']=np.nan
            f[f'cusum_std_{side}']=np.nan;  f[f'cusum_alarm_{side}']=np.nan
        else:
            f[f'cusum_resid_{side}']= float(np.max(np.abs(c)))
            f[f'cusum_mean_{side}'] = float(np.mean(c))
            f[f'cusum_std_{side}']  = float(np.std(c))
            f[f'cusum_alarm_{side}']= int(cusum_alarm(v))
    # Chow
    period = serie_df['period'].to_numpy()
    if (period==1).any():
        brk = int(np.argmax(period==1))
        y = serie_df['value'].to_numpy()
        t = serie_df['time'].to_numpy()
        X = np.column_stack([np.ones_like(t, float), t.astype(float)])
        X1, y1 = X[:brk], y[:brk]; X2, y2 = X[brk:], y[brk:]
        k = X.shape[1]; n = len(y)
        if len(y1) >= (k+1) and len(y2) >= (k+1):
            try:
                beta_full, *_ = np.linalg.lstsq(X, y, rcond=None)
                rss_full = float(np.sum((y - X@beta_full)**2))
                beta1, *_ = np.linalg.lstsq(X1, y1, rcond=None)
                rss1 = float(np.sum((y1 - X1@beta1)**2))
                beta2, *_ = np.linalg.lstsq(X2, y2, rcond=None)
                rss2 = float(np.sum((y2 - X2@beta2)**2))
                num = (rss_full - (rss1 + rss2)) / k
                den = (rss1 + rss2) / max(1, (n - 2*k))
                chow_stat = num/den if den>0 else np.nan
                from scipy.stats import f as fdist
                pval = 1 - fdist.cdf(chow_stat, k, max(1, n-2*k)) if np.isfinite(chow_stat) else np.nan
            except Exception:
                pval = np.nan
        else:
            pval = np.nan
        f['chow_pval'] = pval
    else:
        f['chow_pval'] = np.nan
    f['chow_logp'] = -np.log10(np.clip(f['chow_pval'] if np.isfinite(f['chow_pval']) else 1.0, 1e-300, 1.0))
    return f

def build_impl3_feature_tables(X_train_mi, X_test_mi):
    assert isinstance(X_train_mi.index, pd.MultiIndex)
    tr_ids = X_train_mi.index.get_level_values('id').unique()
    te_ids = X_test_mi.index.get_level_values('id').unique()
    feats_tr = []
    for i in tqdm(tr_ids, desc="Impl3 FE (train)"):
        ser = X_train_mi.loc[i].reset_index()
        feats_tr.append(extract_features_impl3_per_id(ser))
    Ftr = pd.DataFrame(feats_tr, index=tr_ids)
    feats_te = []
    for i in tqdm(te_ids, desc="Impl3 FE (test)"):
        ser = X_test_mi.loc[i].reset_index()
        feats_te.append(extract_features_impl3_per_id(ser))
    Fte = pd.DataFrame(feats_te, index=te_ids)
    return Ftr, Fte

# ============ Curriculum Weights (sin flip/dup) ============
def build_curriculum_and_pseudo(index_ids, oof_teacher, sig_tr_df,
                                pl_pos_q=PL_POS_Q, pl_neg_q=PL_NEG_Q,
                                tt_min=TT_MIN_LOGP, lb_min=LB_DELTA_MIN_LOGP,
                                w_base=CURR_WEIGHT_BASE, w_pos=CURR_WEIGHT_POS, w_neg=CURR_WEIGHT_NEG):
    s = pd.Series(oof_teacher, index=index_ids).rank(pct=True)
    sig = sig_tr_df.reindex(index_ids).fillna(0.0)
    pos_mask = (s.values >= pl_pos_q) & (sig['tt_logp_min'].values >= tt_min) & (sig['lb_logp_delta_max'].values >= lb_min)
    neg_mask = (s.values <= pl_neg_q)
    weights = np.full(len(index_ids), w_base, float)
    weights[pos_mask] = w_pos
    weights[neg_mask] = np.maximum(weights[neg_mask], w_neg)
    return weights, pos_mask, neg_mask

# ============ MI fold-wise ============
def foldwise_mi_select(X, y, folds=FOLDS, topk=TOPK_FULL, seed=SEED):
    """MI por fold con escalado robusto previo para mayor estabilidad."""
    X = _sanitize_for_model(X)
    # robust scale una vez (fit en todo train está bien para selección interna)
    Xs, _ = _robust_scale_fit(X)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_selection import mutual_info_classif
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    counts = pd.Series(0, index=Xs.columns, dtype=int)
    for tr, _ in skf.split(Xs, y):
        mi = mutual_info_classif(Xs.iloc[tr], y[tr], random_state=seed, discrete_features=False)
        top = pd.Series(mi, index=Xs.columns).sort_values(ascending=False).head(topk).index
        counts[top] += 1
    keep = counts[counts >= (folds+1)//2].index.tolist()
    if len(keep) < min(topk, Xs.shape[1]):
        keep = counts.sort_values(ascending=False).head(min(topk, Xs.shape[1])).index.tolist()
    return X[keep].copy(), keep


# ============ Anti-shift KS sin fuga ============
def ks_shift_filter_train_only(X, y, frac=SHIFT_FILTER_FRAC, seed=SEED):
    """
    Estima KS entre dos mitades de train y elimina top 'frac' columnas.
    Protege columnas informativas (whitelist): pint_*, mmd_*, ks_*, log_ks_*, diff_*, chow_*
    """
    if frac <= 0:
        return X.columns.tolist(), []
    # Whitelist protegida
    whitelist = [c for c in X.columns if c.startswith(("pint_", "mmd_", "ks_", "log_ks_", "diff_", "chow_"))]
    candidates = [c for c in X.columns if c not in whitelist]
    if not candidates:
        return X.columns.tolist(), []
    X_non = X[candidates]
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    i1, i2 = next(sss.split(X_non, y))
    A, B = X_non.iloc[i1], X_non.iloc[i2]
    from scipy import stats
    ks_vals = {}
    for c in candidates:
        try:
            ks_vals[c] = float(stats.ks_2samp(A[c].values, B[c].values, method="asymp").statistic)
        except Exception:
            ks_vals[c] = 0.0
    ks_ser = pd.Series(ks_vals).sort_values(ascending=False)
    kdrop = int(round(len(ks_ser) * frac))
    drop_cols = ks_ser.head(kdrop).index.tolist()
    keep_cols = [c for c in X.columns if c not in drop_cols]
    return keep_cols, drop_cols

def ks_shift_filter_train_test(Xtr, Xte, frac=SHIFT_FILTER_FRAC):
    ks_vals = {}
    for c in Xtr.columns:
        try:
            k = stats.ks_2samp(Xtr[c].values, Xte[c].values, method="asymp").statistic
        except Exception:
            k = 0.0
        ks_vals[c] = float(k)
    ks_ser = pd.Series(ks_vals).sort_values(ascending=False)
    kdrop = int(round(len(ks_ser) * frac))
    drop_cols = ks_ser.head(kdrop).index.tolist()
    keep_cols = [c for c in Xtr.columns if c not in drop_cols]
    return keep_cols, drop_cols

# ============ Modelos base Impl2 ============
def oof_hgb_with_test(X, y, X_test, sample_weight=None):
    params = dict(
        learning_rate=0.06 if not FAST_MODE else 0.05,
        max_depth=6, max_leaf_nodes=31, min_samples_leaf=20,
        l2_regularization=0.2, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, scoring='roc_auc', random_state=SEED
    )
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X), float); te_folds=[]
    sw_all = sample_weight if sample_weight is not None else np.ones(len(X), float)
    for tr, va in skf.split(X, y):
        Xtr, Xva = X.iloc[tr], X.iloc[va]; ytr, yva = y[tr], y[va]
        wtr = sw_all[tr]
        clf = HistGradientBoostingClassifier(**params)
        clf.fit(Xtr, ytr, sample_weight=wtr)
        oof[va] = clf.predict_proba(Xva)[:,1]
        te_folds.append(clf.predict_proba(X_test)[:,1])
    te_pred = np.mean(np.column_stack(te_folds), axis=1)
    auc = roc_auc_score(y, oof)
    return oof, te_pred, auc

# ---- Escalado robusto (mediana/IQR) ----
def _robust_scale_fit(X: pd.DataFrame):
    """Escalado robusto que garantiza devolver DataFrame"""
    if not isinstance(X, pd.DataFrame):
        print(f"WARNING: _robust_scale_fit recibió {type(X)}, convirtiendo a DataFrame")
        X = pd.DataFrame(X)
    
    med = X.median(numeric_only=True)
    q75 = X.quantile(0.75, numeric_only=True)
    q25 = X.quantile(0.25, numeric_only=True)
    iqr = (q75 - q25).replace(0.0, np.nan)
    stats = dict(med=med, iqr=iqr)
    Xs = (X - med).divide(iqr + 1e-9)
    Xs = Xs.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    # GARANTIZAR que devuelve DataFrame
    if not isinstance(Xs, pd.DataFrame):
        print(f"WARNING: _robust_scale_fit devolvió {type(Xs)}, convirtiendo a DataFrame")
        Xs = pd.DataFrame(Xs, index=X.index, columns=X.columns)
    
    return Xs, stats

def _robust_scale_apply(X: pd.DataFrame, stats):
    """Escalado robusto que garantiza devolver DataFrame"""
    if not isinstance(X, pd.DataFrame):
        print(f"WARNING: _robust_scale_apply recibió {type(X)}, convirtiendo a DataFrame")
        X = pd.DataFrame(X)
    
    med, iqr = stats['med'], stats['iqr']
    Xs = (X - med).divide(iqr + 1e-9)
    Xs = Xs.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    # GARANTIZAR que devuelve DataFrame
    if not isinstance(Xs, pd.DataFrame):
        print(f"WARNING: _robust_scale_apply devolvió {type(Xs)}, convirtiendo a DataFrame")
        Xs = pd.DataFrame(Xs, index=X.index, columns=X.columns)
    
    return Xs

# ============ PINT Híbrido ==========
def subset_series(series_dict, tbreak_dict, valid_ids_index):
    valid_ids = set(valid_ids_index.tolist())
    s = {gid: series_dict[gid] for gid in series_dict.keys() if gid in valid_ids}
    t = {gid: tbreak_dict[gid] for gid in tbreak_dict.keys() if gid in valid_ids}
    return s, t


def oof_pint_hybrid_with_test(
    X, y, X_test,
    series_source_tr: pd.DataFrame,
    series_source_te: pd.DataFrame,
    n_folds=FOLDS,
    cb_params_pint: dict = None,
    sample_weight=None
):
    """PINT Híbrido: entrenamiento por fold + features robustas integradas en CatBoost"""
    if not (HAS_TORCH and USE_PINT):
        return np.zeros(len(X)), np.zeros(len(X_test)), 0.0

    assert isinstance(series_source_tr.index, pd.MultiIndex), "series_source_tr debe ser MultiIndex"
    assert isinstance(series_source_te.index, pd.MultiIndex), "series_source_te debe ser MultiIndex"

    # Series reales y t_break
    series_dict, tbreak_dict = to_series_dict(series_source_tr)
    te_series_dict, te_tbreak_dict = to_series_dict(series_source_te)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    pint_features_tr = []
    pint_features_te = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_fold_tr = X.iloc[train_idx]
        fold_series, fold_tbreak = subset_series(series_dict, tbreak_dict, X_fold_tr.index)

        fold_model = train_pint_global(fold_series, fold_tbreak)

        # Validación
        for i in val_idx:
            gid = X.index[i]
            if gid in series_dict and gid in tbreak_dict:
                feat = pint_hybrid_rollout_for_id(series_dict[gid], tbreak_dict[gid], fold_model)
                pint_features_tr.append(feat or {})
            else:
                pint_features_tr.append({})

        # Test: acumular features por fold y promediar al final
        te_feats = []
        for gid in X_test.index:
            if gid in te_series_dict and gid in te_tbreak_dict:
                feat = pint_hybrid_rollout_for_id(te_series_dict[gid], te_tbreak_dict[gid], fold_model)
                te_feats.append(feat or {})
            else:
                te_feats.append({})
        # inicializar/acumular
        if pint_features_te is None:
            pint_features_te = te_feats
        else:
            # promedio incremental de diccionarios numéricos
            merged = []
            for d_prev, d_new in zip(pint_features_te, te_feats):
                keys = set(d_prev.keys()) | set(d_new.keys())
                md = {}
                for k in keys:
                    a = d_prev.get(k, 0.0); b = d_new.get(k, 0.0)
                    md[k] = (a + b)
                merged.append(md)
            pint_features_te = merged

    # promedio final por número de folds
    if pint_features_te is not None and len(pint_features_te) == len(X_test.index):
        for d in pint_features_te:
            for k in list(d.keys()):
                d[k] = d[k] / float(n_folds)

    pint_df_tr = pd.DataFrame(pint_features_tr, index=X.index).add_prefix("pint_h_")
    pint_df_te = pd.DataFrame(pint_features_te, index=X_test.index).add_prefix("pint_h_")

    X_with_pint      = X.join(pint_df_tr, how="left").fillna(0)
    X_test_with_pint = X_test.join(pint_df_te, how="left").fillna(0)

    # Parámetros exclusivos de CatBoost para el head PINT
    params = cb_params_pint if cb_params_pint is not None else dict(
        loss_function="Logloss", eval_metric="AUC", auto_class_weights="Balanced",
        iterations=(1200 if FAST_MODE else 2000),  # Tune: iterations
        learning_rate=0.03, depth=6, l2_leaf_reg=5.0,  # Tune: these values
        verbose=False, thread_count=1, rsm=0.85, border_count=128,
        bootstrap_type="Bayesian", bagging_temperature=1.0,
        random_strength=0.5, leaf_estimation_iterations=4
    )

    cb_result = oof_catboost_multi(
        X_with_pint, y, X_test_with_pint,
        seeds=(42,), params=params,  # Tune: add more seeds
        feat_fraction=0.60, label="PINT-Hybrid",
        sample_weight=sample_weight
    )

    if cb_result is not None:
        oof_preds, test_preds, auc_score = cb_result["avg"]
    else:
        oof_preds = np.random.rand(len(X))
        test_preds = np.random.rand(len(X_test))
        auc_score = 0.5

    return oof_preds, test_preds, auc_score

# ============ Dist con HGB + escalado robusto ==========
def oof_dist_hgb_with_test(X, y, X_test, sample_weight=None):
    # Verificaciones básicas
    if not isinstance(X, pd.DataFrame) or X.empty:
        return np.zeros(len(y)), np.zeros(len(X_test)), 0.0
    
    # Alineado e imputación coherente
    Xb, Xb_t = align_and_fill(X.copy(), X_test.copy())
    
    # Escalado robusto (fit en train, apply en test)
    Xs, stats = _robust_scale_fit(Xb)
    Xs_t = _robust_scale_apply(Xb_t.reindex(columns=Xs.columns), stats)
    
    # Arrays para HGB
    Xs_array = Xs.values
    Xs_t_array = Xs_t.values
    y_array = y.values if isinstance(y, pd.Series) else y
    
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(Xs_array), float); te_folds = []
    for tr, va in skf.split(Xs_array, y_array):
        clf = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=8, 
            min_samples_leaf=20, l2_regularization=1.0, 
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
            random_state=SEED
        )
        sw_tr = sample_weight[tr] if sample_weight is not None else None
        clf.fit(Xs_array[tr], y_array[tr], sample_weight=sw_tr)
        oof[va] = clf.predict_proba(Xs_array[va])[:,1]
        te_folds.append(clf.predict_proba(Xs_t_array)[:,1])
    te = np.mean(np.column_stack(te_folds), axis=1)
    auc = roc_auc_score(y_array, oof)
    return oof, te, auc

def oof_logistic_with_test(X, y, X_test, sample_weight=None):
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X), float); te_folds=[]
    sw_all = sample_weight if sample_weight is not None else np.ones(len(X), float)
    for tr, va in skf.split(X, y):
        Xtr, Xva = X.iloc[tr], X.iloc[va]; ytr, yva = y[tr], y[va]
        base = LogisticRegression(max_iter=3000 if not FAST_MODE else 1500, class_weight=None, solver='lbfgs', C=1.0, random_state=SEED)
        cal  = CalibratedClassifierCV(estimator=base, method='isotonic', cv=3)
        cal.fit(Xtr, ytr, sample_weight=sw_all[tr])
        oof[va] = cal.predict_proba(Xva)[:,1]
        te_folds.append(cal.predict_proba(X_test)[:,1])
    te_pred = np.mean(np.column_stack(te_folds), axis=1)
    auc = roc_auc_score(y, oof)
    return oof, te_pred, auc

# ============ XGBoost OOF sobre Xmi ==========
def oof_xgb_on_matrix(X, y, X_test, sample_weight=None):
    if not HAS_XGB:
        print(">> XGBoost (Xmi) omitido.")
        return None
    XgbEarlyStopping = None
    try:
        from xgboost.callback import EarlyStopping as XgbEarlyStopping  # >=1.6
    except Exception:
        try:
            import xgboost as _xgb
            XgbEarlyStopping = _xgb.callback.EarlyStopping
        except Exception:
            XgbEarlyStopping = None
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X), float); te_folds=[]
    sw_all = sample_weight if sample_weight is not None else np.ones(len(X), float)
    X_base = X.replace([np.inf, -np.inf], np.nan)
    med = X_base.median(numeric_only=True)
    X_base = X_base.fillna(med)
    X_test_base = X_test.replace([np.inf, -np.inf], np.nan).fillna(med)
    for tr, va in skf.split(X_base, y):
        Xtr, Xva = X_base.iloc[tr], X_base.iloc[va]
        ytr, yva = y[tr], y[va]
        wtr = sw_all[tr]
        pos = max(1, int(np.sum(ytr==1)))
        neg = max(1, int(np.sum(ytr==0)))
        spw = float(neg/pos)
        params = dict(
            learning_rate=(0.03 if not FAST_MODE else 0.04),
            max_depth=7,
            n_estimators=(6000 if not FAST_MODE else 3000),
            subsample=0.80,
            colsample_bytree=0.80,
            reg_lambda=2.0,
            reg_alpha=1e-3,
            min_child_weight=2,
            random_state=SEED,
            n_jobs=-1,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            scale_pos_weight=spw,
        )
        clf = xgb.XGBClassifier(**params)
        fitted=False
        try:
            clf.fit(Xtr, ytr,
                    sample_weight=wtr,
                    eval_set=[(Xva, yva)],
                    early_stopping_rounds=(300 if not FAST_MODE else 180),
                    verbose=False)
            fitted=True
        except TypeError:
            if XgbEarlyStopping is not None:
                try:
                    es_cb = XgbEarlyStopping(rounds=(300 if not FAST_MODE else 180), save_best=True, maximize=True)
                    clf.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], callbacks=[es_cb], verbose=False)
                    fitted=True
                except Exception:
                    fitted=False
        except Exception:
            fitted=False
        if not fitted:
            clf.set_params(n_estimators=min(params['n_estimators'], 1500))
            clf.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], verbose=False)
        oof[va] = clf.predict_proba(Xva)[:,1]
        te_folds.append(clf.predict_proba(X_test_base)[:,1])
    te = np.mean(np.column_stack(te_folds), axis=1)
    auc = roc_auc_score(y, oof)
    return oof, te, auc

def oof_catboost_multi(X, y, X_test, seeds, params, feat_fraction, label, sample_weight=None):
    if not (USE_CATBOOST and HAS_CATBOOST):
        print(f">> CatBoost {label} omitido.")
        return None
    per_seed=[]
    cols = X.columns.tolist()
    sw_all = sample_weight if sample_weight is not None else np.ones(len(X), float)
    for sd in seeds:
        rng = np.random.default_rng(sd)
        k = max(3, int(np.ceil(len(cols) * feat_fraction)))
        subcols = sorted(rng.choice(cols, size=k, replace=False).tolist())
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=sd)
        oof = np.zeros(len(X), float); te_folds=[]
        for tr, va in skf.split(X, y):
            Xtr, Xva = X.iloc[tr][subcols], X.iloc[va][subcols]
            ytr, yva = y[tr], y[va]; wtr = sw_all[tr]
            mdl = CatBoostClassifier(**{**params, "random_seed": sd})
            mdl.fit(Pool(Xtr, ytr, weight=wtr),
                    eval_set=Pool(Xva, y[va]),
                    early_stopping_rounds=180 if not FAST_MODE else 120, use_best_model=True, verbose=False)
            oof[va] = mdl.predict_proba(X.iloc[va][subcols])[:,1]
            te_folds.append(mdl.predict_proba(X_test[subcols])[:,1])
        te_pred = np.mean(np.column_stack(te_folds), axis=1)
        auc = roc_auc_score(y, oof)
        per_seed.append(dict(seed=sd, oof=oof, te=te_pred, auc=auc))
    oof_avg = np.mean(np.column_stack([e["oof"] for e in per_seed]), axis=1)
    te_avg  = np.mean(np.column_stack([e["te"]  for e in per_seed]), axis=1)
    auc_avg = roc_auc_score(y, oof_avg)
    print(f">> CatBoost {label} OOF AUC (avg): {auc_avg:.4f}")
    return dict(avg=(oof_avg, te_avg, auc_avg), all=per_seed)

def oof_xgb_impl3_with_test(Ftr_raw, y_ser, Fte_raw, sample_weight=None):
    if not (USE_IMPL3_XGB and HAS_XGB):
        print(">> Impl3-XGB omitido.")
        return None
    # EarlyStopping según versión
    XgbEarlyStopping = None
    try:
        from xgboost.callback import EarlyStopping as XgbEarlyStopping  # >=1.6
    except Exception:
        try:
            import xgboost as _xgb
            XgbEarlyStopping = _xgb.callback.EarlyStopping
        except Exception:
            XgbEarlyStopping = None

    y = y_ser.reindex(Ftr_raw.index).astype(int).values
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(Ftr_raw), float); te_bags_all = []

    rng = np.random.default_rng(SEED)
    subs = rng.choice(IMPL3_SUBSAMPLE_GRID, size=IMPL3_N_BAGS, replace=True)
    cols = rng.choice(IMPL3_COLSAMPLE_GRID, size=IMPL3_N_BAGS, replace=True)
    seeds = rng.integers(1, 10_000_000, size=IMPL3_N_BAGS)

    med = Ftr_raw.median(numeric_only=True)
    Ftr_base = Ftr_raw.replace([np.inf, -np.inf], np.nan).fillna(med)
    Fte_base = Fte_raw.replace([np.inf, -np.inf], np.nan).fillna(med)

    sw_all = sample_weight if sample_weight is not None else np.ones(len(Ftr_base), float)

    for fidx, (tr, va) in enumerate(skf.split(Ftr_base, y), start=1):
        idx_tr = Ftr_base.index[tr]; idx_va = Ftr_base.index[va]
        ytr, yva = y[tr], y[va]

        # DiD baseline por fold
        diff_means_tr = Ftr_raw.loc[idx_tr, 'diff_mean'].to_numpy()
        neg_mask = (ytr == 0)
        baseline = float(np.nanmean(diff_means_tr[neg_mask])) if neg_mask.sum() > 0 else 0.0

        def add_did(F, idx):
            D = F.loc[idx].copy()
            D['DiD'] = D['diff_mean'] - baseline
            # versión estandarizada
            sigma = float(np.nanstd(diff_means_tr[neg_mask]) + 1e-9) if neg_mask.sum()>1 else 1.0
            D['DiD_z'] = (D['diff_mean'] - baseline) / sigma
            return D

        Xtr = add_did(Ftr_base, idx_tr); Xva = add_did(Ftr_base, idx_va)
        Xte_fold = add_did(Fte_base, Fte_base.index)

        wtr = sw_all[tr]

        va_preds_bags = []; te_preds_bags = []
        for b in range(IMPL3_N_BAGS):
            params = dict(
                learning_rate=IMPL3_LEARNING_RATE,
                max_depth=IMPL3_MAX_DEPTH,
                n_estimators=IMPL3_MAX_ESTIMATORS,
                subsample=float(subs[b]),
                colsample_bytree=float(cols[b]),
                reg_lambda=IMPL3_REG_LAMBDA,
                random_state=int(seeds[b]),
                n_jobs=-1,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
            )
            clf = xgb.XGBClassifier(**params)

            Xtr_use, ytr_use, wtr_use = Xtr, ytr, wtr
            if USE_SMOTE_IMPL3:
                try:
                    from imblearn.over_sampling import SMOTE
                    sm = SMOTE(random_state=int(seeds[b]))
                    Xtr_sm, ytr_sm = sm.fit_resample(Xtr, ytr)
                    w_mean = np.mean(wtr) if len(wtr)>0 else 1.0
                    wtr_use = np.full(len(ytr_sm), w_mean, float)
                    Xtr_use, ytr_use = Xtr_sm, ytr_sm
                except Exception:
                    Xtr_use, ytr_use, wtr_use = Xtr, ytr, wtr

            fitted = False
            try:
                clf.fit(Xtr_use, ytr_use,
                        sample_weight=wtr_use,
                        eval_set=[(Xva, yva)],
                        early_stopping_rounds=IMPL3_EARLY_STOP_ROUNDS,
                        verbose=False)
                fitted = True
            except TypeError:
                if XgbEarlyStopping is not None:
                    try:
                        es_cb = XgbEarlyStopping(rounds=IMPL3_EARLY_STOP_ROUNDS, save_best=True, maximize=True)
                        clf.fit(Xtr_use, ytr_use,
                                sample_weight=wtr_use,
                                eval_set=[(Xva, yva)],
                                callbacks=[es_cb],
                                verbose=False)
                        fitted = True
                    except Exception:
                        fitted = False
            except Exception:
                fitted = False
            if not fitted:
                clf.set_params(n_estimators=min(IMPL3_MAX_ESTIMATORS, 1200))
                clf.fit(Xtr_use, ytr_use, sample_weight=wtr_use, eval_set=[(Xva, yva)], verbose=False)

            va_preds_bags.append(clf.predict_proba(Xva)[:,1])
            te_preds_bags.append(clf.predict_proba(Xte_fold)[:,1])

        va_mean = np.mean(np.column_stack(va_preds_bags), axis=1)
        te_mean = np.mean(np.column_stack(te_preds_bags), axis=1)
        oof[va] = va_mean
        te_bags_all.append(te_mean)
        fold_auc = roc_auc_score(yva, va_mean)
        print(f"[Impl3-XGB fold {fidx}] AUC={fold_auc:.4f}  (baseline DiD={baseline:.5f})")

    te_pred = np.mean(np.column_stack(te_bags_all), axis=1)
    auc_oof = roc_auc_score(y, oof)
    print(f"Impl3-XGB OOF AUC (bags={IMPL3_N_BAGS}): {auc_oof:.6f}")
    return oof, te_pred, auc_oof

# ============ Meta-stacker LR ============
def meta_stacker_lr(oof_dict, y, test_dict):
    cols = sorted(oof_dict.keys())
    Xmeta = np.column_stack([oof_dict[c] for c in cols])
    Xmeta_test = np.column_stack([test_dict[c] for c in cols])
    lr = LogisticRegression(max_iter=500, solver='lbfgs', class_weight=None, random_state=SEED)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), float); te_folds=[]
    for tr, va in skf.split(Xmeta, y):
        lr.fit(Xmeta[tr], y[tr])
        oof[va] = lr.predict_proba(Xmeta[va])[:,1]
        te_folds.append(lr.predict_proba(Xmeta_test)[:,1])
    te = np.mean(np.column_stack(te_folds), axis=1)
    auc = roc_auc_score(y, oof)
    print(f"[Meta-Stacker LR] OOF AUC = {auc:.6f}")
    return oof, te, auc

# ================== RUN ==================
def run_all(X_train_mi, y_train, X_test_mi, y_test=None):
    # Impl3: precompute features crudas por id
    Ftr_impl3, Fte_impl3 = build_impl3_feature_tables(X_train_mi, X_test_mi)

    # ===== PINT features por id (entrena modelo global en train, aplica a ambos)
    pint_tr_df = pint_te_df = None
    if HAS_TORCH and USE_PINT:
        tr_series_tmp, tr_tb_tmp = to_series_dict(X_train_mi)
        te_series_tmp, te_tb_tmp = to_series_dict(X_test_mi)
        pint_tr_df = build_features_pint(tr_series_tmp, tr_tb_tmp, desc="PINT (train)")
        # Reutiliza el modelo global ya entrenado para test
        if 'model_pint' in globals() and model_pint is not None:
            rows_te = []
            for i in tqdm(list(te_series_tmp.keys()), desc="PINT (test)"):
                f = pint_rollout_errors_for_id(te_series_tmp[i], te_tb_tmp[i])
                rows_te.append(f if f is not None else {})
            pint_te_df = pd.DataFrame(rows_te, index=list(te_series_tmp.keys())).add_prefix("pint_")
        else:
            pint_te_df = pd.DataFrame(index=list(te_series_tmp.keys()))

    # NEW: comprehensive features por id usando create_comprehensive_features_pd
    try:
        df_long_tr = X_train_mi.reset_index()[['id', 'period', 'value']]
        df_long_te = X_test_mi.reset_index()[['id', 'period', 'value']]
        feat_train_cf = create_comprehensive_features_pd(df_long_tr).set_index('id')
        feat_test_cf = create_comprehensive_features_pd(df_long_te).set_index('id')
    except Exception as e:
        print(f"[WARN] comprehensive features falló: {e}")
        feat_train_cf = pd.DataFrame(index=Ftr_impl3.index)
        feat_test_cf = pd.DataFrame(index=Fte_impl3.index)

    # Inyección opcional de Impl3 (sin DiD) en Impl2, extendida con comprehensive features
    if INJECT_IMPL3_FEATURES_IN_IMPL2:
        impl3_no_did_tr = Ftr_impl3.drop(columns=[c for c in ['DiD','DiD_z'] if c in Ftr_impl3.columns], errors="ignore")
        impl3_no_did_te = Fte_impl3.drop(columns=[c for c in ['DiD','DiD_z'] if c in Fte_impl3.columns], errors="ignore")
        extra_impl3 = (impl3_no_did_tr.join(feat_train_cf, how='left'),
                       impl3_no_did_te.join(feat_test_cf,  how='left'))
    else:
        extra_impl3 = None

    print(">> Construyendo features (Impl2)…")
    (Xfull, Xfull_t, Xdist, Xdist_t,
     (tr_series, tr_tb), (te_series, te_tb), sig_tr_df) = build_features_impl2(
        X_train_mi, X_test_mi, extra_impl3_feats=extra_impl3,
        extra_pint_feats_tr=pint_tr_df, extra_pint_feats_te=pint_te_df
    )

    y_vec = to_y_series(y_train, Xfull.index).astype(int).values.ravel()

    # ===== Filtro KS anti-shift sin fuga (o clásico bajo bandera) =====
    removed_cols = []
    if APPLY_SHIFT_FILTER:
        if SHIFT_FILTER_MODE == "train_only":
            keep_cols, drop_cols = ks_shift_filter_train_only(Xfull, y_vec, frac=SHIFT_FILTER_FRAC, seed=SEED)
        else:
            keep_cols, drop_cols = ks_shift_filter_train_test(Xfull, Xfull_t, frac=SHIFT_FILTER_FRAC)

        removed_cols = drop_cols
        if len(drop_cols):
            print(f"[SHIFT] Eliminadas {len(drop_cols)} cols por KS ({SHIFT_FILTER_MODE}, top {SHIFT_FILTER_FRAC*100:.1f}%).")
        else:
            print("[SHIFT] Sin columnas eliminadas por KS.")

        Xks   = Xfull[keep_cols].copy()
        Xks_t = Xfull_t[keep_cols].copy()
    else:
        removed_cols = []
        Xks, Xks_t = Xfull.copy(), Xfull_t.copy()

    # ===== MI fold-wise sobre columnas tras KS =====
    Xmi, keep_mi = foldwise_mi_select(Xks, y_vec, folds=FOLDS, topk=TOPK_FULL, seed=SEED)
    Xmi_t = Xks_t.reindex(columns=keep_mi).fillna(Xks[keep_mi].median(numeric_only=True))

    # === Optuna: tuning de CatBoost A/B y PINT-Head ===
    USE_OPTUNA_TUNE = False

    if USE_OPTUNA_TUNE and USE_CATBOOST and HAS_CATBOOST:
        # A
        bestA, _ = tune_catboost_on_matrix(
            Xmi, y_vec, Xmi_t,
            seeds=CB_SEEDS_A, feat_fraction=CB_FRACTION_A, label="A", n_trials=10
        )
        cbA_params = {
            **bestA,
            "loss_function":"Logloss","eval_metric":"AUC","auto_class_weights":"Balanced",
            "verbose":False,"thread_count":1,"border_count":128
        }

        # B
        bestB, _ = tune_catboost_on_matrix(
            Xmi, y_vec, Xmi_t,
            seeds=CB_SEEDS_B, feat_fraction=CB_FRACTION_B, label="B", n_trials=10
        )
        cbB_params = {
            **bestB,
            "loss_function":"Logloss","eval_metric":"AUC","auto_class_weights":"Balanced",
            "verbose":False,"thread_count":1,"border_count":128
        }

    if USE_OPTUNA_TUNE and HAS_TORCH and USE_PINT and USE_CATBOOST and HAS_CATBOOST:
        # PINT-Head
        bestP, _ = tune_catboost_pint_head(
            Xmi, y_vec, Xmi_t,
            series_source_tr=X_train, series_source_te=X_test,
            n_trials=80
        )
        cbP_params = {
            **bestP,
            "loss_function":"Logloss","eval_metric":"AUC","auto_class_weights":"Balanced",
            "verbose":False,"thread_count":1,"border_count":128
        }

    # ====== Primera pasada (teacher) sin curriculum para obtener OOF base ======
    oof_hgb0,  te_hgb0,  auc_hgb0  = oof_hgb_with_test(Xmi,  y_vec, Xmi_t)
    oof_dist0, te_dist0, auc_dist0 = oof_dist_hgb_with_test(Xdist, y_vec, Xdist_t)

    preds_train0 = {"hgb": oof_hgb0, "dist": oof_dist0}
    preds_test0  = {"hgb": te_hgb0,  "dist": te_dist0}

    # ===== PINT Híbrido (entrenamiento por fold + CatBoost) =====
    if HAS_TORCH and USE_PINT:
        cbP_params = dict(
            loss_function="Logloss", eval_metric="AUC", auto_class_weights="Balanced",
            iterations=(1200 if FAST_MODE else 2000),  # Tune
            learning_rate=0.03, depth=6, l2_leaf_reg=5.0,  # Tune
            verbose=False, thread_count=1, rsm=0.85, border_count=128,
            bootstrap_type="Bayesian", bagging_temperature=1.0,
            random_strength=0.5, leaf_estimation_iterations=4
        )
        oof_pint_hybrid0, te_pint_hybrid0, auc_pint_hybrid0 = oof_pint_hybrid_with_test(
            Xmi, y_vec, Xmi_t,
            series_source_tr=X_train,
            series_source_te=X_test,
            n_folds=FOLDS,
            cb_params_pint=cbP_params
        )
        preds_train0["pint_hybrid"] = oof_pint_hybrid0
        preds_test0["pint_hybrid"]  = te_pint_hybrid0
    else:
        print(">> PINT Híbrido omitido (sin PyTorch o PINT deshabilitado).")

    # ===== CatBoost (teacher)
    cbA_params = dict(loss_function="Logloss", eval_metric="AUC", auto_class_weights="Balanced",
                      iterations=(1500 if not FAST_MODE else 800), learning_rate=0.03, depth=6, l2_leaf_reg=5.0,  # Tune
                      verbose=False, thread_count=1, rsm=0.85, border_count=128,
                      bootstrap_type="Bayesian", bagging_temperature=1.0,
                      random_strength=0.5, leaf_estimation_iterations=4)
    cbB_params = dict(loss_function="Logloss", eval_metric="AUC", auto_class_weights="Balanced",
                      iterations=(2000 if not FAST_MODE else 1000), learning_rate=0.025, depth=7, l2_leaf_reg=3.0,  # Tune
                      verbose=False, thread_count=1, rsm=0.90, border_count=128,
                      bootstrap_type="Bernoulli", subsample=0.80,
                      random_strength=0.5, leaf_estimation_iterations=4)

    if USE_CATBOOST and HAS_CATBOOST:
        cbA0 = oof_catboost_multi(Xmi, y_vec, Xmi_t, seeds=CB_SEEDS_A, params=cbA_params, feat_fraction=CB_FRACTION_A, label="A")
        cbB0 = oof_catboost_multi(Xmi, y_vec, Xmi_t, seeds=CB_SEEDS_B, params=cbB_params, feat_fraction=CB_FRACTION_B, label="B")
        if cbA0 is not None:
            oA, tA, _ = cbA0["avg"]; preds_train0["cbA"] = oA; preds_test0["cbA"] = tA
        if cbB0 is not None:
            oB, tB, _ = cbB0["avg"]; preds_train0["cbB"] = oB; preds_test0["cbB"] = tB

    # ===== Impl3-XGB (teacher)
    if USE_IMPL3_XGB and HAS_XGB:
        xgb_oof0, xgb_te0, auc_xgb0 = oof_xgb_impl3_with_test(Ftr_impl3, to_y_series(y_train, Ftr_impl3.index), Fte_impl3)
        preds_train0["xgb_raw"] = xgb_oof0
        preds_test0["xgb_raw"]  = xgb_te0
    else:
        auc_xgb0 = np.nan

    # XGBoost sobre Xmi (teacher)
    if HAS_XGB:
        xgb_xmi_oof0, xgb_xmi_te0, auc_xgb_xmi0 = oof_xgb_on_matrix(Xmi, y_vec, Xmi_t)
        preds_train0["xgb_xmi"] = xgb_xmi_oof0
        preds_test0["xgb_xmi"]  = xgb_xmi_te0
    
    # ===== Resumen teacher
    print("\nAUC OOF por experto (Teacher):")
    for k, v in preds_train0.items():
        print(f"  {k}: {roc_auc_score(y_vec, v):.4f}")

    auc_rb0, w0 = optimize_rank_blend_dirichlet(preds_train0, y_vec)
    print(f"[Rank-Blend OOF (Dirichlet)] AUC = {auc_rb0:.6f}  | pesos = {w0}")
    auc_sqp0, w_sqp0 = optimize_rank_blend_slsqp(preds_train0, y_vec, w0) if HAS_SLSQP else (None, None)
    if auc_sqp0 is not None and auc_sqp0 > auc_rb0 + 1e-6:
        print(f"[SLSQP refine] Mejora: AUC={auc_sqp0:.6f}")
        w_blend = w_sqp0; auc_blend0 = auc_sqp0
    else:
        print("[SLSQP refine] Sin mejora; se mantienen pesos Dirichlet.")
        w_blend = w0; auc_blend0 = auc_rb0

    # ===== Curriculum Weights (sin flip/dup)
    keys_blend = list(w_blend.keys())
    Wb = np.array([w_blend[k] for k in keys_blend], float); Wb /= Wb.sum()
    Rtr = np.column_stack([rank01(preds_train0[k]) for k in keys_blend])
    teacher_oof = (Rtr * Wb).sum(axis=1)

    sample_weight, mask_pos, mask_neg = build_curriculum_and_pseudo(
        Xmi.index, teacher_oof, sig_tr_df,
        pl_pos_q=PL_POS_Q, pl_neg_q=PL_NEG_Q,
        tt_min=TT_MIN_LOGP, lb_min=LB_DELTA_MIN_LOGP,
        w_base=CURR_WEIGHT_BASE, w_pos=CURR_WEIGHT_POS, w_neg=CURR_WEIGHT_NEG
    )

    # ===== Segunda pasada (student) con curriculum =====
    oof_hgb,  te_hgb,  _ = oof_hgb_with_test(Xmi,  y_vec, Xmi_t,  sample_weight=sample_weight)
    oof_dist, te_dist, _ = oof_dist_hgb_with_test(Xdist, y_vec, Xdist_t, sample_weight=sample_weight)

    preds_train = {"hgb": oof_hgb, "dist": oof_dist}
    preds_test  = {"hgb": te_hgb,  "dist": te_dist}

    # ===== PINT Híbrido (student) =====
    if HAS_TORCH and USE_PINT:
        oof_pint_hybrid, te_pint_hybrid, auc_pint_hybrid = oof_pint_hybrid_with_test(
            Xmi, y_vec, Xmi_t,
            series_source_tr=X_train,
            series_source_te=X_test,
            n_folds=FOLDS,
            cb_params_pint=cbP_params,
            sample_weight=sample_weight
        )
        print(f">> PINT Híbrido (student) OOF AUC: {auc_pint_hybrid:.4f}")
        preds_train["pint_hybrid"] = oof_pint_hybrid
        preds_test["pint_hybrid"]  = te_pint_hybrid

    if USE_CATBOOST and HAS_CATBOOST:
        cbA = oof_catboost_multi(Xmi, y_vec, Xmi_t, seeds=CB_SEEDS_A, params=cbA_params, feat_fraction=CB_FRACTION_A, label="A",
                                 sample_weight=sample_weight)
        cbB = oof_catboost_multi(Xmi, y_vec, Xmi_t, seeds=CB_SEEDS_B, params=cbB_params, feat_fraction=CB_FRACTION_B, label="B",
                                 sample_weight=sample_weight)
        if cbA is not None:
            oA, tA, _ = cbA["avg"]; preds_train["cbA"] = oA; preds_test["cbA"] = tA
        if cbB is not None:
            oB, tB, _ = cbB["avg"]; preds_train["cbB"] = oB; preds_test["cbB"] = tB

    if USE_IMPL3_XGB and HAS_XGB:
        xgb_oof, xgb_te, auc_xgb = oof_xgb_impl3_with_test(Ftr_impl3, to_y_series(y_train, Ftr_impl3.index), Fte_impl3,
                                                           sample_weight=sample_weight)
        preds_train["xgb_raw"] = xgb_oof
        preds_test["xgb_raw"]  = xgb_te
    else:
        auc_xgb = np.nan

    # XGBoost sobre Xmi (student)
    if HAS_XGB:
        xgb_xmi_oof, xgb_xmi_te, auc_xgb_xmi = oof_xgb_on_matrix(Xmi, y_vec, Xmi_t, sample_weight=sample_weight)
        preds_train["xgb_xmi"] = xgb_xmi_oof
        preds_test["xgb_xmi"]  = xgb_xmi_te

    print("\nAUC OOF por experto (Student):")
    for k, v in preds_train.items():
        print(f"  {k}: {roc_auc_score(y_vec, v):.4f}")

    # Rank-blend + SLSQP (student)
    auc_rb, w_init = optimize_rank_blend_dirichlet(preds_train, y_vec)
    print(f"\n[Rank-Blend OOF (Dirichlet)] AUC = {auc_rb:.6f}  | pesos = {w_init}")
    auc_sqp, w_sqp = optimize_rank_blend_slsqp(preds_train, y_vec, w_init) if HAS_SLSQP else (None, None)
    if auc_sqp is not None and auc_sqp > auc_rb + 1e-6:
        print(f"[SLSQP refine] Mejora: AUC={auc_sqp:.6f}")
        w_final = w_sqp; auc_final = auc_sqp
    else:
        print("[SLSQP refine] Sin mejora; se mantienen pesos Dirichlet.")
        w_final = w_init; auc_final = auc_rb

    # Meta-stacker LR (student)
    oof_meta, te_meta, auc_meta = meta_stacker_lr(preds_train, y_vec, preds_test)

    # Meta + RB (mezcla convexa)
    rb_keys = list(w_final.keys())
    Wf = np.array([w_final[k] for k in rb_keys], float); Wf = Wf / Wf.sum()
    Rtr_rb = np.column_stack([rank01(preds_train[k]) for k in rb_keys])
    Rte_rb = np.column_stack([rank01(preds_test[k])  for k in rb_keys])
    s_tr_rb = (Rtr_rb * Wf).sum(axis=1)
    s_te_rb = (Rte_rb * Wf).sum(axis=1)

    Rtr_all = np.column_stack([rank01(s_tr_rb), rank01(oof_meta)])
    Rte_all = np.column_stack([rank01(s_te_rb), rank01(te_meta)])

    best_auc_mix = -1.0; best_alpha = 0.5
    for a in np.linspace(0.0, 1.0, 11):
        s_mix = a * Rtr_all[:, 0] + (1 - a) * Rtr_all[:, 1]
        auc_mix = roc_auc_score(y_vec, s_mix)
        if auc_mix > best_auc_mix:
            best_auc_mix = auc_mix; best_alpha = float(a)
    print(f"[Meta+RB] mejor alpha={best_alpha:.2f} | OOF AUC={best_auc_mix:.6f}")

    # Predicción final de test
    s_test = best_alpha * Rte_all[:, 0] + (1 - best_alpha) * Rte_all[:, 1]
    preds_test_df = pd.Series(s_test, index=Xmi_t.index, name="break_score").sort_index()
    preds_test_df.to_csv(SUBMISSION_NAME, header=True)
    print("\nGuardado:", SUBMISSION_NAME)

    # Eval test si hay y_test
    test_auc = None
    if y_test is not None:
        y_te = to_y_series(y_test, preds_test_df.index).values
        test_auc = roc_auc_score(y_te, preds_test_df.values)
        print("TEST AUC (blend final):", round(test_auc, 6))

    summary = dict(
        keep_cols=keep_mi,
        removed_by_shift=removed_cols,
        oof={k: float(roc_auc_score(y_vec, v)) for k, v in preds_train.items()},
        oof_blend=float(auc_final),
        oof_meta=float(auc_meta),
        oof_meta_mix=float(best_auc_mix),
        weights=w_final,
        test_auc=float(test_auc) if test_auc is not None else None
    )
    return summary

