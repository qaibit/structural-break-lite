#!/usr/bin/env python3
"""
Structural Break Lite — by CONDOR
==================================
Lightweight inference pipeline using HGB, CatBoost and XGBoost.

This is the "Lite" variant of the CONDOR Structural Break Detection system.
It uses the core feature engineering and gradient boosting ensemble from
expertos_8642.py WITHOUT the PINT-Seq neural components (which are
available in the "Complete" variant).

Author: CONDOR — Sovereign Intelligence
Website: https://condor.qaibit.com
"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from expertos_8642 import (
    to_series_dict, build_features_impl2, foldwise_mi_select,
    ks_shift_filter_train_only, oof_hgb_with_test, oof_dist_hgb_with_test,
    oof_catboost_multi, oof_xgb_impl3_with_test, oof_xgb_on_matrix,
    optimize_rank_blend_dirichlet, meta_stacker_lr,
    to_y_series, build_impl3_feature_tables,
    create_comprehensive_features_pd,
    SUBMISSION_NAME, SEED, FOLDS, HAS_CATBOOST, HAS_XGB
)

# Verificar si scipy.optimize está disponible
try:
    from scipy.optimize import minimize
    HAS_SLSQP = True
except Exception:
    HAS_SLSQP = False


def run_lite_inference(
    X_train,
    y_train,
    X_test,
    y_test=None,
    allowed_models=None,
):
    """
    Run the CONDOR Lite ensemble (no PINT-Seq, no neural components).
    
    This pipeline includes:
      1. Feature engineering (Impl2, Impl3, Comprehensive)
      2. KS-Shift filtering for distribution stability
      3. Fold-wise Mutual Information feature selection
      4. Base models: HGB, HGB-Distance, CatBoost A/B, XGBoost
      5. Dirichlet rank-blend + SLSQP optimization
      6. Meta-stacker (Logistic Regression)
    
    Args:
        X_train (pd.DataFrame): Training data with MultiIndex (id, time)
                                 and columns [value, period]
        y_train (pd.Series): Training labels indexed by id
        X_test (pd.DataFrame): Test data with same structure as X_train
        y_test (pd.Series, optional): Test labels for evaluation
        allowed_models (list, optional): Restrict to a subset of models
        
    Returns:
        dict: Summary with OOF AUCs, blend weights, feature importance, and test AUC
    """
    
    print("=" * 80)
    print("🚀 CONDOR STRUCTURAL BREAK LITE")
    print("=" * 80)
    
    # ===== 1. Feature Engineering =====
    print("\n📊 PHASE 1: Feature Engineering...")
    
    # Impl3 features
    Ftr_impl3, Fte_impl3 = build_impl3_feature_tables(X_train, X_test)
    print(f"   ✅ Impl3: {Ftr_impl3.shape}")
    
    # Comprehensive features
    try:
        df_long_tr = X_train.reset_index()[['id', 'period', 'value']]
        df_long_te = X_test.reset_index()[['id', 'period', 'value']]
        feat_train_cf = create_comprehensive_features_pd(df_long_tr).set_index('id')
        feat_test_cf = create_comprehensive_features_pd(df_long_te).set_index('id')
        print(f"   ✅ Comprehensive features: {feat_train_cf.shape}")
    except Exception as e:
        print(f"   ⚠️ Comprehensive features failed: {e}")
        feat_train_cf = pd.DataFrame(index=Ftr_impl3.index)
        feat_test_cf = pd.DataFrame(index=Fte_impl3.index)
    
    # Impl2 features (without PINT injection in Lite mode)
    print("   📊 Building Impl2...")
    pint_tr_df = pd.DataFrame(index=Ftr_impl3.index)
    pint_te_df = pd.DataFrame(index=Fte_impl3.index)
    
    (Xfull, Xfull_t, Xdist, Xdist_t,
     (tr_series, tr_tb), (te_series, te_tb), sig_tr_df) = build_features_impl2(
        X_train, X_test, extra_impl3_feats=None,
        extra_pint_feats_tr=pint_tr_df.join(feat_train_cf, how='left'),
        extra_pint_feats_te=pint_te_df.join(feat_test_cf, how='left')
    )
    print(f"   ✅ Impl2: {Xfull.shape}")
    
    y_vec = to_y_series(y_train, Xfull.index).astype(int).values.ravel()
    
    # KS-Shift Filter
    print("   🧹 Applying KS-shift filter...")
    keep_cols, drop_cols = ks_shift_filter_train_only(Xfull, y_vec, frac=0.05, seed=SEED)
    Xks = Xfull[keep_cols].copy()
    Xks_t = Xfull_t[keep_cols].copy()
    print(f"   ✅ Removed {len(drop_cols)} cols, keeping {len(keep_cols)}")
    
    # MI selection
    print("   🔍 Fold-wise MI selection...")
    Xmi, keep_mi = foldwise_mi_select(Xks, y_vec, folds=FOLDS, topk=420, seed=SEED)
    Xmi_t = Xks_t.reindex(columns=keep_mi).fillna(Xks[keep_mi].median(numeric_only=True))
    print(f"   ✅ MI: {Xmi.shape}")
    
    # ===== 2. Base Models =====
    print("\n🎯 PHASE 2: Base Models (Teacher)...")
    
    preds_train = {}
    preds_test = {}
    
    # HGB
    print("   📈 HGB...")
    oof_hgb, te_hgb, auc_hgb = oof_hgb_with_test(Xmi, y_vec, Xmi_t)
    preds_train["hgb"] = oof_hgb
    preds_test["hgb"] = te_hgb
    print(f"      AUC: {auc_hgb:.4f}")
    
    # Distance HGB
    print("   📈 Distance HGB...")
    oof_dist, te_dist, auc_dist = oof_dist_hgb_with_test(Xdist, y_vec, Xdist_t)
    preds_train["dist"] = oof_dist
    preds_test["dist"] = te_dist
    print(f"      AUC: {auc_dist:.4f}")
    
    # CatBoost A
    if HAS_CATBOOST:
        print("   📈 CatBoost A...")
        cbA = oof_catboost_multi(Xmi, y_vec, Xmi_t, seeds=(42, 1337, 2027), 
                                  params=dict(loss_function="Logloss", eval_metric="AUC",
                                              auto_class_weights="Balanced", iterations=3200,
                                              learning_rate=0.028, depth=7, l2_leaf_reg=12.0,
                                              verbose=False, thread_count=1, rsm=0.92,
                                              border_count=128, bootstrap_type="Bayesian",
                                              bagging_temperature=0.5, random_strength=0.7,
                                              leaf_estimation_iterations=6),
                                  feat_fraction=0.80, label="A")
        if cbA is not None:
            oA, tA, _ = cbA["avg"]
            preds_train["cbA"] = oA
            preds_test["cbA"] = tA
            print(f"      AUC: {roc_auc_score(y_vec, oA):.4f}")
    
    # CatBoost B
    if HAS_CATBOOST:
        print("   📈 CatBoost B...")
        cbB = oof_catboost_multi(Xmi, y_vec, Xmi_t, seeds=(3329, 5153),
                                  params=dict(loss_function="Logloss", eval_metric="AUC",
                                              auto_class_weights="Balanced", iterations=4000,
                                              learning_rate=0.022, depth=8, l2_leaf_reg=9.0,
                                              verbose=False, thread_count=1, rsm=0.95,
                                              border_count=128, bootstrap_type="Bernoulli",
                                              subsample=0.72, random_strength=0.8,
                                              leaf_estimation_iterations=6),
                                  feat_fraction=0.70, label="B")
        if cbB is not None:
            oB, tB, _ = cbB["avg"]
            preds_train["cbB"] = oB
            preds_test["cbB"] = tB
            print(f"      AUC: {roc_auc_score(y_vec, oB):.4f}")
    
    # XGBoost Impl3
    if HAS_XGB:
        print("   📈 XGB Impl3...")
        xgb_oof, xgb_te, auc_xgb = oof_xgb_impl3_with_test(
            Ftr_impl3, to_y_series(y_train, Ftr_impl3.index), Fte_impl3
        )
        preds_train["xgb_raw"] = xgb_oof
        preds_test["xgb_raw"] = xgb_te
        print(f"      AUC: {auc_xgb:.4f}")
    
    # XGBoost Xmi
    if HAS_XGB:
        print("   📈 XGB Xmi...")
        xgb_xmi_oof, xgb_xmi_te, auc_xgb_xmi = oof_xgb_on_matrix(Xmi, y_vec, Xmi_t)
        preds_train["xgb_xmi"] = xgb_xmi_oof
        preds_test["xgb_xmi"] = xgb_xmi_te
        print(f"      AUC: {auc_xgb_xmi:.4f}")
    
    # ===== 3. Blending =====
    print("\n🎯 PHASE 3: Blending...")
    
    if allowed_models is not None:
        allowed_set = set(allowed_models)
        preds_train = {k: v for k, v in preds_train.items() if k in allowed_set}
        preds_test = {k: v for k, v in preds_test.items() if k in allowed_set}
    
    # Rank-blend Dirichlet
    auc_rb, w0 = optimize_rank_blend_dirichlet(preds_train, y_vec)
    print(f"   📊 Rank-Blend (Dirichlet): AUC={auc_rb:.4f}")
    
    # SLSQP refinement
    w_final = w0
    auc_final = auc_rb
    
    if HAS_SLSQP:
        keys = list(w0.keys())
        def rank01_arr(arr):
            ranks = np.argsort(np.argsort(arr))
            return ranks / (len(ranks) - 1 + 1e-12)
        
        R = np.column_stack([rank01_arr(preds_train[k]) for k in keys])
        yv = np.asarray(y_vec, int)
        
        def loss(w):
            w = np.clip(w, 0, 1)
            if w.sum() <= 0: return 1.0
            w = w / w.sum()
            s = (R @ w.reshape(-1, 1)).ravel()
            return 1.0 - roc_auc_score(yv, s)
        
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(np.clip(w, 0, 1)) - 1.0}]
        bnds = [(0.0, 1.0)] * len(keys)
        w0_arr = np.array([w0.get(k, 1.0 / len(keys)) for k in keys], float)
        
        try:
            res = minimize(loss, w0_arr, method='SLSQP', bounds=bnds, constraints=cons, 
                          options={'maxiter': 200, 'ftol': 1e-9, 'disp': False})
            if res.success:
                w_slsqp = np.clip(res.x, 0, 1)
                w_slsqp = w_slsqp / w_slsqp.sum()
                w_dict = {k: float(wi) for k, wi in zip(keys, w_slsqp)}
                auc_slsqp = 1.0 - res.fun
                if auc_slsqp > auc_rb:
                    w_final = w_dict
                    auc_final = auc_slsqp
                    print(f"   ✅ SLSQP improved: {auc_final:.4f}")
        except Exception:
            pass
    
    # Final predictions
    keys_rb = list(w_final.keys())
    Wf = np.array([w_final[k] for k in keys_rb], float)
    Wf = Wf / Wf.sum()
    
    def rank01(arr):
        ranks = np.argsort(np.argsort(arr))
        return ranks / (len(ranks) - 1 + 1e-12)
    
    Rte_rb = np.column_stack([rank01(preds_test[k]) for k in keys_rb])
    s_test = (Rte_rb @ Wf.reshape(-1, 1)).ravel()
    
    preds_test_df = pd.Series(s_test, index=Xmi_t.index, name="break_score").sort_index()
    preds_test_df.to_csv(SUBMISSION_NAME, header=True)
    print(f"\n💾 Saved: {SUBMISSION_NAME}")
    
    # Test AUC if available
    test_auc = None
    if y_test is not None:
        y_te = to_y_series(y_test, preds_test_df.index).values
        test_auc = roc_auc_score(y_te, preds_test_df.values)
        print(f"📊 TEST AUC: {test_auc:.6f}")
    
    summary = dict(
        oof={k: float(roc_auc_score(y_vec, v)) for k, v in preds_train.items()},
        oof_blend=float(auc_final),
        weights=w_final,
        test_auc=float(test_auc) if test_auc is not None else None,
    )
    
    return summary


if __name__ == "__main__":
    print("CONDOR Structural Break Lite")
    print("=" * 40)
    print("Usage:")
    print("  from main import run_lite_inference")
    print("  results = run_lite_inference(X_train, y_train, X_test, y_test)")
    print()
    print("Ensure your data has MultiIndex (id, time) with columns [value, period].")
    print("Refer to README.md for full documentation.")
