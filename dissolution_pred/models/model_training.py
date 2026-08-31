"""Part 3 — Model training: RidgeCV baseline + XGBoost.

Two models only:
    1. RidgeCV (auto-selects regularization via LOO-CV)
    2. XGBoost (config-driven sweep)
"""

import joblib
import numpy as np
import pandas as pd
import yaml
import os
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dissolution_pred.models.baseline_regression import train_baseline
from dissolution_pred.models.xgboost_model import train_xgboost_sweep


_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "model_config.yaml")


def load_training_config(path: str | None = None) -> dict:
    """Load the full model config (xgboost section)."""
    cfg_path = path or _CONFIG_PATH
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def train_models(
    xy_stats: dict,
    xy_latent: dict,
    config_path: str | None = None,
    save_dir: str = "trained_artifacts",
) -> dict:
    """Full Part 3 pipeline.

    - Baseline (RidgeCV): trained on sensor summary stats
    - XGBoost: trained on autoencoder latent variables

    Saves the best model, scaler, and results to *save_dir*.
    Returns dict with results DataFrames and importance.
    """
    print("=" * 60)
    print("PART 3: MODEL TRAINING")
    print("=" * 60)

    config = load_training_config(config_path)
    xgb_config = config.get("xgboost", {})

    # --- Baseline: RidgeCV on sensor stats ---
    print("\n--- RidgeCV (sensor stats) ---")
    baseline_results, baseline_model, baseline_scaler = train_baseline(
        xy_stats["X_train"].values, xy_stats["y_train"],
        xy_stats["X_test"].values, xy_stats["y_test"],
    )
    baseline_results["model"] = baseline_results["model"].apply(lambda x: f"{x} [stats]")

    # --- RidgeCV on latent variables ---
    print("\n--- RidgeCV (latent variables) ---")
    latent_ridge_results, latent_ridge_model, latent_ridge_scaler = train_baseline(
        xy_latent["X_train"].values, xy_latent["y_train"],
        xy_latent["X_test"].values, xy_latent["y_test"],
    )
    latent_ridge_results["model"] = latent_ridge_results["model"].apply(lambda x: f"{x} [latent]")

    # --- XGBoost on latent variables ---
    X_tr = xy_latent["X_train"].values
    y_tr = xy_latent["y_train"]
    X_te = xy_latent["X_test"].values
    y_te = xy_latent["y_test"]

    # Split train into train/val for XGBoost early stopping
    seed = xgb_config.get("seed", 42)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    tr_idx, vl_idx = next(kf.split(X_tr))

    print("\n--- XGBoost (on latent variables) ---")
    xgb_results, importance_df, best_xgb_model = train_xgboost_sweep(
        X_tr[tr_idx], y_tr[tr_idx],
        X_tr[vl_idx], y_tr[vl_idx],
        X_te, y_te,
        xy_latent["feature_names"], xgb_config,
    )

    # --- Combined summary ---
    print("\n" + "=" * 60)
    print("RESULTS (sorted by test R²)")
    print("=" * 60)
    combined = pd.concat([baseline_results, latent_ridge_results, xgb_results], ignore_index=True)
    combined = combined.sort_values("test_R2", ascending=False).reset_index(drop=True)
    print(combined.to_string(index=False))

    best = combined.iloc[0]
    print(f"\nBest model: {best['model']} (test R²={best['test_R2']:.4f})")

    # ── Save all trained artifacts ──
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(best_xgb_model, os.path.join(save_dir, "best_xgboost.pkl"))
    joblib.dump(baseline_model, os.path.join(save_dir, "ridge_stats.pkl"))
    joblib.dump(baseline_scaler, os.path.join(save_dir, "ridge_stats_scaler.pkl"))
    joblib.dump(latent_ridge_model, os.path.join(save_dir, "ridge_latent.pkl"))
    joblib.dump(latent_ridge_scaler, os.path.join(save_dir, "ridge_latent_scaler.pkl"))
    combined.to_csv(os.path.join(save_dir, "model_results.csv"), index=False)
    importance_df.to_csv(os.path.join(save_dir, "feature_importance.csv"), index=False)

    print(f"\nTrained model artifacts saved to {os.path.abspath(save_dir)}/")
    for fname in sorted(os.listdir(save_dir)):
        fpath = os.path.join(save_dir, fname)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {fname} ({size_kb:.1f} KB)")

    return {
        "baseline_results": baseline_results,
        "xgboost_results": xgb_results,
        "combined_results": combined,
        "importance_df": importance_df,
    }
