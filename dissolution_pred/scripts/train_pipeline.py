"""Model training script — loads saved AE artifacts, builds features, trains models,
evaluates on test set, and generates a report folder with all plots + tables.

Usage:
    # First train the autoencoder:
    python -m dissolution_pred.scripts.train_autoencoder

    # Then train models + generate report:
    python -m dissolution_pred.scripts.train_pipeline
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dissolution_pred.data_preparation.data_split import prepare_data
from dissolution_pred.data_preparation.feature_engineering import (
    encode_new_data,
    compute_sensor_stats,
    extract_batch_metadata,
    build_feature_matrix,
    prepare_xy,
    prepare_features,
)
from dissolution_pred.models.model_training import train_models
from dissolution_pred.utils.plotting import (
    plot_latent_target_correlation,
    plot_feature_importance,
    plot_results_comparison,
    generate_report,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train dissolution quality prediction models.",
    )
    default_data_dir = os.path.join(project_root, "..")
    parser.add_argument(
        "--data-dir", type=str, default=default_data_dir,
        help="Directory containing batch_data.csv, lab_data.csv, "
             "sensor_data.csv, tag_offsets.xlsx  (default: %(default)s)",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.60,
        help="Fraction of batches for training (default: 0.60)",
    )
    parser.add_argument(
        "--target", type=str, default="FilteredArea1",
        help="Lab variable to predict (default: FilteredArea1)",
    )
    parser.add_argument(
        "--nan-threshold", type=float, default=0.90,
        help="Drop columns with NaN fraction above this (default: 0.90)",
    )
    parser.add_argument(
        "--ae-save-dir", type=str, default="trained_artifacts",
        help="Directory with saved autoencoder artifacts (default: trained_artifacts)",
    )
    parser.add_argument(
        "--artifacts-dir", type=str, default="trained_artifacts",
        help="Directory to save all trained artifacts (default: trained_artifacts)",
    )
    parser.add_argument(
        "--report-dir", type=str, default="report",
        help="Directory to save report plots + tables (default: report)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)

    # ── Data paths ──
    batch_data_path = os.path.join(data_dir, "batch_data.csv")
    lab_data_path = os.path.join(data_dir, "lab_data.csv")
    sensor_data_path = os.path.join(data_dir, "sensor_data.csv")
    tagoffset_path = os.path.join(data_dir, "tag_offsets.xlsx")

    for p in [batch_data_path, lab_data_path, sensor_data_path, tagoffset_path]:
        if not os.path.isfile(p):
            sys.exit(f"ERROR: File not found: {p}")

    # ── Load raw data ──
    print(f"Loading raw data from {data_dir}/")
    batch_df = pd.read_csv(batch_data_path)
    sensor_df = pd.read_csv(sensor_data_path)
    lab_df = pd.read_csv(lab_data_path)
    tagoffset_df = pd.read_excel(tagoffset_path)
    print(f"  batch:  {batch_df.shape}")
    print(f"  sensor: {sensor_df.shape}")
    print(f"  lab:    {lab_df.shape}")
    print(f"  offsets: {tagoffset_df.shape}")

    # ══════════════════════════════════════════════════════════
    # PART 1: Data Preparation
    # ══════════════════════════════════════════════════════════
    data = prepare_data(
        batch_df, sensor_df, lab_df, tagoffset_df,
        frequency="1min",
        col_nan_threshold=args.nan_threshold,
        train_frac=args.train_frac,
        target_variable=args.target,
    )

    # Split target by train/val/test to prevent leakage
    target_df = data["target_df"]
    target_df["fpbatch"] = target_df["fpbatch"].astype(str).str.strip()
    train_batches_str = [str(b).strip() for b in data["train_batches"]]
    val_batches_str = [str(b).strip() for b in data["val_batches"]]
    test_batches_str = [str(b).strip() for b in data["test_batches"]]

    target_train = target_df[target_df["fpbatch"].isin(train_batches_str)].reset_index(drop=True)
    target_val = target_df[target_df["fpbatch"].isin(val_batches_str)].reset_index(drop=True)
    target_test = target_df[target_df["fpbatch"].isin(test_batches_str)].reset_index(drop=True)

    print(f"\nTarget split (no leakage):")
    print(f"  Train: {len(target_train)} rows, {target_train['fpbatch'].nunique()} batches")
    print(f"  Val:   {len(target_val)} rows, {target_val['fpbatch'].nunique()} batches")
    print(f"  Test:  {len(target_test)} rows, {target_test['fpbatch'].nunique()} batches")

    train_df = data["train_df"]
    val_df = data["val_df"]
    test_df = data["test_df"]

    # ══════════════════════════════════════════════════════════
    # PART 2: Feature Preparation (AE latents + sensor stats)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PART 2: FEATURE PREPARATION")
    print("=" * 60)

    ae_dir = args.ae_save_dir

    # Load pre-trained AE latents for train, encode val/test
    from dissolution_pred.models.autoencoder import load_artifacts
    _, _, ae_cfg, latent_train = load_artifacts(ae_dir)
    latent_val = encode_new_data(val_df, save_dir=ae_dir)
    latent_test = encode_new_data(test_df, save_dir=ae_dir)

    # Sensor summary stats
    stats_train = compute_sensor_stats(train_df)
    stats_val = compute_sensor_stats(val_df)
    stats_test = compute_sensor_stats(test_df)

    # Batch metadata
    batch_meta = extract_batch_metadata(data["batch_lab_merged"])

    # Build feature matrices per split
    stats_train_full = build_feature_matrix(stats_train, batch_meta, target_train)
    stats_val_full = build_feature_matrix(stats_val, batch_meta, target_val)
    stats_test_full = build_feature_matrix(stats_test, batch_meta, target_test)

    latent_train_full = build_feature_matrix(latent_train, batch_meta, target_train)
    latent_val_full = build_feature_matrix(latent_val, batch_meta, target_val)
    latent_test_full = build_feature_matrix(latent_test, batch_meta, target_test)

    print(f"\nStats features  — train: {stats_train_full.shape}, val: {stats_val_full.shape}, test: {stats_test_full.shape}")
    print(f"Latent features — train: {latent_train_full.shape}, val: {latent_val_full.shape}, test: {latent_test_full.shape}")

    # ══════════════════════════════════════════════════════════
    # PART 2b: Baseline models (RidgeCV + XGBoost on separate feature sets)
    # ══════════════════════════════════════════════════════════
    xy_stats = prepare_xy(stats_train_full, stats_val_full)
    xy_latent = prepare_xy(latent_train_full, latent_val_full)

    print("\n--- Baseline models ---")
    baseline_results = train_models(
        xy_stats=xy_stats,
        xy_latent=xy_latent,
        save_dir=args.artifacts_dir,
    )

    # ══════════════════════════════════════════════════════════
    # PART 3: Improved Model (combined features + log1p target)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PART 3: IMPROVED MODEL (combined + log1p)")
    print("=" * 60)

    # Combine latent + stats features
    combined_train = latent_train_full.merge(
        stats_train_full.drop(columns=["target", "lane", "time", "prodname_batch", "batch_duration_hrs"], errors="ignore"),
        on="fpbatch", how="left", suffixes=("", "_stats"),
    )
    combined_val = latent_val_full.merge(
        stats_val_full.drop(columns=["target", "lane", "time", "prodname_batch", "batch_duration_hrs"], errors="ignore"),
        on="fpbatch", how="left", suffixes=("", "_stats"),
    )
    combined_test = latent_test_full.merge(
        stats_test_full.drop(columns=["target", "lane", "time", "prodname_batch", "batch_duration_hrs"], errors="ignore"),
        on="fpbatch", how="left", suffixes=("", "_stats"),
    )

    xy_combined = prepare_xy(combined_train, combined_val)

    # log1p target transform
    y_train_log = np.log1p(xy_combined["y_train"])
    y_val_log = np.log1p(xy_combined["y_test"])
    X_tr = xy_combined["X_train"].values
    X_vl = xy_combined["X_test"].values
    feat_names = xy_combined["feature_names"]

    # XGBoost configs
    xgb_configs = [
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.7},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 10, "subsample": 0.7, "colsample_bytree": 0.7},
        {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.01, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.05, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.02, "min_child_weight": 10, "subsample": 0.7, "colsample_bytree": 0.6},
        {"n_estimators": 500, "max_depth": 2, "learning_rate": 0.01, "min_child_weight": 5, "subsample": 0.9, "colsample_bytree": 0.9},
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, vl_idx = next(kf.split(X_tr))

    all_results = []
    best_xgb = None
    best_r2 = -np.inf

    for cfg in xgb_configs:
        m = xgb.XGBRegressor(**cfg, random_state=42, verbosity=0)
        m.fit(X_tr[tr_idx], y_train_log[tr_idx],
              eval_set=[(X_tr[vl_idx], y_train_log[vl_idx])], verbose=False)

        pred_val = np.expm1(m.predict(X_vl))
        y_val_orig = xy_combined["y_test"]
        r2 = r2_score(y_val_orig, pred_val)
        mae = mean_absolute_error(y_val_orig, pred_val)
        label = f"XGB(n={cfg['n_estimators']},d={cfg['max_depth']},lr={cfg['learning_rate']})"
        all_results.append({"model": label, "val_R2": r2, "val_MAE": mae, "features": "combined+log1p"})
        if r2 > best_r2:
            best_r2 = r2
            best_xgb = m

    res_df = pd.DataFrame(all_results).sort_values("val_R2", ascending=False)
    print("\nIMPROVED RESULTS (sorted by val R²)")
    print("=" * 70)
    print(res_df.to_string(index=False))

    # Feature importance
    imp = pd.DataFrame({"feature": feat_names, "importance": best_xgb.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    print(f"\nBest model val R²={best_r2:.4f}")
    print(f"\nTop 15 features:")
    print(imp.head(15).to_string(index=False))

    # ══════════════════════════════════════════════════════════
    # PART 4: Test Set Evaluation
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PART 4: TEST SET EVALUATION")
    print("=" * 60)

    xy_test = prepare_xy(combined_train, combined_test)
    X_test_arr = xy_test["X_test"].values
    y_test_true = xy_test["y_test"]

    pred_test = np.expm1(best_xgb.predict(X_test_arr))
    test_r2 = r2_score(y_test_true, pred_test)
    test_mae = mean_absolute_error(y_test_true, pred_test)
    test_rmse = float(np.sqrt(mean_squared_error(y_test_true, pred_test)))

    print(f"  Val  R²:  {best_r2:.4f}")
    print(f"  Test R²:  {test_r2:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Samples: {len(y_test_true)}")

    # Save artifacts
    import joblib
    os.makedirs(args.artifacts_dir, exist_ok=True)
    joblib.dump(best_xgb, os.path.join(args.artifacts_dir, "best_xgboost_tuned.pkl"))

    best_params = best_xgb.get_params()
    save_params = {k: v for k, v in best_params.items()
                   if k in ["n_estimators", "max_depth", "learning_rate", "min_child_weight",
                            "subsample", "colsample_bytree", "random_state"]}
    save_params["target_transform"] = "log1p"
    save_params["feature_set"] = "combined (latent + stats)"
    save_params["val_R2"] = float(best_r2)
    save_params["test_R2"] = float(test_r2)
    save_params["test_MAE"] = float(test_mae)
    save_params["test_RMSE"] = test_rmse

    with open(os.path.join(args.artifacts_dir, "best_model_config.json"), "w") as f:
        json.dump(save_params, f, indent=2)

    # ══════════════════════════════════════════════════════════
    # PART 5: Generate Report
    # ══════════════════════════════════════════════════════════
    y_val_pred = np.expm1(best_xgb.predict(X_vl))

    generate_report(
        target_df=target_df,
        target_train=target_train,
        target_val=target_val,
        target_test=target_test,
        latent_train=latent_train,
        latent_val=latent_val,
        latent_test=latent_test,
        ae_config=ae_cfg,
        best_model=best_xgb,
        feat_names=feat_names,
        imp_df=imp,
        y_val_true=xy_combined["y_test"],
        y_val_pred=y_val_pred,
        y_test_true=y_test_true,
        y_test_pred=pred_test,
        val_r2=best_r2,
        test_r2=test_r2,
        test_mae=test_mae,
        test_rmse=test_rmse,
        model_params=save_params,
        res_df=res_df,
        report_dir=args.report_dir,
    )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
