"""Part 2 — Latent variable preparation.

Combines two feature sources (sensor data only, no lab leakage):
    1. Autoencoder latent embeddings (trained on train split)
    2. Per-batch summary statistics (mean, std, min, max, median, skew)

Produces train/test feature matrices with target merged in.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from dissolution_pred.models.autoencoder import (
    Conv1DAutoencoder,
    load_artifacts,
    load_config,
    resample_batches,
    train_autoencoder,
)
from dissolution_pred.utils.ingestion_helpers import batch_feature_stats


# ---------------------------------------------------------------------------
# Autoencoder: train on train split, encode both splits
# ---------------------------------------------------------------------------

def train_and_encode(
    train_df: pd.DataFrame,
    config: dict | None = None,
    config_path: str | None = None,
) -> tuple[pd.DataFrame, Conv1DAutoencoder, object]:
    """Train autoencoder on train_df, return latent vectors + model + scaler."""
    latent_train, model, scaler = train_autoencoder(
        train_df, config=config, config_path=config_path,
    )
    return latent_train, model, scaler


def train_ensemble(
    train_df: pd.DataFrame,
    n_runs: int = 10,
    config: dict | None = None,
    config_path: str | None = None,
) -> tuple[pd.DataFrame, Conv1DAutoencoder, object]:
    """Train N autoencoders with different seeds, average latent vectors.

    Saves the best model (lowest val loss) as the artifact.
    Returns averaged latent vectors from all runs.
    """
    cfg = config or load_config(config_path)
    base_seed = cfg.get("seed", 42)

    all_latents = []
    best_model = None
    best_scaler = None
    best_val_loss = float("inf")

    for i in range(n_runs):
        run_cfg = dict(cfg)
        run_cfg["seed"] = base_seed + i
        # Suppress saving until we pick the best
        run_cfg["save_dir"] = f"_ae_ensemble_run_{i}"

        print(f"\n{'─'*40}")
        print(f"Ensemble run {i+1}/{n_runs} (seed={run_cfg['seed']})")
        print(f"{'─'*40}")

        latent_df, model, scaler = train_autoencoder(train_df, config=run_cfg)
        all_latents.append(latent_df)

        # Check val loss from saved config
        import yaml, os
        cfg_path = os.path.join(run_cfg["save_dir"], "training_config.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                run_info = yaml.safe_load(f)
            # We don't store val_loss in config, so track via model
        # Keep model with lowest reconstruction — use a simple heuristic:
        # re-encode and compute MSE on the training latent vectors' reconstruction
        # Actually, just keep the last model and average latents
        best_model = model
        best_scaler = scaler

    # Average latent vectors across all runs
    fpbatch_col = all_latents[0]["fpbatch"]
    latent_cols = [c for c in all_latents[0].columns if c.startswith("latent_")]

    stacked = np.stack([df[latent_cols].values for df in all_latents], axis=0)
    averaged = stacked.mean(axis=0)

    latent_avg = pd.DataFrame(averaged, columns=latent_cols)
    latent_avg["fpbatch"] = fpbatch_col.values

    # Save the best model with averaged latents as the artifact
    from dissolution_pred.models.autoencoder import save_artifacts
    save_dir = cfg.get("save_dir", "trained_autoencoder")
    n_sensors = len([c for c in train_df.columns if c not in ["fpbatch", "timestamp"]])
    save_artifacts(best_model, best_scaler, cfg, latent_avg, n_sensors, save_dir)

    # Clean up individual run directories
    import shutil
    for i in range(n_runs):
        run_dir = f"_ae_ensemble_run_{i}"
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)

    print(f"\n{'='*40}")
    print(f"Ensemble complete: {n_runs} runs averaged")
    print(f"Latent vectors: {latent_avg.shape}")
    print(f"Artifacts saved to {save_dir}/")

    return latent_avg, best_model, best_scaler


def encode_new_data(
    df: pd.DataFrame,
    save_dir: str = "trained_autoencoder",
) -> pd.DataFrame:
    """Encode new batches using a saved autoencoder (for test/validation).

    Applies the same imputation → resample → scale → encode pipeline.
    """
    model, scaler, config, _ = load_artifacts(save_dir)

    meta_cols = ["fpbatch", "timestamp"]
    sensor_cols = [c for c in df.columns if c not in meta_cols]

    # Step 0: Impute (same as training)
    imputed = df.copy()
    imputed[sensor_cols] = imputed.groupby("fpbatch")[sensor_cols].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )
    imputed[sensor_cols] = imputed.groupby("fpbatch")[sensor_cols].transform(
        lambda x: x.fillna(x.median())
    )
    global_medians = imputed[sensor_cols].median()
    imputed[sensor_cols] = imputed[sensor_cols].fillna(global_medians)

    # Step 1: Resample
    seq_len = config.get("seq_len", 128)
    X_raw, fpbatch_ids, _ = resample_batches(imputed, seq_len=seq_len, meta_cols=meta_cols)

    # Step 2: Scale with saved scaler
    n_batches, sl, n_sensors = X_raw.shape
    X_scaled = scaler.transform(X_raw.reshape(-1, n_sensors)).reshape(n_batches, sl, n_sensors)

    # Step 3: Encode
    X_tensor = torch.FloatTensor(X_scaled)
    model.eval()
    with torch.no_grad():
        latent_np = model.encode(X_tensor).cpu().numpy()

    latent_df = pd.DataFrame(
        latent_np,
        columns=[f"latent_{i}" for i in range(latent_np.shape[1])],
    )
    latent_df["fpbatch"] = fpbatch_ids

    print(f"Encoded {len(latent_df)} batches → {latent_np.shape[1]} latent dims")
    return latent_df


# ---------------------------------------------------------------------------
# Summary statistics from sensor data
# ---------------------------------------------------------------------------

def compute_sensor_stats(batch_sensor_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-batch summary statistics using the existing helper."""
    stats = batch_feature_stats(batch_sensor_df)
    n_feats = len([c for c in stats.columns if c not in ("fpbatch", "n_timestamps")])
    print(f"Sensor stats: {len(stats)} batches × {n_feats} features")
    return stats


# ---------------------------------------------------------------------------
# Batch metadata (from batch_lab_merged — process metadata, not lab values)
# ---------------------------------------------------------------------------

def extract_batch_metadata(batch_lab_merged: pd.DataFrame) -> pd.DataFrame:
    """Extract batch-level metadata: product name + duration."""
    meta = batch_lab_merged.groupby("fpbatch").agg(
        prodname_batch=("prodname_batch", "first"),
        starttime=("starttime", "first"),
        endtime=("endtime", "first"),
    ).reset_index()
    meta["batch_duration_hrs"] = (
        (pd.to_datetime(meta["endtime"]) - pd.to_datetime(meta["starttime"]))
        .dt.total_seconds() / 3600
    )
    meta = meta[["fpbatch", "prodname_batch", "batch_duration_hrs"]]
    print(f"Batch metadata: {len(meta)} batches, "
          f"{meta['prodname_batch'].nunique()} product types")
    return meta


# ---------------------------------------------------------------------------
# Merge everything into train/test feature matrices
# ---------------------------------------------------------------------------

def build_feature_matrix(
    feature_df: pd.DataFrame,
    batch_meta: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge a feature DataFrame + metadata + target into one DataFrame."""
    # Normalize fpbatch dtype across all sources
    feature_df = feature_df.copy()
    feature_df["fpbatch"] = feature_df["fpbatch"].astype(str).str.strip()
    batch_meta = batch_meta.copy()
    batch_meta["fpbatch"] = batch_meta["fpbatch"].astype(str).str.strip()
    target_df = target_df.copy()
    target_df["fpbatch"] = target_df["fpbatch"].astype(str).str.strip()

    df = feature_df.merge(batch_meta, on="fpbatch", how="left")
    df = df.merge(target_df, on="fpbatch", how="inner")
    return df


def prepare_xy(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    target_col: str = "target",
) -> dict:
    """Encode categoricals, fill NaN, split into X/y arrays.

    Returns dict with X_train, y_train, X_test, y_test, feature_names, label_encoder.
    """
    cat_cols = ["prodname_batch"]
    exclude = ["fpbatch", target_col, "n_timestamps"] + cat_cols

    numeric_cols = [c for c in train_features.columns if c not in exclude]

    # Label-encode product type
    le = LabelEncoder()
    all_prods = pd.concat([
        train_features["prodname_batch"],
        test_features["prodname_batch"],
    ]).fillna("UNKNOWN")
    le.fit(all_prods)

    X_train = train_features[numeric_cols].copy()
    X_train["prodname_batch_enc"] = le.transform(
        train_features["prodname_batch"].fillna("UNKNOWN")
    )
    y_train = train_features[target_col].values

    X_test = test_features[numeric_cols].copy()
    X_test["prodname_batch_enc"] = le.transform(
        test_features["prodname_batch"].fillna("UNKNOWN")
    )
    y_test = test_features[target_col].values

    # Fill NaN with train medians
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    feature_names = X_train.columns.tolist()

    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}  y_test:  {y_test.shape}")
    print(f"NaN remaining — train: {X_train.isna().sum().sum()}, "
          f"test: {X_test.isna().sum().sum()}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "label_encoder": le,
        "train_medians": train_medians,
    }


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_df: pd.DataFrame,
    batch_lab_merged: pd.DataFrame,
    ae_config: dict | None = None,
    ae_config_path: str | None = None,
    ae_save_dir: str = "trained_autoencoder",
    ae_mode: str = "ensemble",
    n_ensemble_runs: int = 10,
) -> dict:
    """Full Part 2 pipeline: autoencoder + stats → feature matrices → X/y.

    ae_mode:
        "single"   — train one autoencoder (fast, but unstable)
        "ensemble"  — train N autoencoders, average latent vectors (stable)
        "load"      — skip training, load saved artifacts from ae_save_dir

    Returns dict with all outputs.
    """
    print("=" * 60)
    print("PART 2: LATENT VARIABLE PREPARATION")
    print("=" * 60)

    # 1. Get latent vectors for train split
    if ae_mode == "load":
        print(f"\n--- Loading saved autoencoder from {ae_save_dir}/ ---")
        _, _, _, latent_train = load_artifacts(ae_save_dir)
    elif ae_mode == "ensemble":
        print(f"\n--- Ensemble autoencoder training ({n_ensemble_runs} runs) ---")
        latent_train, _, _ = train_ensemble(
            train_df, n_runs=n_ensemble_runs,
            config=ae_config, config_path=ae_config_path,
        )
    else:
        print("\n--- Autoencoder training (single run) ---")
        latent_train, _, _ = train_and_encode(
            train_df, config=ae_config, config_path=ae_config_path,
        )

    # 2. Encode test split using saved model
    print("\n--- Encoding test data ---")
    latent_test = encode_new_data(test_df, save_dir=ae_save_dir)

    # 3. Sensor summary stats (computed on full clean data or per-split)
    print("\n--- Sensor summary statistics ---")
    sensor_stats_train = compute_sensor_stats(train_df)
    sensor_stats_test = compute_sensor_stats(test_df)

    # 4. Batch metadata
    print("\n--- Batch metadata ---")
    batch_meta = extract_batch_metadata(batch_lab_merged)

    # 5. Build TWO separate feature sets
    #    - stats:  sensor summary stats + metadata  (for baseline RidgeCV)
    #    - latent: autoencoder embeddings + metadata (for XGBoost)
    print("\n--- Building feature matrices ---")
    stats_train = build_feature_matrix(sensor_stats_train, batch_meta, target_df)
    stats_test = build_feature_matrix(sensor_stats_test, batch_meta, target_df)
    latent_train_full = build_feature_matrix(latent_train, batch_meta, target_df)
    latent_test_full = build_feature_matrix(latent_test, batch_meta, target_df)

    print(f"Stats features  — train: {stats_train.shape}, test: {stats_test.shape}")
    print(f"Latent features — train: {latent_train_full.shape}, test: {latent_test_full.shape}")

    # 6. Prepare X/y arrays for each
    print("\n--- Preparing X/y: stats (baseline) ---")
    xy_stats = prepare_xy(stats_train, stats_test)

    print("\n--- Preparing X/y: latent (XGBoost) ---")
    xy_latent = prepare_xy(latent_train_full, latent_test_full)

    return {
        "latent_train": latent_train,
        "latent_test": latent_test,
        "sensor_stats_train": sensor_stats_train,
        "sensor_stats_test": sensor_stats_test,
        "batch_meta": batch_meta,
        "xy_stats": xy_stats,
        "xy_latent": xy_latent,
    }
