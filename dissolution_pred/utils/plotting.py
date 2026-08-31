"""Plotting helpers for the dissolution prediction pipeline."""

import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_latent_target_correlation(
    latent_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_col: str = "target",
    title_suffix: str = "",
) -> None:
    """Bar chart + heatmap showing latent dimension correlations with target."""
    latent_df = latent_df.copy()
    target_df = target_df.copy()
    latent_df["fpbatch"] = latent_df["fpbatch"].astype(str).str.strip()
    target_df["fpbatch"] = target_df["fpbatch"].astype(str).str.strip()
    merged = latent_df.merge(target_df, on="fpbatch", how="inner")
    merged[target_col] = pd.to_numeric(merged[target_col], errors="coerce")
    merged = merged.dropna(subset=[target_col])

    latent_cols = [c for c in latent_df.columns if c.startswith("latent_")]
    corr = merged[latent_cols + [target_col]].corr()
    target_corr = corr[target_col].drop(target_col).sort_values(key=abs, ascending=False)

    print(f"Latent feature correlations with {target_col} "
          f"({len(merged)} batches){title_suffix}:")
    print(target_corr.round(4))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Bar chart
    ax = axes[0]
    colors = ["red" if v < 0 else "steelblue" for v in target_corr.values]
    ax.barh(target_corr.index, target_corr.values, color=colors)
    ax.set_title(f"Latent dim correlations with {target_col}")
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Pearson r")

    # Heatmap
    ax = axes[1]
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, square=True, ax=ax, annot_kws={"size": 9},
    )
    ax.set_title(f"Latent features + {target_col} correlation")

    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
) -> None:
    """Horizontal bar plot of top feature importances."""
    top = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.3)))
    ax.barh(top["feature"], top["importance"], color="steelblue")
    ax.set_title(f"Feature Importance (top {top_n})")
    ax.set_xlabel("Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: str | None = None,
) -> plt.Figure:
    """Scatter plot: predicted vs actual + residuals."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pred vs actual
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.3, s=8, edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{model_name}: Predicted vs Actual")
    ax.legend()

    # Residuals
    ax = axes[1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.3, s=8, edgecolors="none")
    ax.axhline(0, color="red", ls="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title(f"{model_name}: Residuals")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def generate_report(
    *,
    target_df: pd.DataFrame,
    target_train: pd.DataFrame,
    target_val: pd.DataFrame,
    target_test: pd.DataFrame,
    latent_train: pd.DataFrame,
    latent_val: pd.DataFrame,
    latent_test: pd.DataFrame,
    ae_config: dict,
    best_model,
    feat_names: list[str],
    imp_df: pd.DataFrame,
    y_val_true: np.ndarray,
    y_val_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    val_r2: float,
    test_r2: float,
    test_mae: float,
    test_rmse: float,
    model_params: dict,
    res_df: pd.DataFrame,
    report_dir: str = "report",
) -> str:
    """Generate a complete report with all plots and tables saved to one folder.

    Parameters
    ----------
    target_df : full target DataFrame (all splits)
    target_train/val/test : split target DataFrames
    latent_train/val/test : latent vector DataFrames per split
    ae_config : autoencoder configuration dict
    best_model : trained XGBoost model
    feat_names : list of feature names
    imp_df : feature importance DataFrame (columns: feature, importance)
    y_val_true/pred : validation actuals and predictions
    y_test_true/pred : test actuals and predictions
    val_r2, test_r2, test_mae, test_rmse : metrics
    model_params : best model hyperparameters dict
    res_df : model comparison results DataFrame
    report_dir : output directory

    Returns
    -------
    str : path to the report directory
    """
    os.makedirs(report_dir, exist_ok=True)
    plots_dir = os.path.join(report_dir, "plots")
    tables_dir = os.path.join(report_dir, "tables")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    print("=" * 60)
    print(f"GENERATING REPORT → {os.path.abspath(report_dir)}")
    print("=" * 60)

    # ── 1. Target distribution (6-panel) ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("FilteredArea1 — Target Distribution", fontsize=16, fontweight="bold")

    sns.histplot(target_df["target"], kde=True, bins=50, ax=axes[0, 0], color="steelblue")
    axes[0, 0].set_title("Overall Distribution")
    axes[0, 0].axvline(target_df["target"].mean(), color="red", ls="--",
                        label=f'Mean={target_df["target"].mean():.2f}')
    axes[0, 0].axvline(target_df["target"].median(), color="orange", ls="--",
                        label=f'Median={target_df["target"].median():.2f}')
    axes[0, 0].legend()

    sns.boxplot(y=target_df["target"], ax=axes[0, 1], color="steelblue")
    axes[0, 1].set_title("Boxplot")

    if "time" in target_df.columns:
        sns.boxplot(data=target_df, x="time", y="target", ax=axes[0, 2], palette="viridis")
        axes[0, 2].set_title("Distribution by Time Point")
        axes[0, 2].tick_params(axis="x", rotation=45)

    if "lane" in target_df.columns:
        sns.boxplot(data=target_df, x="lane", y="target", ax=axes[1, 0], palette="Set2")
        axes[1, 0].set_title("Distribution by Lane")

    batch_means = target_df.groupby("fpbatch")["target"].mean().sort_values()
    axes[1, 1].bar(range(len(batch_means)), batch_means.values, color="steelblue", width=1.0)
    axes[1, 1].set_title(f"Mean Target per Batch (n={len(batch_means)})")
    axes[1, 1].set_xlabel("Batch (sorted)")

    if "time" in target_df.columns:
        tp = target_df.groupby("time")["target"].agg(["mean", "std"]).reset_index()
        axes[1, 2].errorbar(tp["time"], tp["mean"], yerr=tp["std"],
                             marker="o", capsize=4, color="steelblue")
        axes[1, 2].set_title("Dissolution Profile (mean ± std)")
        axes[1, 2].set_xlabel("Time (min)")

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "target_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [1/6] Target distribution plot")

    # ── 2. Latent-target correlation heatmap ──
    def _batch_corr(lat_df, tgt_df):
        lat = lat_df.copy()
        tgt = tgt_df.copy()
        lat["fpbatch"] = lat["fpbatch"].astype(str).str.strip()
        tgt["fpbatch"] = tgt["fpbatch"].astype(str).str.strip()
        batch_tgt = tgt.groupby("fpbatch")["target"].mean()
        merged = lat.merge(batch_tgt.rename("target"), on="fpbatch", how="inner")
        lcols = [c for c in lat.columns if c.startswith("latent_")]
        return merged[lcols].corrwith(merged["target"])

    corr_train = _batch_corr(latent_train, target_train)
    corr_val = _batch_corr(latent_val, target_val)
    corr_test = _batch_corr(latent_test, target_test)

    latent_cols = [c for c in latent_train.columns if c.startswith("latent_")]

    # Heatmap
    lat_m = latent_train.copy()
    lat_m["fpbatch"] = lat_m["fpbatch"].astype(str).str.strip()
    tgt_m = target_train.copy()
    tgt_m["fpbatch"] = tgt_m["fpbatch"].astype(str).str.strip()
    batch_tgt_m = tgt_m.groupby("fpbatch")["target"].mean()
    merged_m = lat_m.merge(batch_tgt_m.rename("target"), on="fpbatch", how="inner")
    corr_matrix = merged_m[latent_cols + ["target"]].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, ax=ax, annot_kws={"size": 9})
    ax.set_title("Latent Features + Target Correlation (Train)")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "latent_target_correlation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [2/6] Latent-target correlation heatmap")

    # ── 3. Predicted vs Actual (val) ──
    plot_predictions(y_val_true, y_val_pred, "Validation",
                     save_path=os.path.join(plots_dir, "pred_vs_actual_val.png"))
    plt.close("all")
    print("  [3/6] Predicted vs actual (validation)")

    # ── 4. Predicted vs Actual (test) ──
    plot_predictions(y_test_true, y_test_pred, "Test",
                     save_path=os.path.join(plots_dir, "pred_vs_actual_test.png"))
    plt.close("all")
    print("  [4/6] Predicted vs actual (test)")

    # ── 5. Feature importance bar chart ──
    top = imp_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="steelblue")
    ax.set_title("Top 20 Feature Importances (XGBoost)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [5/6] Feature importance chart")

    # ── 6. Target distribution by split ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, tgt, color in [("Train", target_train, "#2196F3"),
                               ("Val", target_val, "#FF9800"),
                               ("Test", target_test, "#4CAF50")]:
        ax.hist(tgt["target"], bins=50, alpha=0.5, label=f"{name} (n={len(tgt)})", color=color)
    ax.set_title("Target Distribution by Split")
    ax.set_xlabel("FilteredArea1")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "target_by_split.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [6/6] Target distribution by split")

    # ── Tables ──
    # Correlation table
    corr_table = pd.DataFrame({
        "Latent Dim": latent_cols,
        "Train Corr": corr_train.values,
        "Val Corr": corr_val.values,
        "Test Corr": corr_test.values,
        "Train |Corr|": corr_train.abs().values,
    }).sort_values("Train |Corr|", ascending=False)

    summary_row = pd.DataFrame([{
        "Latent Dim": "MEAN |corr|",
        "Train Corr": corr_train.abs().mean(),
        "Val Corr": corr_val.abs().mean(),
        "Test Corr": corr_test.abs().mean(),
        "Train |Corr|": corr_train.abs().mean(),
    }])
    corr_table = pd.concat([corr_table, summary_row], ignore_index=True)
    corr_table.to_csv(os.path.join(tables_dir, "latent_target_correlations.csv"), index=False)

    # Model comparison
    res_df.to_csv(os.path.join(tables_dir, "model_comparison.csv"), index=False)

    # Feature importances
    imp_df.to_csv(os.path.join(tables_dir, "feature_importances.csv"), index=False)

    # Best model config
    config_out = dict(model_params)
    config_out["val_R2"] = float(val_r2)
    config_out["test_R2"] = float(test_r2)
    config_out["test_MAE"] = float(test_mae)
    config_out["test_RMSE"] = float(test_rmse)
    config_out["ae_config"] = {k: v for k, v in ae_config.items()
                                if k in ["latent_dim", "channels", "seq_len", "lr", "epochs"]}
    with open(os.path.join(tables_dir, "best_model_config.json"), "w") as f:
        json.dump(config_out, f, indent=2, default=str)

    # Target stats by split
    split_stats = []
    for name, tgt in [("Train", target_train), ("Val", target_val), ("Test", target_test)]:
        t = tgt["target"]
        split_stats.append({
            "Split": name, "N": len(t), "Batches": tgt["fpbatch"].nunique(),
            "Mean": t.mean(), "Median": t.median(), "Std": t.std(),
            "Min": t.min(), "Max": t.max(),
        })
    pd.DataFrame(split_stats).to_csv(os.path.join(tables_dir, "target_stats_by_split.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"REPORT COMPLETE — {os.path.abspath(report_dir)}")
    print(f"{'='*60}")
    print(f"  plots/  ({len(os.listdir(plots_dir))} files)")
    for f in sorted(os.listdir(plots_dir)):
        print(f"    {f}")
    print(f"  tables/ ({len(os.listdir(tables_dir))} files)")
    for f in sorted(os.listdir(tables_dir)):
        print(f"    {f}")

    return os.path.abspath(report_dir)


def plot_results_comparison(
    combined_results: pd.DataFrame,
) -> None:
    """Bar chart comparing all models on test R²."""
    df = combined_results.sort_values("test_R2", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.35)))
    colors = ["steelblue" if "XGB" not in m else "darkorange" for m in df["model"]]
    ax.barh(df["model"], df["test_R2"], color=colors)
    ax.set_xlabel("Test R²")
    ax.set_title("Model Comparison — Test R²")
    ax.axvline(0, color="black", lw=0.5)
    plt.tight_layout()
    plt.show()
