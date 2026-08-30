"""Plotting helpers for the dissolution prediction pipeline."""

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
) -> None:
    """Scatter plot: predicted vs actual + residuals."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pred vs actual
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{model_name}: Predicted vs Actual")
    ax.legend()

    # Residuals
    ax = axes[1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidth=0.5)
    ax.axhline(0, color="red", ls="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title(f"{model_name}: Residuals")

    plt.tight_layout()
    plt.show()


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
