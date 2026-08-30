"""Baseline regression model: RidgeCV with automatic alpha selection."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(model, X, y) -> dict:
    """Compute MAE, RMSE, R² for a fitted model."""
    pred = model.predict(X)
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "R2": r2_score(y, pred),
    }


def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[pd.DataFrame, RidgeCV]:
    """Train a RidgeCV baseline (auto-selects regularization via LOO-CV).

    Returns (results_df, fitted_model).
    """
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    alphas = np.logspace(-2, 5, 50)
    model = RidgeCV(alphas=alphas, cv=None).fit(X_tr_sc, y_train)  # LOO-CV

    train_metrics = evaluate(model, X_tr_sc, y_train)
    test_metrics = evaluate(model, X_te_sc, y_test)

    results = pd.DataFrame([{
        "model": f"RidgeCV(α={model.alpha_:.2f})",
        "train_MAE": train_metrics["MAE"],
        "train_R2": train_metrics["R2"],
        "test_MAE": test_metrics["MAE"],
        "test_R2": test_metrics["R2"],
    }])

    print("=== Baseline: RidgeCV ===")
    print(f"  Selected α: {model.alpha_:.2f}")
    print(f"  Train — MAE: {train_metrics['MAE']:.4f}  R²: {train_metrics['R2']:.4f}")
    print(f"  Test  — MAE: {test_metrics['MAE']:.4f}  R²: {test_metrics['R2']:.4f}")

    return results, model
