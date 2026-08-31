"""XGBoost model sweep with feature importance tracking."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "R2": r2_score(y, pred),
    }


def train_xgboost_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a config-driven sweep of XGBoost models.

    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val     : validation data
    X_test, y_test   : test data
    feature_names    : list of feature column names
    config           : dict from config.yaml["xgboost"]

    Returns
    -------
    results_df      : pd.DataFrame of metrics per config
    importance_df   : pd.DataFrame of feature importances from best model
    """
    seed = config.get("seed", 42)
    results = []
    best_model, best_val_r2 = None, -np.inf

    for cfg in config.get("configs", []):
        m = xgb.XGBRegressor(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            min_child_weight=cfg.get("min_child_weight", 5),
            subsample=cfg.get("subsample", 0.8),
            colsample_bytree=cfg.get("colsample_bytree", 0.8),
            random_state=seed,
            verbosity=0,
        )
        m.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        ev = evaluate(m, X_val, y_val)
        et = evaluate(m, X_test, y_test)
        label = f"XGB(n={cfg['n_estimators']},d={cfg['max_depth']},lr={cfg['learning_rate']})"
        results.append({
            "model": label,
            "val_MAE": ev["MAE"], "val_R2": ev["R2"], "test_R2": et["R2"],
        })
        if ev["R2"] > best_val_r2:
            best_val_r2 = ev["R2"]
            best_model = m

    results_df = pd.DataFrame(results).sort_values("val_R2", ascending=False).reset_index(drop=True)
    print("=== XGBoost Sweep ===")
    print(results_df.to_string(index=False))

    # Feature importance from best model
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": best_model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print(f"\nTop 10 features (best model, val R²={best_val_r2:.4f}):")
    print(importance_df.head(10).to_string(index=False))

    return results_df, importance_df, best_model
