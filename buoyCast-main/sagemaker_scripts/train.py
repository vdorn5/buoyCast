"""
SageMaker Training script for buoyCast (scikit-learn).

Reads CSV from the SageMaker training channel and writes a joblib model
artifact to SM_MODEL_DIR.

Expected input schema:
  - features: numeric columns
  - target column: target_E_star
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected file at {path}")
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--max_iter", type=int, default=400)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--min_samples_leaf", type=int, default=30)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    val_dir = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    train_path = os.path.join(train_dir, "train.csv")
    val_path = os.path.join(val_dir, "validation.csv")

    df_train = _read_csv(train_path)
    df_val = _read_csv(val_path) if os.path.exists(val_path) else None

    target_col = "target_E_star"
    if target_col not in df_train.columns:
        raise KeyError(f"Expected '{target_col}' in training data columns")

    feature_cols = [c for c in df_train.columns if c != target_col]
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        max_iter=args.max_iter,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )

    print("[INFO] Training model...")
    model.fit(X_train, y_train)

    # Optional validation metrics
    metrics: dict[str, Any] = {}
    if df_val is not None and len(df_val) > 0 and target_col in df_val.columns:
        X_val = df_val[feature_cols]
        y_val = df_val[target_col]
        pred = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
        mae = float(mean_absolute_error(y_val, pred))
        metrics.update({"val_rmse_E_star": rmse, "val_mae_E_star": mae})
        print(f"[INFO] Validation RMSE(E*): {rmse:.4f} | MAE(E*): {mae:.4f}")

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)

    meta = {
        "target_col": target_col,
        "feature_cols": feature_cols,
        "algorithm": "HistGradientBoostingRegressor",
        "hyperparameters": vars(args),
        "metrics": metrics,
    }
    with open(os.path.join(model_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Saved model to {model_path}")


if __name__ == "__main__":
    main()
