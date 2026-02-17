"""
SageMaker Processing script: evaluate a trained model.

Inputs (ProcessingInput):
  - /opt/ml/processing/model/model.tar.gz  (from TrainingStep model artifacts)
  - /opt/ml/processing/test/test.csv       (from PreprocessStep)

Outputs (ProcessingOutput):
  - /opt/ml/processing/evaluation/evaluation.json

We compute metrics both on:
  - E_star (energy proxy)
  - Hs = sqrt(E_star) (wave height proxy), after clipping E_star >= 0
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _extract_model(artifact_path: str, extract_dir: str) -> str:
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(artifact_path) as tar:
        tar.extractall(path=extract_dir)

    model_path = os.path.join(extract_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Expected model.joblib inside artifact. Looked for: {model_path}"
        )
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifact", type=str, default="/opt/ml/processing/model/model.tar.gz")
    parser.add_argument("--test-csv", type=str, default="/opt/ml/processing/test/test.csv")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/evaluation")
    parser.add_argument("--target-col", type=str, default="target_E_star")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_dir = os.path.join(args.output_dir, "model_extracted")
    model_path = _extract_model(args.model_artifact, model_dir)

    model = joblib.load(model_path)

    df = pd.read_csv(args.test_csv)
    if args.target_col not in df.columns:
        raise KeyError(f"Expected '{args.target_col}' in test data columns")

    feature_cols = [c for c in df.columns if c != args.target_col]
    X = df[feature_cols]
    y = df[args.target_col].to_numpy(dtype=float)

    pred_E = model.predict(X)
    pred_E = np.asarray(pred_E, dtype=float)

    # Physics guardrail: energy proxy must be non-negative
    pred_E_clipped = np.clip(pred_E, 0.0, None)

    rmse_E = float(np.sqrt(mean_squared_error(y, pred_E_clipped)))
    mae_E = float(mean_absolute_error(y, pred_E_clipped))

    hs_true = np.sqrt(np.clip(y, 0.0, None))
    hs_pred = np.sqrt(pred_E_clipped)

    rmse_hs = float(np.sqrt(mean_squared_error(hs_true, hs_pred)))
    mae_hs = float(mean_absolute_error(hs_true, hs_pred))

    neg_frac = float(np.mean(pred_E < 0.0))

    evaluation = {
        "regression_metrics": {
            "rmse_E_star": rmse_E,
            "mae_E_star": mae_E,
            "rmse_Hs": rmse_hs,
            "mae_Hs": mae_hs,
            "neg_prediction_fraction": neg_frac,
            "n_test_rows": int(len(df)),
            "n_features": int(len(feature_cols)),
        }
    }

    out_path = os.path.join(args.output_dir, "evaluation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)

    print("[INFO] Evaluation complete:")
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()
