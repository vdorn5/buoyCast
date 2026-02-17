"""
SageMaker inference script for the buoyCast scikit-learn model.

This enables Model Registry -> Endpoint deployment with a custom handler.

Supported input content types:
  - text/csv: rows of feature values (no header)
  - application/json: {"instances": [[...], [...]]}

Output:
  - application/json: {"predictions": [...]}  where predictions are E_star (>=0)
"""

from __future__ import annotations

import io
import json
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd


def model_fn(model_dir: str):
    model_path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model file at {model_path}")
    return joblib.load(model_path)


def input_fn(request_body: bytes | str, request_content_type: str) -> np.ndarray:
    if isinstance(request_body, bytes):
        body = request_body.decode("utf-8")
    else:
        body = request_body

    ct = (request_content_type or "").lower()

    if ct.startswith("text/csv"):
        # CSV with no header
        df = pd.read_csv(io.StringIO(body), header=None)
        return df.to_numpy(dtype=float)

    if ct.startswith("application/json"):
        payload = json.loads(body)
        if isinstance(payload, dict) and "instances" in payload:
            return np.asarray(payload["instances"], dtype=float)
        if isinstance(payload, list):
            return np.asarray(payload, dtype=float)
        raise ValueError("JSON payload must be a list or a dict with key 'instances'.")

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: np.ndarray, model) -> Any:
    pred = model.predict(input_data)
    pred = np.asarray(pred, dtype=float)

    # Physics guardrail: E_star cannot be negative.
    pred = np.clip(pred, 0.0, None)
    return pred


def output_fn(prediction: Any, accept: str) -> tuple[str, str]:
    accept = (accept or "application/json").lower()

    if accept.startswith("application/json"):
        out = {"predictions": np.asarray(prediction).tolist()}
        return json.dumps(out), "application/json"

    if accept.startswith("text/csv"):
        arr = np.asarray(prediction).reshape(-1, 1)
        df = pd.DataFrame(arr)
        return df.to_csv(index=False, header=False), "text/csv"

    # Default
    out = {"predictions": np.asarray(prediction).tolist()}
    return json.dumps(out), "application/json"
