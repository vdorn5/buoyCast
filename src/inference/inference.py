# inference/inference.py
import os, json
import joblib
import numpy as np

def model_fn(model_dir):
    # Training creates model artifact; sklearn container extracts into model_dir
    for fname in ["model.joblib", "model.pkl"]:
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError(f"No model file found in {model_dir}")

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        payload = json.loads(request_body)
        X = payload["instances"] if isinstance(payload, dict) and "instances" in payload else payload
        return np.asarray(X, dtype=np.float32)

    if request_content_type in ("text/csv", "application/csv"):
        rows = [r for r in request_body.strip().splitlines() if r.strip()]
        X = [[float(x) for x in r.split(",")] for r in rows]
        return np.asarray(X, dtype=np.float32)

    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        return json.dumps({"predictions": np.asarray(prediction).tolist()}), response_content_type
    return "\n".join(map(str, prediction)), "text/plain"
