"""
SageMaker Processing script: build supervised features and train/val/test splits.

This script is designed to run inside a SageMaker Processing Job (e.g., as part
of a SageMaker Pipeline). It expects curated NDBC data (CSV) staged into a local
input directory.

Input layout (example):
  /opt/ml/processing/input/curated/buoy=46086/stdmet.csv
  /opt/ml/processing/input/curated/buoy=46042/stdmet.csv
  ...

Outputs:
  /opt/ml/processing/train/train.csv
  /opt/ml/processing/validation/validation.csv
  /opt/ml/processing/test/test.csv
  /opt/ml/processing/baseline/baseline.csv
  /opt/ml/processing/baseline/inference_sample.csv
  /opt/ml/processing/report/data_report.json
  /opt/ml/processing/report/feature_cols.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import asdict, dataclass
from typing import Sequence, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Feature engineering utils
# -------------------------

def make_supervised(
    df: pd.DataFrame,
    entity_col: str = "station_id",
    time_col: str = "timestamp",
    lead_hours: int = 1,
    lags: Sequence[int] = (1, 2, 3, 6),
    include_current_features: bool = False,
) -> Tuple[pd.DataFrame, list[str]]:
    """Convert a buoy time series into a supervised learning table.

    Physics-safe target:
      E_star = Hs^2 (non-negative wave energy proxy)

    Target:
      target_E_star = E_star shifted -lead_hours
    """
    if entity_col not in df.columns:
        raise KeyError(f"Expected entity_col='{entity_col}' in df.columns")
    if time_col not in df.columns:
        raise KeyError(f"Expected time_col='{time_col}' in df.columns")

    out = df.sort_values([entity_col, time_col]).copy()

    # Base physics-safe state
    out["E_star"] = out["wave_height"] ** 2

    # Forecast target
    out["target_E_star"] = out.groupby(entity_col)["E_star"].shift(-lead_hours)

    base_features = [
        "wave_height",
        "E_star",
        "wind_speed",
        "dominant_wave_period",
    ]

    feature_cols: list[str] = []
    if include_current_features:
        feature_cols.extend([c for c in base_features if c in out.columns])

    for lag in lags:
        for col in base_features:
            if col not in out.columns:
                continue
            lag_col = f"{col}_lag_{lag}"
            out[lag_col] = out.groupby(entity_col)[col].shift(lag)
            feature_cols.append(lag_col)

    required = feature_cols + ["target_E_star"]
    out = out.dropna(subset=required).reset_index(drop=True)
    return out, feature_cols


def time_split(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-ordered train/val/test split."""
    if not np.isclose(sum(splits), 1.0):
        raise ValueError(f"splits must sum to 1.0, got {splits}")

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce", utc=True)
    out = out.dropna(subset=[time_col]).sort_values(time_col)

    n = len(out)
    train_end = int(n * splits[0])
    val_end = int(n * (splits[0] + splits[1]))

    train = out.iloc[:train_end].copy()
    val = out.iloc[train_end:val_end].copy()
    test = out.iloc[val_end:].copy()
    return train, val, test


# -------------------------
# Reporting
# -------------------------

@dataclass(frozen=True)
class DataReport:
    n_rows_raw: int
    n_rows_supervised: int
    n_rows_train: int
    n_rows_val: int
    n_rows_test: int
    n_buoys: int
    buoy_ids: list[str]
    missing_fraction_raw: dict[str, float]
    missing_fraction_supervised: dict[str, float]


def _missing_fraction(df: pd.DataFrame) -> dict[str, float]:
    if len(df) == 0:
        return {}
    return {c: float(df[c].isna().mean()) for c in df.columns}


def main() -> None:
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input/curated")
    parser.add_argument("--train-dir", type=str, default="/opt/ml/processing/train")
    parser.add_argument("--validation-dir", type=str, default="/opt/ml/processing/validation")
    parser.add_argument("--test-dir", type=str, default="/opt/ml/processing/test")
    parser.add_argument("--baseline-dir", type=str, default="/opt/ml/processing/baseline")
    parser.add_argument("--report-dir", type=str, default="/opt/ml/processing/report")

    # Feature engineering
    parser.add_argument("--lead-hours", type=int, default=1)
    parser.add_argument("--lags", type=str, default="1,2,3,6")
    parser.add_argument("--include-current-features", action="store_true")

    # Split
    parser.add_argument("--splits", type=str, default="0.8,0.1,0.1")

    args = parser.parse_args()

    lags = [int(x.strip()) for x in args.lags.split(",") if x.strip()]
    splits = tuple(float(x.strip()) for x in args.splits.split(","))  # type: ignore

    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.validation_dir, exist_ok=True)
    os.makedirs(args.test_dir, exist_ok=True)
    os.makedirs(args.baseline_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    # Locate all CSV files under the input directory.
    csv_paths = glob.glob(os.path.join(args.input_dir, "**", "*.csv"), recursive=True)
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found under input-dir={args.input_dir}. "
            "Expected something like .../buoy=XXXXX/stdmet.csv"
        )

    dfs = []
    buoy_ids: list[str] = []
    for p in sorted(csv_paths):
        df = pd.read_csv(p)

        # Attempt to infer buoy id from the folder name: buoy=XXXXX
        match = None
        for part in os.path.normpath(p).split(os.sep):
            if part.startswith("buoy="):
                match = part.split("=", 1)[1]
                break
        if "station_id" not in df.columns and match is not None:
            df["station_id"] = str(match)

        if "timestamp" not in df.columns:
            raise KeyError(f"Expected 'timestamp' column in {p}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])

        buoy_ids.append(str(df["station_id"].iloc[0]) if "station_id" in df.columns and len(df) else (match or "unknown"))
        dfs.append(df)

    raw_all = pd.concat(dfs, ignore_index=True)
    raw_all = raw_all.sort_values(["station_id", "timestamp"])

    # Basic guardrails: require the core columns.
    required_cols = ["station_id", "timestamp", "wave_height", "wind_speed", "dominant_wave_period"]
    missing_required = [c for c in required_cols if c not in raw_all.columns]
    if missing_required:
        raise KeyError(f"Missing required columns in curated input: {missing_required}")

    supervised, feature_cols = make_supervised(
        raw_all,
        entity_col="station_id",
        time_col="timestamp",
        lead_hours=args.lead_hours,
        lags=lags,
        include_current_features=args.include_current_features,
    )

    # Add Feature Store-friendly metadata (optional for training, useful for tracing).
    supervised["event_time"] = (
        pd.to_datetime(supervised["timestamp"], utc=True).astype("int64") / 1e9
    ).astype(float)

    supervised["record_id"] = (
        supervised["station_id"].astype(str) + "_" + supervised["event_time"].astype(int).astype(str)
    )

    train_df, val_df, test_df = time_split(supervised, time_col="timestamp", splits=splits)

    # Training/eval tables: feature columns + target only (keeps training scripts simple).
    train_out = train_df[feature_cols + ["target_E_star"]].copy()
    val_out = val_df[feature_cols + ["target_E_star"]].copy()
    test_out = test_df[feature_cols + ["target_E_star"]].copy()

    train_out.to_csv(os.path.join(args.train_dir, "train.csv"), index=False)
    val_out.to_csv(os.path.join(args.validation_dir, "validation.csv"), index=False)
    test_out.to_csv(os.path.join(args.test_dir, "test.csv"), index=False)

    # Baseline dataset for Model Monitor (features only).
    baseline = train_df[feature_cols].copy()
    baseline.to_csv(os.path.join(args.baseline_dir, "baseline.csv"), index=False)

    # Small sample for endpoint invocation demos.
    sample = baseline.head(10)
    sample.to_csv(os.path.join(args.baseline_dir, "inference_sample.csv"), index=False)

    # Reports
    report = DataReport(
        n_rows_raw=int(len(raw_all)),
        n_rows_supervised=int(len(supervised)),
        n_rows_train=int(len(train_out)),
        n_rows_val=int(len(val_out)),
        n_rows_test=int(len(test_out)),
        n_buoys=int(raw_all["station_id"].nunique()),
        buoy_ids=sorted([str(x) for x in raw_all["station_id"].unique()]),
        missing_fraction_raw=_missing_fraction(raw_all),
        missing_fraction_supervised=_missing_fraction(supervised[feature_cols + ["target_E_star"]]),
    )

    with open(os.path.join(args.report_dir, "data_report.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    with open(os.path.join(args.report_dir, "feature_cols.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_cols": feature_cols}, f, indent=2)

    print("[INFO] Preprocess complete.")
    print(f"[INFO] Raw rows: {len(raw_all)} | Supervised rows: {len(supervised)}")
    print(f"[INFO] Train/Val/Test: {len(train_out)}/{len(val_out)}/{len(test_out)}")
    print(f"[INFO] Features: {len(feature_cols)}")


if __name__ == "__main__":
    main()
