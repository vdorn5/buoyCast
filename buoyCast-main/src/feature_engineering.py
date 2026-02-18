"""Feature engineering utilities.

The core idea in this project is to predict a *physics-safe* quantity:

- **E\_star = Hs^2** (a simple, non-negative wave-energy proxy)

We then learn to forecast E\_star at some lead time using lagged wind/wave
features.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd


def make_supervised(
    df: pd.DataFrame,
    entity_col: str = "station_id",
    time_col: str = "timestamp",
    lead_hours: int = 1,
    lags: Sequence[int] = (1, 2, 3, 6),
    include_current_features: bool = False,
) -> Tuple[pd.DataFrame, list[str]]:
    """Convert a buoy time series into a supervised learning table.

    Parameters
    ----------
    df:
        Input dataframe.
    entity_col:
        Entity/group column (e.g., buoy/station id).
    time_col:
        Timestamp column.
    lead_hours:
        Forecast horizon in hours (assumes your data is hourly or close enough
        that an integer shift is reasonable).
    lags:
        Lag steps (in rows/hours) to include as features.
    include_current_features:
        If True, include the current (t) values of the base features in the
        returned `feature_cols` list in addition to lagged features.

    Returns
    -------
    (df_supervised, feature_cols)
        df_supervised contains E_star, target_E_star, lagged columns, and any
        original columns from `df`.

        feature_cols is the list of feature column names suitable for modeling.
    """

    if entity_col not in df.columns:
        raise KeyError(f"make_supervised expected entity_col='{entity_col}' in df.columns")
    if time_col not in df.columns:
        raise KeyError(f"make_supervised expected time_col='{time_col}' in df.columns")

    out = df.sort_values([entity_col, time_col]).copy()

    # Base physics-safe state (non-negative)
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

    # Only drop rows that are unusable for modeling (avoid dropping due to
    # unrelated columns).
    required_for_training = feature_cols + ["target_E_star"]
    out = out.dropna(subset=required_for_training).reset_index(drop=True)

    return out, feature_cols


def time_split(
    df: pd.DataFrame,
    time_col: str = "event_time",
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-ordered train/val/test split.

    This is a *simple* splitter that preserves chronological order.

    Parameters
    ----------
    df:
        Input DataFrame.
    time_col:
        Name of the time column. If not present, a small set of common
        alternatives are attempted.
    splits:
        (train, val, test) fractions that must sum to 1.0.

    Returns
    -------
    (train, val, test)
    """

    if not np.isclose(sum(splits), 1.0):
        raise ValueError(f"splits must sum to 1.0, got {splits}")

    if time_col not in df.columns:
        for alt in ("event_time", "timestamp", "time", "datetime"):
            if alt in df.columns:
                time_col = alt
                break
        else:
            raise KeyError(
                f"time_col='{time_col}' not found, and no fallback time column was found in df.columns"
            )

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
