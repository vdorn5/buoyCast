"""Utilities for fetching and cleaning NOAA NDBC buoy data.

This module is written to be *importable* even when optional runtime dependencies
(like `ndbc_api`) are not installed. That makes local development and CI runs
much smoother.

The fetch function is therefore the only place that requires `ndbc_api`.
"""

from __future__ import annotations

import pandas as pd


def fetch_ndbc_data(
    buoy_id: str,
    start_time: str,
    end_time: str,
    mode: str = "stdmet",
) -> pd.DataFrame:
    """Fetch NDBC data for a given buoy and time range.

    Parameters
    ----------
    buoy_id:
        Buoy identifier (e.g., "46042").
    start_time:
        Start date in YYYY-MM-DD format.
    end_time:
        End date in YYYY-MM-DD format.
    mode:
        Type of data to fetch (e.g., "stdmet").

    Returns
    -------
    pd.DataFrame
        DataFrame with the requested data. Returns an empty DataFrame if no
        results are returned.
    """

    # Import here so `import data_utils` doesn't fail in environments that
    # don't have the optional dependency installed (e.g., GitHub Actions CI).
    try:
        from ndbc_api import NdbcApi  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Optional dependency 'ndbc_api' is required for fetch_ndbc_data(). "
            "Install it (or add it to your requirements) to enable live downloads."
        ) from e

    ndbc = NdbcApi()
    df = ndbc.get_data(
        buoy_id,
        mode=mode,
        start_time=start_time,
        end_time=end_time,
        as_df=True,
    )

    # Some clients return `None` for empty queries.
    if df is None:
        return pd.DataFrame()

    return df


def clean_ndbc_data(raw_df: pd.DataFrame, buoy_id: str) -> pd.DataFrame:
    """
    Strict NDBC cleaner for *wave + wind only*.

    Policy (as requested):
      1) Drop non-wave/wind columns (pressure/temp/dewpoint/etc).
      2) If ANY remaining wave/wind value is missing OR equals sentinel values
         (MM, 99, 999, 9999) -> drop the entire row.
      3) Return a dataset with NO NaN/None in the kept wave/wind variables.

    Kept columns (if present):
      - wind_direction (WDIR)
      - wind_speed (WSPD)
      - wind_gust (GST)
      - wave_height (WVHT)
      - dominant_wave_period (DPD)
      - average_wave_period (APD)
      - mean_wave_direction (MWD)

    Always included:
      - station_id
      - timestamp (UTC)

    Notes:
      - This is intentionally strict and can reduce row count significantly.
      - Handles DatetimeIndex / MultiIndex safely without reset_index collisions.
    """
    import numpy as np
    import pandas as pd

    df = raw_df.copy()

    # -----------------------------
    # 1) Rename NDBC columns
    # -----------------------------
    df = df.rename(
        columns={
            "WDIR": "wind_direction",
            "WSPD": "wind_speed",
            "GST": "wind_gust",
            "WVHT": "wave_height",
            "DPD": "dominant_wave_period",
            "APD": "average_wave_period",
            "MWD": "mean_wave_direction",
            # we intentionally do NOT keep these anymore:
            "PRES": "pressure",
            "ATMP": "air_temperature",
            "WTMP": "water_temperature",
            "DEWP": "dewpoint_temperature",
            "VIS": "visibility",
            "TIDE": "tide",
        }
    )

    # -----------------------------
    # 2) Build timestamp safely (no reset_index collisions)
    # -----------------------------
    if "timestamp" in df.columns:
        ts = df["timestamp"]
    elif isinstance(df.index, pd.DatetimeIndex):
        ts = df.index
    elif isinstance(df.index, pd.MultiIndex):
        # prefer datetime-typed level
        ts = None
        for i in range(df.index.nlevels):
            lv = df.index.get_level_values(i)
            try:
                if np.issubdtype(lv.dtype, np.datetime64):
                    ts = lv
                    break
            except Exception:
                pass
        # fallback by name
        if ts is None:
            for name in ["datetime", "date_time", "time", "date", "timestamp"]:
                if name in df.index.names:
                    ts = df.index.get_level_values(name)
                    break
        if ts is None:
            ts = df.index
    else:
        ts = df.index

    df["timestamp"] = pd.to_datetime(ts, errors="coerce", utc=True)

    # -----------------------------
    # 3) Ensure station_id exists
    # -----------------------------
    if "station_id" not in df.columns:
        if isinstance(df.index, pd.MultiIndex) and "station_id" in df.index.names:
            df["station_id"] = df.index.get_level_values("station_id").astype(str)
        else:
            df["station_id"] = str(buoy_id)
    else:
        df["station_id"] = df["station_id"].astype(str).fillna(str(buoy_id))

    # Flatten index WITHOUT inserting index levels as columns
    df = df.reset_index(drop=True)

    # -----------------------------
    # 4) Keep ONLY wave + wind variables
    # -----------------------------
    wave_wind_cols = [
        "wind_direction",
        "wind_speed",
        "wind_gust",
        "wave_height",
        "dominant_wave_period",
        "average_wave_period",
        "mean_wave_direction",
    ]

    keep_cols = ["timestamp", "station_id"] + [c for c in wave_wind_cols if c in df.columns]
    df = df[keep_cols].copy()

    # -----------------------------
    # 5) Strict invalid handling (MM, 99, 999, 9999) -> drop row
    # -----------------------------
    # IMPORTANT nuance:
    # In NDBC, 99 can be a real wind direction (99 degrees).
    # Missing directions are typically 999.
    # We'll treat 99 as missing for *non-direction* fields, and 999 for directions.
    # If you truly want "99 anywhere kills the row", tell me and I'll flip one line.

    direction_cols = {"wind_direction", "mean_wave_direction"}

    sentinels_common = {99, 99.0, 999, 999.0, 9999, 9999.0}
    sentinels_direction = {999, 999.0, 9999, 9999.0}

    for c in [col for col in wave_wind_cols if col in df.columns]:
        # normalize objects -> replace MM/None-ish -> numeric
        if df[c].dtype == object:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .replace({"MM": np.nan, "": np.nan, "None": np.nan, "nan": np.nan, "NaN": np.nan})
            )

        df[c] = pd.to_numeric(df[c], errors="coerce")

        # apply sentinel rules
        if c in direction_cols:
            df.loc[df[c].isin(sentinels_direction), c] = np.nan
        else:
            df.loc[df[c].isin(sentinels_common), c] = np.nan

    # -----------------------------
    # 6) Drop rows missing even ONE remaining wave/wind variable
    # -----------------------------
    required = [c for c in wave_wind_cols if c in df.columns]
    df = df.dropna(subset=["timestamp"] + required)

    # Optional: basic plausibility filters (keeps junk out)
    if "wind_speed" in df.columns:
        df = df[df["wind_speed"].between(0, 200)]
    if "wind_gust" in df.columns:
        df = df[df["wind_gust"].between(0, 200)]
    if "wave_height" in df.columns:
        df = df[df["wave_height"].between(0, 30)]
    if "dominant_wave_period" in df.columns:
        df = df[df["dominant_wave_period"].between(0, 30)]
    if "average_wave_period" in df.columns:
        df = df[df["average_wave_period"].between(0, 30)]
    if "wind_direction" in df.columns:
        df = df[df["wind_direction"].between(0, 360)]
    if "mean_wave_direction" in df.columns:
        df = df[df["mean_wave_direction"].between(0, 360)]

    # sort + de-dupe
    df = df.sort_values(["station_id", "timestamp"]).drop_duplicates(
        subset=["station_id", "timestamp"], keep="last"
    ).reset_index(drop=True)

    return df
