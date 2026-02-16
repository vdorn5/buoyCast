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
    """Clean and feature-engineer NDBC data.

    Steps
    -----
    - Rename columns to human-readable snake_case.
    - Replace sentinel values (99, 999, 9999) with NA.
    - Drop columns we don't use (if present).
    - Drop rows missing core fields.
    - Compute derived features:
        * wind_speed_ms (knots -> m/s)
        * wave_energy (simple proxy: Hs^2 * Tp)
    - Ensure station_id exists.
    - Reset index and expose timestamp.

    Parameters
    ----------
    raw_df:
        Raw NDBC DataFrame.
    buoy_id:
        Buoy identifier.

    Returns
    -------
    pd.DataFrame
        Cleaned and feature-engineered DataFrame.
    """

    df = raw_df.copy()

    # Rename columns
    df = df.rename(
        columns={
            "WDIR": "wind_direction",
            "WSPD": "wind_speed",
            "GST": "wind_gust",
            "WVHT": "wave_height",
            "DPD": "dominant_wave_period",
            "APD": "average_wave_period",
            "MWD": "mean_wave_direction",
            "PRES": "pressure",
            "ATMP": "air_temperature",
            "WTMP": "water_temperature",
            "DEWP": "dewpoint_temperature",
            "VIS": "visibility",
            "TIDE": "tide",
        }
    )

    # Sentinel values to NA (applies across all columns)
    df.replace(
        to_replace=[99, 99.0, 999, 999.0, 9999, 9999.0],
        value=pd.NA,
        inplace=True,
    )

    # Drop columns that may or may not exist (keep it resilient)
    df.drop(columns=["visibility", "PTDY", "tide"], inplace=True, errors="ignore")

    # Ensure the entity identifier exists (used downstream by make_supervised)
    if "station_id" not in df.columns:
        df["station_id"] = str(buoy_id)

    # Drop rows missing core fields needed for the wave-energy features.
    required = ["wave_height", "wind_speed", "dominant_wave_period"]
    required_present = [c for c in required if c in df.columns]
    if required_present:
        df = df.dropna(subset=required_present)

    # Derived features
    if "wind_speed" in df.columns:
        # NDBC wind speed is typically reported in knots.
        df["wind_speed_ms"] = df["wind_speed"] * 0.514444

    if "wave_height" in df.columns and "dominant_wave_period" in df.columns:
        # Simple energy proxy used throughout this project.
        df["wave_energy"] = (df["wave_height"] ** 2) * df["dominant_wave_period"]

    # Expose timestamp (NDBC typically uses a DateTimeIndex)
    df = df.reset_index().rename(columns={"index": "timestamp"})

    return df
