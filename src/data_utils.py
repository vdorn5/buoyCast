# data_utils.py

from ndbc_api import NdbcApi  # adjust import if your NdbcApi is elsewhere
import pandas as pd


def fetch_ndbc_data(buoy_id: str, start_time: str, end_time: str, mode: str = "stdmet") -> pd.DataFrame:
    """
    Fetch NDBC data for a given buoy and time range.

    Args:
        buoy_id (str): Buoy identifier (e.g., "46042").
        start_time (str): Start date in YYYY-MM-DD format.
        end_time (str): End date in YYYY-MM-DD format.
        mode (str, optional): Type of data to fetch. Defaults to "stdmet".

    Returns:
        pd.DataFrame: DataFrame with the requested data.
    """
    ndbc = NdbcApi()
    df = ndbc.get_data(
        buoy_id,
        mode=mode,
        start_time=start_time,
        end_time=end_time,
        as_df=True
    )
    return df

def clean_ndbc_data(raw_df: pd.DataFrame, buoy_id: str) -> pd.DataFrame:
    """
    Clean and feature-engineer NDBC data.

    Steps:
    - Rename columns to human-readable names.
    - Replace sentinel values (99, 999, 9999) with NA.
    - Drop rows with missing wave height.
    - Compute derived features: wind speed in m/s and wave energy.
    - Add buoy ID as a column.

    Args:
        raw_df (pd.DataFrame): Raw NDBC DataFrame.
        buoy_id (str): Buoy identifier.

    Returns:
        pd.DataFrame: Cleaned and feature-engineered DataFrame.
    """
    # Rename columns
    df = raw_df.rename(
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

    # Sentinel value cleanup
    df.replace(
        to_replace=[99, 99.0, 999, 999.0, 9999, 9999.0],
        value=pd.NA,
        inplace=True
    )

    # List of columns to drop
    cols_to_drop = ["visibility", "PTDY", "tide"]
    df.drop(columns=cols_to_drop, inplace=True)

    # Drop rows with missing data
    df.dropna(inplace=True)

    # Derived features
    df["wind_speed_ms"] = df["wind_speed"] * 0.514444
    df["wave_energy"] = (df["wave_height"] ** 2) * df["dominant_wave_period"]

    # Add buoy column
    df["buoy"] = buoy_id

    return df