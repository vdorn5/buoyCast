import pandas as pd
import numpy as np

def make_supervised(
    df,
    entity_col="station_id",
    time_col="timestamp",
    lead_hours=1,
    lags=[1, 2, 3, 6],
):
    df = df.sort_values([entity_col, time_col]).copy()

    # Base physics-safe target
    df["E_star"] = df["wave_height"] ** 2
    df["target_E_star"] = (
        df.groupby(entity_col)["E_star"].shift(-lead_hours)
    )

    feature_cols = []

    base_features = [
        "wave_height",
        "E_star",
        "wind_speed",
        "dominant_wave_period",
    ]

    for lag in lags:
        for col in base_features:
            lag_col = f"{col}_lag_{lag}"
            df[lag_col] = (
                df.groupby(entity_col)[col].shift(lag)
            )
            feature_cols.append(lag_col)

    # Drop rows with incomplete history or target
    df = df.dropna().reset_index(drop=True)

    return df, feature_cols

def time_split(df, time_col="event_time", splits=(0.8, 0.1, 0.1)):
    assert sum(splits) == 1.0

    df = df.sort_values(time_col)
    n = len(df)

    train_end = int(n * splits[0])
    val_end = int(n * (splits[0] + splits[1]))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train, val, test
