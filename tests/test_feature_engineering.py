import pandas as pd

from src.feature_engineering import make_supervised, time_split


def test_make_supervised_creates_expected_columns_and_no_missing_in_features():
    # 6 hourly samples -> should yield rows where lags=2 and target exist
    df = pd.DataFrame(
        {
            "station_id": ["A"] * 6,
            "timestamp": pd.date_range("2023-01-01", periods=6, freq="H", tz="UTC"),
            "wave_height": [1.0, 1.1, 1.2, 1.1, 1.3, 1.4],
            "wind_speed": [5, 6, 7, 6, 8, 9],
            "dominant_wave_period": [10, 10, 11, 11, 12, 12],
        }
    )

    out, feature_cols = make_supervised(df, lead_hours=1, lags=(1, 2))

    # Core columns
    assert "E_star" in out.columns
    assert "target_E_star" in out.columns

    # Lag columns exist and are listed as features
    expected_lags = [
        "wave_height_lag_1",
        "E_star_lag_1",
        "wind_speed_lag_1",
        "dominant_wave_period_lag_1",
        "wave_height_lag_2",
        "E_star_lag_2",
        "wind_speed_lag_2",
        "dominant_wave_period_lag_2",
    ]
    for c in expected_lags:
        assert c in out.columns
        assert c in feature_cols

    # No missing values in the features/target subset
    assert out[feature_cols + ["target_E_star"]].isna().sum().sum() == 0


def test_time_split_respects_order_and_sizes():
    df = pd.DataFrame(
        {
            "event_time": pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC"),
            "x": range(10),
        }
    )

    train, val, test = time_split(df, time_col="event_time", splits=(0.6, 0.2, 0.2))

    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2

    # Chronological order
    assert train["event_time"].max() <= val["event_time"].min()
    assert val["event_time"].max() <= test["event_time"].min()
