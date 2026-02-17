import pandas as pd

from src.data_utils import clean_ndbc_data


def test_clean_ndbc_data_renames_and_adds_features_and_station_id():
    raw = pd.DataFrame(
        {
            "WDIR": [180, 190],
            "WSPD": [10.0, 12.0],
            "GST": [15.0, 18.0],
            "WVHT": [2.0, 2.5],
            "DPD": [10.0, 11.0],
            "APD": [9.0, 10.0],
            "MWD": [200, 210],
            "PRES": [1013.0, 1012.0],
            "ATMP": [15.0, 16.0],
            "WTMP": [14.0, 14.5],
            "DEWP": [12.0, 12.5],
        }
    )

    out = clean_ndbc_data(raw, buoy_id="46086")

    # Renamed columns
    assert "wind_speed" in out.columns
    assert "wave_height" in out.columns
    assert "dominant_wave_period" in out.columns

    # Derived
    assert "wind_speed_ms" in out.columns
    assert "wave_energy" in out.columns

    # Station id is present
    assert "station_id" in out.columns
    assert set(out["station_id"].unique()) == {"46086"}
