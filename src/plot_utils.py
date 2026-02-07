# plot_utils.py

import matplotlib.pyplot as plt
import pandas as pd

def plot_ndbc_histograms(df: pd.DataFrame, bins: int = 60) -> None:
    """
    Visual QC check: histograms of NDBC buoy data columns.
    
    Plots distributions for:
      - WVHT: Significant Wave Height [m]
      - DPD: Dominant Wave Period [s]
      - WSPD: Wind Speed [m/s]
      - MWD: Mean Wave Direction [deg]
      - WDIR: Wind Direction [deg]

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned NDBC data with columns: ['WVHT', 'DPD', 'WSPD', 'MWD', 'WDIR'].
    bins : int, default=60
        Number of histogram bins.

    Returns
    -------
    None
    """
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    axs = axs.flatten()

    plots = [
        ("wave_height", "Significant Wave Height Hs [m]"),
        ("dominant_wave_period",  "Dominant Wave Period Tp [s]"),
        ("wind_speed", "Wind Speed [m/s]"),
        ("mean_wave_direction",  "Mean Wave Direction [deg]"),
        ("wind_direction", "Wind Direction [deg]"),
    ]

    for ax, (col, label) in zip(axs, plots):
        if col in df.columns:
            ax.hist(df[col].dropna(), bins=bins, density=True, alpha=0.75)
            ax.set_xlabel(label)
            ax.set_ylabel("PDF")
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, f"{col} not in DataFrame")
