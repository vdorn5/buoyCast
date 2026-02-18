"""Plot helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_ndbc_histograms(df: pd.DataFrame, bins: int = 60) -> None:
    """Visual QC: histograms of key NDBC buoy variables.

    Plots distributions for:
      - wave_height: Significant Wave Height Hs [m]
      - dominant_wave_period: Dominant Wave Period Tp [s]
      - wind_speed: Wind Speed [knots] (or whatever your cleaned units are)
      - mean_wave_direction: Mean Wave Direction [deg]
      - wind_direction: Wind Direction [deg]

    Parameters
    ----------
    df:
        Cleaned NDBC dataframe.
    bins:
        Number of histogram bins.
    """

    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    axs = axs.flatten()

    plots = [
        ("wave_height", "Significant Wave Height Hs [m]"),
        ("dominant_wave_period", "Dominant Wave Period Tp [s]"),
        ("wind_speed", "Wind Speed"),
        ("mean_wave_direction", "Mean Wave Direction [deg]"),
        ("wind_direction", "Wind Direction [deg]"),
    ]

    for ax, (col, label) in zip(axs, plots):
        if col in df.columns:
            ax.hist(df[col].dropna(), bins=bins, density=True, alpha=0.75)
            ax.set_xlabel(label)
            ax.set_ylabel("PDF")
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, f"{col} not in DataFrame", ha="center", va="center")
            ax.axis("off")

    # Turn off any unused subplots (e.g., the 6th panel).
    for ax in axs[len(plots) :]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()
