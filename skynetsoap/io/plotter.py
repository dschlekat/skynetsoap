"""Lightcurve plotting.

Migrated from models/plotter.py with extended filter colors.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from ..core.result import PhotometryResult

FILTER_COLORS = {
    "Open": "k",
    # SDSS
    "uprime": "tab:blue",
    "gprime": "tab:green",
    "rprime": "tab:red",
    "iprime": "tab:orange",
    "zprime": "tab:purple",
    "u": "tab:blue",
    "g": "tab:green",
    "r": "tab:red",
    "i": "tab:orange",
    "z": "tab:purple",
    # Johnson-Cousins
    "U": "darkviolet",
    "B": "blue",
    "V": "green",
    "R": "red",
    "I": "darkred",
}

TITLE = {
    "flux": "Flux vs Time",
    "calibrated_mag": "Calibrated Magnitude vs Time",
    "ins_mag": "Instrumental Magnitude vs Time",
}

YLABEL = {
    "flux": "Flux (counts)",
    "calibrated_mag": "Magnitude (mag)",
    "ins_mag": "Instrumental Magnitude (mag)",
}


def plot_lightcurve(
    result: PhotometryResult,
    units: str = "calibrated_mag",
    path: str | Path | None = None,
    show: bool = False,
    **kwargs,
) -> plt.Figure:
    """Plot a lightcurve from a PhotometryResult.

    Parameters
    ----------
    result : PhotometryResult
    units : str
        Column to plot on the y-axis ("flux", "calibrated_mag", "ins_mag").
    path : str or Path, optional
        If provided, save the figure to this path.
    show : bool
        If True, call plt.show().
    **kwargs
        Passed to ``ax.errorbar``.

    Returns
    -------
    Figure
    """
    table = result.table
    err_col = units + "_err" if units != "flux" else "flux_err"

    fig, ax = plt.subplots()
    filters = np.unique(table["filter"])

    for filt in filters:
        mask = table["filter"] == filt
        subset = table[mask]
        color = FILTER_COLORS.get(filt, "gray")
        ax.errorbar(
            np.array(subset["mjd"]),
            np.array(subset[units]),
            yerr=np.array(subset[err_col]) if err_col in subset.colnames else None,
            fmt="o",
            color=color,
            label=filt,
            markersize=4,
            **kwargs,
        )

    ax.set_xlabel("MJD")
    ax.set_ylabel(YLABEL.get(units, units))
    ax.set_title(TITLE.get(units, f"{units} vs Time"))
    ax.legend()

    if "mag" in units:
        ax.invert_yaxis()

    fig.tight_layout()

    if path is not None:
        fig.savefig(str(path), dpi=150)

    if show:
        plt.show()

    return fig
