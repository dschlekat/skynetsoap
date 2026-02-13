"""Lightcurve plotting and debugging visualization.

Migrated from models/plotter.py with extended filter colors.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from astropy.coordinates import SkyCoord

if TYPE_CHECKING:
    from ..core.image import FITSImage
    from ..core.result import PhotometryResult
    from ..extraction.background import BackgroundResult

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


def create_debug_plot(
    image: FITSImage,
    bkg_result: BackgroundResult,
    source_coords: np.ndarray | None = None,
    aperture_radius: float | None = None,
    ref_coords: SkyCoord | None = None,
    matched_mask: np.ndarray | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Create multi-panel debugging plot for a single image.

    Parameters
    ----------
    image : FITSImage
        The original FITS image.
    bkg_result : BackgroundResult
        Background estimation result.
    source_coords : ndarray, optional
        Array of (x, y) pixel coordinates for extracted sources.
    aperture_radius : float, optional
        Aperture radius in pixels.
    ref_coords : SkyCoord, optional
        Reference catalog coordinates (matched to sources).
    matched_mask : ndarray, optional
        Boolean mask indicating which sources matched reference stars.
    save_path : Path, optional
        If provided, save the plot to this path.

    Returns
    -------
    Figure
        The matplotlib figure object.
    """
    # Use WCS projection if available
    if image.has_wcs:
        fig = plt.figure(figsize=(14, 11))

        # Create subplots with WCS projection where appropriate
        ax1 = fig.add_subplot(2, 2, 1, projection=image.wcs)
        ax2 = fig.add_subplot(2, 2, 2, projection=image.wcs)
        ax3 = fig.add_subplot(2, 2, 3, projection=image.wcs)
        ax4 = fig.add_subplot(2, 2, 4, projection=image.wcs)
        axes = np.array([[ax1, ax2], [ax3, ax4]])

        # Configure WCS axes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.coords[0].set_axislabel("RA")
            ax.coords[1].set_axislabel("Dec")
            ax.coords.grid(color="white", alpha=0.3, linestyle="solid", linewidth=0.5)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    fig.suptitle(f"Debug: {image.path.name}", fontsize=14, fontweight="bold")

    # Panel 1: Original image
    ax1 = axes[0, 0]
    vmin, vmax = np.percentile(image.data[~np.isnan(image.data)], [1, 99])
    im1 = ax1.imshow(image.data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax1.set_title("Original Image")
    if not image.has_wcs:
        ax1.set_xlabel("X (pixels)")
        ax1.set_ylabel("Y (pixels)")
    plt.colorbar(im1, ax=ax1, label="Counts")

    # Panel 2: Background map
    ax2 = axes[0, 1]
    im2 = ax2.imshow(bkg_result.background, origin="lower", cmap="viridis")
    ax2.set_title(f"Background (RMS={bkg_result.global_rms:.2f})")
    if not image.has_wcs:
        ax2.set_xlabel("X (pixels)")
        ax2.set_ylabel("Y (pixels)")
    plt.colorbar(im2, ax=ax2, label="Counts")

    # Panel 3: Background-subtracted image with sources
    ax3 = axes[1, 0]
    data_sub = image.data - bkg_result.background
    vmin_sub, vmax_sub = np.percentile(data_sub[~np.isnan(data_sub)], [1, 99])
    im3 = ax3.imshow(
        data_sub, origin="lower", cmap="gray", vmin=vmin_sub, vmax=vmax_sub
    )
    ax3.set_title("Background-Subtracted + Sources")
    if not image.has_wcs:
        ax3.set_xlabel("X (pixels)")
        ax3.set_ylabel("Y (pixels)")
    plt.colorbar(im3, ax=ax3, label="Counts")

    # Overlay extracted sources
    if source_coords is not None:
        x, y = source_coords[:, 0], source_coords[:, 1]
        ax3.scatter(
            x,
            y,
            s=50,
            facecolors="none",
            edgecolors="red",
            linewidths=1,
            label="Extracted",
            transform=ax3.get_transform("pixel") if image.has_wcs else ax3.transData,
        )

        # Draw apertures if radius provided
        if aperture_radius is not None:
            for xi, yi in zip(x, y):
                circle = Circle(
                    (xi, yi),
                    aperture_radius,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1,
                    alpha=0.7,
                    transform=ax3.get_transform("pixel")
                    if image.has_wcs
                    else ax3.transData,
                )
                ax3.add_patch(circle)

        ax3.legend(loc="upper right")

    # Panel 4: Background-subtracted with apertures highlighted
    ax4 = axes[1, 1]
    im4 = ax4.imshow(
        data_sub, origin="lower", cmap="gray", vmin=vmin_sub, vmax=vmax_sub
    )
    ax4.set_title("Aperture Photometry")
    if not image.has_wcs:
        ax4.set_xlabel("X (pixels)")
        ax4.set_ylabel("Y (pixels)")
    plt.colorbar(im4, ax=ax4, label="Counts")

    # Overlay all sources with apertures
    if source_coords is not None:
        x, y = source_coords[:, 0], source_coords[:, 1]

        # Plot source positions
        ax4.scatter(
            x,
            y,
            s=30,
            facecolors="none",
            edgecolors="lime",
            linewidths=1.5,
            label="Sources",
            transform=ax4.get_transform("pixel") if image.has_wcs else ax4.transData,
        )

        # Draw apertures if radius provided
        if aperture_radius is not None:
            for xi, yi in zip(x, y):
                circle = Circle(
                    (xi, yi),
                    aperture_radius,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.5,
                    alpha=0.8,
                    transform=ax4.get_transform("pixel")
                    if image.has_wcs
                    else ax4.transData,
                )
                ax4.add_patch(circle)

            # Add aperture info text in corner
            aperture_text = f"Aperture: {aperture_radius:.2f} px"
            ax4.text(
                0.02,
                0.98,
                aperture_text,
                transform=ax4.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round", facecolor="black", alpha=0.7, edgecolor="cyan"
                ),
                color="cyan",
                fontweight="bold",
            )

        ax4.legend(loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
