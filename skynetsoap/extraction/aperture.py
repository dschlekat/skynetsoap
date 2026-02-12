"""Aperture photometry and optimal aperture selection using sep."""

from __future__ import annotations

import numpy as np
import sep


def sum_circle(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    r: float,
    err: float | np.ndarray | None = None,
    gain: float = 1.0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wrapper around ``sep.sum_circle``.

    Returns
    -------
    flux, flux_err, flag : ndarray
    """
    return sep.sum_circle(data, x, y, r, err=err, gain=gain, **kwargs)


def compute_optimal_aperture(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    err: float | np.ndarray,
    gain: float = 1.0,
    min_r: float = 3.0,
    max_r: float = 20.0,
    step: float = 0.5,
) -> float:
    """Find the aperture radius that maximises median SNR.

    Parameters
    ----------
    data : ndarray
        Background-subtracted image.
    x, y : ndarray
        Source positions.
    err, gain : float
        Error and gain for ``sep.sum_circle``.
    min_r, max_r, step : float
        Range and step size for the search.

    Returns
    -------
    float
        Optimal aperture radius in pixels.
    """
    radii = np.arange(min_r, max_r + step, step)
    best_r = min_r
    best_snr = -np.inf

    for r in radii:
        flux, flux_err, _ = sep.sum_circle(data, x, y, r, err=err, gain=gain)
        valid = flux_err > 0
        if not np.any(valid):
            continue
        snr = flux[valid] / flux_err[valid]
        median_snr = float(np.median(snr))
        if median_snr > best_snr:
            best_snr = median_snr
            best_r = float(r)

    return best_r


def fwhm_scaled_radius(
    fwhm: float,
    scale: float = 2.5,
    min_r: float = 3.0,
    max_r: float = 20.0,
) -> float:
    """Compute an aperture radius from the FWHM, clamped to [min_r, max_r].

    Parameters
    ----------
    fwhm : float
        Measured FWHM in pixels.
    scale : float
        Multiplier (aperture = scale * fwhm / 2).
    min_r, max_r : float
        Bounds.

    Returns
    -------
    float
    """
    if np.isnan(fwhm) or fwhm <= 0:
        return min_r
    r = scale * fwhm / 2.0
    return float(np.clip(r, min_r, max_r))
