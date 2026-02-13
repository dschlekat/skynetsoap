"""CCD error model for calibrated magnitude uncertainties.

Ported from tic_photometry.ipynb.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Custom exception hierarchy
# ---------------------------------------------------------------------------


class SOAPError(Exception):
    """Base exception for the SOAP pipeline."""


class ExtractionError(SOAPError):
    """Raised when source extraction fails."""


class CalibrationError(SOAPError):
    """Raised when photometric calibration fails."""


class AstrometryError(SOAPError):
    """Raised when astrometric solving fails."""


class ConfigError(SOAPError):
    """Raised for configuration validation errors."""


class ImageError(SOAPError):
    """Raised for issues loading or processing FITS images."""


# ---------------------------------------------------------------------------
# Error models
# ---------------------------------------------------------------------------


def ccd_magnitude_error(
    flux: float | np.ndarray,
    gain: float,
    n_pix: float,
    background: float,
    rdnoise: float,
    n_bkgpix: float,
    sigma_bkg: float,
    sigma_zp: float,
) -> float | np.ndarray:
    """Compute total error on a calibrated stellar magnitude.

    Parameters
    ----------
    flux : float or array
        Net stellar flux in the aperture (ADU).
    gain : float
        CCD gain (e-/ADU).
    n_pix : float
        Number of pixels in the photometric aperture.
    background : float
        Mean background level per pixel (ADU).
    rdnoise : float
        CCD read noise per pixel (e-).
    n_bkgpix : float
        Number of pixels in the background region.
    sigma_bkg : float
        Standard deviation of background per pixel (ADU).
    sigma_zp : float
        Zeropoint uncertainty (mag).

    Returns
    -------
    sigma_m : float or array
        Total 1-sigma calibrated magnitude error (mag).
    """
    var_flux = (
        flux / gain
        + n_pix * (rdnoise**2 + background / gain)
        + (n_pix**2) * (sigma_bkg**2) / n_bkgpix
    )
    sigma_flux = np.sqrt(var_flux)
    sigma_inst = (2.5 / np.log(10)) * (sigma_flux / flux)
    sigma_m = np.sqrt(sigma_inst**2 + sigma_zp**2)
    return sigma_m


def compute_limiting_magnitude(
    zeropoint: float,
    background_rms: float,
    aperture_radius: float,
    n_sigma: float = 5.0,
) -> float:
    """Compute the limiting magnitude for an image.

    The limiting magnitude is the faintest magnitude detectable at a given
    confidence level (typically 5-sigma) based on background noise.

    Formula: m_lim = ZP - 2.5 * log10(n_sigma * sigma_sky * sqrt(n_pix))

    Parameters
    ----------
    zeropoint : float
        Photometric zeropoint (mag).
    background_rms : float
        Background RMS noise (counts/pixel).
    aperture_radius : float
        Aperture radius in pixels.
    n_sigma : float
        Detection significance threshold (default: 5.0).

    Returns
    -------
    float
        Limiting magnitude (mag).
    """
    if np.isnan(zeropoint) or zeropoint == 0:
        return np.nan

    # Aperture area in pixels
    n_pix = np.pi * aperture_radius**2

    # Minimum detectable flux (in sky background units)
    min_flux = n_sigma * background_rms * np.sqrt(n_pix)

    # Convert to magnitude
    if min_flux <= 0:
        return np.nan

    limiting_mag = zeropoint - 2.5 * np.log10(min_flux)
    return float(limiting_mag)
