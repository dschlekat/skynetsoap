"""CCD error model for calibrated magnitude uncertainties.

Ported from tic_photometry.ipynb.
"""

from __future__ import annotations

import numpy as np


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
        + n_pix * (rdnoise ** 2 + background / gain)
        + (n_pix ** 2) * (sigma_bkg ** 2) / n_bkgpix
    )
    sigma_flux = np.sqrt(var_flux)
    sigma_inst = (2.5 / np.log(10)) * (sigma_flux / flux)
    sigma_m = np.sqrt(sigma_inst ** 2 + sigma_zp ** 2)
    return sigma_m
