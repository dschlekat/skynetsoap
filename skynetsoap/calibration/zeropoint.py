"""Inverse-variance weighted zeropoint computation.

Ported from tic_photometry.ipynb.
"""

from __future__ import annotations

import numpy as np


def compute_zeropoint(
    m_inst: np.ndarray,
    m_cat: np.ndarray,
    sigma_inst: np.ndarray,
    sigma_cat: np.ndarray,
    sigma_clip: float = 3.0,
    max_iter: int = 5,
) -> tuple[float, float, np.ndarray]:
    """Compute a photometric zeropoint with inverse-variance weighting and sigma clipping.

    Parameters
    ----------
    m_inst : array
        Instrumental magnitudes of calibration stars.
    m_cat : array
        Catalog magnitudes in the target system.
    sigma_inst : array
        Uncertainties on instrumental magnitudes.
    sigma_cat : array
        Uncertainties on catalog magnitudes.
    sigma_clip : float
        Sigma threshold for outlier rejection.
    max_iter : int
        Maximum clipping iterations.

    Returns
    -------
    ZP : float
        Best-fit zeropoint.
    sigma_ZP : float
        Zeropoint uncertainty.
    mask : ndarray (bool)
        Stars kept in the final solution.
    """
    m_inst = np.asarray(m_inst, dtype=float)
    m_cat = np.asarray(m_cat, dtype=float)
    sigma_inst = np.asarray(sigma_inst, dtype=float)
    sigma_cat = np.asarray(sigma_cat, dtype=float)

    delta = m_cat - m_inst
    sigma = np.sqrt(sigma_inst ** 2 + sigma_cat ** 2)

    mask = np.ones_like(delta, dtype=bool)

    for _ in range(max_iter):
        d = delta[mask]
        s = sigma[mask]

        if len(d) == 0:
            return np.nan, np.nan, mask

        w = 1.0 / (s ** 2)
        ZP = np.sum(w * d) / np.sum(w)
        sigma_ZP = np.sqrt(1.0 / np.sum(w))

        resid = d - ZP
        new_mask = np.abs(resid) < sigma_clip * s

        if np.all(new_mask == mask[mask]):
            break

        mask_indices = np.where(mask)[0]
        mask[mask_indices] = new_mask

    return ZP, sigma_ZP, mask
