"""Source extraction using sep."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sep
from astropy.stats import sigma_clipped_stats


@dataclass
class ExtractionResult:
    """Result of source extraction."""

    objects: np.ndarray
    n_sources: int
    fwhm: float


def compute_fwhm(objects: np.ndarray) -> float:
    """Compute the sigma-clipped median FWHM from sep ellipse parameters.

    Uses ``fwhm = 2 * sqrt(a * b)`` where a, b are the semi-major/minor
    axes from sep.extract.
    """
    if len(objects) == 0:
        return np.nan
    raw_fwhm = 2.0 * np.sqrt(objects["a"] * objects["b"])
    _, median, _ = sigma_clipped_stats(raw_fwhm, sigma=3.0)
    return float(median)


def extract_sources(
    data: np.ndarray,
    err: float | np.ndarray,
    threshold: float = 1.5,
    min_area: int = 5,
    **kwargs,
) -> ExtractionResult:
    """Extract sources from background-subtracted data.

    Parameters
    ----------
    data : ndarray
        Background-subtracted 2D image.
    err : float or ndarray
        Global RMS or per-pixel error array.
    threshold : float
        Detection threshold in units of *err*.
    min_area : int
        Minimum number of connected pixels.
    **kwargs
        Passed through to ``sep.extract``.

    Returns
    -------
    ExtractionResult
    """
    objects = sep.extract(data, threshold, err=err, minarea=min_area, **kwargs)
    fwhm = compute_fwhm(objects)
    return ExtractionResult(objects=objects, n_sources=len(objects), fwhm=fwhm)
