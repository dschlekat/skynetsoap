"""Filter transformation functions (Jordi+2006).

Generalized to read coefficients from config, but also usable standalone.
"""

from __future__ import annotations

import numpy as np


def apply_transform(
    mag1: float | np.ndarray,
    mag2: float | np.ndarray,
    err1: float | np.ndarray,
    err2: float | np.ndarray,
    a: float,
    sigma_a: float,
    b: float,
    sigma_b: float,
    formula: str = "color_plus_offset",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a photometric transformation with error propagation.

    Parameters
    ----------
    mag1, mag2 : float or ndarray
        Input magnitudes (source bands).
    err1, err2 : float or ndarray
        Uncertainties on input magnitudes.
    a, sigma_a : float
        Slope coefficient and its uncertainty.
    b, sigma_b : float
        Offset coefficient and its uncertainty.
    formula : str
        ``"color_plus_offset"`` : target = mag1 + a*(mag1 - mag2) + b
        ``"color_only"``        : target_color = a*(mag1 - mag2) + b

    Returns
    -------
    result, result_err : ndarray
        Transformed magnitude (or color) and its uncertainty.
    """
    mag1 = np.asarray(mag1, dtype=float)
    mag2 = np.asarray(mag2, dtype=float)
    err1 = np.asarray(err1, dtype=float)
    err2 = np.asarray(err2, dtype=float)

    C = mag1 - mag2
    sigma_C2 = err1 ** 2 + err2 ** 2

    if formula == "color_plus_offset":
        result = mag1 + a * C + b
        result_err = np.sqrt(
            err1 ** 2
            + (a ** 2) * sigma_C2
            + (C ** 2) * (sigma_a ** 2)
            + sigma_b ** 2
        )
    elif formula == "color_only":
        result = a * C + b
        result_err = np.sqrt(
            (a ** 2) * sigma_C2
            + (C ** 2) * (sigma_a ** 2)
            + sigma_b ** 2
        )
    else:
        raise ValueError(f"Unknown formula type: {formula!r}")

    return result, result_err
