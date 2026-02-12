"""Background estimation using sep."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sep


@dataclass
class BackgroundResult:
    """Result of background estimation."""

    background: np.ndarray
    rms: np.ndarray
    global_back: float
    global_rms: float


def estimate_background(data: np.ndarray, **kwargs) -> BackgroundResult:
    """Estimate and return the background for a 2D image.

    Parameters
    ----------
    data : ndarray
        2D image array (float32).
    **kwargs
        Passed through to ``sep.Background``.

    Returns
    -------
    BackgroundResult
    """
    bkg = sep.Background(data, **kwargs)
    return BackgroundResult(
        background=bkg.back(),
        rms=bkg.rms(),
        global_back=bkg.globalback,
        global_rms=bkg.globalrms,
    )
