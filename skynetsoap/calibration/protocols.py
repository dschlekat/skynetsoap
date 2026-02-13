"""Protocol definition for calibration backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord


@runtime_checkable
class CalibrationProtocol(Protocol):
    """Interface that any calibration backend must satisfy."""

    def calibrate_image(
        self,
        image,
        ins_mags: np.ndarray,
        ins_mag_errs: np.ndarray,
        source_coords: SkyCoord,
        filter_band: str,
    ) -> tuple[float, float, np.ndarray]:
        """Compute the photometric zeropoint for a single image.

        Parameters
        ----------
        image : FITSImage
            The image being calibrated.
        ins_mags : ndarray
            Instrumental magnitudes of extracted sources.
        ins_mag_errs : ndarray
            Uncertainties on instrumental magnitudes.
        source_coords : SkyCoord
            Sky coordinates of extracted sources.
        filter_band : str
            Canonical filter band name (e.g. "V", "R").

        Returns
        -------
        zp : float
            Photometric zeropoint.
        zp_err : float
            Zeropoint uncertainty.
        mask : ndarray (bool)
            Which calibration stars were used in the final solution.
        """
        ...

    def get_reference_catalog(
        self,
        center: SkyCoord,
        radius_arcmin: float,
        filter_band: str,
    ) -> pd.DataFrame:
        """Query and return the reference catalog for a field.

        Returns
        -------
        DataFrame with at least RAJ2000, DEJ2000, and magnitude columns.
        """
        ...
