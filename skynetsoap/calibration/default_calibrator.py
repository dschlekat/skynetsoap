"""Default calibration implementation using Vizier catalogs + zeropoint."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
import astropy.units as u
from astropy.coordinates import SkyCoord

from ..config.loader import SOAPConfig
from .catalog import ReferenceCatalog
from .zeropoint import compute_zeropoint

logger = logging.getLogger("soap")


class DefaultCalibrator:
    """Default ``CalibrationProtocol`` implementation.

    Wires together ReferenceCatalog queries, source-to-catalog matching,
    and inverse-variance weighted zeropoint computation.
    """

    def __init__(self, config: SOAPConfig):
        self.config = config
        self.match_radius_arcsec = config.calibration_match_radius_arcsec
        self.sigma_clip = config.calibration_sigma_clip
        self.max_iter = config.calibration_max_iter
        self.default_cat_error = config.calibration_default_cat_error
        self.ref_catalog = ReferenceCatalog(
            catalogs_config=config.catalogs,
            filters_config=config.filters,
            default_error=config.calibration_default_cat_error,
            merge_tolerance_arcsec=config.calibration_merge_tolerance_arcsec,
        )
        self._cached_ref: pl.DataFrame | None = None
        self._cached_center: SkyCoord | None = None

    def get_reference_catalog(
        self,
        center: SkyCoord,
        radius_arcmin: float = 10.0,
        filter_band: str = "V",
    ) -> pl.DataFrame:
        """Query and return the reference catalog for a field."""
        canonical_band = self._canonicalize_filter_band(filter_band)
        self._cached_ref = self.ref_catalog.query(center, radius_arcmin, canonical_band)
        self._cached_center = center
        return self._cached_ref

    def calibrate_image(
        self,
        image,
        ins_mags: np.ndarray,
        ins_mag_errs: np.ndarray,
        source_coords: SkyCoord,
        filter_band: str,
    ) -> tuple[float, float, np.ndarray]:
        """Compute the photometric zeropoint for a single image.

        Matches extracted sources to the reference catalog within
        ``match_radius_arcsec``, then computes an inverse-variance
        weighted zeropoint.

        Returns
        -------
        zp, zp_err, match_mask
        """
        canonical_band = self._canonicalize_filter_band(filter_band)

        # Ensure we have a reference catalog
        if self._cached_ref is None or self._cached_ref.is_empty():
            center = image.wcs.pixel_to_world(image.shape[1] / 2, image.shape[0] / 2)
            self.get_reference_catalog(center, filter_band=canonical_band)

        ref = self._cached_ref
        if ref is None or ref.is_empty():
            logger.warning("No reference catalog available for calibration.")
            return np.nan, np.nan, np.zeros(len(ins_mags), dtype=bool)

        mag_col = f"{canonical_band}mag"
        err_col = f"e_{canonical_band}mag"

        if mag_col not in ref.columns:
            logger.warning(
                "Filter band %s (canonical: %s) not found in reference catalog.",
                filter_band,
                canonical_band,
            )
            return np.nan, np.nan, np.zeros(len(ins_mags), dtype=bool)

        # Build reference coordinates
        ref_coords = SkyCoord(
            ra=ref["RAJ2000"].to_numpy() * u.deg,
            dec=ref["DEJ2000"].to_numpy() * u.deg,
        )

        # Match extracted sources to catalog
        idx, d2d, _ = source_coords.match_to_catalog_sky(ref_coords)
        match_mask = d2d.arcsec < self.match_radius_arcsec

        if not np.any(match_mask):
            logger.warning(
                "No sources matched within %.1f arcsec.", self.match_radius_arcsec
            )
            return np.nan, np.nan, match_mask

        # Select matched reference stars using polars row selection
        matched_indices = idx[match_mask].tolist()
        matched_refs = ref[matched_indices]
        matched_ins_mags = ins_mags[match_mask]
        matched_ins_mag_errs = ins_mag_errs[match_mask]

        # Get catalog magnitudes and errors
        cat_mags = matched_refs[mag_col].to_numpy()
        if err_col in matched_refs.columns:
            cat_errs = matched_refs[err_col].to_numpy()
            cat_errs = np.where(np.isnan(cat_errs), self.default_cat_error, cat_errs)
        else:
            cat_errs = np.full(len(cat_mags), self.default_cat_error)

        # Filter out NaN values
        valid = ~np.isnan(cat_mags) & ~np.isnan(matched_ins_mags)
        if not np.any(valid):
            return np.nan, np.nan, match_mask

        zp, zp_err, _ = compute_zeropoint(
            m_inst=matched_ins_mags[valid],
            m_cat=cat_mags[valid],
            sigma_inst=matched_ins_mag_errs[valid],
            sigma_cat=cat_errs[valid],
            sigma_clip=self.sigma_clip,
            max_iter=self.max_iter,
        )

        return zp, zp_err, match_mask

    def _canonicalize_filter_band(self, filter_band: str) -> str:
        """Normalize input filter names using configured aliases."""
        aliases = self.config.filters.get("aliases", {})
        return aliases.get(filter_band, filter_band)
