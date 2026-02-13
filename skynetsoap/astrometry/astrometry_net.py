"""Astrometry.net solver via astroquery."""

from __future__ import annotations

import logging
import os

from astropy.wcs import WCS

from .wcs_utils import validate_wcs

logger = logging.getLogger("soap")


class AstrometryNetSolver:
    """``AstrometryProtocol`` implementation using astrometry.net.

    Requires ``astroquery`` and an API key set via the
    ``ASTROMETRY_NET_API_KEY`` environment variable or passed directly.
    """

    def __init__(self, api_key: str | None = None, timeout: int = 120):
        self.api_key = api_key or os.environ.get("ASTROMETRY_NET_API_KEY")
        self.timeout = timeout
        self._solver = None

    def _get_solver(self):
        if self._solver is None:
            from astroquery.astrometry_net import AstrometryNet
            solver = AstrometryNet()
            if self.api_key:
                solver.api_key = self.api_key
            self._solver = solver
        return self._solver

    def is_solved(self, image) -> bool:
        """Check whether the image already has a valid WCS."""
        return validate_wcs(image.header)

    def solve(self, image) -> WCS | None:
        """Attempt to solve astrometry for *image*.

        Returns
        -------
        WCS or None
        """
        if self.is_solved(image):
            logger.info("Image already has WCS: %s", image.path.name)
            return image.wcs

        try:
            solver = self._get_solver()
            logger.info("Submitting %s to astrometry.net...", image.path.name)
            wcs_header = solver.solve_from_image(
                str(image.path),
                solve_timeout=self.timeout,
                force_image_upload=True,
            )
            if wcs_header:
                logger.info("Solved: %s", image.path.name)
                return WCS(wcs_header)
            else:
                logger.warning("Failed to solve: %s", image.path.name)
                return None
        except Exception as e:
            logger.warning("Astrometry solving error for %s: %s", image.path.name, e)
            return None
