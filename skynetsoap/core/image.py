"""FITSImage dataclass for lazy-loaded FITS file handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from .coordinates import coord_in_image_fov

if TYPE_CHECKING:
    from astropy.io.fits import Header


@dataclass
class FITSImage:
    """Lazy-loaded wrapper around a single FITS image."""

    path: Path

    _data: np.ndarray | None = field(default=None, repr=False, init=False)
    _header: Header | None = field(default=None, repr=False, init=False)
    _wcs: WCS | None = field(default=None, repr=False, init=False)

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self._data is None or self._header is None:
            with fits.open(self.path) as hdul:
                self._data = hdul[0].data.astype(np.float32)
                self._header = hdul[0].header

    @property
    def data(self) -> np.ndarray:
        self._load()
        return self._data

    @property
    def header(self) -> Header:
        self._load()
        return self._header

    @property
    def wcs(self) -> WCS:
        if self._wcs is None:
            self._wcs = WCS(self.header)
        return self._wcs

    # ------------------------------------------------------------------
    # Convenience properties from header
    # ------------------------------------------------------------------
    @property
    def filter_name(self) -> str:
        return self.header.get("FILTER", "Unknown")

    @property
    def gain(self) -> float:
        return float(self.header.get("GAIN", 1.0))

    @property
    def rdnoise(self) -> float:
        return float(self.header.get("RDNOISE", 5.0))

    @property
    def exptime(self) -> float:
        return float(self.header.get("EXPTIME", 0.0))

    @property
    def mjd(self) -> float:
        return float(self.wcs.wcs.mjdobs)

    @property
    def mid_jd(self) -> float:
        """Mid-exposure Julian Date."""
        jd = self.header.get("JD")
        if jd is not None:
            return float(jd) + self.exptime / 2.0 / 86400.0
        # Fallback: convert MJD to JD
        return self.mjd + 2400000.5 + self.exptime / 2.0 / 86400.0

    @property
    def telescope(self) -> str:
        return self.header.get("TELESCOP", "Unknown")

    @property
    def has_wcs(self) -> bool:
        return self.header.get("CTYPE1") is not None and self.header.get("CTYPE2") is not None

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------
    def target_pixel_position(self, coord: SkyCoord) -> tuple[float, float]:
        """Convert a sky coordinate to pixel position (x, y)."""
        x, y = self.wcs.all_world2pix(coord.ra.deg, coord.dec.deg, 0)
        return float(x), float(y)

    def coord_in_fov(self, coord: SkyCoord, margin: float = 0.0) -> bool:
        """Check if a sky coordinate falls within this image's FOV."""
        return coord_in_image_fov(coord, self.wcs, self.shape, margin)

    @classmethod
    def load(cls, path: str | Path) -> FITSImage:
        """Create a FITSImage from a file path."""
        return cls(path=Path(path))
