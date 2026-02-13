"""Coordinate parsing and FOV validation utilities."""

from __future__ import annotations

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS


def parse_coordinates(ra, dec) -> SkyCoord:
    """Parse RA/Dec into a SkyCoord.

    Accepts:
    - SkyCoord (returned as-is)
    - Two strings (interpreted as hourangle, deg)
    - Two floats (interpreted as degrees)
    """
    if isinstance(ra, SkyCoord):
        return ra

    if isinstance(ra, str) and isinstance(dec, str):
        return SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg), frame="icrs")

    if isinstance(ra, (int, float)) and isinstance(dec, (int, float)):
        return SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame="icrs")

    raise TypeError(f"Cannot parse coordinates from ra={ra!r}, dec={dec!r}")


def coord_in_image_fov(coord: SkyCoord, wcs: WCS, shape: tuple[int, int], margin: float = 0.0) -> bool:
    """Check whether *coord* falls within the image FOV defined by *wcs* and *shape*.

    Parameters
    ----------
    coord : SkyCoord
    wcs : WCS
    shape : (ny, nx)
    margin : float
        Fractional margin (0-1) shrinking the accepted region inward.
    """
    ny, nx = shape
    x, y = wcs.all_world2pix(coord.ra.deg, coord.dec.deg, 0)
    x, y = float(x), float(y)
    m = margin * min(nx, ny)
    return (m <= x <= nx - m) and (m <= y <= ny - m)
