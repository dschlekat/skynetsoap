"""WCS validation and FOV checking utilities."""

from __future__ import annotations

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from ..core.coordinates import coord_in_image_fov


def validate_wcs(header) -> bool:
    """Check whether a FITS header contains a valid WCS.

    Looks for CTYPE1 and CTYPE2 keywords.
    """
    return header.get("CTYPE1") is not None and header.get("CTYPE2") is not None


def target_in_fov(
    coord: SkyCoord,
    wcs: WCS,
    shape: tuple[int, int],
    margin: float = 0.0,
) -> bool:
    """Check whether a sky coordinate falls within the image FOV.

    Parameters
    ----------
    coord : SkyCoord
    wcs : WCS
    shape : (ny, nx)
    margin : float
        Fractional margin (0-1) shrinking the accepted region.
    """
    return coord_in_image_fov(coord, wcs, shape, margin)
