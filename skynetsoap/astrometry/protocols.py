"""Protocol definition for astrometry backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from astropy.wcs import WCS


@runtime_checkable
class AstrometryProtocol(Protocol):
    """Interface that any astrometry solver must satisfy."""

    def solve(self, image) -> WCS | None:
        """Attempt to solve the astrometry for *image*.

        Returns a new WCS on success, or None on failure.
        """
        ...

    def is_solved(self, image) -> bool:
        """Check whether *image* already has a valid WCS."""
        ...
