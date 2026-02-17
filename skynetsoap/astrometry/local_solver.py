"""Local astrometry.net solver via solve-field command."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS

from .wcs_utils import validate_wcs

logger = logging.getLogger("soap")


class LocalAstrometryNetSolver:
    """``AstrometryProtocol`` implementation using local solve-field command.

    Requires a local installation of astrometry.net with the solve-field
    executable available in PATH or at a specified location.

    Parameters
    ----------
    binary_path : str, optional
        Path to solve-field executable. If None, searches PATH.
    timeout : int, default 300
        Timeout in seconds for solve-field execution.
    scale_low : float, optional
        Lower bound for image scale in arcsec/pixel.
    scale_high : float, optional
        Upper bound for image scale in arcsec/pixel.
    depth : int or list of int, optional
        Search depth(s) to pass to solve-field.
    downsample : int, optional
        Downsample factor for faster solving.
    extra_args : list of str, optional
        Additional arguments to pass to solve-field.

    Examples
    --------
    >>> solver = LocalAstrometryNetSolver()
    >>> wcs = solver.solve(image)

    >>> # With scale hints for faster solving
    >>> solver = LocalAstrometryNetSolver(scale_low=0.5, scale_high=2.0)
    >>> wcs = solver.solve(image)
    """

    def __init__(
        self,
        binary_path: str | None = None,
        timeout: int = 300,
        scale_low: float | None = None,
        scale_high: float | None = None,
        depth: int | list[int] | None = None,
        downsample: int | None = None,
        extra_args: list[str] | None = None,
    ):
        self.binary_path = self._find_binary(binary_path)
        self.timeout = timeout
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.depth = depth
        self.downsample = downsample
        self.extra_args = extra_args or []

    @staticmethod
    def is_available() -> bool:
        """Check if solve-field is available on the system.

        Returns
        -------
        bool
            True if solve-field executable can be found in PATH.
        """
        return shutil.which("solve-field") is not None

    def _find_binary(self, path: str | None) -> str:
        """Locate solve-field executable.

        Parameters
        ----------
        path : str, optional
            Path to solve-field executable or command name.

        Returns
        -------
        str
            Path to solve-field executable.

        Raises
        ------
        FileNotFoundError
            If solve-field cannot be found.
        """
        if path is None:
            path = "solve-field"

        # If absolute path, verify it exists and is executable
        if os.path.isabs(path):
            if not os.path.isfile(path):
                msg = f"solve-field not found at {path}"
                raise FileNotFoundError(msg)
            if not os.access(path, os.X_OK):
                msg = f"solve-field at {path} is not executable"
                raise PermissionError(msg)
            return path

        # Otherwise search PATH
        found = shutil.which(path)
        if found is None:
            msg = f"solve-field not found in PATH (searched for '{path}')"
            raise FileNotFoundError(msg)
        return found

    def is_solved(self, image) -> bool:
        """Check whether the image already has a valid WCS.

        Parameters
        ----------
        image : FITSImage
            Image to check.

        Returns
        -------
        bool
            True if image has valid WCS.
        """
        return validate_wcs(image.header)

    def solve(self, image) -> WCS | None:
        """Attempt to solve astrometry for *image* using local solve-field.

        Parameters
        ----------
        image : FITSImage
            Image to solve.

        Returns
        -------
        WCS or None
            WCS object on success, None on failure.
        """
        if self.is_solved(image):
            logger.info("Image already has WCS: %s", image.path.name)
            return image.wcs

        logger.info("Solving %s with local solve-field...", image.path.name)

        try:
            with tempfile.TemporaryDirectory(prefix="soap_astrom_") as tmpdir:
                tmpdir = Path(tmpdir)
                basename = image.path.stem

                # Build command
                cmd = self._build_command(image.path, tmpdir, basename)

                # Execute with timeout
                try:
                    result = subprocess.run(
                        cmd,
                        timeout=self.timeout,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    logger.info(
                        "Local solve timed out after %ds for %s",
                        self.timeout,
                        image.path.name,
                    )
                    return None

                # Check return code
                if result.returncode != 0:
                    logger.debug(
                        "solve-field failed for %s: %s",
                        image.path.name,
                        result.stderr.strip() if result.stderr else "unknown error",
                    )
                    return None

                # Parse WCS from output
                wcs = self._parse_wcs_from_output(tmpdir, basename)
                if wcs is not None:
                    logger.info("Solved %s with local solver", image.path.name)
                return wcs

        except Exception as e:
            logger.warning(
                "Local astrometry solving error for %s: %s", image.path.name, e
            )
            return None

    def _build_command(
        self,
        image_path: Path,
        output_dir: Path,
        basename: str,
    ) -> list[str]:
        """Construct solve-field command line arguments.

        Parameters
        ----------
        image_path : Path
            Path to FITS image to solve.
        output_dir : Path
            Directory for solve-field output files.
        basename : str
            Base name for output files.

        Returns
        -------
        list of str
            Command line arguments.
        """
        cmd = [
            self.binary_path,
            "--no-plots",  # Don't generate PNG plots
            "--overwrite",  # Overwrite existing output files
            "--dir",
            str(output_dir),
            "--new-fits",
            "none",  # Don't write new FITS file
            "--wcs",
            str(output_dir / f"{basename}.wcs"),  # Write WCS to this file
        ]

        # Add scale hints if provided
        if self.scale_low is not None:
            cmd.extend(["--scale-low", str(self.scale_low)])
        if self.scale_high is not None:
            cmd.extend(["--scale-high", str(self.scale_high)])
        if self.scale_low is not None or self.scale_high is not None:
            cmd.append("--scale-units")
            cmd.append("arcsecperpix")

        # Add depth if provided
        if self.depth is not None:
            if isinstance(self.depth, list):
                cmd.append("--depth")
                cmd.append(",".join(str(d) for d in self.depth))
            else:
                cmd.extend(["--depth", str(self.depth)])

        # Add downsample if provided
        if self.downsample is not None:
            cmd.extend(["--downsample", str(self.downsample)])

        # Add any extra user-specified arguments
        cmd.extend(self.extra_args)

        # Add image path
        cmd.append(str(image_path))

        return cmd

    def _parse_wcs_from_output(self, output_dir: Path, basename: str) -> WCS | None:
        """Extract WCS from solve-field output files.

        solve-field creates basename.wcs containing the WCS FITS header.

        Parameters
        ----------
        output_dir : Path
            Directory containing solve-field output files.
        basename : str
            Base name of output files.

        Returns
        -------
        WCS or None
            Parsed WCS object or None if parsing failed.
        """
        wcs_file = output_dir / f"{basename}.wcs"

        if not wcs_file.exists():
            logger.debug("WCS file not found: %s", wcs_file)
            return None

        try:
            # Read WCS header from .wcs file
            with fits.open(wcs_file) as hdul:
                header = hdul[0].header
                wcs = WCS(header)
                return wcs
        except Exception as e:
            logger.debug("Failed to parse WCS from %s: %s", wcs_file, e)
            return None
