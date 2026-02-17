"""Hybrid astrometry solver with local and API fallback support."""

from __future__ import annotations

import logging

from astropy.wcs import WCS

from .astrometry_net import AstrometryNetSolver
from .local_solver import LocalAstrometryNetSolver
from .wcs_utils import validate_wcs

logger = logging.getLogger("soap")


class HybridAstrometryNetSolver:
    """``AstrometryProtocol`` implementation with local and API solver support.

    Orchestrates between local solve-field and astrometry.net API with
    configurable priority and fallback behavior.

    Parameters
    ----------
    mode : str, default "auto"
        Solver mode:
        - "auto": Try local first if available, fallback to API
        - "local": Try local first, fallback to API if enabled
        - "api": Try API first, fallback to local if enabled
        - "local_only": Only use local solver (no fallback)
        - "api_only": Only use API solver (no fallback)
    local_config : dict, optional
        Configuration dictionary for LocalAstrometryNetSolver.
        Keys: binary_path, timeout, scale_low, scale_high, depth,
        downsample, extra_args.
    api_config : dict, optional
        Configuration dictionary for AstrometryNetSolver.
        Keys: api_key, timeout.
    fallback_enabled : bool, default True
        Whether to try fallback solver if primary fails.

    Examples
    --------
    >>> # Auto mode: prefer local if available
    >>> solver = HybridAstrometryNetSolver(mode="auto")
    >>> wcs = solver.solve(image)

    >>> # Local-only mode
    >>> solver = HybridAstrometryNetSolver(
    ...     mode="local_only",
    ...     local_config={"timeout": 120, "scale_low": 0.5, "scale_high": 2.0}
    ... )
    >>> wcs = solver.solve(image)

    >>> # API with local fallback
    >>> solver = HybridAstrometryNetSolver(
    ...     mode="api",
    ...     api_config={"api_key": "YOUR_KEY", "timeout": 500},  # pragma: allowlist secret
    ...     fallback_enabled=True
    ... )
    >>> wcs = solver.solve(image)
    """

    VALID_MODES = ("auto", "local", "api", "local_only", "api_only")

    def __init__(
        self,
        mode: str = "auto",
        local_config: dict | None = None,
        api_config: dict | None = None,
        fallback_enabled: bool = True,
    ):
        if mode not in self.VALID_MODES:
            msg = f"Invalid mode '{mode}'. Must be one of {self.VALID_MODES}"
            raise ValueError(msg)

        self.mode = mode
        self.fallback_enabled = fallback_enabled
        self._local_config = local_config or {}
        self._api_config = api_config or {}
        self._local_solver = None
        self._api_solver = None

    def _get_local_solver(self) -> LocalAstrometryNetSolver | None:
        """Lazy-initialize local solver.

        Returns
        -------
        LocalAstrometryNetSolver or None
            Local solver instance, or None if not available.
        """
        if self._local_solver is None:
            if not LocalAstrometryNetSolver.is_available():
                logger.debug("Local solver (solve-field) not available")
                return None

            try:
                self._local_solver = LocalAstrometryNetSolver(**self._local_config)
            except (FileNotFoundError, PermissionError) as e:
                logger.warning("Failed to initialize local solver: %s", e)
                return None

        return self._local_solver

    def _get_api_solver(self) -> AstrometryNetSolver:
        """Lazy-initialize API solver.

        Returns
        -------
        AstrometryNetSolver
            API solver instance.
        """
        if self._api_solver is None:
            self._api_solver = AstrometryNetSolver(**self._api_config)
        return self._api_solver

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
        """Attempt to solve astrometry using configured strategy.

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

        # Determine solver priority based on mode
        primary, secondary = self._determine_solver_priority()

        # Try primary solver
        wcs = self._try_solver(primary, image)
        if wcs is not None:
            logger.info("Solved %s using %s solver", image.path.name, primary)
            return wcs

        # Try fallback if enabled and available
        if self.fallback_enabled and secondary is not None:
            logger.info(
                "Primary %s solver failed, trying %s fallback for %s",
                primary,
                secondary,
                image.path.name,
            )
            wcs = self._try_solver(secondary, image)
            if wcs is not None:
                logger.info(
                    "Solved %s using %s solver (fallback)",
                    image.path.name,
                    secondary,
                )
                return wcs

        logger.warning("Failed to solve astrometry for %s", image.path.name)
        return None

    def _determine_solver_priority(self) -> tuple[str, str | None]:
        """Determine primary and secondary solvers based on mode.

        Returns
        -------
        tuple of (str, str or None)
            Primary solver name and secondary solver name (or None).
        """
        if self.mode == "auto":
            # Auto mode: prefer local if available, else API
            if LocalAstrometryNetSolver.is_available():
                return ("local", "api")
            else:
                return ("api", "local")

        elif self.mode == "local":
            return ("local", "api")

        elif self.mode == "api":
            return ("api", "local")

        elif self.mode == "local_only":
            return ("local", None)

        elif self.mode == "api_only":
            return ("api", None)

        else:
            # Should never reach here due to validation in __init__
            msg = f"Unknown mode: {self.mode}"
            raise ValueError(msg)

    def _try_solver(self, solver_name: str, image) -> WCS | None:
        """Attempt to solve with specified solver.

        Parameters
        ----------
        solver_name : str
            Solver name: "local" or "api".
        image : FITSImage
            Image to solve.

        Returns
        -------
        WCS or None
            WCS object on success, None on failure.
        """
        try:
            if solver_name == "local":
                solver = self._get_local_solver()
                if solver is None:
                    logger.debug("Local solver not available")
                    return None
                return solver.solve(image)

            elif solver_name == "api":
                solver = self._get_api_solver()
                return solver.solve(image)

            else:
                logger.error("Unknown solver name: %s", solver_name)
                return None

        except Exception as e:
            logger.debug("Solver %s error for %s: %s", solver_name, image.path.name, e)
            return None
