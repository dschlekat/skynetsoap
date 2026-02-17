"""SOAP v2 -- Field-wide photometry pipeline orchestrator."""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from tqdm import tqdm

from .config.loader import SOAPConfig, load_config
from .config.photometry import (
    canonicalize_filter_band,
    get_ab_minus_vega_offsets,
    infer_mag_system_for_filter,
)
from .core.image import FITSImage
from .core.errors import ccd_magnitude_error, compute_limiting_magnitude
from .core.limiting_mag import compute_robust_limiting_magnitude
from .core.result import PhotometryResult
from .extraction.background import estimate_background
from .extraction.extractor import extract_sources
from .extraction.aperture import (
    sum_circle,
    compute_optimal_aperture,
    fwhm_scaled_radius,
)
from .calibration.default_calibrator import DefaultCalibrator
from .io.skynet_api import SkynetAPI
from .io.plotter import plot_lightcurve, create_debug_plot
from .io.cache_admin import (
    get_cache_info,
    clear_observation_cache,
    dir_stats as cache_dir_stats,
    prune_cache as prune_cache_admin,
    cleanup_observation as cleanup_observation_admin,
)
from .utils.logging import setup_logging

import warnings
from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter("ignore", category=AstropyWarning)


class Soap:
    """Field-wide photometry pipeline runner.

    This class orchestrates the full extraction + calibration pipeline
    across all sources in a field. Supports single/multi-aperture photometry,
    forced photometry at specified positions, and limiting magnitude calculation.

    Parameters
    ----------
    observation_id : int
        Skynet observation ID.
    config : SOAPConfig, optional
        Pipeline configuration. If None, loads defaults.
    config_path : str or Path, optional
        Path to a user TOML config file.
    calibrator : CalibrationProtocol, optional
        Custom calibration backend. If None, uses DefaultCalibrator.
    solver : AstrometryProtocol, optional
        Custom astrometry backend. If None, astrometry is skipped
        unless ``config.astrometry_enabled`` is True.
    verbose : bool
        Enable verbose logging and progress bars.
    image_dir : str or Path
        Directory for downloaded FITS images.
    result_dir : str or Path
        Directory for output results.

    Photometry Modes
    ----------------
    - **fixed**: Single aperture with user-defined radius
    - **fwhm_scaled**: Aperture scaled to measured FWHM
    - **optimal**: Searches for radius that maximizes median SNR across all sources in the observation (default)
    - **multi**: Tests multiple radii for curve-of-growth analysis

    By default, all modes return one row per source (best aperture selected).
    Set `aperture_keep_all=True` to keep all tested apertures.

    Forced Photometry
    -----------------
    Measure flux at specified sky positions even if no source is detected.
    Pass `forced_positions` (list of SkyCoord) to `run()` method.
    Forced measurements are flagged with `is_forced=True`.

    Limiting Magnitude
    ------------------
    Automatically calculated for all measurements (normal and forced) based on
    5-sigma detection threshold. Default method is analytic (background RMS +
    aperture size). Optional robust blank-sky aperture sampling is configurable
    via ``limiting_mag.method = "robust"``.
    Stored in `limiting_mag` column, with diagnostics in metadata and per-row
    `limiting_mag_analytic` / `limiting_mag_robust` columns.

    Methods
    -------
    download(after=None, before=None, days_ago=None)
        Download FITS images from Skynet with optional date filters.
    run(images=None, forced_positions=None, after=None, before=None, days_ago=None, convert_vega_to_ab=None)
        Run the full photometry pipeline. Optionally include forced photometry.
    plot(units="calibrated_mag", path=None, show=False, **kwargs)
        Plot the light curve in the specified units.
    export(format="csv", path=None, **kwargs)
        Export the photometry results to a file in the specified format.
    debug_image(image_path, save_path=None, show=False)
        Create diagnostic 4-panel plot for a single image.
    cache_info()
        Get information about cached files for this observation.
    clear_cache(images=True, results=True, confirm=True)
        Clear cached files for this observation.
    cleanup_observation(observation_id, image_dir="soap_images", result_dir="soap_results", ...)
        Remove cached files for a specific observation ID.
    prune_cache(max_total_size_mb, image_dir="soap_images", result_dir="soap_results", ...)
        Prune oldest observation caches to stay under a disk budget.

    Examples
    --------
    Basic field-wide run:

    >>> from skynetsoap import Soap
    >>> s = Soap(observation_id=11920699, verbose=True)
    >>> s.download(after="2025-01-12")
    >>> result = s.run()
    >>> result.to_csv("all_sources.csv")

    Forced photometry at a target position:

    >>> from astropy.coordinates import SkyCoord
    >>> target = SkyCoord("12:49:37.598", "-63:32:09.8", unit=("hourangle", "deg"))
    >>> forced = s.run(forced_positions=[target])
    >>> target_result = forced.extract_target(target, forced_photometry=True)
    >>> target_result.to_csv("target_forced.csv")
    """

    def __init__(
        self,
        observation_id: int,
        config: SOAPConfig | None = None,
        config_path: str | Path | None = None,
        calibrator=None,
        solver=None,
        verbose: bool = False,
        image_dir: str | Path = "soap_images",
        result_dir: str | Path = "soap_results",
    ):
        self.observation_id = observation_id

        # Config
        if config is not None:
            self.config = config
        else:
            self.config = load_config(user_config_path=config_path)

        self.verbose = verbose
        self.logger = setup_logging(
            level=self.config.log_level,
            verbose=verbose,
        )

        # Calibrator
        if calibrator is not None:
            self.calibrator = calibrator
        else:
            self.calibrator = DefaultCalibrator(self.config)

        # Astrometry solver
        if solver is not None:
            self.solver = solver  # User-provided solver takes precedence
        elif self.config.astrometry_enabled:
            self.solver = self._create_default_solver()  # Auto-create from config
        else:
            self.solver = None

        # Directories
        self.image_dir = Path(image_dir) / str(observation_id)
        self.result_dir = Path(result_dir) / str(observation_id)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        # State
        self._result: PhotometryResult | None = None
        self._api: SkynetAPI | None = None

    def _create_default_solver(self):
        """Create default HybridAstrometryNetSolver from config.

        Returns
        -------
        HybridAstrometryNetSolver
            Configured hybrid solver instance.
        """
        from .astrometry import HybridAstrometryNetSolver

        return HybridAstrometryNetSolver(
            mode=self.config.astrometry_solver,
            fallback_enabled=self.config.astrometry_fallback_enabled,
            local_config={
                "binary_path": self.config.astrometry_local_binary_path,
                "timeout": self.config.astrometry_local_timeout,
                "scale_low": self.config.astrometry_local_scale_low,
                "scale_high": self.config.astrometry_local_scale_high,
                "depth": self.config.astrometry_local_depth,
                "downsample": self.config.astrometry_local_downsample,
                "extra_args": self.config.astrometry_local_extra_args,
            },
            api_config={
                "api_key": self.config.astrometry_api_key,
                "timeout": self.config.astrometry_api_timeout,
            },
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def result(self) -> PhotometryResult:
        """The most recent pipeline result."""
        if self._result is None:
            raise RuntimeError("No results yet. Call .run() first.")
        return self._result

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    def download(
        self,
        after: str | None = None,
        before: str | None = None,
        days_ago: float | None = None,
    ) -> list[Path]:
        """Download images from Skynet.

        Parameters
        ----------
        after, before : str, optional
            ISO datetime bounds.
        days_ago : float, optional
            Only images from the last N days.

        Returns
        -------
        list[Path]
        """
        if self._api is None:
            self._api = SkynetAPI()

        obs = self._api.get_observation(self.observation_id)
        self.logger.info(
            "Downloading images for observation %d (%s)", self.observation_id, obs.name
        )

        images = self._api.download_images(
            obs,
            path=self.image_dir,
            after=after,
            before=before,
            days_ago=days_ago,
            verbose=self.verbose,
        )
        return images

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def run(
        self,
        images: list[str | Path] | None = None,
        after: str | None = None,
        before: str | None = None,
        days_ago: float | None = None,
        forced_positions: list[SkyCoord] | None = None,
        debug: bool | None = None,
        convert_vega_to_ab: bool | None = None,
    ) -> PhotometryResult:
        """Run the full field-wide photometry pipeline.

        Parameters
        ----------
        images : list, optional
            Explicit list of FITS file paths. If None, uses all .fits
            files in ``image_dir``.
        after, before, days_ago
            Date filters (applied to image MJD after loading).
        forced_positions : list[SkyCoord], optional
            List of sky coordinates for forced photometry. If provided,
            measurements will be made at these exact positions.
        debug : bool, optional
            Enable debug plotting for this run. If None, uses config setting.
            If True, generates 4-panel diagnostic plots for each image.
        convert_vega_to_ab : bool, optional
            If True, convert Vega-based calibrated magnitudes/zeropoints to AB.
            If None, uses config ``calibration.convert_vega_to_ab``.

        Returns
        -------
        PhotometryResult
        """
        if images is None:
            image_paths = sorted(
                Path(p) for p in glob.glob(str(self.image_dir / "*.fits"))
            )
        else:
            image_paths = [Path(p) for p in images]

        if not image_paths:
            self.logger.warning("No FITS images found in %s", self.image_dir)
            self._result = PhotometryResult(
                observation_id=self.observation_id,
                result_dir=self.result_dir,
            )
            return self._result

        self.logger.info("Processing %d images", len(image_paths))

        result = PhotometryResult(
            observation_id=self.observation_id,
            result_dir=self.result_dir,
        )
        ref_catalog_initialized = False

        # Determine debug mode (parameter overrides config)
        debug_mode = debug if debug is not None else self.config.debug_mode
        convert_vega_to_ab_enabled = (
            self.config.calibration_convert_vega_to_ab
            if convert_vega_to_ab is None
            else convert_vega_to_ab
        )

        loop = tqdm(image_paths, desc="Processing", disable=not self.verbose)
        for img_path in loop:
            loop.set_description(f"Processing {img_path.name}")
            try:
                self._process_single_image(
                    img_path,
                    result,
                    ref_catalog_initialized,
                    forced_positions,
                    debug_mode,
                    convert_vega_to_ab_enabled,
                )
                ref_catalog_initialized = True
            except Exception as e:
                self.logger.warning("Error processing %s: %s", img_path.name, e)
                continue

        # Sort by time
        if len(result) > 0:
            result = result.sort_by_time()

        self._result = result
        self.logger.info("Pipeline complete: %d measurements", len(result))
        return result

    def _process_single_image(
        self,
        img_path: Path,
        result: PhotometryResult,
        ref_catalog_initialized: bool,
        forced_positions: list[SkyCoord] | None = None,
        debug_mode: bool = False,
        convert_vega_to_ab: bool = False,
    ) -> None:
        """Process a single FITS image through the pipeline."""
        image = FITSImage.load(img_path)

        # Check for valid WCS
        if not image.has_wcs:
            if self.solver is not None and self.config.astrometry_enabled:
                wcs = self.solver.solve(image)
                if wcs is None:
                    self.logger.info(
                        "Skipping %s: no WCS and solving failed.", img_path.name
                    )
                    return
                image._wcs = wcs
            else:
                self.logger.info("Skipping %s: no WCS.", img_path.name)
                return

        # Initialize reference catalog from first good image
        if not ref_catalog_initialized:
            center = image.wcs.pixel_to_world(
                image.shape[1] / 2.0, image.shape[0] / 2.0
            )
            self.calibrator.get_reference_catalog(center, filter_band=image.filter_name)

        # Background subtraction
        bkg_result = estimate_background(image.data)
        data_sub = image.data - bkg_result.background

        # Source extraction
        ext_result = extract_sources(
            data_sub,
            err=bkg_result.global_rms,
            threshold=self.config.extraction_threshold,
            min_area=self.config.extraction_min_area,
        )

        if ext_result.n_sources == 0:
            self.logger.info("No sources extracted from %s", img_path.name)
            return

        objects = ext_result.objects

        # Get aperture radii (may be multiple for multi-aperture mode)
        aperture_radii = self._get_aperture_radii(
            data_sub, objects, bkg_result.global_rms, image.gain, ext_result.fwhm
        )

        # For calibration, we need to perform one measurement to get zeropoint
        # Use the first aperture radius for this
        calib_aperture_r = aperture_radii[0]
        flux_cal, flux_err_cal, _ = sum_circle(
            data_sub,
            objects["x"],
            objects["y"],
            calib_aperture_r,
            err=bkg_result.global_rms,
            gain=image.gain,
        )

        # Filter valid sources for calibration
        valid_cal = flux_cal > 0
        if not np.any(valid_cal):
            self.logger.info("No valid sources for calibration in %s", img_path.name)
            return

        # Get calibration zeropoint using valid sources
        flux_cal_valid = flux_cal[valid_cal]
        flux_err_cal_valid = flux_err_cal[valid_cal]
        x_cal = objects["x"][valid_cal]
        y_cal = objects["y"][valid_cal]

        ins_mag_cal = -2.5 * np.log10(flux_cal_valid)
        ins_mag_err_cal = 1.0857 * (flux_err_cal_valid / flux_cal_valid)

        world_coords_cal = image.wcs.pixel_to_world(x_cal, y_cal)
        if not hasattr(world_coords_cal, "__len__"):
            world_coords_cal = SkyCoord([world_coords_cal])

        zp, zp_err, match_mask = self.calibrator.calibrate_image(
            image, ins_mag_cal, ins_mag_err_cal, world_coords_cal, image.filter_name
        )
        n_cal = int(np.sum(match_mask)) if match_mask is not None else 0

        canonical_filter = canonicalize_filter_band(
            image.filter_name, self.config.filters
        )
        cal_mag_system = self._infer_calibrated_mag_system(canonical_filter)
        if convert_vega_to_ab and cal_mag_system == "Vega":
            ab_minus_vega = self._ab_minus_vega_offset(canonical_filter)
            if ab_minus_vega is None:
                self.logger.warning(
                    "Requested Vega->AB conversion but no offset is defined for filter %s.",
                    canonical_filter,
                )
            else:
                zp = zp + ab_minus_vega
                cal_mag_system = "AB"
                self.logger.info(
                    "Converted calibration to AB for filter %s using AB-Vega offset %.3f mag.",
                    canonical_filter,
                    ab_minus_vega,
                )

        limiting_method = self._resolve_limiting_method()
        cal_limiting_mag = compute_limiting_magnitude(
            zp, bkg_result.global_rms, calib_aperture_r, n_sigma=5.0
        )
        robust_extra_mask = self._build_robust_extra_mask(image, limiting_method)
        self.logger.info(
            "ZP = %.3f +/- %.3f from %d stars (filter %s), limiting mag (analytic) = %.3f",
            zp,
            zp_err,
            n_cal,
            canonical_filter,
            cal_limiting_mag,
        )
        if limiting_method == "robust":
            self.logger.info(
                "Using robust limiting magnitude for %s with pipeline-selected apertures.",
                img_path.name,
            )

        # Multi-aperture photometry loop
        for aperture_id, aperture_r in enumerate(aperture_radii):
            # Aperture photometry on all sources
            flux, flux_err, flag = sum_circle(
                data_sub,
                objects["x"],
                objects["y"],
                aperture_r,
                err=bkg_result.global_rms,
                gain=image.gain,
            )

            # Filter out negative/zero flux
            valid = flux > 0
            if not np.any(valid):
                continue

            flux_valid = flux[valid]
            flux_err_valid = flux_err[valid]
            flag_valid = flag[valid]
            x = objects["x"][valid]
            y = objects["y"][valid]

            # Convert to sky coordinates
            world_coords = image.wcs.pixel_to_world(x, y)
            if not hasattr(world_coords, "__len__"):
                world_coords = SkyCoord([world_coords])

            ins_mag, ins_mag_err, snr, cal_mag, cal_mag_err = (
                self._compute_photometry_quantities(
                    flux=flux_valid,
                    flux_err=flux_err_valid,
                    zp=zp,
                    zp_err=zp_err,
                    aperture_r=aperture_r,
                    bkg_global_back=bkg_result.global_back,
                    bkg_global_rms=bkg_result.global_rms,
                    gain=image.gain,
                    rdnoise=image.rdnoise,
                )
            )
            limiting_mag_analytic, limiting_mag_robust, limiting_mag = (
                self._compute_limiting_magnitude_for_radius(
                    img_path=img_path,
                    result=result,
                    data_sub=data_sub,
                    bkg_global_rms=bkg_result.global_rms,
                    zp=zp,
                    aperture_radius=aperture_r,
                    limiting_method=limiting_method,
                    robust_extra_mask=robust_extra_mask,
                    aperture_id=aperture_id,
                    is_forced=False,
                )
            )

            self._append_measurement_rows(
                result=result,
                image=image,
                img_path=img_path,
                x_pix=np.asarray(x, dtype=float),
                y_pix=np.asarray(y, dtype=float),
                ra_deg=np.asarray(world_coords.ra.deg, dtype=float),
                dec_deg=np.asarray(world_coords.dec.deg, dtype=float),
                flux=np.asarray(flux_valid, dtype=float),
                flux_err=np.asarray(flux_err_valid, dtype=float),
                snr=np.asarray(snr, dtype=float),
                ins_mag=np.asarray(ins_mag, dtype=float),
                ins_mag_err=np.asarray(ins_mag_err, dtype=float),
                calibrated_mag=np.asarray(cal_mag, dtype=float),
                calibrated_mag_err=np.asarray(cal_mag_err, dtype=float),
                cal_mag_system=cal_mag_system,
                zp=zp,
                zp_err=zp_err,
                aperture_id=aperture_id,
                aperture_radius=aperture_r,
                fwhm=ext_result.fwhm,
                n_cal_stars=n_cal,
                limiting_mag_analytic=limiting_mag_analytic,
                limiting_mag_robust=limiting_mag_robust,
                limiting_mag=limiting_mag,
                is_forced=False,
                flags=np.asarray(flag_valid, dtype=int),
            )

        # Forced photometry at specified positions
        if forced_positions is not None:
            forced_aperture_r = self.config.forced_photometry_aperture_radius

            forced_x, forced_y = self._forced_positions_to_pixels(
                image, forced_positions
            )

            # Perform aperture photometry at forced positions
            flux_forced, flux_err_forced, flag_forced = sum_circle(
                data_sub,
                forced_x,
                forced_y,
                forced_aperture_r,
                err=bkg_result.global_rms,
                gain=image.gain,
            )

            (
                forced_ins_mag,
                forced_ins_mag_err,
                forced_snr,
                forced_cal_mag,
                forced_cal_mag_err,
            ) = self._compute_photometry_quantities(
                flux=np.asarray(flux_forced, dtype=float),
                flux_err=np.asarray(flux_err_forced, dtype=float),
                zp=zp,
                zp_err=zp_err,
                aperture_r=forced_aperture_r,
                bkg_global_back=bkg_result.global_back,
                bkg_global_rms=bkg_result.global_rms,
                gain=image.gain,
                rdnoise=image.rdnoise,
            )
            (
                limiting_mag_forced_analytic,
                limiting_mag_forced_robust,
                limiting_mag_forced,
            ) = self._compute_limiting_magnitude_for_radius(
                img_path=img_path,
                result=result,
                data_sub=data_sub,
                bkg_global_rms=bkg_result.global_rms,
                zp=zp,
                aperture_radius=forced_aperture_r,
                limiting_method=limiting_method,
                robust_extra_mask=robust_extra_mask,
                aperture_id=0,
                is_forced=True,
            )

            forced_ra = np.array(
                [coord.ra.deg for coord in forced_positions], dtype=float
            )
            forced_dec = np.array(
                [coord.dec.deg for coord in forced_positions], dtype=float
            )
            self._append_measurement_rows(
                result=result,
                image=image,
                img_path=img_path,
                x_pix=np.asarray(forced_x, dtype=float),
                y_pix=np.asarray(forced_y, dtype=float),
                ra_deg=forced_ra,
                dec_deg=forced_dec,
                flux=np.asarray(flux_forced, dtype=float),
                flux_err=np.asarray(flux_err_forced, dtype=float),
                snr=np.asarray(forced_snr, dtype=float),
                ins_mag=np.asarray(forced_ins_mag, dtype=float),
                ins_mag_err=np.asarray(forced_ins_mag_err, dtype=float),
                calibrated_mag=np.asarray(forced_cal_mag, dtype=float),
                calibrated_mag_err=np.asarray(forced_cal_mag_err, dtype=float),
                cal_mag_system=cal_mag_system,
                zp=zp,
                zp_err=zp_err,
                aperture_id=0,
                aperture_radius=forced_aperture_r,
                fwhm=ext_result.fwhm,
                n_cal_stars=n_cal,
                limiting_mag_analytic=limiting_mag_forced_analytic,
                limiting_mag_robust=limiting_mag_forced_robust,
                limiting_mag=limiting_mag_forced,
                is_forced=True,
                flags=np.asarray(flag_forced, dtype=int),
            )

        # Save intermediate products if enabled
        if self.config.save_intermediates:
            # Use data from last aperture for intermediate products
            self._save_intermediates(
                img_path,
                image,
                bkg_result,
                data_sub,
                ext_result,
                x_cal,
                y_cal,
                world_coords_cal,
                flux_cal_valid,
                ins_mag_cal,
                zp,
            )

        # Generate debug plot if enabled
        if debug_mode:
            debug_dir = Path(self.config.debug_dir) / str(self.observation_id)
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_path = debug_dir / f"{img_path.stem}_debug.png"

            # Prepare data for debug plot (use calibration aperture)
            source_coords = np.column_stack([x_cal, y_cal])
            if (
                hasattr(self.calibrator, "_cached_ref")
                and self.calibrator._cached_ref is not None
            ):
                import astropy.units as u

                ref = self.calibrator._cached_ref
                ref_coords = SkyCoord(
                    ra=ref["RAJ2000"].to_numpy() * u.deg,
                    dec=ref["DEJ2000"].to_numpy() * u.deg,
                )
            else:
                ref_coords = None

            create_debug_plot(
                image=image,
                bkg_result=bkg_result,
                source_coords=source_coords,
                aperture_radius=calib_aperture_r,
                ref_coords=ref_coords,
                matched_mask=match_mask,
                save_path=debug_path,
            )

    def _resolve_limiting_method(self) -> str:
        """Return limiting-magnitude method with validation and fallback."""
        limiting_method = self.config.limiting_mag_method.lower().strip()
        if limiting_method not in {"analytic", "robust"}:
            self.logger.warning(
                "Unknown limiting_mag.method=%r; falling back to analytic.",
                self.config.limiting_mag_method,
            )
            return "analytic"
        return limiting_method

    def _infer_calibrated_mag_system(self, canonical_filter: str) -> str:
        """Infer calibrated magnitude system from canonical filter band."""
        return infer_mag_system_for_filter(canonical_filter, self.config.filters)

    def _ab_minus_vega_offset(self, canonical_filter: str) -> float | None:
        """Return AB-Vega offset for a canonical filter, if available."""
        band = canonicalize_filter_band(canonical_filter, self.config.filters)
        offsets = get_ab_minus_vega_offsets()
        return offsets.get(band)

    def _build_robust_extra_mask(
        self, image: FITSImage, limiting_method: str
    ) -> np.ndarray | None:
        """Build extra bad-pixel/saturation mask for robust limiting magnitude."""
        if limiting_method != "robust":
            return None

        robust_extra_mask = ~np.isfinite(image.data)
        sat_level = image.header.get("SATURATE")
        if sat_level is not None:
            robust_extra_mask = robust_extra_mask | (image.data >= float(sat_level))
        return robust_extra_mask

    def _compute_photometry_quantities(
        self,
        flux: np.ndarray,
        flux_err: np.ndarray,
        zp: float,
        zp_err: float,
        aperture_r: float,
        bkg_global_back: float,
        bkg_global_rms: float,
        gain: float,
        rdnoise: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute instrumental/calibrated magnitude quantities from flux arrays."""
        flux = np.asarray(flux, dtype=float)
        flux_err = np.asarray(flux_err, dtype=float)

        ins_mag = np.full_like(flux, np.nan, dtype=float)
        ins_mag_err = np.full_like(flux, np.nan, dtype=float)
        snr = np.full_like(flux, np.nan, dtype=float)
        cal_mag = np.full_like(flux, np.nan, dtype=float)
        cal_mag_err = np.full_like(flux, np.nan, dtype=float)

        valid = flux > 0
        if not np.any(valid):
            return ins_mag, ins_mag_err, snr, cal_mag, cal_mag_err

        ins_mag[valid] = -2.5 * np.log10(flux[valid])
        ins_mag_err[valid] = 1.0857 * (flux_err[valid] / flux[valid])
        snr[valid] = flux[valid] / flux_err[valid]

        if np.isnan(zp):
            return ins_mag, ins_mag_err, snr, cal_mag, cal_mag_err

        cal_mag[valid] = ins_mag[valid] + zp
        n_pix = np.pi * aperture_r**2
        n_bkgpix = np.pi * (3 * aperture_r) ** 2
        cal_mag_err_valid = ccd_magnitude_error(
            flux=flux[valid],
            gain=gain,
            n_pix=n_pix,
            background=bkg_global_back,
            rdnoise=rdnoise,
            n_bkgpix=n_bkgpix,
            sigma_bkg=bkg_global_rms,
            sigma_zp=zp_err,
        )
        cal_mag_err[valid] = np.atleast_1d(cal_mag_err_valid)
        return ins_mag, ins_mag_err, snr, cal_mag, cal_mag_err

    def _compute_limiting_magnitude_for_radius(
        self,
        *,
        img_path: Path,
        result: PhotometryResult,
        data_sub: np.ndarray,
        bkg_global_rms: float,
        zp: float,
        aperture_radius: float,
        limiting_method: str,
        robust_extra_mask: np.ndarray | None,
        aperture_id: int,
        is_forced: bool,
    ) -> tuple[float, float, float]:
        """Compute analytic/robust/selected limiting magnitudes for one aperture."""
        limiting_mag_analytic = compute_limiting_magnitude(
            zp, bkg_global_rms, aperture_radius, n_sigma=5.0
        )
        if limiting_method != "robust":
            return limiting_mag_analytic, np.nan, limiting_mag_analytic

        robust_result = compute_robust_limiting_magnitude(
            data_sub=data_sub,
            zeropoint=zp,
            aperture_radius_pixels=aperture_radius,
            err=bkg_global_rms,
            extraction_threshold=self.config.extraction_threshold,
            extraction_min_area=self.config.extraction_min_area,
            n_samples=self.config.limiting_mag_robust_n_samples,
            mask_dilate_pixels=self.config.limiting_mag_robust_mask_dilate_pixels,
            edge_buffer_pixels=self.config.limiting_mag_robust_edge_buffer_pixels,
            sigma_estimator=self.config.limiting_mag_robust_sigma_estimator,
            max_draws_multiplier=self.config.limiting_mag_robust_max_draws_multiplier,
            random_seed=self.config.limiting_mag_robust_random_seed,
            extra_mask=robust_extra_mask,
        )
        limiting_mag_robust = robust_result.limiting_mag
        context = "forced_photometry" if is_forced else f"aperture_id={aperture_id}"
        for warning_text in robust_result.warnings:
            self.logger.warning("%s (%s, %s)", warning_text, img_path.name, context)

        self._record_limiting_mag_diagnostic(
            result,
            {
                "image_file": img_path.name,
                "aperture_id": aperture_id,
                "is_forced": is_forced,
                "method": "robust",
                "n_samples_requested": robust_result.n_samples_requested,
                "n_samples_used": robust_result.n_samples_used,
                "aperture_radius_pixels": robust_result.aperture_radius_pixels,
                "sigma_ap": robust_result.sigma_ap,
                "f5": robust_result.flux_limit,
                "fraction_masked": robust_result.fraction_masked,
                "warnings": robust_result.warnings,
            },
        )
        self.logger.info(
            "Robust limiting mag (%s, %s): m5=%.3f, sigma_ap=%.3f, "
            "f5=%.3f, n=%d/%d, r=%.2f px, masked=%.1f%%",
            img_path.name,
            context,
            robust_result.limiting_mag,
            robust_result.sigma_ap,
            robust_result.flux_limit,
            robust_result.n_samples_used,
            robust_result.n_samples_requested,
            robust_result.aperture_radius_pixels,
            100.0 * robust_result.fraction_masked,
        )

        if np.isfinite(limiting_mag_robust):
            return limiting_mag_analytic, limiting_mag_robust, limiting_mag_robust

        self.logger.warning(
            "Robust limiting magnitude failed for %s (%s); using analytic value.",
            img_path.name,
            context,
        )
        return limiting_mag_analytic, limiting_mag_robust, limiting_mag_analytic

    def _forced_positions_to_pixels(
        self, image: FITSImage, forced_positions: list[SkyCoord]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert forced sky positions to pixel arrays."""
        if len(forced_positions) == 1:
            forced_pix = image.wcs.world_to_pixel(forced_positions[0])
            return np.array([forced_pix[0]]), np.array([forced_pix[1]])

        combined_coords = SkyCoord(
            ra=[c.ra for c in forced_positions],
            dec=[c.dec for c in forced_positions],
        )
        forced_pix = image.wcs.world_to_pixel(combined_coords)
        return np.array(forced_pix[0]), np.array(forced_pix[1])

    def _append_measurement_rows(
        self,
        *,
        result: PhotometryResult,
        image: FITSImage,
        img_path: Path,
        x_pix: np.ndarray,
        y_pix: np.ndarray,
        ra_deg: np.ndarray,
        dec_deg: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        snr: np.ndarray,
        ins_mag: np.ndarray,
        ins_mag_err: np.ndarray,
        calibrated_mag: np.ndarray,
        calibrated_mag_err: np.ndarray,
        cal_mag_system: str,
        zp: float,
        zp_err: float,
        aperture_id: int,
        aperture_radius: float,
        fwhm: float,
        n_cal_stars: int,
        limiting_mag_analytic: float,
        limiting_mag_robust: float,
        limiting_mag: float,
        is_forced: bool,
        flags: np.ndarray,
    ) -> None:
        """Append one row per source/position to the result table."""
        for i in range(len(flux)):
            result.add_measurement(
                image_file=img_path.name,
                telescope=image.telescope,
                filter=image.filter_name,
                exptime=image.exptime,
                mjd=image.mjd,
                jd=image.mid_jd,
                x_pix=float(x_pix[i]),
                y_pix=float(y_pix[i]),
                ra=float(ra_deg[i]),
                dec=float(dec_deg[i]),
                flux=float(flux[i]),
                flux_err=float(flux_err[i]),
                snr=float(snr[i]) if np.isfinite(snr[i]) else np.nan,
                ins_mag=float(ins_mag[i]) if np.isfinite(ins_mag[i]) else np.nan,
                ins_mag_err=float(ins_mag_err[i])
                if np.isfinite(ins_mag_err[i])
                else np.nan,
                calibrated_mag=float(calibrated_mag[i])
                if np.isfinite(calibrated_mag[i])
                else np.nan,
                calibrated_mag_err=float(calibrated_mag_err[i])
                if np.isfinite(calibrated_mag_err[i])
                else np.nan,
                cal_mag_system=cal_mag_system,
                zeropoint=float(zp) if not np.isnan(zp) else np.nan,
                zeropoint_err=float(zp_err) if not np.isnan(zp_err) else np.nan,
                aperture_id=aperture_id,
                aperture_radius=aperture_radius,
                fwhm=fwhm,
                n_cal_stars=n_cal_stars,
                limiting_mag_analytic=limiting_mag_analytic,
                limiting_mag_robust=limiting_mag_robust,
                limiting_mag=limiting_mag,
                is_forced=is_forced,
                flag=int(flags[i]),
            )

    def _record_limiting_mag_diagnostic(
        self, result: PhotometryResult, diagnostic: dict[str, object]
    ) -> None:
        """Append limiting-magnitude diagnostics to table metadata."""
        diagnostics = result.table.meta.setdefault("limiting_mag_diagnostics", [])
        diagnostics.append(diagnostic)

    def _get_aperture_radii(
        self,
        data: np.ndarray,
        objects: np.ndarray,
        err: float,
        gain: float,
        fwhm: float,
    ) -> list[float]:
        """Get aperture radii based on config mode.

        Returns
        -------
        list[float]
            List of aperture radii in pixels. For multi mode, returns all
            radii from config. For other modes, returns a single-element list.
        """
        mode = self.config.aperture_mode

        if mode == "multi":
            return self.config.aperture_radii

        if mode == "fixed":
            return [self.config.aperture_fixed_radius]

        if mode == "optimal":
            r = compute_optimal_aperture(
                data,
                objects["x"],
                objects["y"],
                err=err,
                gain=gain,
                min_r=self.config.aperture_min_radius,
                max_r=self.config.aperture_max_radius,
                step=self.config.aperture_step,
            )
            return [r]

        # Default: fwhm_scaled
        r = fwhm_scaled_radius(
            fwhm,
            scale=self.config.aperture_scale,
            min_r=self.config.aperture_min_radius,
            max_r=self.config.aperture_max_radius,
        )
        return [r]

    def _select_aperture(
        self,
        data: np.ndarray,
        objects: np.ndarray,
        err: float,
        gain: float,
        fwhm: float,
    ) -> float:
        """Select aperture radius based on config mode.

        Deprecated: Use _get_aperture_radii instead.
        """
        radii = self._get_aperture_radii(data, objects, err, gain, fwhm)
        return radii[0]

    def _save_intermediates(
        self,
        img_path: Path,
        image: FITSImage,
        bkg_result,
        data_sub: np.ndarray,
        ext_result,
        x: np.ndarray,
        y: np.ndarray,
        world_coords: SkyCoord,
        flux: np.ndarray,
        ins_mag: np.ndarray,
        zp: float,
    ) -> None:
        """Save intermediate products to disk."""
        from astropy.io import fits
        from astropy.table import Table

        inter_dir = Path(self.config.intermediates_dir) / str(self.observation_id)
        inter_dir.mkdir(parents=True, exist_ok=True)

        # Save background-subtracted FITS
        bgsub_path = inter_dir / f"{img_path.stem}_bgsub.fits"
        hdu = fits.PrimaryHDU(data=data_sub, header=image.header)
        hdu.header["BKGMEAN"] = (bkg_result.global_back, "Background mean (counts)")
        hdu.header["BKGRMS"] = (bkg_result.global_rms, "Background RMS (counts)")
        hdu.writeto(bgsub_path, overwrite=True)

        # Save source catalog
        catalog_path = inter_dir / f"{img_path.stem}_sources.ecsv"
        catalog = Table()
        catalog["x_pix"] = x
        catalog["y_pix"] = y
        if world_coords is not None:
            catalog["ra"] = world_coords.ra.deg
            catalog["dec"] = world_coords.dec.deg
        catalog["flux"] = flux
        catalog["ins_mag"] = ins_mag
        catalog["fwhm"] = ext_result.fwhm
        catalog.write(catalog_path, format="ascii.ecsv", overwrite=True)

        # Save calibrated FITS (if zeropoint available)
        if not np.isnan(zp):
            cal_path = inter_dir / f"{img_path.stem}_cal.fits"
            hdu_cal = fits.PrimaryHDU(data=data_sub, header=image.header)
            hdu_cal.header["ZP"] = (zp, "Photometric zeropoint (mag)")
            hdu_cal.header["BKGMEAN"] = (
                bkg_result.global_back,
                "Background mean (counts)",
            )
            hdu_cal.header["BKGRMS"] = (
                bkg_result.global_rms,
                "Background RMS (counts)",
            )
            hdu_cal.writeto(cal_path, overwrite=True)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    def plot(
        self,
        units: str = "calibrated_mag",
        path: str | Path | None = None,
        show: bool = False,
        **kwargs,
    ):
        """Plot the pipeline results.

        Parameters
        ----------
        units : str
            "flux", "calibrated_mag", or "ins_mag".
        path : str or Path, optional
            Save path.
        show : bool
            Call plt.show().
        """
        if path is None:
            path = self.result_dir / f"{units}_plot.png"
        return plot_lightcurve(self.result, units=units, path=path, show=show, **kwargs)

    def export(
        self,
        format: str = "csv",
        path: str | Path | None = None,
        **kwargs,
    ) -> Path:
        """Export results to file.

        Parameters
        ----------
        format : str
            "csv", "ecsv", "parquet", "json", "gcn".
        path : str or Path, optional
            Output path. Auto-generated if not provided.

        Returns
        -------
        Path
        """
        if path is None:
            ext = {
                "csv": ".csv",
                "ecsv": ".ecsv",
                "parquet": ".parquet",
                "json": ".json",
                "gcn": ".txt",
            }
            path = self.result_dir / f"photometry{ext.get(format, '.csv')}"

        path = Path(path)
        export_fn = getattr(self.result, f"to_{format}", None)
        if export_fn is None:
            raise ValueError(f"Unknown export format: {format!r}")
        return export_fn(path, **kwargs)

    # ------------------------------------------------------------------
    # Cache and cleanup utilities
    # ------------------------------------------------------------------
    def cache_info(self) -> dict:
        """Get information about cached files for this observation."""
        return get_cache_info(self.image_dir, self.result_dir)

    def clear_cache(
        self,
        images: bool = True,
        results: bool = True,
        confirm: bool = True,
    ) -> dict:
        """Clear cached files for this observation."""
        stats = clear_observation_cache(
            image_dir=self.image_dir,
            result_dir=self.result_dir,
            observation_id=self.observation_id,
            images=images,
            results=results,
            confirm=confirm,
        )
        if stats["images_deleted"] > 0:
            self.logger.info(
                "Deleted %d images from %s", stats["images_deleted"], self.image_dir
            )
        if stats["results_deleted"] > 0:
            self.logger.info(
                "Deleted %d results from %s", stats["results_deleted"], self.result_dir
            )
        return stats

    @staticmethod
    def _dir_stats(path: Path) -> tuple[int, int, float]:
        """Return (n_files, total_size_bytes, latest_mtime) for a path tree."""
        return cache_dir_stats(path)

    @staticmethod
    def prune_cache(
        max_total_size_mb: float,
        image_dir: str | Path = "soap_images",
        result_dir: str | Path = "soap_results",
        keep_recent: int = 1,
        confirm: bool = True,
    ) -> dict:
        """Prune old observation caches until total disk usage is below a limit."""
        return prune_cache_admin(
            max_total_size_mb=max_total_size_mb,
            image_dir=image_dir,
            result_dir=result_dir,
            keep_recent=keep_recent,
            confirm=confirm,
        )

    @staticmethod
    def cleanup_observation(
        observation_id: int,
        image_dir: str | Path = "soap_images",
        result_dir: str | Path = "soap_results",
        images: bool = True,
        results: bool = True,
        confirm: bool = True,
    ) -> dict:
        """Clean up all files for a specific observation (static method)."""
        return cleanup_observation_admin(
            observation_id=observation_id,
            image_dir=image_dir,
            result_dir=result_dir,
            images=images,
            results=results,
            confirm=confirm,
        )

    # ------------------------------------------------------------------
    # Debugging utilities
    # ------------------------------------------------------------------
    def debug_image(
        self,
        image_path: str | Path,
        save_path: str | Path | None = None,
        show: bool = False,
    ):
        """Create debugging visualization for a single image.

        Generates a 4-panel plot showing:
        1. Original image
        2. Background map
        3. Background-subtracted image with extracted sources and apertures
        4. Matched vs unmatched calibration stars

        Parameters
        ----------
        image_path : str or Path
            Path to a FITS image file.
        save_path : str or Path, optional
            If provided, save the plot to this path.
            If None and debug_mode is enabled, saves to debug_dir.
        show : bool
            If True, display the plot interactively.

        Returns
        -------
        Figure
            The matplotlib figure object.
        """
        import matplotlib.pyplot as plt
        import astropy.units as u

        img_path = Path(image_path)
        image = FITSImage.load(img_path)

        if not image.has_wcs:
            self.logger.warning(
                "Image %s has no WCS, debug plot may be limited.", img_path.name
            )

        # Background subtraction
        bkg_result = estimate_background(image.data)
        data_sub = image.data - bkg_result.background

        # Source extraction
        ext_result = extract_sources(
            data_sub,
            err=bkg_result.global_rms,
            threshold=self.config.extraction_threshold,
            min_area=self.config.extraction_min_area,
        )

        if ext_result.n_sources == 0:
            self.logger.warning("No sources extracted from %s", img_path.name)
            source_coords = None
            aperture_r = None
            ref_coords = None
            match_mask = None
        else:
            objects = ext_result.objects
            aperture_radii = self._get_aperture_radii(
                data_sub, objects, bkg_result.global_rms, image.gain, ext_result.fwhm
            )
            # Use first aperture for debugging
            aperture_r = aperture_radii[0]

            # Get source coordinates
            x = objects["x"]
            y = objects["y"]
            source_coords = np.column_stack([x, y])

            # Convert to sky coordinates if WCS available
            if image.has_wcs:
                world_coords = image.wcs.pixel_to_world(x, y)
                if not hasattr(world_coords, "__len__"):
                    world_coords = SkyCoord([world_coords])

                # Get reference catalog if available
                if (
                    hasattr(self.calibrator, "_cached_ref")
                    and self.calibrator._cached_ref is not None
                ):
                    ref = self.calibrator._cached_ref
                    ref_coords = SkyCoord(
                        ra=ref["RAJ2000"].to_numpy() * u.deg,
                        dec=ref["DEJ2000"].to_numpy() * u.deg,
                    )
                    idx, d2d, _ = world_coords.match_to_catalog_sky(ref_coords)
                    match_mask = d2d.arcsec < self.calibrator.match_radius_arcsec
                else:
                    ref_coords = None
                    match_mask = None
            else:
                ref_coords = None
                match_mask = None

        # Determine save path
        if save_path is None:
            debug_dir = Path(self.config.debug_dir) / str(self.observation_id)
            debug_dir.mkdir(parents=True, exist_ok=True)
            final_save_path = debug_dir / f"{img_path.stem}_debug.png"
        else:
            final_save_path = Path(save_path)

        # Create the plot
        fig = create_debug_plot(
            image=image,
            bkg_result=bkg_result,
            source_coords=source_coords,
            aperture_radius=aperture_r,
            ref_coords=ref_coords,
            matched_mask=match_mask,
            save_path=final_save_path,
        )

        if show:
            plt.show()

        return fig
