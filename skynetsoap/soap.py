"""SOAP v2 -- Field-wide photometry pipeline orchestrator."""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from tqdm import tqdm

from .config.loader import SOAPConfig, load_config
from .core.image import FITSImage
from .core.errors import ccd_magnitude_error, compute_limiting_magnitude
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
from .utils.logging import setup_logging

import warnings
from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter("ignore", category=AstropyWarning)

# TODO: Add clarity for magnitude systems in the results, e.g. by including a "cal_mag_system" column that specifies the system of the calibrated magnitudes (e.g. "AB", "Vega"). Stick with the system used by the input data, but provide a way to convert if needed.
# TODO: Add debugging method to inspect individual images with plotting of sources, apertures, etc.
# TODO: Add better cache management for intermediate products, especially downloaded images, to speed up repeated runs with different configs or parameters.
# TODO: Add support for parallel processing of images to speed up the pipeline on large datasets, with careful handling of shared resources like the reference catalog.
# TODO: Add util methods to clean up downloaded images and results for a given observation ID, or to manage disk usage across multiple runs.
# TODO: Add options to save intermediate products like calibrated images, source catalogs, etc.
# TODO: Add forced photometry mode for known or theorized transient target positions.
# TODO: Add more robust handling of edge cases like no sources detected, no calibrators, failed astrometry, etc.
# TODO: Add support for multi-aperture photometry and curve-of-growth analysis, as well as debugging modes, for better aperture selection.
# TODO: Add an optional limiting magnitude calculation based on background noise and aperture size for non-detections.
# TODO: Add example usage within the class docstring and in the README.
# TODO: Rename package and core class to skynetphot and SkynetPhot for better clarity and discoverability.


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
    - **fwhm_scaled**: Aperture scaled to measured FWHM (default)
    - **optimal**: Searches for radius that maximizes median SNR
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
    5-sigma detection threshold using background RMS and aperture size.
    Stored in `limiting_mag` column.

    Methods
    -------
    download(after=None, before=None, days_ago=None)
        Download FITS images from Skynet with optional date filters.
    run(images=None, forced_positions=None, after=None, before=None, days_ago=None)
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
        self.solver = solver

        # Directories
        self.image_dir = Path(image_dir) / str(observation_id)
        self.result_dir = Path(result_dir) / str(observation_id)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        # State
        self._result: PhotometryResult | None = None
        self._api: SkynetAPI | None = None

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

        loop = tqdm(image_paths, desc="Processing", disable=not self.verbose)
        for img_path in loop:
            loop.set_description(f"Processing {img_path.name}")
            try:
                self._process_single_image(
                    img_path, result, ref_catalog_initialized, forced_positions
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
        cal_limiting_mag = compute_limiting_magnitude(
            zp, bkg_result.global_rms, calib_aperture_r, n_sigma=5.0
        )
        aliases = self.config.filters.get("aliases", {})
        canonical_filter = aliases.get(image.filter_name, image.filter_name)
        self.logger.info(
            "ZP = %.3f +/- %.3f from %d stars (filter %s), limiting mag = %.3f",
            zp,
            zp_err,
            n_cal,
            canonical_filter,
            cal_limiting_mag,
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

            # Instrumental magnitudes
            ins_mag = -2.5 * np.log10(flux_valid)
            ins_mag_err = 1.0857 * (flux_err_valid / flux_valid)

            # Convert to sky coordinates
            world_coords = image.wcs.pixel_to_world(x, y)
            if not hasattr(world_coords, "__len__"):
                world_coords = SkyCoord([world_coords])

            # Calibrated magnitudes
            if np.isnan(zp):
                cal_mag = np.full_like(ins_mag, np.nan)
                cal_mag_err = np.full_like(ins_mag_err, np.nan)
            else:
                cal_mag = ins_mag + zp
                n_pix = np.pi * aperture_r**2
                n_bkgpix = np.pi * (3 * aperture_r) ** 2
                cal_mag_err = ccd_magnitude_error(
                    flux=flux_valid,
                    gain=image.gain,
                    n_pix=n_pix,
                    background=bkg_result.global_back,
                    rdnoise=image.rdnoise,
                    n_bkgpix=n_bkgpix,
                    sigma_bkg=bkg_result.global_rms,
                    sigma_zp=zp_err,
                )
                cal_mag_err = np.atleast_1d(cal_mag_err)

            snr = flux_valid / flux_err_valid

            # Limiting magnitude
            limiting_mag = compute_limiting_magnitude(
                zp, bkg_result.global_rms, aperture_r, n_sigma=5.0
            )

            # Append all sources to result
            for i in range(len(flux_valid)):
                result.add_measurement(
                    image_file=img_path.name,
                    telescope=image.telescope,
                    filter=image.filter_name,
                    exptime=image.exptime,
                    mjd=image.mjd,
                    jd=image.mid_jd,
                    x_pix=float(x[i]),
                    y_pix=float(y[i]),
                    ra=float(world_coords[i].ra.deg),
                    dec=float(world_coords[i].dec.deg),
                    flux=float(flux_valid[i]),
                    flux_err=float(flux_err_valid[i]),
                    snr=float(snr[i]),
                    ins_mag=float(ins_mag[i]),
                    ins_mag_err=float(ins_mag_err[i]),
                    calibrated_mag=float(cal_mag[i]),
                    calibrated_mag_err=float(cal_mag_err[i]),
                    zeropoint=float(zp) if not np.isnan(zp) else np.nan,
                    zeropoint_err=float(zp_err) if not np.isnan(zp_err) else np.nan,
                    aperture_id=aperture_id,
                    aperture_radius=aperture_r,
                    fwhm=ext_result.fwhm,
                    n_cal_stars=n_cal,
                    limiting_mag=limiting_mag,
                    is_forced=False,
                    flag=int(flag_valid[i]),
                )

        # Forced photometry at specified positions
        if forced_positions is not None:
            forced_aperture_r = self.config.forced_photometry_aperture_radius

            # Convert forced sky positions to pixel coordinates
            # Handle both single and multiple positions
            if len(forced_positions) == 1:
                forced_pix = image.wcs.world_to_pixel(forced_positions[0])
                forced_x = np.array([forced_pix[0]])
                forced_y = np.array([forced_pix[1]])
            else:
                # Combine into single SkyCoord for batch conversion
                combined_coords = SkyCoord(
                    ra=[c.ra for c in forced_positions],
                    dec=[c.dec for c in forced_positions],
                )
                forced_pix = image.wcs.world_to_pixel(combined_coords)
                forced_x = np.array(forced_pix[0])
                forced_y = np.array(forced_pix[1])

            # Perform aperture photometry at forced positions
            flux_forced, flux_err_forced, flag_forced = sum_circle(
                data_sub,
                forced_x,
                forced_y,
                forced_aperture_r,
                err=bkg_result.global_rms,
                gain=image.gain,
            )

            # Calculate limiting magnitude for forced photometry
            limiting_mag_forced = compute_limiting_magnitude(
                zp, bkg_result.global_rms, forced_aperture_r, n_sigma=5.0
            )

            # Process each forced position
            for i in range(len(forced_positions)):
                flux_i = flux_forced[i]
                flux_err_i = flux_err_forced[i]

                # Handle negative/zero flux
                if flux_i <= 0:
                    ins_mag_i = np.nan
                    ins_mag_err_i = np.nan
                    snr_i = np.nan
                else:
                    ins_mag_i = -2.5 * np.log10(flux_i)
                    ins_mag_err_i = 1.0857 * (flux_err_i / flux_i)
                    snr_i = flux_i / flux_err_i

                # Calibrated magnitude
                if np.isnan(zp) or np.isnan(ins_mag_i):
                    cal_mag_i = np.nan
                    cal_mag_err_i = np.nan
                else:
                    cal_mag_i = ins_mag_i + zp
                    n_pix = np.pi * forced_aperture_r**2
                    n_bkgpix = np.pi * (3 * forced_aperture_r) ** 2
                    cal_mag_err_i = ccd_magnitude_error(
                        flux=flux_i,
                        gain=image.gain,
                        n_pix=n_pix,
                        background=bkg_result.global_back,
                        rdnoise=image.rdnoise,
                        n_bkgpix=n_bkgpix,
                        sigma_bkg=bkg_result.global_rms,
                        sigma_zp=zp_err,
                    )

                result.add_measurement(
                    image_file=img_path.name,
                    telescope=image.telescope,
                    filter=image.filter_name,
                    exptime=image.exptime,
                    mjd=image.mjd,
                    jd=image.mid_jd,
                    x_pix=float(forced_x[i]),
                    y_pix=float(forced_y[i]),
                    ra=float(forced_positions[i].ra.deg),
                    dec=float(forced_positions[i].dec.deg),
                    flux=float(flux_i),
                    flux_err=float(flux_err_i),
                    snr=float(snr_i) if not np.isnan(snr_i) else np.nan,
                    ins_mag=float(ins_mag_i) if not np.isnan(ins_mag_i) else np.nan,
                    ins_mag_err=float(ins_mag_err_i)
                    if not np.isnan(ins_mag_err_i)
                    else np.nan,
                    calibrated_mag=float(cal_mag_i)
                    if not np.isnan(cal_mag_i)
                    else np.nan,
                    calibrated_mag_err=float(cal_mag_err_i)
                    if not np.isnan(cal_mag_err_i)
                    else np.nan,
                    zeropoint=float(zp) if not np.isnan(zp) else np.nan,
                    zeropoint_err=float(zp_err) if not np.isnan(zp_err) else np.nan,
                    aperture_id=0,
                    aperture_radius=forced_aperture_r,
                    fwhm=ext_result.fwhm,
                    n_cal_stars=n_cal,
                    limiting_mag=limiting_mag_forced,
                    is_forced=True,
                    flag=int(flag_forced[i]),
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
        if self.config.debug_mode:
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
        """Get information about cached files for this observation.

        Returns
        -------
        dict
            Dictionary with cache statistics:
            - n_images: Number of FITS files in image_dir
            - n_results: Number of result files in result_dir
            - total_size_mb: Total size in megabytes
            - image_dir: Path to image directory
            - result_dir: Path to result directory
        """
        image_files = list(self.image_dir.glob("*.fits"))
        result_files = (
            list(self.result_dir.glob("*")) if self.result_dir.exists() else []
        )

        image_size = sum(f.stat().st_size for f in image_files if f.is_file())
        result_size = sum(f.stat().st_size for f in result_files if f.is_file())
        total_size_mb = (image_size + result_size) / (1024 * 1024)

        return {
            "n_images": len(image_files),
            "n_results": len(result_files),
            "total_size_mb": round(total_size_mb, 2),
            "image_dir": str(self.image_dir),
            "result_dir": str(self.result_dir),
        }

    def clear_cache(
        self,
        images: bool = True,
        results: bool = True,
        confirm: bool = True,
    ) -> dict:
        """Clear cached files for this observation.

        Parameters
        ----------
        images : bool
            Clear downloaded FITS images from image_dir.
        results : bool
            Clear result files from result_dir.
        confirm : bool
            If True, requires user confirmation before deletion.

        Returns
        -------
        dict
            Dictionary with counts of deleted files:
            - images_deleted: Number of FITS files removed
            - results_deleted: Number of result files removed
        """
        if confirm:
            msg = f"Delete cache for observation {self.observation_id}?"
            if images and results:
                msg += f"\n  - {self.image_dir} (images)"
                msg += f"\n  - {self.result_dir} (results)"
            elif images:
                msg += f"\n  - {self.image_dir} (images only)"
            elif results:
                msg += f"\n  - {self.result_dir} (results only)"
            msg += "\nProceed? (y/n): "

            response = input(msg)
            if response.lower() != "y":
                self.logger.info("Cache clear cancelled.")
                return {"images_deleted": 0, "results_deleted": 0}

        images_deleted = 0
        results_deleted = 0

        if images and self.image_dir.exists():
            image_files = list(self.image_dir.glob("*.fits"))
            for f in image_files:
                f.unlink()
                images_deleted += 1
            self.logger.info(
                "Deleted %d images from %s", images_deleted, self.image_dir
            )

        if results and self.result_dir.exists():
            result_files = list(self.result_dir.glob("*"))
            for f in result_files:
                if f.is_file():
                    f.unlink()
                    results_deleted += 1
            self.logger.info(
                "Deleted %d results from %s", results_deleted, self.result_dir
            )

        return {"images_deleted": images_deleted, "results_deleted": results_deleted}

    @staticmethod
    def cleanup_observation(
        observation_id: int,
        image_dir: str | Path = "soap_images",
        result_dir: str | Path = "soap_results",
        confirm: bool = True,
    ) -> dict:
        """Clean up all files for a specific observation (static method).

        Parameters
        ----------
        observation_id : int
            Observation ID to clean up.
        image_dir : str or Path
            Base directory for images.
        result_dir : str or Path
            Base directory for results.
        confirm : bool
            Require confirmation before deletion.

        Returns
        -------
        dict
            Deletion statistics.
        """
        import shutil

        img_path = Path(image_dir) / str(observation_id)
        res_path = Path(result_dir) / str(observation_id)

        if confirm:
            msg = f"Delete all files for observation {observation_id}?"
            if img_path.exists():
                msg += f"\n  - {img_path}"
            if res_path.exists():
                msg += f"\n  - {res_path}"
            msg += "\nProceed? (y/n): "

            response = input(msg)
            if response.lower() != "y":
                return {"images_deleted": 0, "results_deleted": 0}

        images_deleted = 0
        results_deleted = 0

        if img_path.exists():
            files = list(img_path.glob("*.fits"))
            images_deleted = len(files)
            shutil.rmtree(img_path)

        if res_path.exists():
            files = list(res_path.glob("*"))
            results_deleted = len([f for f in files if f.is_file()])
            shutil.rmtree(res_path)

        return {"images_deleted": images_deleted, "results_deleted": results_deleted}

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
            aperture_r = self._select_aperture(
                data_sub, objects, bkg_result.global_rms, image.gain, ext_result.fwhm
            )

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
        if save_path is None and self.config.debug_mode:
            debug_dir = Path(self.config.debug_dir) / str(self.observation_id)
            debug_dir.mkdir(parents=True, exist_ok=True)
            save_path = debug_dir / f"{img_path.stem}_debug.png"

        # Create the plot
        fig = create_debug_plot(
            image=image,
            bkg_result=bkg_result,
            source_coords=source_coords,
            aperture_radius=aperture_r,
            ref_coords=ref_coords,
            matched_mask=match_mask,
            save_path=save_path,
        )

        if show:
            plt.show()

        return fig
