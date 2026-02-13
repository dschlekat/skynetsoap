"""SOAP v2 -- Field-wide photometry pipeline orchestrator."""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from tqdm import tqdm

from .config.loader import SOAPConfig, load_config
from .core.image import FITSImage
from .core.errors import ccd_magnitude_error
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
from .io.plotter import plot_lightcurve
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
    across all sources in a field. Target extraction is a post-processing
    step on the returned ``PhotometryResult``.

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

    Methods
    -------
    download(after=None, before=None, days_ago=None)
        Download FITS images from Skynet with optional date filters.
    run(images=None, after=None, before=None, days_ago=None)
        Run the full photometry pipeline on the specified images or all images in ``image_dir``.
    plot(units="calibrated_mag", path=None, show=False, **kwargs)
        Plot the light curve in the specified units.
    export(format="csv", path=None, **kwargs)
        Export the photometry results to a file in the specified format.
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
    ) -> PhotometryResult:
        """Run the full field-wide photometry pipeline.

        Parameters
        ----------
        images : list, optional
            Explicit list of FITS file paths. If None, uses all .fits
            files in ``image_dir``.
        after, before, days_ago
            Date filters (applied to image MJD after loading).

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
                self._process_single_image(img_path, result, ref_catalog_initialized)
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

        # Aperture selection
        aperture_r = self._select_aperture(
            data_sub, objects, bkg_result.global_rms, image.gain, ext_result.fwhm
        )

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
            return

        flux = flux[valid]
        flux_err = flux_err[valid]
        flag = flag[valid]
        x = objects["x"][valid]
        y = objects["y"][valid]

        # Instrumental magnitudes
        ins_mag = -2.5 * np.log10(flux)
        ins_mag_err = 1.0857 * (flux_err / flux)

        # Convert to sky coordinates
        world_coords = image.wcs.pixel_to_world(x, y)
        if not hasattr(world_coords, "__len__"):
            world_coords = SkyCoord([world_coords])

        # Calibration
        zp, zp_err, match_mask = self.calibrator.calibrate_image(
            image, ins_mag, ins_mag_err, world_coords, image.filter_name
        )

        if np.isnan(zp):
            cal_mag = np.full_like(ins_mag, np.nan)
            cal_mag_err = np.full_like(ins_mag_err, np.nan)
            n_cal = 0
        else:
            cal_mag = ins_mag + zp
            n_pix = np.pi * aperture_r**2
            n_bkgpix = np.pi * (3 * aperture_r) ** 2
            cal_mag_err = ccd_magnitude_error(
                flux=flux,
                gain=image.gain,
                n_pix=n_pix,
                background=bkg_result.global_back,
                rdnoise=image.rdnoise,
                n_bkgpix=n_bkgpix,
                sigma_bkg=bkg_result.global_rms,
                sigma_zp=zp_err,
            )
            cal_mag_err = np.atleast_1d(cal_mag_err)
            n_cal = int(np.sum(match_mask)) if match_mask is not None else 0

        snr = flux / flux_err

        # Append all sources to result
        for i in range(len(flux)):
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
                flux=float(flux[i]),
                flux_err=float(flux_err[i]),
                snr=float(snr[i]),
                ins_mag=float(ins_mag[i]),
                ins_mag_err=float(ins_mag_err[i]),
                calibrated_mag=float(cal_mag[i]),
                calibrated_mag_err=float(cal_mag_err[i]),
                zeropoint=float(zp) if not np.isnan(zp) else np.nan,
                zeropoint_err=float(zp_err) if not np.isnan(zp_err) else np.nan,
                aperture_radius=aperture_r,
                fwhm=ext_result.fwhm,
                n_cal_stars=n_cal,
                is_forced=False,
                flag=int(flag[i]),
            )

    def _select_aperture(
        self,
        data: np.ndarray,
        objects: np.ndarray,
        err: float,
        gain: float,
        fwhm: float,
    ) -> float:
        """Select aperture radius based on config mode."""
        mode = self.config.aperture_mode

        if mode == "fixed":
            return self.config.aperture_fixed_radius

        if mode == "optimal":
            return compute_optimal_aperture(
                data,
                objects["x"],
                objects["y"],
                err=err,
                gain=gain,
                min_r=self.config.aperture_min_radius,
                max_r=self.config.aperture_max_radius,
                step=self.config.aperture_step,
            )

        return fwhm_scaled_radius(
            fwhm,
            scale=self.config.aperture_scale,
            min_r=self.config.aperture_min_radius,
            max_r=self.config.aperture_max_radius,
        )

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
