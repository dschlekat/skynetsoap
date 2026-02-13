"""Integration tests for the SOAP photometry pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord

from skynetsoap import Soap, PhotometryResult, PhotometryTarget


# ---------------------------------------------------------------------------
# Mock calibrator
# ---------------------------------------------------------------------------
class MockCalibrator:
    """Calibrator stub that returns a fixed zeropoint."""

    def __init__(self, zp: float = 25.0, zp_err: float = 0.01):
        self.zp = zp
        self.zp_err = zp_err

    def get_reference_catalog(
        self, center, radius_arcmin: float = 10.0, filter_band: str = "V"
    ):
        return pd.DataFrame()

    def calibrate_image(
        self, image, ins_mags, ins_mag_errs, source_coords, filter_band
    ):
        match_mask = np.ones(len(ins_mags), dtype=bool)
        return self.zp, self.zp_err, match_mask


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipelineEndToEnd:
    """Full pipeline integration test with mock data and mock calibrator."""

    def test_pipeline_end_to_end(self, mock_fits_image, tmp_path, soap_config):
        """Run the pipeline on a synthetic image and verify results."""
        s = Soap(
            observation_id=99999,
            config=soap_config,
            calibrator=MockCalibrator(zp=25.0, zp_err=0.01),
            image_dir=str(tmp_path / "images"),
            result_dir=str(tmp_path / "results"),
        )

        # Copy the mock image into Soap's image_dir
        shutil.copy(mock_fits_image, s.image_dir / "test_image.fits")

        result = s.run()

        # Result is a PhotometryResult with at least one measurement
        assert isinstance(result, PhotometryResult)
        assert len(result) > 0

        # All expected columns are present
        expected_columns = [
            "image_file",
            "telescope",
            "filter",
            "exptime",
            "mjd",
            "jd",
            "x_pix",
            "y_pix",
            "ra",
            "dec",
            "flux",
            "flux_err",
            "snr",
            "ins_mag",
            "ins_mag_err",
            "calibrated_mag",
            "calibrated_mag_err",
            "zeropoint",
            "zeropoint_err",
            "aperture_radius",
            "fwhm",
            "n_cal_stars",
            "is_forced",
            "flag",
        ]
        for col in expected_columns:
            assert col in result.table.colnames, f"Missing column: {col}"

        # Calibrated magnitude should be ins_mag + ZP (=25.0)
        ins_mag = np.array(result.table["ins_mag"])
        cal_mag = np.array(result.table["calibrated_mag"])
        np.testing.assert_allclose(cal_mag, ins_mag + 25.0, atol=0.01)

        # Limiting magnitude should be fainter (numerically larger)
        # than all measured/calibrated stellar magnitudes.
        lim_mag = np.array(result.table["limiting_mag"], dtype=float)
        finite = np.isfinite(cal_mag) & np.isfinite(lim_mag)
        assert np.any(finite)
        assert np.all(lim_mag[finite] > cal_mag[finite])

        # Export produces a CSV file
        csv_path = s.export(format="csv")
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0


class TestPipelineNoImages:
    """Pipeline should handle an empty image directory gracefully."""

    def test_pipeline_no_images(self, tmp_path, soap_config):
        """Run the pipeline with no images; expect an empty result."""
        s = Soap(
            observation_id=99999,
            config=soap_config,
            calibrator=MockCalibrator(),
            image_dir=str(tmp_path / "images"),
            result_dir=str(tmp_path / "results"),
        )

        result = s.run()

        assert isinstance(result, PhotometryResult)
        assert len(result) == 0


class TestPipelineSkipsNoWCS:
    """Pipeline should skip images that lack WCS headers."""

    def test_pipeline_skips_no_wcs(self, mock_fits_image_no_wcs, tmp_path, soap_config):
        """An image without WCS is skipped; result is empty."""
        s = Soap(
            observation_id=99999,
            config=soap_config,
            calibrator=MockCalibrator(),
            image_dir=str(tmp_path / "images"),
            result_dir=str(tmp_path / "results"),
        )

        shutil.copy(mock_fits_image_no_wcs, s.image_dir / "nowcs.fits")

        result = s.run()

        assert isinstance(result, PhotometryResult)
        assert len(result) == 0


class TestTargetExtraction:
    """Target extraction from a pre-populated PhotometryResult."""

    def test_pipeline_target_extraction(self, photometry_result):
        """extract_target returns measurements near the given coordinate."""
        # The sample measurements are at roughly RA=51.95, Dec=74.66
        target_coord = SkyCoord(ra=51.950, dec=74.660, unit="deg")

        target = photometry_result.extract_target(target_coord, radius_arcsec=5.0)

        assert isinstance(target, PhotometryTarget)
        assert len(target) >= 1

        # Every returned measurement should be within the match radius
        source_coords = SkyCoord(
            ra=np.array(target.table["ra"]),
            dec=np.array(target.table["dec"]),
            unit="deg",
        )
        separations = source_coords.separation(target_coord).arcsec
        assert np.all(separations < 5.0)


class TestFilterByBand:
    """Filtering by photometric band."""

    def test_pipeline_filter_by_band(self, photometry_result):
        """filter_by_band returns only rows with the requested filter."""
        v_result = photometry_result.filter_by_band("V")

        assert len(v_result) > 0
        for row in v_result.table:
            assert row["filter"] == "V"


class TestExportFormats:
    """Export methods produce non-empty files."""

    @pytest.mark.parametrize(
        "fmt,ext",
        [
            ("csv", ".csv"),
            ("ecsv", ".ecsv"),
            ("json", ".json"),
        ],
    )
    def test_pipeline_export_formats(self, photometry_result, tmp_path, fmt, ext):
        """to_csv, to_ecsv, and to_json each create a non-empty output file."""
        out_path = tmp_path / f"output{ext}"
        export_fn = getattr(photometry_result, f"to_{fmt}")
        returned_path = export_fn(out_path)

        assert Path(returned_path).exists()
        assert Path(returned_path).stat().st_size > 0


class TestMultiAperturePhotometry:
    """Multi-aperture photometry tests."""

    def test_multi_aperture_mode(self, mock_fits_image, tmp_path, soap_config):
        """Run pipeline in multi-aperture mode and verify N rows per source."""
        # Set multi-aperture mode with 3 radii
        soap_config.aperture_mode = "multi"
        soap_config.aperture_radii = [3.0, 5.0, 7.0]
        soap_config.aperture_keep_all = True

        s = Soap(
            observation_id=99999,
            config=soap_config,
            calibrator=MockCalibrator(zp=25.0, zp_err=0.01),
            image_dir=str(tmp_path / "images"),
            result_dir=str(tmp_path / "results"),
        )

        shutil.copy(mock_fits_image, s.image_dir / "test_image.fits")

        result = s.run()

        # Verify we have measurements
        assert len(result) > 0

        # Verify aperture_id column exists and has multiple values
        aperture_ids = np.unique(result.table["aperture_id"])
        assert len(aperture_ids) == 3
        assert set(aperture_ids) == {0, 1, 2}

        # Each source should have 3 measurements (one per aperture)
        # Group by (image_file, x_pix, y_pix) and count
        unique_sources = set()
        for row in result.table:
            unique_sources.add((row["image_file"], row["x_pix"], row["y_pix"]))

        # Total measurements should be num_sources * num_apertures
        expected_total = len(unique_sources) * 3
        assert len(result) == expected_total

    def test_get_best_aperture_integration(
        self, mock_fits_image, tmp_path, soap_config
    ):
        """Verify get_best_aperture reduces multi-aperture results."""
        # Set multi-aperture mode
        soap_config.aperture_mode = "multi"
        soap_config.aperture_radii = [3.0, 5.0, 7.0]
        soap_config.aperture_keep_all = True

        s = Soap(
            observation_id=99999,
            config=soap_config,
            calibrator=MockCalibrator(zp=25.0, zp_err=0.01),
            image_dir=str(tmp_path / "images"),
            result_dir=str(tmp_path / "results"),
        )

        shutil.copy(mock_fits_image, s.image_dir / "test_image.fits")

        result = s.run()
        assert len(result) > 0

        # Get the number of unique sources
        unique_sources = set()
        for row in result.table:
            unique_sources.add((row["image_file"], row["x_pix"], row["y_pix"]))
        n_sources = len(unique_sources)

        # Apply get_best_aperture
        best = result.get_best_aperture(criterion="snr")

        # Should have exactly n_sources measurements (one per source)
        assert len(best) == n_sources

        # All aperture_ids should be valid
        assert all(0 <= aid <= 2 for aid in best.table["aperture_id"])

    def test_filter_by_aperture_integration(
        self, mock_fits_image, tmp_path, soap_config
    ):
        """Verify filter_by_aperture returns correct subset."""
        soap_config.aperture_mode = "multi"
        soap_config.aperture_radii = [3.0, 5.0, 7.0]
        soap_config.aperture_keep_all = True

        s = Soap(
            observation_id=99999,
            config=soap_config,
            calibrator=MockCalibrator(zp=25.0, zp_err=0.01),
            image_dir=str(tmp_path / "images"),
            result_dir=str(tmp_path / "results"),
        )

        shutil.copy(mock_fits_image, s.image_dir / "test_image.fits")

        result = s.run()
        assert len(result) > 0

        # Filter for aperture_id = 1
        filtered = result.filter_by_aperture(1)

        # All results should have aperture_id = 1
        assert all(row["aperture_id"] == 1 for row in filtered.table)

        # Should have approximately 1/3 of the total measurements
        assert len(filtered) > 0
        assert len(filtered) < len(result)


class TestForcedPhotometry:
    """Forced photometry tests."""

    def test_forced_photometry(
        self, mock_fits_image, tmp_path, soap_config, known_sources
    ):
        """Run pipeline with forced photometry at known positions."""
        s = Soap(
            observation_id=99999,
            config=soap_config,
            calibrator=MockCalibrator(zp=25.0, zp_err=0.01),
            image_dir=str(tmp_path / "images"),
            result_dir=str(tmp_path / "results"),
        )

        shutil.copy(mock_fits_image, s.image_dir / "test_image.fits")

        # Use first 3 injected source positions for forced photometry
        forced_coords = [known_sources.sky_coords[i] for i in range(3)]

        result = s.run(forced_positions=forced_coords)

        # Verify we have measurements
        assert len(result) > 0

        # Verify is_forced column exists
        assert "is_forced" in result.table.colnames

        # Count forced photometry measurements
        n_forced = sum(result.table["is_forced"])
        assert n_forced == 3

        # Verify forced measurements are at correct positions
        forced_mask = result.table["is_forced"]
        forced_result = result.table[forced_mask]

        for i, coord in enumerate(forced_coords):
            # Find the measurement closest to this coordinate
            forced_coords_result = SkyCoord(
                ra=forced_result["ra"],
                dec=forced_result["dec"],
                unit="deg",
            )
            seps = forced_coords_result.separation(coord).arcsec

            # At least one forced measurement should be very close to the input position
            assert np.min(seps) < 1.0  # Within 1 arcsec

    def test_forced_photometry_with_limiting_mag(
        self, mock_fits_image, tmp_path, soap_config, known_sources
    ):
        """Verify forced photometry includes limiting magnitude."""
        s = Soap(
            observation_id=99999,
            config=soap_config,
            calibrator=MockCalibrator(zp=25.0, zp_err=0.01),
            image_dir=str(tmp_path / "images"),
            result_dir=str(tmp_path / "results"),
        )

        shutil.copy(mock_fits_image, s.image_dir / "test_image.fits")

        # Use first injected source position
        forced_coords = [known_sources.sky_coords[0]]

        result = s.run(forced_positions=forced_coords)

        # Get forced measurements
        forced_mask = result.table["is_forced"]
        forced_result = result.table[forced_mask]

        assert len(forced_result) > 0

        # Verify limiting_mag is not NaN
        assert all(~np.isnan(forced_result["limiting_mag"]))

        # Limiting magnitude should be fainter than calibrated magnitude
        for row in forced_result:
            if not np.isnan(row["calibrated_mag"]):
                assert row["limiting_mag"] > row["calibrated_mag"]


class TestLimitingMagnitude:
    """Limiting magnitude calculation tests."""

    def test_limiting_magnitude_calculated(
        self, mock_fits_image, tmp_path, soap_config
    ):
        """Verify limiting magnitude is calculated for all measurements."""
        s = Soap(
            observation_id=99999,
            config=soap_config,
            calibrator=MockCalibrator(zp=25.0, zp_err=0.01),
            image_dir=str(tmp_path / "images"),
            result_dir=str(tmp_path / "results"),
        )

        shutil.copy(mock_fits_image, s.image_dir / "test_image.fits")

        result = s.run()

        assert len(result) > 0

        # Verify limiting_mag column exists
        assert "limiting_mag" in result.table.colnames

        # All non-forced measurements should have limiting_mag
        normal_mask = ~result.table["is_forced"]
        normal_result = result.table[normal_mask]

        if len(normal_result) > 0:
            # All should have limiting_mag (may be NaN if zeropoint failed)
            assert all(~np.isnan(normal_result["limiting_mag"]))

            # Limiting magnitude should be fainter than detected magnitudes
            for row in normal_result:
                if not np.isnan(row["calibrated_mag"]):
                    assert row["limiting_mag"] > row["calibrated_mag"]

    def test_limiting_magnitude_default_method_is_analytic(
        self, mock_fits_image, tmp_path, soap_config
    ):
        """Default config should report analytic limiting magnitude as primary."""
        s = Soap(
            observation_id=99999,
            config=soap_config,
            calibrator=MockCalibrator(zp=25.0, zp_err=0.01),
            image_dir=str(tmp_path / "images"),
            result_dir=str(tmp_path / "results"),
        )
        shutil.copy(mock_fits_image, s.image_dir / "test_image.fits")

        result = s.run()
        assert len(result) > 0
        assert "limiting_mag_analytic" in result.table.colnames
        assert "limiting_mag_robust" in result.table.colnames

        np.testing.assert_allclose(
            np.array(result.table["limiting_mag"], dtype=float),
            np.array(result.table["limiting_mag_analytic"], dtype=float),
            equal_nan=True,
        )
        assert np.all(
            np.isnan(np.array(result.table["limiting_mag_robust"], dtype=float))
        )

    def test_limiting_magnitude_robust_method_selection(
        self, mock_fits_image, tmp_path, soap_config
    ):
        """Robust mode should populate robust limit and select it as primary."""
        soap_config.limiting_mag_method = "robust"
        soap_config.limiting_mag_robust_n_samples = 250
        soap_config.limiting_mag_robust_mask_dilate_pixels = 2
        soap_config.limiting_mag_robust_edge_buffer_pixels = 10
        soap_config.limiting_mag_robust_sigma_estimator = "mad"
        soap_config.limiting_mag_robust_max_draws_multiplier = 20
        soap_config.limiting_mag_robust_random_seed = 11

        s = Soap(
            observation_id=99999,
            config=soap_config,
            calibrator=MockCalibrator(zp=25.0, zp_err=0.01),
            image_dir=str(tmp_path / "images"),
            result_dir=str(tmp_path / "results"),
        )
        shutil.copy(mock_fits_image, s.image_dir / "test_image.fits")

        result = s.run()
        assert len(result) > 0

        robust_vals = np.array(result.table["limiting_mag_robust"], dtype=float)
        assert np.any(np.isfinite(robust_vals))
        np.testing.assert_allclose(
            np.array(result.table["limiting_mag"], dtype=float),
            robust_vals,
            equal_nan=True,
        )

        diagnostics = result.table.meta.get("limiting_mag_diagnostics", [])
        assert len(diagnostics) >= 1
        assert diagnostics[0]["method"] == "robust"

    def test_limiting_magnitude_robust_uses_pipeline_apertures(
        self, mock_fits_image, tmp_path, soap_config
    ):
        """Robust method should use the pipeline photometry apertures."""
        soap_config.aperture_mode = "multi"
        soap_config.aperture_radii = [3.0, 7.0]
        soap_config.aperture_keep_all = True
        soap_config.limiting_mag_method = "robust"
        soap_config.limiting_mag_robust_n_samples = 200
        soap_config.limiting_mag_robust_mask_dilate_pixels = 2
        soap_config.limiting_mag_robust_edge_buffer_pixels = 10
        soap_config.limiting_mag_robust_sigma_estimator = "mad"
        soap_config.limiting_mag_robust_max_draws_multiplier = 20
        soap_config.limiting_mag_robust_random_seed = 13

        s = Soap(
            observation_id=99999,
            config=soap_config,
            calibrator=MockCalibrator(zp=25.0, zp_err=0.01),
            image_dir=str(tmp_path / "images"),
            result_dir=str(tmp_path / "results"),
        )
        shutil.copy(mock_fits_image, s.image_dir / "test_image.fits")

        result = s.run()
        assert len(result) > 0

        diagnostics = result.table.meta.get("limiting_mag_diagnostics", [])
        non_forced = [d for d in diagnostics if not d.get("is_forced", False)]
        assert len(non_forced) >= 2

        diag_radii = sorted(
            {
                round(float(d["aperture_radius_pixels"]), 2)
                for d in non_forced
                if np.isfinite(d["aperture_radius_pixels"])
            }
        )
        assert diag_radii == [3.0, 7.0]
