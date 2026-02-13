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
