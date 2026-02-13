"""Unit tests for skynetsoap.core.result."""

from astropy.coordinates import SkyCoord

from skynetsoap.core.result import PhotometryResult, PhotometryTarget


class TestPhotometryResult:
    """Tests for PhotometryResult."""

    def test_empty_result(self):
        """Verify len == 0 for a fresh result."""
        result = PhotometryResult()

        assert len(result) == 0

    def test_add_measurement(self):
        """Verify len increases by 1 after adding a measurement."""
        result = PhotometryResult()
        result.add_measurement(
            ra=51.95,
            dec=74.66,
            flux=1000.0,
            flux_err=10.0,
            snr=100.0,
            ins_mag=-10.0,
            ins_mag_err=0.01,
            calibrated_mag=15.0,
            calibrated_mag_err=0.02,
            filter="V",
            mjd=60000.0,
            jd=2460000.5,
            image_file="test.fits",
        )

        assert len(result) == 1

    def test_filter_by_band(self, sample_result):
        """Verify only matching band is returned."""
        # Add measurements with different bands
        result = PhotometryResult()
        for band in ["V", "V", "B", "R", "V"]:
            result.add_measurement(
                filter=band,
                ra=51.95,
                dec=74.66,
                mjd=60000.0,
                jd=2460000.5,
            )

        filtered = result.filter_by_band("V")

        assert len(filtered) == 3
        assert all(row["filter"] == "V" for row in filtered.table)

    def test_sort_by_time(self):
        """Verify mjd values are sorted after sort_by_time."""
        result = PhotometryResult()
        mjds = [60003.0, 60001.0, 60005.0, 60002.0]
        for mjd in mjds:
            result.add_measurement(
                mjd=mjd,
                jd=mjd + 2400000.5,
                ra=51.95,
                dec=74.66,
            )

        sorted_result = result.sort_by_time()
        sorted_mjds = sorted_result.table["mjd"].value

        assert list(sorted_mjds) == sorted(mjds)

    def test_extract_target(self):
        """Verify returns PhotometryTarget with correct measurements."""
        result = PhotometryResult()
        # Add a measurement near target
        result.add_measurement(
            ra=51.95,
            dec=74.66,
            mjd=60000.0,
            jd=2460000.5,
            flux=1000.0,
            image_file="img1.fits",
        )
        # Add a measurement far away
        result.add_measurement(
            ra=100.0,
            dec=10.0,
            mjd=60001.0,
            jd=2460001.5,
            flux=2000.0,
            image_file="img2.fits",
        )

        coord = SkyCoord(ra=51.95, dec=74.66, unit="deg")
        target = result.extract_target(coord, radius_arcsec=5.0)

        assert isinstance(target, PhotometryTarget)
        assert len(target) == 1

    def test_export_csv(self, tmp_path):
        """Verify CSV file is created with correct content."""
        result = PhotometryResult()
        result.add_measurement(
            ra=51.95,
            dec=74.66,
            filter="V",
            calibrated_mag=15.0,
            mjd=60000.0,
            jd=2460000.5,
        )

        out = tmp_path / "test_output.csv"
        result.to_csv(out)

        assert out.exists()
        text = out.read_text()
        assert "ra" in text
        assert "dec" in text

    def test_export_ecsv(self, tmp_path):
        """Verify ECSV file is created."""
        result = PhotometryResult()
        result.add_measurement(
            ra=51.95,
            dec=74.66,
            mjd=60000.0,
            jd=2460000.5,
        )

        out = tmp_path / "test_output.ecsv"
        result.to_ecsv(out)

        assert out.exists()
        text = out.read_text()
        # ECSV files have a header with format metadata
        assert "ecsv" in text.lower() or "datatype" in text.lower()

    def test_to_pandas(self):
        """Verify returns DataFrame with correct columns."""
        import pandas as pd

        result = PhotometryResult()
        result.add_measurement(
            ra=51.95,
            dec=74.66,
            flux=1000.0,
            filter="V",
            mjd=60000.0,
            jd=2460000.5,
        )

        df = result.to_pandas()

        assert isinstance(df, pd.DataFrame)
        assert "ra" in df.columns
        assert "dec" in df.columns
        assert "flux" in df.columns
        assert len(df) == 1
