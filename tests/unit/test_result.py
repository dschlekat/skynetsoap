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

    def test_filter_by_aperture(self):
        """Verify filter_by_aperture returns only the specified aperture."""
        result = PhotometryResult()

        # Add measurements with different aperture IDs
        for aperture_id in [0, 1, 2, 0, 1]:
            result.add_measurement(
                ra=51.95,
                dec=74.66,
                flux=1000.0,
                filter="V",
                mjd=60000.0,
                jd=2460000.5,
                aperture_id=aperture_id,
            )

        # Filter for aperture_id = 1
        filtered = result.filter_by_aperture(1)

        assert len(filtered) == 2
        assert all(row["aperture_id"] == 1 for row in filtered.table)

    def test_get_best_aperture_snr(self):
        """Verify get_best_aperture selects the aperture with highest SNR."""
        result = PhotometryResult()

        # Add multiple aperture measurements for the same source in the same image
        # Aperture 0: SNR = 100
        result.add_measurement(
            image_file="test.fits",
            x_pix=100.0,
            y_pix=100.0,
            ra=51.95,
            dec=74.66,
            flux=10000.0,
            flux_err=100.0,
            snr=100.0,
            filter="V",
            mjd=60000.0,
            jd=2460000.5,
            aperture_id=0,
        )
        # Aperture 1: SNR = 150 (best)
        result.add_measurement(
            image_file="test.fits",
            x_pix=100.0,
            y_pix=100.0,
            ra=51.95,
            dec=74.66,
            flux=15000.0,
            flux_err=100.0,
            snr=150.0,
            filter="V",
            mjd=60000.0,
            jd=2460000.5,
            aperture_id=1,
        )
        # Aperture 2: SNR = 120
        result.add_measurement(
            image_file="test.fits",
            x_pix=100.0,
            y_pix=100.0,
            ra=51.95,
            dec=74.66,
            flux=12000.0,
            flux_err=100.0,
            snr=120.0,
            filter="V",
            mjd=60000.0,
            jd=2460000.5,
            aperture_id=2,
        )

        best = result.get_best_aperture(criterion="snr")

        # Should return only one measurement (the one with SNR=150)
        assert len(best) == 1
        assert best.table["snr"][0] == 150.0
        assert best.table["aperture_id"][0] == 1

    def test_get_best_aperture_flux(self):
        """Verify get_best_aperture selects the aperture with highest flux."""
        result = PhotometryResult()

        # Add multiple aperture measurements for the same source
        for aperture_id, flux in enumerate([10000.0, 8000.0, 12000.0]):
            result.add_measurement(
                image_file="test.fits",
                x_pix=100.0,
                y_pix=100.0,
                ra=51.95,
                dec=74.66,
                flux=flux,
                flux_err=100.0,
                snr=flux / 100.0,
                filter="V",
                mjd=60000.0,
                jd=2460000.5,
                aperture_id=aperture_id,
            )

        best = result.get_best_aperture(criterion="flux")

        # Should return only one measurement (the one with flux=12000)
        assert len(best) == 1
        assert best.table["flux"][0] == 12000.0
        assert best.table["aperture_id"][0] == 2

    def test_get_best_aperture_multi_image(self):
        """Verify get_best_aperture works correctly with multiple images."""
        result = PhotometryResult()

        # Image 1: 3 apertures, best SNR = 150 (aperture 1)
        for aperture_id, snr in enumerate([100.0, 150.0, 120.0]):
            result.add_measurement(
                image_file="img1.fits",
                x_pix=100.0,
                y_pix=100.0,
                ra=51.95,
                dec=74.66,
                flux=snr * 100.0,
                flux_err=100.0,
                snr=snr,
                filter="V",
                mjd=60000.0,
                jd=2460000.5,
                aperture_id=aperture_id,
            )

        # Image 2: 3 apertures, best SNR = 200 (aperture 2)
        for aperture_id, snr in enumerate([180.0, 190.0, 200.0]):
            result.add_measurement(
                image_file="img2.fits",
                x_pix=100.0,
                y_pix=100.0,
                ra=51.95,
                dec=74.66,
                flux=snr * 100.0,
                flux_err=100.0,
                snr=snr,
                filter="V",
                mjd=60001.0,
                jd=2460001.5,
                aperture_id=aperture_id,
            )

        best = result.get_best_aperture(criterion="snr")

        # Should return 2 measurements (one per image)
        assert len(best) == 2
        # Check that we got the best from each image
        assert 150.0 in best.table["snr"]
        assert 200.0 in best.table["snr"]

    def test_multi_aperture_pipeline(self):
        """Verify multi-aperture mode produces N rows per source."""
        # This is more of an integration test, but we can test the result structure
        result = PhotometryResult()

        n_apertures = 5
        n_sources = 3

        # Simulate multi-aperture photometry: each source gets N measurements
        for source_id in range(n_sources):
            for aperture_id in range(n_apertures):
                result.add_measurement(
                    image_file="test.fits",
                    x_pix=100.0 + source_id * 10,
                    y_pix=100.0 + source_id * 10,
                    ra=51.95 + source_id * 0.01,
                    dec=74.66 + source_id * 0.01,
                    flux=1000.0 * (aperture_id + 1),
                    flux_err=100.0,
                    snr=10.0 * (aperture_id + 1),
                    filter="V",
                    mjd=60000.0,
                    jd=2460000.5,
                    aperture_id=aperture_id,
                )

        # Total should be n_sources * n_apertures
        assert len(result) == n_sources * n_apertures

        # get_best_aperture should reduce to n_sources
        best = result.get_best_aperture(criterion="snr")
        assert len(best) == n_sources

    def test_limiting_magnitude(self):
        """Verify limiting_mag is calculated and stored correctly."""

        result = PhotometryResult()

        limiting_mag_value = 18.5

        result.add_measurement(
            ra=51.95,
            dec=74.66,
            flux=1000.0,
            filter="V",
            mjd=60000.0,
            jd=2460000.5,
            limiting_mag=limiting_mag_value,
        )

        assert len(result) == 1
        # Compare values, not Quantity objects
        assert float(result.table["limiting_mag"][0].value) == limiting_mag_value

    def test_forced_photometry_flag(self):
        """Verify is_forced flag is stored correctly."""
        result = PhotometryResult()

        # Add a normal detection
        result.add_measurement(
            ra=51.95,
            dec=74.66,
            flux=1000.0,
            filter="V",
            mjd=60000.0,
            jd=2460000.5,
            is_forced=False,
        )

        # Add a forced photometry measurement
        result.add_measurement(
            ra=51.96,
            dec=74.67,
            flux=100.0,
            filter="V",
            mjd=60000.0,
            jd=2460000.5,
            is_forced=True,
        )

        assert len(result) == 2
        assert not result.table["is_forced"][0]
        assert result.table["is_forced"][1]
