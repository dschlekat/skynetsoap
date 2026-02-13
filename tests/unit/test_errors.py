"""Unit tests for skynetsoap.core.errors."""

from skynetsoap.core.errors import ccd_magnitude_error


class TestCCDMagnitudeError:
    """Tests for the CCD magnitude error model."""

    def test_ccd_magnitude_error_positive(self):
        """Verify result is always positive for positive flux."""
        sigma = ccd_magnitude_error(
            flux=10000.0,
            gain=1.5,
            n_pix=100.0,
            background=200.0,
            rdnoise=5.0,
            n_bkgpix=5000.0,
            sigma_bkg=10.0,
            sigma_zp=0.02,
        )

        assert sigma > 0

    def test_ccd_magnitude_error_increases_with_noise(self):
        """Verify error increases when rdnoise increases."""
        common = dict(
            flux=10000.0,
            gain=1.5,
            n_pix=100.0,
            background=200.0,
            n_bkgpix=5000.0,
            sigma_bkg=10.0,
            sigma_zp=0.02,
        )

        err_low = ccd_magnitude_error(rdnoise=3.0, **common)
        err_high = ccd_magnitude_error(rdnoise=15.0, **common)

        assert err_high > err_low

    def test_ccd_magnitude_error_decreases_with_flux(self):
        """Verify error decreases for brighter sources."""
        common = dict(
            gain=1.5,
            n_pix=100.0,
            background=200.0,
            rdnoise=5.0,
            n_bkgpix=5000.0,
            sigma_bkg=10.0,
            sigma_zp=0.02,
        )

        err_faint = ccd_magnitude_error(flux=1000.0, **common)
        err_bright = ccd_magnitude_error(flux=50000.0, **common)

        assert err_faint > err_bright

    def test_ccd_magnitude_error_zp_contribution(self):
        """Verify error includes sigma_zp contribution."""
        common = dict(
            flux=50000.0,
            gain=1.5,
            n_pix=100.0,
            background=200.0,
            rdnoise=5.0,
            n_bkgpix=5000.0,
            sigma_bkg=10.0,
        )

        err_no_zp = ccd_magnitude_error(sigma_zp=0.0, **common)
        err_with_zp = ccd_magnitude_error(sigma_zp=0.1, **common)

        assert err_with_zp > err_no_zp
