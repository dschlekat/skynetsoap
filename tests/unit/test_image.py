"""Unit tests for skynetsoap.core.image."""

import numpy as np
import pytest

from skynetsoap.core.image import FITSImage


class TestFITSImage:
    """Tests for FITSImage loading and properties."""

    def test_load_fits_image(self, mock_fits_image):
        """Verify FITSImage.load returns valid image with correct properties."""
        img = FITSImage.load(mock_fits_image)

        assert img.path == mock_fits_image
        assert img.data is not None
        assert img.header is not None

    def test_fits_image_data_shape(self, mock_fits_image):
        """Verify shape matches expected 256x256."""
        img = FITSImage.load(mock_fits_image)

        assert img.shape == (256, 256)

    def test_fits_image_wcs(self, mock_fits_image):
        """Verify WCS works (pixel_to_world does not crash)."""
        img = FITSImage.load(mock_fits_image)

        assert img.has_wcs
        # Should not raise
        sky = img.wcs.pixel_to_world(128, 128)
        assert sky is not None

    def test_fits_image_properties(self, mock_fits_image):
        """Verify filter_name, gain, rdnoise, exptime match header values."""
        img = FITSImage.load(mock_fits_image)

        assert img.filter_name == "V"
        assert img.gain == pytest.approx(1.5)
        assert img.rdnoise == pytest.approx(5.0)
        assert img.exptime == pytest.approx(30.0)

    def test_fits_image_no_wcs(self, mock_fits_no_wcs):
        """Verify has_wcs is False for image without WCS headers."""
        img = FITSImage.load(mock_fits_no_wcs)

        assert not img.has_wcs

    def test_fits_image_mjd(self, mock_fits_image):
        """Verify mjd property works."""
        img = FITSImage.load(mock_fits_image)

        mjd = img.mjd
        assert np.isfinite(mjd)
        assert mjd > 0
