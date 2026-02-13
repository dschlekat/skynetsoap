"""Unit tests for skynetsoap.extraction.aperture."""

import numpy as np
import pytest

from skynetsoap.extraction.aperture import (
    compute_optimal_aperture,
    fwhm_scaled_radius,
    sum_circle,
)


def _make_source_image(shape=(128, 128), x0=64.0, y0=64.0, flux=10000.0, fwhm=4.0):
    """Create a simple image with a single Gaussian source."""
    rng = np.random.default_rng(42)
    sig = fwhm / 2.355
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    data = flux * np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / sig**2)
    data += rng.normal(loc=100.0, scale=5.0, size=shape)
    return data.astype(np.float32)


class TestSumCircle:
    """Tests for circular aperture summation."""

    def test_sum_circle_positive_flux(self):
        """Verify flux > 0 at source positions."""
        data = _make_source_image()
        # Subtract background
        data_sub = (data - np.median(data)).astype(np.float32)
        x = np.array([64.0])
        y = np.array([64.0])

        flux, flux_err, flag = sum_circle(data_sub, x, y, r=8.0, err=5.0)

        assert flux[0] > 0


class TestFwhmScaledRadius:
    """Tests for FWHM-scaled radius computation."""

    def test_fwhm_scaled_radius_normal(self):
        """Verify returns scale*fwhm/2 clamped to bounds."""
        fwhm = 6.0
        scale = 2.5
        expected = scale * fwhm / 2.0  # 7.5

        result = fwhm_scaled_radius(fwhm, scale=scale)

        assert result == pytest.approx(expected)

    def test_fwhm_scaled_radius_nan(self):
        """Verify returns min_r for NaN fwhm."""
        result = fwhm_scaled_radius(np.nan, min_r=3.0)

        assert result == 3.0

    def test_fwhm_scaled_radius_clamp(self):
        """Verify clamping to min_r and max_r."""
        # Very small FWHM -> should clamp to min_r
        result_low = fwhm_scaled_radius(0.5, scale=2.5, min_r=3.0, max_r=20.0)
        assert result_low == 3.0

        # Very large FWHM -> should clamp to max_r
        result_high = fwhm_scaled_radius(50.0, scale=2.5, min_r=3.0, max_r=20.0)
        assert result_high == 20.0


class TestComputeOptimalAperture:
    """Tests for optimal aperture search."""

    def test_compute_optimal_aperture(self):
        """Verify returns a value in [min_r, max_r]."""
        data = _make_source_image()
        data_sub = (data - np.median(data)).astype(np.float32)
        x = np.array([64.0])
        y = np.array([64.0])

        r = compute_optimal_aperture(data_sub, x, y, err=5.0, min_r=3.0, max_r=15.0)

        assert 3.0 <= r <= 15.0
