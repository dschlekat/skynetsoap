"""Unit tests for skynetsoap.calibration.zeropoint."""

import numpy as np
import pytest

from skynetsoap.calibration.zeropoint import compute_zeropoint


class TestComputeZeropoint:
    """Tests for zeropoint computation."""

    def test_zeropoint_perfect(self):
        """With zero scatter (delta all equal), verify ZP matches exactly."""
        n = 10
        m_inst = np.full(n, -10.0)
        m_cat = np.full(n, 15.0)
        sigma_inst = np.full(n, 0.01)
        sigma_cat = np.full(n, 0.01)

        ZP, sigma_ZP, mask = compute_zeropoint(m_inst, m_cat, sigma_inst, sigma_cat)

        expected_zp = 25.0  # m_cat - m_inst
        assert ZP == pytest.approx(expected_zp, abs=1e-10)
        assert np.all(mask)

    def test_zeropoint_with_outlier(self):
        """Inject one outlier, verify sigma clipping removes it."""
        n = 30
        rng = np.random.default_rng(42)
        true_zp = 25.0
        m_inst = rng.uniform(-12, -8, n)
        m_cat = m_inst + true_zp + rng.normal(0, 0.02, n)
        sigma_inst = np.full(n, 0.03)
        sigma_cat = np.full(n, 0.03)

        # Inject outlier at index 0 with +1.5 mag offset (more realistic than +5.0)
        m_cat[0] = m_inst[0] + true_zp + 1.5

        ZP, sigma_ZP, mask = compute_zeropoint(
            m_inst, m_cat, sigma_inst, sigma_cat, sigma_clip=3.0
        )

        # Outlier should be clipped
        assert not mask[0]
        assert ZP == pytest.approx(true_zp, abs=0.1)

    def test_zeropoint_empty(self):
        """Verify returns NaN for empty arrays."""
        ZP, sigma_ZP, mask = compute_zeropoint(
            np.array([]), np.array([]), np.array([]), np.array([])
        )

        assert np.isnan(ZP)
        assert np.isnan(sigma_ZP)

    def test_zeropoint_uncertainty(self):
        """Verify sigma_ZP decreases with more stars."""
        true_zp = 25.0
        sigma = 0.05

        _, sigma_small, _ = compute_zeropoint(
            np.array([-10.0, -11.0]),
            np.array([15.0, 14.0]),
            np.full(2, sigma),
            np.full(2, sigma),
        )

        _, sigma_large, _ = compute_zeropoint(
            np.arange(-10, 0, dtype=float),
            np.arange(-10, 0, dtype=float) + true_zp,
            np.full(10, sigma),
            np.full(10, sigma),
        )

        assert sigma_large < sigma_small
