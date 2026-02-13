"""Unit tests for robust limiting-magnitude utilities."""

from __future__ import annotations

import numpy as np

from skynetsoap.core.limiting_mag import compute_robust_limiting_magnitude


def _inject_gaussian(
    data: np.ndarray, x0: float, y0: float, amp: float, sigma: float
) -> np.ndarray:
    yy, xx = np.mgrid[0 : data.shape[0], 0 : data.shape[1]]
    return data + amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma**2))


class TestRobustLimitingMagnitude:
    """Tests for blank-sky aperture limiting-magnitude estimation."""

    def test_robust_method_returns_finite_value(self):
        """Robust method should return finite values on a simple synthetic image."""
        rng = np.random.default_rng(123)
        data = rng.normal(0.0, 3.0, (128, 128)).astype(np.float32)
        data = _inject_gaussian(data, 40.0, 40.0, amp=200.0, sigma=2.0)
        data = _inject_gaussian(data, 90.0, 70.0, amp=180.0, sigma=2.5)

        out = compute_robust_limiting_magnitude(
            data_sub=data,
            zeropoint=25.0,
            aperture_radius_pixels=4.0,
            err=3.0,
            extraction_threshold=1.5,
            extraction_min_area=5,
            n_samples=400,
            mask_dilate_pixels=2,
            edge_buffer_pixels=10,
            sigma_estimator="mad",
            max_draws_multiplier=20,
            random_seed=5,
        )

        assert np.isfinite(out.limiting_mag)
        assert np.isfinite(out.sigma_ap)
        assert np.isfinite(out.flux_limit)
        assert out.n_samples_used > 0
        assert out.n_samples_used <= out.n_samples_requested

    def test_restricted_area_handles_insufficient_samples(self):
        """Large edge/mask restrictions should be handled gracefully."""
        rng = np.random.default_rng(456)
        data = rng.normal(0.0, 2.0, (64, 64)).astype(np.float32)
        data = _inject_gaussian(data, 32.0, 32.0, amp=500.0, sigma=6.0)

        out = compute_robust_limiting_magnitude(
            data_sub=data,
            zeropoint=25.0,
            aperture_radius_pixels=5.0,
            err=2.0,
            extraction_threshold=1.5,
            extraction_min_area=5,
            n_samples=200,
            mask_dilate_pixels=4,
            edge_buffer_pixels=30,
            sigma_estimator="mad",
            max_draws_multiplier=10,
            random_seed=7,
        )

        assert out.n_samples_used < out.n_samples_requested
        assert len(out.warnings) > 0
