"""Unit tests for skynetsoap.extraction.background."""

import numpy as np
import pytest

from skynetsoap.extraction.background import estimate_background


class TestEstimateBackground:
    """Tests for background estimation."""

    def test_estimate_background_shape(self):
        """Verify background and rms arrays match input shape."""
        rng = np.random.default_rng(42)
        data = (rng.normal(loc=100.0, scale=5.0, size=(256, 256))).astype(np.float32)

        result = estimate_background(data)

        assert result.background.shape == data.shape
        assert result.rms.shape == data.shape

    def test_estimate_background_values(self):
        """Verify global_back is close to the injected background mean."""
        rng = np.random.default_rng(42)
        sky_level = 200.0
        data = (rng.normal(loc=sky_level, scale=5.0, size=(256, 256))).astype(
            np.float32
        )

        result = estimate_background(data)

        assert result.global_back == pytest.approx(sky_level, abs=5.0)

    def test_estimate_background_rms(self):
        """Verify global_rms is close to the injected noise standard deviation."""
        rng = np.random.default_rng(42)
        noise_sigma = 8.0
        data = (rng.normal(loc=100.0, scale=noise_sigma, size=(256, 256))).astype(
            np.float32
        )

        result = estimate_background(data)

        assert result.global_rms == pytest.approx(noise_sigma, abs=2.0)
