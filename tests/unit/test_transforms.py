"""Unit tests for skynetsoap.calibration.transforms."""

import numpy as np
import pytest

from skynetsoap.calibration.transforms import apply_transform


class TestApplyTransform:
    """Tests for photometric transformation."""

    def test_color_plus_offset(self):
        """Verify formula: result = mag1 + a*(mag1-mag2) + b."""
        mag1 = np.array([14.0, 15.0])
        mag2 = np.array([13.5, 14.8])
        err1 = np.array([0.01, 0.02])
        err2 = np.array([0.01, 0.02])
        a, b = 0.3, -0.05
        sigma_a, sigma_b = 0.0, 0.0  # zero coeff errors for exact check

        result, result_err = apply_transform(
            mag1,
            mag2,
            err1,
            err2,
            a,
            sigma_a,
            b,
            sigma_b,
            formula="color_plus_offset",
        )

        C = mag1 - mag2
        expected = mag1 + a * C + b
        np.testing.assert_allclose(result, expected)

    def test_color_only(self):
        """Verify formula: result = a*(mag1-mag2) + b."""
        mag1 = np.array([14.0, 15.0])
        mag2 = np.array([13.5, 14.8])
        err1 = np.array([0.01, 0.02])
        err2 = np.array([0.01, 0.02])
        a, b = 1.1, 0.02
        sigma_a, sigma_b = 0.0, 0.0

        result, result_err = apply_transform(
            mag1,
            mag2,
            err1,
            err2,
            a,
            sigma_a,
            b,
            sigma_b,
            formula="color_only",
        )

        C = mag1 - mag2
        expected = a * C + b
        np.testing.assert_allclose(result, expected)

    def test_unknown_formula(self):
        """Verify ValueError raised for unknown formula."""
        with pytest.raises(ValueError, match="Unknown formula type"):
            apply_transform(1.0, 2.0, 0.01, 0.01, 0.3, 0.01, 0.0, 0.01, formula="bad")

    def test_error_propagation(self):
        """Verify errors are non-negative and increase with input errors."""
        mag1 = np.array([14.0])
        mag2 = np.array([13.5])

        # Small errors
        _, err_small = apply_transform(
            mag1,
            mag2,
            np.array([0.01]),
            np.array([0.01]),
            0.3,
            0.01,
            0.0,
            0.01,
        )

        # Larger errors
        _, err_large = apply_transform(
            mag1,
            mag2,
            np.array([0.1]),
            np.array([0.1]),
            0.3,
            0.01,
            0.0,
            0.01,
        )

        assert err_small[0] >= 0
        assert err_large[0] > err_small[0]
