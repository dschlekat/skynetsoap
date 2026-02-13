"""Unit tests for skynetsoap.extraction.extractor."""

import numpy as np

from skynetsoap.extraction.extractor import compute_fwhm, extract_sources


def _make_image_with_sources(rng, shape=(256, 256), n_sources=10, sky=100.0, sigma=5.0):
    """Create a synthetic image with Gaussian sources."""
    data = rng.normal(loc=sky, scale=sigma, size=shape).astype(np.float32)
    ys = rng.uniform(30, shape[0] - 30, n_sources)
    xs = rng.uniform(30, shape[1] - 30, n_sources)
    fluxes = rng.uniform(5000, 20000, n_sources)

    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    for x0, y0, flux in zip(xs, ys, fluxes):
        fwhm = 4.0
        sig = fwhm / 2.355
        gauss = flux * np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / sig**2)
        data += gauss.astype(np.float32)

    return data, xs, ys


class TestExtractSources:
    """Tests for source extraction."""

    def test_extract_sources_finds_sources(self):
        """Verify n_sources > 0 on a mock image with injected sources."""
        rng = np.random.default_rng(42)
        data, _, _ = _make_image_with_sources(rng)
        data_sub = data - np.median(data)

        result = extract_sources(data_sub.astype(np.float32), err=5.0)

        assert result.n_sources > 0

    def test_extract_sources_positions(self):
        """Verify extracted positions are near injected positions (within 4 px)."""
        rng = np.random.default_rng(42)
        data, xs_true, ys_true = _make_image_with_sources(rng, n_sources=5)
        data_sub = data - np.median(data)

        result = extract_sources(data_sub.astype(np.float32), err=5.0)

        # For each injected source, check that at least one extracted source is nearby
        for xt, yt in zip(xs_true, ys_true):
            dists = np.sqrt(
                (result.objects["x"] - xt) ** 2 + (result.objects["y"] - yt) ** 2
            )
            assert np.min(dists) < 4.0, (
                f"No extracted source within 4 px of ({xt:.1f}, {yt:.1f})"
            )

    def test_extract_sources_fwhm(self):
        """Verify fwhm is finite and positive."""
        rng = np.random.default_rng(42)
        data, _, _ = _make_image_with_sources(rng)
        data_sub = data - np.median(data)

        result = extract_sources(data_sub.astype(np.float32), err=5.0)

        assert np.isfinite(result.fwhm)
        assert result.fwhm > 0

    def test_extract_sources_empty_image(self):
        """Verify n_sources == 0 on a uniform noise image."""
        rng = np.random.default_rng(42)
        data = rng.normal(loc=100.0, scale=5.0, size=(128, 128)).astype(np.float32)
        data_sub = data - np.median(data)

        result = extract_sources(data_sub, err=5.0, threshold=10.0)

        assert result.n_sources == 0


class TestComputeFwhm:
    """Tests for FWHM computation."""

    def test_compute_fwhm_empty(self):
        """Verify returns nan for empty objects array."""
        empty = np.array([])
        assert np.isnan(compute_fwhm(empty))
