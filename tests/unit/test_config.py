"""Unit tests for skynetsoap.config.loader."""

from skynetsoap.config.loader import SOAPConfig, _deep_merge, load_config


class TestLoadConfig:
    """Tests for configuration loading."""

    def test_load_default_config(self, default_config):
        """Verify load_config() returns SOAPConfig with expected defaults."""
        assert isinstance(default_config, SOAPConfig)
        assert default_config.extraction_threshold == 1.5
        assert default_config.extraction_min_area == 5
        assert default_config.aperture_mode == "optimal"
        assert default_config.calibration_sigma_clip == 3.0
        assert default_config.limiting_mag_method == "analytic"
        assert default_config.limiting_mag_robust_n_samples == 2000
        assert default_config.limiting_mag_robust_random_seed is None

    def test_config_from_dict(self):
        """Verify SOAPConfig.from_dict with custom values."""
        d = {
            "extraction": {"threshold": 3.0, "min_area": 10},
            "aperture": {"mode": "optimal", "scale": 3.0},
        }

        cfg = SOAPConfig.from_dict(d)

        assert cfg.extraction_threshold == 3.0
        assert cfg.extraction_min_area == 10
        assert cfg.aperture_mode == "optimal"
        assert cfg.aperture_scale == 3.0
        # Defaults should remain for unset values
        assert cfg.calibration_sigma_clip == 3.0

    def test_deep_merge(self):
        """Test _deep_merge with nested dicts."""
        base = {"a": 1, "b": {"c": 2, "d": 3}, "e": 5}
        override = {"b": {"c": 99}, "f": 6}

        result = _deep_merge(base, override)

        assert result["a"] == 1
        assert result["b"]["c"] == 99
        assert result["b"]["d"] == 3
        assert result["e"] == 5
        assert result["f"] == 6
        # Original dicts should be unchanged
        assert base["b"]["c"] == 2

    def test_config_has_filters_and_catalogs(self, default_config):
        """Verify filters and catalogs dicts are populated."""
        assert isinstance(default_config.filters, dict)
        assert isinstance(default_config.catalogs, dict)
        assert len(default_config.filters) > 0
        assert len(default_config.catalogs) > 0

    def test_config_override(self):
        """Verify overrides dict takes precedence."""
        cfg = load_config(overrides={"extraction": {"threshold": 5.0}})

        assert cfg.extraction_threshold == 5.0
