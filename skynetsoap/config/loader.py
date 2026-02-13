"""Configuration loading and validation for SOAP pipeline."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CONFIG_DIR = Path(__file__).parent


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass
class SOAPConfig:
    """Holds all pipeline configuration."""

    # Extraction
    extraction_threshold: float = 1.5
    extraction_min_area: int = 5

    # Aperture
    aperture_mode: str = "fwhm_scaled"
    aperture_scale: float = 2.5
    aperture_fixed_radius: float = 5.0
    aperture_min_radius: float = 3.0
    aperture_max_radius: float = 20.0
    aperture_step: float = 0.5
    aperture_radii: list[float] = field(
        default_factory=lambda: [2.0, 3.0, 5.0, 7.0, 10.0]
    )
    aperture_keep_all: bool = False

    # Calibration
    calibration_sigma_clip: float = 3.0
    calibration_max_iter: int = 5
    calibration_match_radius_arcsec: float = 3.0
    calibration_default_cat_error: float = 0.03
    calibration_merge_tolerance_arcsec: float = 2.0

    # Forced photometry
    forced_photometry_enabled: bool = False
    forced_photometry_snr_threshold: float = 3.0
    forced_photometry_aperture_radius: float = 5.0

    # Astrometry
    astrometry_enabled: bool = False
    astrometry_timeout: int = 120

    # Debugging & Intermediate Products
    debug_mode: bool = False
    debug_dir: str = "soap_debug"
    save_intermediates: bool = False
    intermediates_dir: str = "soap_intermediates"

    # Logging
    log_level: str = "WARNING"

    # Raw TOML data for filters/catalogs
    filters: dict = field(default_factory=dict)
    catalogs: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> SOAPConfig:
        """Create a SOAPConfig from a flat or nested dictionary."""
        cfg = cls()

        ext = d.get("extraction", {})
        cfg.extraction_threshold = ext.get("threshold", cfg.extraction_threshold)
        cfg.extraction_min_area = ext.get("min_area", cfg.extraction_min_area)

        ap = d.get("aperture", {})
        cfg.aperture_mode = ap.get("mode", cfg.aperture_mode)
        cfg.aperture_scale = ap.get("scale", cfg.aperture_scale)
        cfg.aperture_fixed_radius = ap.get("fixed_radius", cfg.aperture_fixed_radius)
        cfg.aperture_min_radius = ap.get("min_radius", cfg.aperture_min_radius)
        cfg.aperture_max_radius = ap.get("max_radius", cfg.aperture_max_radius)
        cfg.aperture_step = ap.get("step", cfg.aperture_step)
        cfg.aperture_radii = ap.get("radii", cfg.aperture_radii)
        cfg.aperture_keep_all = ap.get("keep_all", cfg.aperture_keep_all)

        cal = d.get("calibration", {})
        cfg.calibration_sigma_clip = cal.get("sigma_clip", cfg.calibration_sigma_clip)
        cfg.calibration_max_iter = cal.get("max_iter", cfg.calibration_max_iter)
        cfg.calibration_match_radius_arcsec = cal.get(
            "match_radius_arcsec", cfg.calibration_match_radius_arcsec
        )
        cfg.calibration_default_cat_error = cal.get(
            "default_cat_error", cfg.calibration_default_cat_error
        )
        cfg.calibration_merge_tolerance_arcsec = cal.get(
            "merge_tolerance_arcsec", cfg.calibration_merge_tolerance_arcsec
        )

        fp = d.get("forced_photometry", {})
        cfg.forced_photometry_enabled = fp.get("enabled", cfg.forced_photometry_enabled)
        cfg.forced_photometry_snr_threshold = fp.get(
            "snr_threshold", cfg.forced_photometry_snr_threshold
        )
        cfg.forced_photometry_aperture_radius = fp.get(
            "aperture_radius", cfg.forced_photometry_aperture_radius
        )

        ast = d.get("astrometry", {})
        cfg.astrometry_enabled = ast.get("enabled", cfg.astrometry_enabled)
        cfg.astrometry_timeout = ast.get("timeout", cfg.astrometry_timeout)

        dbg = d.get("debug", {})
        cfg.debug_mode = dbg.get("enabled", cfg.debug_mode)
        cfg.debug_dir = dbg.get("debug_dir", cfg.debug_dir)
        cfg.save_intermediates = dbg.get("save_intermediates", cfg.save_intermediates)
        cfg.intermediates_dir = dbg.get("intermediates_dir", cfg.intermediates_dir)

        log = d.get("logging", {})
        cfg.log_level = log.get("level", cfg.log_level)

        cfg.filters = d.get("filters", cfg.filters)
        cfg.catalogs = d.get("catalogs", cfg.catalogs)

        return cfg


def load_config(
    user_config_path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> SOAPConfig:
    """Load default config, merge with user config and overrides.

    Parameters
    ----------
    user_config_path : path, optional
        Path to a user TOML file that overrides defaults.
    overrides : dict, optional
        Additional overrides applied last.

    Returns
    -------
    SOAPConfig
    """
    defaults = _load_toml(_CONFIG_DIR / "defaults.toml")
    filters = _load_toml(_CONFIG_DIR / "filters.toml")
    catalogs = _load_toml(_CONFIG_DIR / "catalogs.toml")

    merged = defaults.copy()
    merged["filters"] = filters
    merged["catalogs"] = catalogs

    if user_config_path is not None:
        user = _load_toml(Path(user_config_path))
        merged = _deep_merge(merged, user)

    if overrides is not None:
        merged = _deep_merge(merged, overrides)

    return SOAPConfig.from_dict(merged)
