"""Shared photometric-system mappings loaded from filters.toml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import tomllib

_CONFIG_DIR = Path(__file__).parent
_FILTERS_TOML = _CONFIG_DIR / "filters.toml"


@lru_cache(maxsize=1)
def _filters_config() -> dict:
    with open(_FILTERS_TOML, "rb") as f:
        return tomllib.load(f)


def get_filter_aliases(filters_config: dict | None = None) -> dict[str, str]:
    cfg = _filters_config() if filters_config is None else filters_config
    return dict(cfg.get("aliases", {}))


def canonicalize_filter_band(
    filter_band: str, filters_config: dict | None = None
) -> str:
    aliases = get_filter_aliases(filters_config)
    return aliases.get(filter_band, filter_band)


def get_band_mag_system_map(filters_config: dict | None = None) -> dict[str, str]:
    cfg = _filters_config() if filters_config is None else filters_config
    return dict(cfg.get("photometry", {}).get("band_mag_system", {}))


def infer_mag_system_for_filter(
    filter_band: str, filters_config: dict | None = None
) -> str:
    canonical_band = canonicalize_filter_band(filter_band, filters_config)
    system_map = get_band_mag_system_map(filters_config)
    mag_system = system_map.get(canonical_band, "Unknown")
    if mag_system in {"AB", "Vega"}:
        return mag_system
    return "Unknown"


def get_ab_minus_vega_offsets() -> dict[str, float]:
    cfg = _filters_config()
    raw = cfg.get("photometry", {}).get("ab_minus_vega_offsets", {})
    return {str(k): float(v) for k, v in raw.items()}
