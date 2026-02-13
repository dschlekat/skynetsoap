"""File-based result caching for the SOAP pipeline."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from astropy.table import QTable

logger = logging.getLogger("soap")


class ResultCache:
    """Hash-based cache for QTable results stored as ECSV.

    Keys are derived from image paths and a config hash.
    """

    def __init__(self, cache_dir: str | Path = ".soap_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _make_key(image_path: str | Path, config_hash: str = "") -> str:
        raw = f"{Path(image_path).resolve()}:{config_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _path_for(self, key: str) -> Path:
        return self.cache_dir / f"{key}.ecsv"

    def has(self, image_path: str | Path, config_hash: str = "") -> bool:
        key = self._make_key(image_path, config_hash)
        return self._path_for(key).exists()

    def load(self, image_path: str | Path, config_hash: str = "") -> QTable | None:
        key = self._make_key(image_path, config_hash)
        p = self._path_for(key)
        if p.exists():
            logger.debug("Cache hit: %s", p)
            return QTable.read(p, format="ascii.ecsv")
        return None

    def save(self, image_path: str | Path, table: QTable, config_hash: str = "") -> None:
        key = self._make_key(image_path, config_hash)
        p = self._path_for(key)
        table.write(p, format="ascii.ecsv", overwrite=True)
        logger.debug("Cached: %s", p)

    def invalidate(self, image_path: str | Path, config_hash: str = "") -> None:
        key = self._make_key(image_path, config_hash)
        p = self._path_for(key)
        if p.exists():
            p.unlink()
            logger.debug("Invalidated cache: %s", p)
