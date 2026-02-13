"""Skynet API wrapper for downloading observation images.

Migrated from utils/skynet_api.py with logging and pathlib support.
"""

from __future__ import annotations

import logging
import os
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger("soap")


class SkynetAPI:
    """Download FITS images from Skynet."""

    def __init__(self, api_token: str | None = None):
        from skynetapi import ObservationRequest, DownloadRequest

        self.api_token = api_token or os.getenv("SKYNET_API_TOKEN")
        if not self.api_token:
            raise ValueError(
                "Skynet API token missing. Set SKYNET_API_TOKEN environment variable "
                "or pass api_token directly."
            )
        self.observation_request = ObservationRequest(token=self.api_token)
        self.download_request = DownloadRequest(token=self.api_token)

    def get_observation(self, observation_id: int):
        """Retrieve an observation by ID."""
        return self.observation_request.get(observation_id)

    @staticmethod
    def _manifest_path(path: Path) -> Path:
        return path / ".download_manifest.json"

    @classmethod
    def _load_manifest(cls, path: Path) -> dict[str, str]:
        manifest_path = cls._manifest_path(path)
        if not manifest_path.exists():
            return {}
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception:
            logger.debug("Could not read download manifest at %s", manifest_path)
        return {}

    @classmethod
    def _save_manifest(cls, path: Path, manifest: dict[str, str]) -> None:
        manifest_path = cls._manifest_path(path)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

    def download_images(
        self,
        observation,
        path: str | Path = "soap_images/",
        after: str | None = None,
        before: str | None = None,
        days_ago: float | None = None,
        verbose: bool = True,
    ) -> list[Path]:
        """Download FITS images for exposures within a date range.

        Parameters
        ----------
        observation
            Skynet observation object.
        path : str or Path
            Output directory.
        after, before : str, optional
            ISO datetime strings for filtering.
        days_ago : float, optional
            Download only images from this many days ago until now.
        verbose : bool
            Show progress bar.

        Returns
        -------
        list[Path]
            Paths to downloaded FITS files.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if days_ago is not None:
            after_dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
        elif after is not None:
            after_dt = datetime.fromisoformat(after)
        else:
            after_dt = None

        before_dt = datetime.fromisoformat(before) if before is not None else None

        eligible_exps = []
        for exp in observation.exps:
            if exp.center_time is None:
                logger.debug("Skipping exposure %s: not yet taken.", exp.id)
                continue
            try:
                center_time = datetime.fromisoformat(exp.center_time)
            except (TypeError, ValueError):
                logger.debug("Skipping exposure %s: invalid datetime.", exp.id)
                continue
            if after_dt is not None and center_time < after_dt:
                continue
            if before_dt is not None and center_time > before_dt:
                continue
            eligible_exps.append(exp)

        # Backward-compatible cache bootstrap for directories already populated
        # by earlier versions with non-deterministic filenames.
        manifest = self._load_manifest(path)
        existing_fits = sorted(path.glob("*.fits"))
        if not manifest and eligible_exps and len(existing_fits) >= len(eligible_exps):
            logger.info(
                "Using %d cached images already present in %s", len(existing_fits), path
            )
            return existing_fits

        downloaded: list[Path] = []
        skipped: int = 0
        manifest_changed = False
        loop = tqdm(eligible_exps, disable=not verbose)
        for exp in loop:
            exp_key = str(exp.id)

            # Primary cache check via persisted manifest.
            cached_name = manifest.get(exp_key)
            if cached_name:
                cached_path = path / cached_name
                if cached_path.exists():
                    logger.debug(
                        "Skipping %s: already cached as %s", exp.id, cached_name
                    )
                    downloaded.append(cached_path)
                    skipped += 1
                    loop.set_description(f"Cached {exp.id}")
                    continue

            # Check if file already exists (cache check)
            expected_filename = f"r{exp.id}.fits"
            expected_path = path / expected_filename
            if expected_path.exists():
                logger.debug("Skipping %s: already exists", expected_filename)
                downloaded.append(expected_path)
                skipped += 1
                if manifest.get(exp_key) != expected_filename:
                    manifest[exp_key] = expected_filename
                    manifest_changed = True
                loop.set_description(f"Cached {exp.id}")
                continue

            loop.set_description(f"Downloading {exp.id}")
            filepath = self.download_request.get_fits(
                out_dir=str(path), reducequiet=1, image=f"r{exp.id}"
            )
            file_path_obj = Path(filepath)
            downloaded.append(file_path_obj)
            if manifest.get(exp_key) != file_path_obj.name:
                manifest[exp_key] = file_path_obj.name
                manifest_changed = True

        if manifest_changed:
            self._save_manifest(path, manifest)

        if skipped > 0:
            logger.info(
                "Found %d cached images, downloaded %d new images to %s",
                skipped,
                len(downloaded) - skipped,
                path,
            )
        else:
            logger.info("Downloaded %d images to %s", len(downloaded), path)
        return downloaded
