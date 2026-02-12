"""Skynet API wrapper for downloading observation images.

Migrated from utils/skynet_api.py with logging and pathlib support.
"""

from __future__ import annotations

import logging
import os
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

        downloaded: list[Path] = []
        loop = tqdm(observation.exps, disable=not verbose)
        for exp in loop:
            loop.set_description(f"Downloading {exp.id}")

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

            filepath = self.download_request.get_fits(
                out_dir=str(path), reducequiet=1, image=f"r{exp.id}"
            )
            downloaded.append(Path(filepath))

        logger.info("Downloaded %d images to %s", len(downloaded), path)
        return downloaded
