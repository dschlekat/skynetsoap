"""Unit tests for skynetsoap.io.skynet_api."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from skynetsoap.io.skynet_api import SkynetAPI


def _make_api() -> SkynetAPI:
    api = SkynetAPI.__new__(SkynetAPI)
    api._token = "test-token"
    return api


def _obs_with_exp_ids(*ids: int):
    return SimpleNamespace(
        exps=[
            SimpleNamespace(id=i, center_time="2025-01-01T00:00:00+00:00") for i in ids
        ]
    )


def _fake_download_fits(file_by_image: dict[str, str] | None = None):
    """Returns a _download_fits replacement that writes stub files to disk."""
    file_by_image = file_by_image or {}

    def _impl(self, exp_id: int, out_dir: str) -> str:
        image_key = f"r{exp_id}"
        filename = file_by_image.get(image_key, f"{image_key}.fits")
        filepath = Path(out_dir) / filename
        filepath.write_text("fits", encoding="utf-8")
        return str(filepath)

    return _impl


class TestSkynetAPIDownloadCaching:
    def test_uses_existing_files_without_manifest_when_counts_match(self, tmp_path):
        # Simulate old behavior: files exist with non-r<ID>.fits names.
        cached_a = tmp_path / "obs_file_000.fits"
        cached_b = tmp_path / "obs_file_001.fits"
        cached_a.write_text("fits", encoding="utf-8")
        cached_b.write_text("fits", encoding="utf-8")

        api = _make_api()
        obs = _obs_with_exp_ids(1, 2)

        with patch.object(SkynetAPI, "_download_fits", _fake_download_fits()):
            files = api.download_images(obs, path=tmp_path, verbose=False)

        assert files == sorted([cached_a, cached_b])

    def test_skips_download_when_manifest_entry_exists(self, tmp_path):
        cached = tmp_path / "cached_name.fits"
        cached.write_text("fits", encoding="utf-8")
        manifest_path = tmp_path / ".download_manifest.json"
        manifest_path.write_text(json.dumps({"123": cached.name}), encoding="utf-8")

        api = _make_api()
        obs = _obs_with_exp_ids(123)

        calls: list[int] = []

        def _tracking_download(self, exp_id: int, out_dir: str) -> str:
            calls.append(exp_id)
            filepath = Path(out_dir) / f"r{exp_id}.fits"
            filepath.write_text("fits", encoding="utf-8")
            return str(filepath)

        with patch.object(SkynetAPI, "_download_fits", _tracking_download):
            files = api.download_images(obs, path=tmp_path, verbose=False)

        assert calls == []
        assert files == [cached]

    def test_downloads_only_missing_and_updates_manifest(self, tmp_path):
        cached = tmp_path / "cached_a.fits"
        cached.write_text("fits", encoding="utf-8")
        manifest_path = tmp_path / ".download_manifest.json"
        manifest_path.write_text(json.dumps({"10": cached.name}), encoding="utf-8")

        api = _make_api()
        obs = _obs_with_exp_ids(10, 20)

        calls: list[int] = []

        def _tracking_download(self, exp_id: int, out_dir: str) -> str:
            calls.append(exp_id)
            filename = "downloaded_b.fits" if exp_id == 20 else f"r{exp_id}.fits"
            filepath = Path(out_dir) / filename
            filepath.write_text("fits", encoding="utf-8")
            return str(filepath)

        with patch.object(SkynetAPI, "_download_fits", _tracking_download):
            files = api.download_images(obs, path=tmp_path, verbose=False)

        assert calls == [20]
        assert files[0] == cached
        assert files[1].name == "downloaded_b.fits"

        updated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert updated_manifest["10"] == "cached_a.fits"
        assert updated_manifest["20"] == "downloaded_b.fits"
