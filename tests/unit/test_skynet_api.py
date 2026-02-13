"""Unit tests for skynetsoap.io.skynet_api."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from skynetsoap.io.skynet_api import SkynetAPI


class _FakeDownloadRequest:
    def __init__(self, out_dir: Path, file_by_image: dict[str, str] | None = None):
        self.out_dir = out_dir
        self.file_by_image = file_by_image or {}
        self.calls: list[str] = []

    def get_fits(self, out_dir: str, reducequiet: int, image: str) -> str:
        self.calls.append(image)
        filename = self.file_by_image.get(image, f"{image}.fits")
        filepath = Path(out_dir) / filename
        filepath.write_text("fits", encoding="utf-8")
        return str(filepath)


def _make_api(download_request: _FakeDownloadRequest) -> SkynetAPI:
    api = SkynetAPI.__new__(SkynetAPI)
    api.download_request = download_request
    return api


def _obs_with_exp_ids(*ids: int):
    return SimpleNamespace(
        exps=[
            SimpleNamespace(id=i, center_time="2025-01-01T00:00:00+00:00") for i in ids
        ]
    )


class TestSkynetAPIDownloadCaching:
    def test_uses_existing_files_without_manifest_when_counts_match(self, tmp_path):
        # Simulate old behavior: files exist with non-r<ID>.fits names.
        cached_a = tmp_path / "obs_file_000.fits"
        cached_b = tmp_path / "obs_file_001.fits"
        cached_a.write_text("fits", encoding="utf-8")
        cached_b.write_text("fits", encoding="utf-8")

        fake_download = _FakeDownloadRequest(tmp_path)
        api = _make_api(fake_download)
        obs = _obs_with_exp_ids(1, 2)

        files = api.download_images(obs, path=tmp_path, verbose=False)

        assert fake_download.calls == []
        assert files == sorted([cached_a, cached_b])

    def test_skips_download_when_manifest_entry_exists(self, tmp_path):
        cached = tmp_path / "cached_name.fits"
        cached.write_text("fits", encoding="utf-8")
        manifest_path = tmp_path / ".download_manifest.json"
        manifest_path.write_text(json.dumps({"123": cached.name}), encoding="utf-8")

        fake_download = _FakeDownloadRequest(tmp_path)
        api = _make_api(fake_download)
        obs = _obs_with_exp_ids(123)

        files = api.download_images(obs, path=tmp_path, verbose=False)

        assert fake_download.calls == []
        assert files == [cached]

    def test_downloads_only_missing_and_updates_manifest(self, tmp_path):
        cached = tmp_path / "cached_a.fits"
        cached.write_text("fits", encoding="utf-8")
        manifest_path = tmp_path / ".download_manifest.json"
        manifest_path.write_text(json.dumps({"10": cached.name}), encoding="utf-8")

        fake_download = _FakeDownloadRequest(
            tmp_path,
            file_by_image={"r20": "downloaded_b.fits"},
        )
        api = _make_api(fake_download)
        obs = _obs_with_exp_ids(10, 20)

        files = api.download_images(obs, path=tmp_path, verbose=False)

        assert fake_download.calls == ["r20"]
        assert files[0] == cached
        assert files[1].name == "downloaded_b.fits"

        updated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert updated_manifest["10"] == "cached_a.fits"
        assert updated_manifest["20"] == "downloaded_b.fits"
