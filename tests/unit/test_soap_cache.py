"""Unit tests for SOAP cache cleanup helpers."""

from __future__ import annotations

import os
from pathlib import Path

from skynetsoap.soap import Soap


def _write_file(path: Path, size_bytes: int, ts: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size_bytes)
    os.utime(path, (ts, ts))


def _set_dir_mtime(path: Path, ts: int) -> None:
    os.utime(path, (ts, ts))


def test_cleanup_observation_selective(tmp_path: Path):
    obs_id = 12345
    img_obs = tmp_path / "images" / str(obs_id)
    res_obs = tmp_path / "results" / str(obs_id)
    _write_file(img_obs / "a.fits", 64, 1000)
    _write_file(res_obs / "photometry.csv", 128, 1000)
    _set_dir_mtime(img_obs, 1000)
    _set_dir_mtime(res_obs, 1000)

    stats = Soap.cleanup_observation(
        observation_id=obs_id,
        image_dir=tmp_path / "images",
        result_dir=tmp_path / "results",
        images=True,
        results=False,
        confirm=False,
    )

    assert stats["images_deleted"] == 1
    assert stats["results_deleted"] == 0
    assert stats["bytes_deleted"] == 64
    assert not img_obs.exists()
    assert res_obs.exists()


def test_prune_cache_deletes_oldest_while_keeping_recent(tmp_path: Path):
    image_base = tmp_path / "images"
    result_base = tmp_path / "results"

    # Three observations with equal sizes; timestamps make 1001 oldest, 1003 newest.
    for obs_id, ts in [(1001, 1000), (1002, 2000), (1003, 3000)]:
        img_obs = image_base / str(obs_id)
        res_obs = result_base / str(obs_id)
        _write_file(img_obs / "frame.fits", 1000, ts)
        _write_file(res_obs / "photometry.csv", 1000, ts)
        _set_dir_mtime(img_obs, ts)
        _set_dir_mtime(res_obs, ts)

    stats = Soap.prune_cache(
        max_total_size_mb=0.004,  # ~4194 bytes
        image_dir=image_base,
        result_dir=result_base,
        keep_recent=1,
        confirm=False,
    )

    assert stats["total_before_mb"] > stats["target_mb"]
    assert 1001 in stats["observations_deleted"]
    assert 1003 not in stats["observations_deleted"]
    assert not (image_base / "1001").exists()
    assert not (result_base / "1001").exists()
    assert (image_base / "1003").exists()
    assert (result_base / "1003").exists()
