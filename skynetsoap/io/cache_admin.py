"""Cache management helpers for SOAP observation products."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def get_cache_info(image_dir: Path, result_dir: Path) -> dict:
    """Get cache statistics for an observation's image/result directories."""
    image_files = list(image_dir.glob("*.fits"))
    result_files = list(result_dir.glob("*")) if result_dir.exists() else []

    image_size = sum(f.stat().st_size for f in image_files if f.is_file())
    result_size = sum(f.stat().st_size for f in result_files if f.is_file())
    total_size_mb = (image_size + result_size) / (1024 * 1024)

    return {
        "n_images": len(image_files),
        "n_results": len(result_files),
        "total_size_mb": round(total_size_mb, 2),
        "image_dir": str(image_dir),
        "result_dir": str(result_dir),
    }


def clear_observation_cache(
    image_dir: Path,
    result_dir: Path,
    observation_id: int,
    images: bool = True,
    results: bool = True,
    confirm: bool = True,
) -> dict:
    """Clear cached image/result files for one observation."""
    if confirm:
        msg = f"Delete cache for observation {observation_id}?"
        if images and results:
            msg += f"\n  - {image_dir} (images)"
            msg += f"\n  - {result_dir} (results)"
        elif images:
            msg += f"\n  - {image_dir} (images only)"
        elif results:
            msg += f"\n  - {result_dir} (results only)"
        msg += "\nProceed? (y/n): "

        if input(msg).strip().lower() != "y":
            return {"images_deleted": 0, "results_deleted": 0}

    images_deleted = 0
    results_deleted = 0

    if images and image_dir.exists():
        image_files = list(image_dir.glob("*.fits"))
        for f in image_files:
            f.unlink()
            images_deleted += 1

    if results and result_dir.exists():
        result_files = list(result_dir.glob("*"))
        for f in result_files:
            if f.is_file():
                f.unlink()
                results_deleted += 1

    return {"images_deleted": images_deleted, "results_deleted": results_deleted}


def dir_stats(path: Path) -> tuple[int, int, float]:
    """Return (n_files, total_size_bytes, latest_mtime) for a path tree."""
    if not path.exists():
        return 0, 0, 0.0

    n_files = 0
    total_size = 0
    try:
        latest_mtime = path.stat().st_mtime
    except OSError:
        latest_mtime = 0.0

    stack: list[Path] = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                            latest_mtime = max(
                                latest_mtime,
                                entry.stat(follow_symlinks=False).st_mtime,
                            )
                        elif entry.is_file(follow_symlinks=False):
                            st = entry.stat(follow_symlinks=False)
                            n_files += 1
                            total_size += st.st_size
                            latest_mtime = max(latest_mtime, st.st_mtime)
                    except OSError:
                        continue
        except OSError:
            continue

    return n_files, total_size, latest_mtime


def prune_cache(
    max_total_size_mb: float,
    image_dir: str | Path = "soap_images",
    result_dir: str | Path = "soap_results",
    keep_recent: int = 1,
    confirm: bool = True,
) -> dict:
    """Prune old observation caches until total disk usage is below a limit."""
    if max_total_size_mb <= 0:
        raise ValueError("max_total_size_mb must be > 0")

    keep_recent = max(0, int(keep_recent))
    max_total_size_bytes = int(max_total_size_mb * 1024 * 1024)
    image_base = Path(image_dir)
    result_base = Path(result_dir)

    obs_ids: set[str] = set()
    if image_base.exists():
        obs_ids.update(p.name for p in image_base.iterdir() if p.is_dir())
    if result_base.exists():
        obs_ids.update(p.name for p in result_base.iterdir() if p.is_dir())

    if not obs_ids:
        return {
            "observations_deleted": [],
            "files_deleted": 0,
            "bytes_deleted": 0,
            "total_before_mb": 0.0,
            "total_after_mb": 0.0,
            "target_mb": round(max_total_size_mb, 2),
        }

    usage = []
    total_before = 0
    for obs_id in obs_ids:
        img_path = image_base / obs_id
        res_path = result_base / obs_id
        n_img, size_img, mtime_img = dir_stats(img_path)
        n_res, size_res, mtime_res = dir_stats(res_path)
        obs_size = size_img + size_res
        if obs_size == 0:
            continue
        total_before += obs_size
        usage.append(
            {
                "obs_id": obs_id,
                "size": obs_size,
                "files": n_img + n_res,
                "mtime": max(mtime_img, mtime_res),
                "img_path": img_path,
                "res_path": res_path,
            }
        )

    if total_before <= max_total_size_bytes:
        total_mb = round(total_before / (1024 * 1024), 2)
        return {
            "observations_deleted": [],
            "files_deleted": 0,
            "bytes_deleted": 0,
            "total_before_mb": total_mb,
            "total_after_mb": total_mb,
            "target_mb": round(max_total_size_mb, 2),
        }

    usage.sort(key=lambda x: x["mtime"], reverse=True)
    keep_ids = {u["obs_id"] for u in usage[:keep_recent]}
    candidates = [
        u
        for u in sorted(usage, key=lambda x: x["mtime"])
        if u["obs_id"] not in keep_ids
    ]

    if confirm:
        current_mb = total_before / (1024 * 1024)
        msg = (
            f"Prune SOAP cache to <= {max_total_size_mb:.2f} MB?"
            f"\nCurrent usage: {current_mb:.2f} MB across {len(usage)} observations."
            f"\nProceed? (y/n): "
        )
        if input(msg).strip().lower() != "y":
            return {
                "observations_deleted": [],
                "files_deleted": 0,
                "bytes_deleted": 0,
                "total_before_mb": round(current_mb, 2),
                "total_after_mb": round(current_mb, 2),
                "target_mb": round(max_total_size_mb, 2),
            }

    deleted_obs: list[int | str] = []
    files_deleted = 0
    bytes_deleted = 0
    total_after = total_before

    for obs in candidates:
        if total_after <= max_total_size_bytes:
            break

        if obs["img_path"].exists():
            shutil.rmtree(obs["img_path"], ignore_errors=True)
        if obs["res_path"].exists():
            shutil.rmtree(obs["res_path"], ignore_errors=True)

        total_after -= obs["size"]
        bytes_deleted += obs["size"]
        files_deleted += obs["files"]
        try:
            deleted_obs.append(int(obs["obs_id"]))
        except ValueError:
            deleted_obs.append(obs["obs_id"])

    return {
        "observations_deleted": deleted_obs,
        "files_deleted": files_deleted,
        "bytes_deleted": bytes_deleted,
        "total_before_mb": round(total_before / (1024 * 1024), 2),
        "total_after_mb": round(max(total_after, 0) / (1024 * 1024), 2),
        "target_mb": round(max_total_size_mb, 2),
    }


def cleanup_observation(
    observation_id: int,
    image_dir: str | Path = "soap_images",
    result_dir: str | Path = "soap_results",
    images: bool = True,
    results: bool = True,
    confirm: bool = True,
) -> dict:
    """Remove cached files for a specific observation ID."""
    img_path = Path(image_dir) / str(observation_id)
    res_path = Path(result_dir) / str(observation_id)
    n_img, bytes_img, _ = dir_stats(img_path)
    n_res, bytes_res, _ = dir_stats(res_path)

    if confirm:
        msg = f"Delete all files for observation {observation_id}?"
        if images and img_path.exists():
            msg += f"\n  - {img_path}"
        if results and res_path.exists():
            msg += f"\n  - {res_path}"
        msg += "\nProceed? (y/n): "

        if input(msg).strip().lower() != "y":
            return {"images_deleted": 0, "results_deleted": 0, "bytes_deleted": 0}

    images_deleted = 0
    results_deleted = 0
    bytes_deleted = 0

    if images and img_path.exists():
        images_deleted = n_img
        bytes_deleted += bytes_img
        shutil.rmtree(img_path, ignore_errors=True)

    if results and res_path.exists():
        results_deleted = n_res
        bytes_deleted += bytes_res
        shutil.rmtree(res_path, ignore_errors=True)

    return {
        "images_deleted": images_deleted,
        "results_deleted": results_deleted,
        "bytes_deleted": bytes_deleted,
    }
