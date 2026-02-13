"""Limiting-magnitude utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sep
from scipy.ndimage import binary_dilation


@dataclass
class RobustLimitingMagResult:
    """Output of robust blank-sky aperture limiting-magnitude estimation."""

    limiting_mag: float
    sigma_ap: float
    flux_limit: float
    n_samples_requested: int
    n_samples_used: int
    aperture_radius_pixels: float
    fraction_masked: float
    warnings: list[str] = field(default_factory=list)


def _build_blank_sky_mask(
    data_sub: np.ndarray,
    err: float | np.ndarray,
    extraction_threshold: float,
    extraction_min_area: int,
    mask_dilate_pixels: int,
    edge_buffer_pixels: int,
    extra_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Build a boolean mask where True means unusable for blank-sky sampling."""
    mask = ~np.isfinite(data_sub)
    if extra_mask is not None:
        mask = mask | np.asarray(extra_mask, dtype=bool)

    # Build a source mask from SEP detections on background-subtracted data.
    data_extract = np.nan_to_num(data_sub, nan=0.0, posinf=0.0, neginf=0.0)
    objects = sep.extract(
        data_extract,
        extraction_threshold,
        err=err,
        minarea=extraction_min_area,
    )
    if len(objects) > 0:
        sep.mask_ellipse(
            mask,
            objects["x"],
            objects["y"],
            objects["a"],
            objects["b"],
            objects["theta"],
            r=2.0,
        )

    if mask_dilate_pixels > 0:
        mask = binary_dilation(mask, iterations=int(mask_dilate_pixels))

    edge = int(edge_buffer_pixels)
    if edge > 0:
        mask[:edge, :] = True
        mask[-edge:, :] = True
        mask[:, :edge] = True
        mask[:, -edge:] = True

    return mask.astype(bool), float(np.mean(mask))


def _sample_blank_positions(
    blank_mask: np.ndarray,
    n_samples: int,
    aperture_radius_pixels: float,
    max_draws_multiplier: int,
    random_seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rejection-sample blank-sky aperture centers from unmasked image regions."""
    if n_samples <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    ny, nx = blank_mask.shape
    valid = ~blank_mask.copy()

    # Ensure the full aperture footprint stays on-image.
    margin = int(np.ceil(aperture_radius_pixels))
    if margin > 0:
        valid[:margin, :] = False
        valid[-margin:, :] = False
        valid[:, :margin] = False
        valid[:, -margin:] = False

    if not np.any(valid):
        return np.array([], dtype=float), np.array([], dtype=float)

    rng = np.random.default_rng(random_seed)
    max_draws = int(max_draws_multiplier) * int(n_samples)
    max_draws = max(max_draws, n_samples)

    xs: list[float] = []
    ys: list[float] = []
    draws = 0
    while len(xs) < n_samples and draws < max_draws:
        x = float(rng.uniform(0, nx))
        y = float(rng.uniform(0, ny))
        draws += 1
        if valid[int(y), int(x)]:
            xs.append(x)
            ys.append(y)

    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def compute_robust_limiting_magnitude(
    data_sub: np.ndarray,
    zeropoint: float,
    aperture_radius_pixels: float,
    *,
    err: float | np.ndarray,
    extraction_threshold: float,
    extraction_min_area: int,
    n_samples: int,
    mask_dilate_pixels: int,
    edge_buffer_pixels: int,
    sigma_estimator: str,
    max_draws_multiplier: int,
    random_seed: int | None,
    extra_mask: np.ndarray | None = None,
) -> RobustLimitingMagResult:
    """Measure robust 5-sigma limiting magnitude from blank-sky apertures."""
    warnings: list[str] = []
    n_samples_i = int(n_samples)

    if np.isnan(zeropoint) or zeropoint == 0:
        warnings.append(
            "Zeropoint is NaN/zero; robust limiting magnitude is undefined."
        )
        return RobustLimitingMagResult(
            limiting_mag=np.nan,
            sigma_ap=np.nan,
            flux_limit=np.nan,
            n_samples_requested=n_samples_i,
            n_samples_used=0,
            aperture_radius_pixels=float(aperture_radius_pixels),
            fraction_masked=np.nan,
            warnings=warnings,
        )

    if not np.isfinite(aperture_radius_pixels) or aperture_radius_pixels <= 0:
        warnings.append(
            "Aperture radius is invalid; robust limiting magnitude is undefined."
        )
        return RobustLimitingMagResult(
            limiting_mag=np.nan,
            sigma_ap=np.nan,
            flux_limit=np.nan,
            n_samples_requested=n_samples_i,
            n_samples_used=0,
            aperture_radius_pixels=float(aperture_radius_pixels),
            fraction_masked=np.nan,
            warnings=warnings,
        )

    blank_mask, fraction_masked = _build_blank_sky_mask(
        data_sub=data_sub,
        err=err,
        extraction_threshold=extraction_threshold,
        extraction_min_area=extraction_min_area,
        mask_dilate_pixels=mask_dilate_pixels,
        edge_buffer_pixels=edge_buffer_pixels,
        extra_mask=extra_mask,
    )

    xs, ys = _sample_blank_positions(
        blank_mask=blank_mask,
        n_samples=n_samples_i,
        aperture_radius_pixels=float(aperture_radius_pixels),
        max_draws_multiplier=max_draws_multiplier,
        random_seed=random_seed,
    )

    if len(xs) == 0:
        warnings.append("No valid blank-sky aperture centers were found.")
        return RobustLimitingMagResult(
            limiting_mag=np.nan,
            sigma_ap=np.nan,
            flux_limit=np.nan,
            n_samples_requested=n_samples_i,
            n_samples_used=0,
            aperture_radius_pixels=float(aperture_radius_pixels),
            fraction_masked=fraction_masked,
            warnings=warnings,
        )

    sums, _, flags = sep.sum_circle(
        data_sub,
        xs,
        ys,
        float(aperture_radius_pixels),
        err=err,
        gain=1.0,
        mask=blank_mask,
    )

    valid = np.isfinite(sums) & (flags == 0)
    sums = np.asarray(sums[valid], dtype=float)
    n_used = len(sums)
    if n_used < n_samples_i:
        warnings.append(
            f"Only {n_used}/{n_samples_i} accepted sky apertures after masking/flags."
        )

    sigma_mode = sigma_estimator.lower().strip()
    if n_used == 0:
        sigma_ap = np.nan
    elif sigma_mode == "std":
        sigma_ap = float(np.std(sums, ddof=1)) if n_used > 1 else np.nan
    else:
        if sigma_mode != "mad":
            warnings.append(f"Unknown sigma_estimator={sigma_estimator!r}; using MAD.")
        med = float(np.median(sums))
        mad = float(np.median(np.abs(sums - med)))
        sigma_ap = 1.4826 * mad

    flux_limit = 5.0 * sigma_ap if np.isfinite(sigma_ap) else np.nan
    if not np.isfinite(flux_limit) or flux_limit <= 0:
        limiting_mag = np.nan
    else:
        limiting_mag = float(zeropoint - 2.5 * np.log10(flux_limit))

    return RobustLimitingMagResult(
        limiting_mag=limiting_mag,
        sigma_ap=float(sigma_ap) if np.isfinite(sigma_ap) else np.nan,
        flux_limit=float(flux_limit) if np.isfinite(flux_limit) else np.nan,
        n_samples_requested=n_samples_i,
        n_samples_used=n_used,
        aperture_radius_pixels=float(aperture_radius_pixels),
        fraction_masked=fraction_masked,
        warnings=warnings,
    )
