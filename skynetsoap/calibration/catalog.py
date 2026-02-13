"""Reference catalog querying and merging via Vizier.

Ported from tic_photometry.ipynb / grb250424A_phot.ipynb.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

from .transforms import apply_transform

logger = logging.getLogger("soap")


class ReferenceCatalog:
    """Query, transform, and merge Vizier reference catalogs.

    Parameters
    ----------
    catalogs_config : dict
        The ``catalogs`` section from SOAPConfig (loaded from catalogs.toml).
    filters_config : dict
        The ``filters`` section from SOAPConfig (loaded from filters.toml).
    default_error : float
        Default magnitude error when catalog errors are unavailable.
    merge_tolerance_arcsec : float
        Angular tolerance for deduplicating merged catalogs.
    """

    def __init__(
        self,
        catalogs_config: dict,
        filters_config: dict,
        default_error: float = 0.03,
        merge_tolerance_arcsec: float = 2.0,
    ):
        self.catalogs_config = catalogs_config.get("catalogs", catalogs_config)
        self.filters_config = filters_config
        self.default_error = default_error
        self.merge_tolerance_arcsec = merge_tolerance_arcsec
        self._cache: dict[str, pl.DataFrame] = {}

    def query(
        self,
        center: SkyCoord,
        radius_arcmin: float = 10.0,
        filter_band: str = "V",
    ) -> pl.DataFrame:
        """Query configured catalogs, apply transforms, and merge.

        Parameters
        ----------
        center : SkyCoord
            Field center.
        radius_arcmin : float
            Search radius in arcminutes.
        filter_band : str
            Target filter band (e.g. "V", "R").

        Returns
        -------
        Polars DataFrame with RAJ2000, DEJ2000, and JC magnitude columns.
        """
        canonical_band = self._canonicalize_filter_band(filter_band)
        cache_key = (
            f"{center.ra.deg:.6f}_{center.dec.deg:.6f}_{radius_arcmin}_{canonical_band}"
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        catalogs = []
        selected_catalogs = self._select_catalogs_for_band(canonical_band)
        for cat_name, cat_cfg in selected_catalogs.items():
            try:
                df = self._query_single(
                    cat_name,
                    cat_cfg,
                    center,
                    radius_arcmin,
                    canonical_band,
                )
                if df is not None and len(df) > 0:
                    catalogs.append(df)
            except Exception as e:
                logger.warning("Failed to query catalog %s: %s", cat_name, e)

        if not catalogs:
            logger.warning("No reference stars found in any catalog.")
            return pl.DataFrame()

        if len(catalogs) == 1:
            result = catalogs[0]
        else:
            result = catalogs[0]
            for other in catalogs[1:]:
                result = self._merge_catalogs(result, other)

        self._cache[cache_key] = result
        return result

    def _canonicalize_filter_band(self, filter_band: str) -> str:
        """Normalize input filter names using configured aliases."""
        aliases = self.filters_config.get("aliases", {})
        return aliases.get(filter_band, filter_band)

    def _select_catalogs_for_band(self, filter_band: str) -> dict:
        """Choose only catalogs relevant to the requested filter band."""
        jc_bands = set(
            self.filters_config.get("systems", {})
            .get("johnson_cousins", {})
            .get("bands", [])
        )
        sdss_bands = set(
            self.filters_config.get("systems", {}).get("sdss", {}).get("bands", [])
        )

        if filter_band in jc_bands:
            preferred = ("apass",)
        elif filter_band in sdss_bands:
            preferred = ("panstarrs", "skymapper")
        else:
            return self.catalogs_config

        selected = {
            name: self.catalogs_config[name]
            for name in preferred
            if name in self.catalogs_config
        }
        return selected or self.catalogs_config

    def _query_single(
        self,
        cat_name: str,
        cat_cfg: dict,
        center: SkyCoord,
        radius_arcmin: float,
        filter_band: str,
    ) -> pl.DataFrame | None:
        """Query a single Vizier catalog and apply transformations."""
        vizier_id = cat_cfg["vizier_id"]
        ra_col = cat_cfg["ra_column"]
        dec_col = cat_cfg["dec_column"]

        v = Vizier(columns=["*"], row_limit=1000)
        radius = cat_cfg.get("search_radius_arcmin", radius_arcmin)
        result = v.query_region(center, radius=f"{radius}m", catalog=vizier_id)

        if not result or len(result) == 0:
            logger.info("No results from %s", cat_name)
            return None

        # Convert astropy table to polars via pandas (astropy doesn't have direct polars support yet)
        df = pl.from_pandas(result[0].to_pandas())
        logger.info("Retrieved %d stars from %s", len(df), cat_name)

        # Apply renames
        renames = cat_cfg.get("renames", {})
        if renames:
            df = df.rename(renames)
            ra_col = renames.get(ra_col, ra_col)
            dec_col = renames.get(dec_col, dec_col)

        # Fill missing error columns with configured default error.
        # ``self.default_error`` comes from calibration_default_cat_error in defaults.toml.
        for err_col in cat_cfg.get("errors", {}).values():
            err_col = renames.get(err_col, err_col)
            if err_col and err_col not in df.columns:
                df = df.with_columns(pl.lit(self.default_error).alias(err_col))

        # Drop rows missing required columns
        required = cat_cfg.get("required_columns", {}).get("columns", [])
        # Apply renames to required column names too
        required_renamed = [renames.get(c, c) for c in required]
        if required_renamed:
            df = df.drop_nulls(subset=required_renamed)

        # Apply color cuts
        color_cuts = cat_cfg.get("color_cuts", {})
        for col, bounds in color_cuts.items():
            if col in df.columns:
                df = df.filter(
                    (pl.col(col) >= bounds["min"]) & (pl.col(col) <= bounds["max"])
                )

        if len(df) == 0:
            return None

        logger.info("%d stars after filtering for %s", len(df), cat_name)

        # Normalize native catalog band columns to canonical "<band>mag" columns.
        df = self._normalize_band_columns(df, cat_cfg, renames)

        # Apply photometric transformations to get JC magnitudes
        df = self._apply_transforms(df, cat_cfg, filter_band)

        # Keep only stars that can be used for the requested calibration band.
        target_mag_col = f"{filter_band}mag"
        target_err_col = f"e_{filter_band}mag"
        if target_mag_col not in df.columns:
            logger.info(
                "Catalog %s has no %s column after normalization/transforms.",
                cat_name,
                target_mag_col,
            )
            return None
        df = df.drop_nulls(subset=[target_mag_col])
        if target_err_col not in df.columns:
            df = df.with_columns(pl.lit(self.default_error).alias(target_err_col))
        else:
            df = df.with_columns(pl.col(target_err_col).fill_null(self.default_error))
        if len(df) == 0:
            return None

        # Normalize coordinate columns
        if ra_col != "RAJ2000":
            df = df.rename({ra_col: "RAJ2000"})
        if dec_col != "DEJ2000":
            df = df.rename({dec_col: "DEJ2000"})

        return df

    def _normalize_band_columns(
        self, df: pl.DataFrame, cat_cfg: dict, renames: dict
    ) -> pl.DataFrame:
        """Map native catalog bands (e.g. rPSF) to canonical <band>mag columns."""
        bands_cfg = cat_cfg.get("bands", {})
        errors_cfg = cat_cfg.get("errors", {})
        for band_key, src_col in bands_cfg.items():
            src_col = renames.get(src_col, src_col)
            mag_col = f"{band_key}mag"
            if src_col in df.columns and mag_col not in df.columns:
                df = df.with_columns(pl.col(src_col).alias(mag_col))

            err_src = errors_cfg.get(band_key)
            if err_src is None:
                continue
            err_src = renames.get(err_src, err_src)
            err_col = f"e_{band_key}mag"
            if err_src in df.columns:
                df = df.with_columns(
                    pl.col(err_src).fill_null(self.default_error).alias(err_col)
                )
            elif err_col not in df.columns:
                df = df.with_columns(pl.lit(self.default_error).alias(err_col))
        return df

    def _apply_transforms(
        self, df: pl.DataFrame, cat_cfg: dict, filter_band: str
    ) -> pl.DataFrame:
        """Apply Jordi+2006 transforms to produce JC magnitudes."""
        if filter_band not in set(
            self.filters_config.get("systems", {})
            .get("johnson_cousins", {})
            .get("bands", [])
        ):
            return df

        transforms = self.filters_config.get("transforms", {}).get("ugriz_to_jc", {})
        bands_cfg = cat_cfg.get("bands", {})
        errors_cfg = cat_cfg.get("errors", {})

        t_cfg = transforms.get(filter_band)
        if not t_cfg:
            return df
        for jc_band in [filter_band]:
            if f"{jc_band}mag" in df.columns:
                continue

            src1_key, src2_key = t_cfg["source_bands"]
            src1_col = bands_cfg.get(src1_key)
            src2_col = bands_cfg.get(src2_key)

            if src1_col is None or src2_col is None:
                continue
            if src1_col not in df.columns or src2_col not in df.columns:
                continue

            err1_col = errors_cfg.get(src1_key)
            err2_col = errors_cfg.get(src2_key)
            err1 = (
                df[err1_col].to_numpy()
                if (err1_col and err1_col in df.columns)
                else np.full(len(df), self.default_error)
            )
            err2 = (
                df[err2_col].to_numpy()
                if (err2_col and err2_col in df.columns)
                else np.full(len(df), self.default_error)
            )

            mag, mag_err = apply_transform(
                df[src1_col].to_numpy(),
                df[src2_col].to_numpy(),
                err1,
                err2,
                a=t_cfg["a"],
                sigma_a=t_cfg["sigma_a"],
                b=t_cfg["b"],
                sigma_b=t_cfg["sigma_b"],
                formula=t_cfg["formula"],
            )

            if t_cfg["formula"] == "color_only":
                # This gives a color (e.g. U-B); need to add B to get U
                if jc_band == "U" and "Bmag" in df.columns:
                    b_err = (
                        df["e_Bmag"].to_numpy()
                        if "e_Bmag" in df.columns
                        else np.full(len(df), self.default_error)
                    )
                    df = df.with_columns(
                        [
                            pl.Series(f"{jc_band}mag", mag + df["Bmag"].to_numpy()),
                            pl.Series(
                                f"e_{jc_band}mag", np.sqrt(mag_err**2 + b_err**2)
                            ),
                        ]
                    )
                else:
                    df = df.with_columns(
                        [
                            pl.Series(f"{jc_band}mag", mag),
                            pl.Series(f"e_{jc_band}mag", mag_err),
                        ]
                    )
            else:
                df = df.with_columns(
                    [
                        pl.Series(f"{jc_band}mag", mag),
                        pl.Series(f"e_{jc_band}mag", mag_err),
                    ]
                )

        return df

    def _merge_catalogs(
        self,
        df1: pl.DataFrame,
        df2: pl.DataFrame,
    ) -> pl.DataFrame:
        """Merge two catalogs, removing duplicates within angular tolerance.

        Prefers df1 (higher priority) when sources overlap.
        """
        tol = self.merge_tolerance_arcsec * u.arcsec

        pref_coords = SkyCoord(
            ra=df1["RAJ2000"].to_numpy() * u.deg,
            dec=df1["DEJ2000"].to_numpy() * u.deg,
        )
        other_coords = SkyCoord(
            ra=df2["RAJ2000"].to_numpy() * u.deg,
            dec=df2["DEJ2000"].to_numpy() * u.deg,
        )

        _, sep2d, _ = other_coords.match_to_catalog_sky(pref_coords)
        keep_mask = sep2d > tol

        # Filter df2 to keep only unique sources
        other_unique = df2.filter(pl.Series(keep_mask))
        merged = pl.concat([df1, other_unique], how="vertical")

        # Ensure standard columns exist
        for col in [
            "RAJ2000",
            "DEJ2000",
            "Umag",
            "e_Umag",
            "Bmag",
            "e_Bmag",
            "Vmag",
            "e_Vmag",
            "Rmag",
            "e_Rmag",
            "Imag",
            "e_Imag",
        ]:
            if col not in merged.columns:
                merged = merged.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))

        return merged
