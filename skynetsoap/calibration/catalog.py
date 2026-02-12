"""Reference catalog querying and merging via Vizier.

Ported from tic_photometry.ipynb / grb250424A_phot.ipynb.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

from .transforms import apply_transform

logger = logging.getLogger("soap")

# TODO: Add support for more catalogs, e.g. Pan-STARRS, ATLAS, etc. Add args for user-specified catalogs.
# TODO: Only query catalogs relevant to the target filter band (e.g. if filter_band="R", only query catalogs with R or similar bands).


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
        self._cache: dict[str, pd.DataFrame] = {}

    def query(
        self,
        center: SkyCoord,
        radius_arcmin: float = 10.0,
        filter_band: str = "V",
    ) -> pd.DataFrame:
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
        DataFrame with RAJ2000, DEJ2000, and JC magnitude columns.
        """
        cache_key = f"{center.ra.deg:.6f}_{center.dec.deg:.6f}_{radius_arcmin}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        catalogs = []
        for cat_name, cat_cfg in self.catalogs_config.items():
            try:
                df = self._query_single(cat_name, cat_cfg, center, radius_arcmin)
                if df is not None and len(df) > 0:
                    catalogs.append(df)
            except Exception as e:
                logger.warning("Failed to query catalog %s: %s", cat_name, e)

        if not catalogs:
            logger.warning("No reference stars found in any catalog.")
            return pd.DataFrame()

        if len(catalogs) == 1:
            result = catalogs[0]
        else:
            result = catalogs[0]
            for other in catalogs[1:]:
                result = self._merge_catalogs(result, other)

        self._cache[cache_key] = result
        return result

    def _query_single(
        self,
        cat_name: str,
        cat_cfg: dict,
        center: SkyCoord,
        radius_arcmin: float,
    ) -> pd.DataFrame | None:
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

        df = result[0].to_pandas()
        logger.info("Retrieved %d stars from %s", len(df), cat_name)

        # Apply renames
        renames = cat_cfg.get("renames", {})
        if renames:
            df.rename(columns=renames, inplace=True)
            ra_col = renames.get(ra_col, ra_col)
            dec_col = renames.get(dec_col, dec_col)

        # Drop rows missing required columns
        required = cat_cfg.get("required_columns", {}).get("columns", [])
        # Apply renames to required column names too
        required_renamed = [renames.get(c, c) for c in required]
        if required_renamed:
            df.dropna(subset=required_renamed, inplace=True)

        # Apply color cuts
        color_cuts = cat_cfg.get("color_cuts", {})
        for col, bounds in color_cuts.items():
            if col in df.columns:
                df = df[(df[col] >= bounds["min"]) & (df[col] <= bounds["max"])]

        if df.empty:
            return None

        logger.info("%d stars after filtering for %s", len(df), cat_name)

        # Apply photometric transformations to get JC magnitudes
        df = self._apply_transforms(df, cat_cfg)

        # Normalize coordinate columns
        if ra_col != "RAJ2000":
            df.rename(columns={ra_col: "RAJ2000"}, inplace=True)
        if dec_col != "DEJ2000":
            df.rename(columns={dec_col: "DEJ2000"}, inplace=True)

        return df

    def _apply_transforms(self, df: pd.DataFrame, cat_cfg: dict) -> pd.DataFrame:
        """Apply Jordi+2006 transforms to produce JC magnitudes."""
        transforms = self.filters_config.get("transforms", {}).get("ugriz_to_jc", {})
        bands_cfg = cat_cfg.get("bands", {})
        errors_cfg = cat_cfg.get("errors", {})

        for jc_band, t_cfg in transforms.items():
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
            err1 = df[err1_col].values if (err1_col and err1_col in df.columns) else np.full(len(df), self.default_error)
            err2 = df[err2_col].values if (err2_col and err2_col in df.columns) else np.full(len(df), self.default_error)

            mag, mag_err = apply_transform(
                df[src1_col].values,
                df[src2_col].values,
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
                    b_err = df["e_Bmag"].values if "e_Bmag" in df.columns else np.full(len(df), self.default_error)
                    df[f"{jc_band}mag"] = mag + df["Bmag"].values
                    df[f"e_{jc_band}mag"] = np.sqrt(mag_err ** 2 + b_err ** 2)
                else:
                    df[f"{jc_band}mag"] = mag
                    df[f"e_{jc_band}mag"] = mag_err
            else:
                df[f"{jc_band}mag"] = mag
                df[f"e_{jc_band}mag"] = mag_err

        return df

    def _merge_catalogs(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge two catalogs, removing duplicates within angular tolerance.

        Prefers df1 (higher priority) when sources overlap.
        """
        tol = self.merge_tolerance_arcsec * u.arcsec

        pref_coords = SkyCoord(
            ra=np.asarray(df1["RAJ2000"]) * u.deg,
            dec=np.asarray(df1["DEJ2000"]) * u.deg,
        )
        other_coords = SkyCoord(
            ra=np.asarray(df2["RAJ2000"]) * u.deg,
            dec=np.asarray(df2["DEJ2000"]) * u.deg,
        )

        _, sep2d, _ = other_coords.match_to_catalog_sky(pref_coords)
        keep_mask = sep2d > tol

        other_unique = df2.loc[keep_mask].copy()
        merged = pd.concat([df1, other_unique], ignore_index=True)

        # Ensure standard columns exist
        for col in ["RAJ2000", "DEJ2000", "Umag", "e_Umag", "Bmag", "e_Bmag",
                     "Vmag", "e_Vmag", "Rmag", "e_Rmag", "Imag", "e_Imag"]:
            if col not in merged.columns:
                merged[col] = np.nan

        return merged
