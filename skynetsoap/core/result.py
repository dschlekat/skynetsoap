"""PhotometryResult: QTable wrapper for pipeline output."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable

from ..utils.time_utils import filter_table_by_date

logger = logging.getLogger("soap")

# Column definitions with units
_COLUMNS = {
    "image_file": (str, None),
    "telescope": (str, None),
    "filter": (str, None),
    "exptime": (float, u.s),
    "mjd": (float, u.d),
    "jd": (float, u.d),
    "x_pix": (float, None),
    "y_pix": (float, None),
    "ra": (float, u.deg),
    "dec": (float, u.deg),
    "flux": (float, None),
    "flux_err": (float, None),
    "snr": (float, None),
    "ins_mag": (float, u.mag),
    "ins_mag_err": (float, u.mag),
    "calibrated_mag": (float, u.mag),
    "calibrated_mag_err": (float, u.mag),
    "zeropoint": (float, u.mag),
    "zeropoint_err": (float, u.mag),
    "aperture_radius": (float, None),
    "fwhm": (float, None),
    "n_cal_stars": (int, None),
    "is_forced": (bool, None),
    "flag": (int, None),
}


def _empty_table() -> QTable:
    """Create an empty QTable with the standard columns."""
    t = QTable()
    for name, (dtype, unit) in _COLUMNS.items():
        if dtype is str:
            t[name] = np.array([], dtype="U256")
        elif dtype is bool:
            t[name] = np.array([], dtype=bool)
        elif dtype is int:
            t[name] = np.array([], dtype=np.int64)
        else:
            col = np.array([], dtype=np.float64)
            if unit is not None:
                col = col * unit
            t[name] = col
    return t


class PhotometryResult:
    """Wrapper around a QTable holding photometry results.

    Provides convenience methods for filtering, exporting, and
    extracting single-target lightcurves.
    """

    def __init__(
        self,
        table: QTable | None = None,
        observation_id: int | None = None,
        result_dir: str | Path | None = None,
    ):
        self.table = table if table is not None else _empty_table()
        self.observation_id = observation_id
        self.result_dir = Path(result_dir) if result_dir is not None else None

    def __len__(self) -> int:
        return len(self.table)

    def __repr__(self) -> str:
        return f"PhotometryResult({len(self)} measurements)"

    def _new_result(self, table: QTable) -> PhotometryResult:
        return PhotometryResult(
            table,
            observation_id=self.observation_id,
            result_dir=self.result_dir,
        )

    def _new_target(self, table: QTable) -> PhotometryTarget:
        return PhotometryTarget(
            table,
            observation_id=self.observation_id,
            result_dir=self.result_dir,
        )

    # ------------------------------------------------------------------
    # Adding data
    # ------------------------------------------------------------------
    def add_measurement(self, **kwargs: Any) -> None:
        """Append a single measurement row."""
        row = {}
        for name, (dtype, unit) in _COLUMNS.items():
            val = kwargs.get(name)
            if val is None:
                if dtype is str:
                    val = ""
                elif dtype is bool:
                    val = False
                elif dtype is int:
                    val = 0
                else:
                    val = np.nan
            if unit is not None and dtype is float:
                val = val * unit if not hasattr(val, "unit") else val
            row[name] = val
        self.table.add_row(row)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def filter_by_date(
        self,
        after: str | float | None = None,
        before: str | float | None = None,
        days_ago: float | None = None,
    ) -> PhotometryResult:
        """Return a new PhotometryResult filtered by date."""
        filtered = filter_table_by_date(
            self.table, after=after, before=before, days_ago=days_ago
        )
        return self._new_result(filtered)

    def filter_by_band(self, band: str) -> PhotometryResult:
        """Return a new PhotometryResult for a single filter band."""
        mask = self.table["filter"] == band
        return self._new_result(self.table[mask])

    def sort_by_time(self) -> PhotometryResult:
        """Return a copy sorted by MJD."""
        t = self.table.copy()
        t.sort("mjd")
        return self._new_result(t)

    def extract_target(
        self,
        coord: SkyCoord,
        radius_arcsec: float = 3.0,
        forced_photometry: bool = False,
        snr_threshold: float = 3.0,
    ) -> PhotometryTarget:
        """Extract measurements for a single sky coordinate.

        Parameters
        ----------
        coord : SkyCoord
            Target position.
        radius_arcsec : float
            Match radius.
        forced_photometry : bool
            If True, include forced-photometry measurements.
        snr_threshold : float
            Minimum SNR for forced photometry.

        Returns
        -------
        PhotometryTarget containing only the target.
        """
        source_coords = SkyCoord(
            ra=self.table["ra"],
            dec=self.table["dec"],
        )
        sep = source_coords.separation(coord)
        mask = sep.arcsec < radius_arcsec

        if not forced_photometry:
            mask = mask & ~self.table["is_forced"]

        result = self.table[mask]

        if forced_photometry and snr_threshold > 0:
            snr_ok = result["snr"] >= snr_threshold
            not_forced = ~result["is_forced"]
            result = result[snr_ok | not_forced]

        # Deduplicate: keep closest match per image
        if len(result) > 0:
            seps = SkyCoord(ra=result["ra"], dec=result["dec"]).separation(coord).arcsec

            # Group by image_file and keep minimum separation
            unique_images = np.unique(result["image_file"])
            keep_indices = []
            for img in unique_images:
                img_mask = result["image_file"] == img
                img_indices = np.where(img_mask)[0]
                img_seps = seps[img_mask]
                best = img_indices[np.argmin(img_seps)]
                keep_indices.append(best)
            result = result[keep_indices]

        return self._new_target(result)

    # ------------------------------------------------------------------
    # Export methods
    # ------------------------------------------------------------------
    def to_csv(self, path: str | Path) -> Path:
        """Export to CSV."""
        path = Path(path)
        self.table.write(path, format="csv", overwrite=True)
        return path

    def to_ecsv(self, path: str | Path) -> Path:
        """Export to ECSV (preserves units and metadata)."""
        path = Path(path)
        self.table.write(path, format="ascii.ecsv", overwrite=True)
        return path

    def to_parquet(self, path: str | Path) -> Path:
        """Export to Parquet via pyarrow."""
        path = Path(path)
        self.to_pandas().to_parquet(path, index=False)
        return path

    def to_json(self, path: str | Path) -> Path:
        """Export to JSON."""
        path = Path(path)
        df = self.to_pandas()
        df.to_json(path, orient="records", indent=2)
        return path

    def to_pandas(self):
        """Convert to a pandas DataFrame."""
        return self.table.to_pandas()

    def to_polars(self):
        """Convert to a polars DataFrame."""
        import polars as pl

        return pl.from_pandas(self.to_pandas())

    def to_gcn(
        self,
        path: str | Path,
        start_time: float | None = None,
        all_results: bool = False,
    ) -> Path:
        """Export in GCN circular format."""
        from ..io.table import write_gcn_table

        return write_gcn_table(
            self.table, path, start_time=start_time, all_results=all_results
        )


class PhotometryTarget(PhotometryResult):
    """Single-target photometry result with convenience export."""

    def __repr__(self) -> str:
        return f"PhotometryTarget({len(self)} measurements)"

    def export(
        self,
        format: str = "csv",
        path: str | Path | None = None,
        **kwargs,
    ) -> Path:
        """Export target results to file.

        Parameters
        ----------
        format : str
            "csv", "ecsv", "parquet", "json", "gcn".
        path : str or Path, optional
            Output path. Auto-generated if not provided.
        """
        if path is None:
            ext = {
                "csv": ".csv",
                "ecsv": ".ecsv",
                "parquet": ".parquet",
                "json": ".json",
                "gcn": ".txt",
            }
            default_name = f"target_photometry{ext.get(format, '.csv')}"
            if self.result_dir is not None:
                path = self.result_dir / default_name
            elif self.observation_id is not None:
                path = Path("soap_results") / str(self.observation_id) / default_name
            else:
                path = Path(default_name)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        export_fn = getattr(self, f"to_{format}", None)
        if export_fn is None:
            raise ValueError(f"Unknown export format: {format!r}")
        return export_fn(path, **kwargs)
