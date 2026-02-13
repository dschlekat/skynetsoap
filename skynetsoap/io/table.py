"""GCN circular format table export.

Ported from models/table.py.
"""

from __future__ import annotations

from pathlib import Path

from astropy.table import QTable


def write_gcn_table(
    table: QTable,
    path: str | Path,
    start_time: float | None = None,
    all_results: bool = False,
) -> Path:
    """Write a GCN-formatted text table.

    Parameters
    ----------
    table : QTable
        Photometry results table with columns: mjd, telescope, filter,
        exptime, calibrated_mag, calibrated_mag_err.
    path : str or Path
        Output file path.
    start_time : float, optional
        Reference MJD. If provided, the time column shows seconds since
        this time instead of MJD.
    all_results : bool
        If False, only the first result per filter is included.

    Returns
    -------
    Path
    """
    path = Path(path)

    if start_time is None:
        header = "MJD             | Telescope | Filter | Exposure Duration | Mag     | Mag Error"
    else:
        header = "Time Since GRB | Telescope | Filter | Exposure Duration | Mag     | Mag Error"

    rows: list[list[str]] = []
    seen_filters: set[str] = set()

    for row in table:
        filter_name = str(row["filter"])

        if not all_results:
            if filter_name in seen_filters:
                continue
            seen_filters.add(filter_name)

        if start_time is None:
            mjd_val = float(row["mjd"])
            time_str = f"{mjd_val:.5f} MJD"
        else:
            delta_s = (float(row["mjd"]) - start_time) * 86400
            time_str = f"{int(delta_s)}s"

        telescope = str(row["telescope"])
        exp_len = f"{int(float(row['exptime']))}s"
        mag = f"{float(row['calibrated_mag']):.3f}"
        mag_err = f"{float(row['calibrated_mag_err']):.3f}"

        rows.append([time_str, telescope, filter_name, exp_len, mag, mag_err])

    with open(path, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write("{:<14} | {:<9} | {:<6} | {:<17} | {:<7} | {:<9}\n".format(*r))

    return path
