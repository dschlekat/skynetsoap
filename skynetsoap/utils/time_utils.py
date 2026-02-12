"""Date filtering helpers for astropy Tables."""

from __future__ import annotations

from astropy.table import Table
from astropy.time import Time, TimeDelta


def filter_table_by_date(
    table: Table,
    after: str | float | None = None,
    before: str | float | None = None,
    days_ago: float | None = None,
    time_column: str = "mjd",
) -> Table:
    """Filter an astropy Table by date range.

    Parameters
    ----------
    table : Table
        Table with a time column in MJD.
    after : str or float, optional
        Keep rows after this time (ISO string or MJD float).
    before : str or float, optional
        Keep rows before this time (ISO string or MJD float).
    days_ago : float, optional
        Keep rows from this many days ago until now. Mutually exclusive
        with *after*/*before*.
    time_column : str
        Name of the MJD column.

    Returns
    -------
    Table
        Filtered copy of the input table.
    """
    if days_ago is not None:
        if after is not None or before is not None:
            raise ValueError("Specify either 'days_ago' or 'after'/'before', not both.")
        after_mjd = (Time.now() - TimeDelta(days_ago, format="jd")).mjd
        return table[table[time_column] > after_mjd]

    mask = [True] * len(table)

    if after is not None:
        after_mjd = Time(after).mjd if isinstance(after, str) else float(after)
        mask = table[time_column] > after_mjd

    result = table[mask]

    if before is not None:
        before_mjd = Time(before).mjd if isinstance(before, str) else float(before)
        result = result[result[time_column] < before_mjd]

    return result
