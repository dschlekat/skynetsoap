"""Logging configuration for SOAP pipeline."""

from __future__ import annotations

import logging

_LOGGER_NAME = "soap"


def setup_logging(level: str = "WARNING", verbose: bool = False) -> logging.Logger:
    """Configure and return the SOAP logger.

    Parameters
    ----------
    level : str
        Logging level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    verbose : bool
        If True, forces INFO level and enables tqdm progress bars.
    """
    logger = logging.getLogger(_LOGGER_NAME)

    if verbose:
        level = "INFO"

    logger.setLevel(getattr(logging, level.upper(), logging.WARNING))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_logger() -> logging.Logger:
    """Return the SOAP logger (must call setup_logging first)."""
    return logging.getLogger(_LOGGER_NAME)
