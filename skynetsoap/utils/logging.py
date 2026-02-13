"""Logging configuration for SOAP pipeline."""

from __future__ import annotations

import logging

from tqdm import tqdm

_LOGGER_NAME = "soap"


class _TqdmLoggingHandler(logging.Handler):
    """Logging handler that preserves tqdm progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


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
    logger.propagate = False

    if verbose:
        level = "INFO"

    logger.setLevel(getattr(logging, level.upper(), logging.WARNING))

    needs_tqdm = verbose
    has_tqdm_handler = any(isinstance(h, _TqdmLoggingHandler) for h in logger.handlers)
    has_stream_handler = any(isinstance(h, logging.StreamHandler) and not isinstance(h, _TqdmLoggingHandler) for h in logger.handlers)

    if (needs_tqdm and not has_tqdm_handler) or (not needs_tqdm and not has_stream_handler):
        logger.handlers.clear()

        handler: logging.Handler = _TqdmLoggingHandler() if needs_tqdm else logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_logger() -> logging.Logger:
    """Return the SOAP logger (must call setup_logging first)."""
    return logging.getLogger(_LOGGER_NAME)
