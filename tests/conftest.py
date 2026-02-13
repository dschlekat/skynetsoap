"""Shared pytest fixtures for the SOAP photometry pipeline test suite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from skynetsoap.config import SOAPConfig, load_config
from skynetsoap.core.result import PhotometryResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RNG_SEED = 42
_IMAGE_SIZE = 256
_BACKGROUND_MEAN = 100.0
_BACKGROUND_STD = 5.0
_CENTER_RA = 51.95  # degrees
_CENTER_DEC = 74.66  # degrees
_PIXEL_SCALE = 1.0 / 3600.0  # 1 arcsec/pixel in degrees

# Injected source definitions: (x, y, peak_flux, sigma)
_INJECTED_SOURCES = [
    (50.0, 50.0, 5000.0, 2.5),
    (200.0, 30.0, 3000.0, 2.2),
    (128.0, 128.0, 8000.0, 3.0),
    (80.0, 200.0, 2000.0, 2.0),
    (180.0, 180.0, 6000.0, 2.8),
    (30.0, 150.0, 1500.0, 2.3),
    (220.0, 100.0, 4000.0, 2.6),
    (100.0, 60.0, 3500.0, 2.4),
    (160.0, 220.0, 2500.0, 2.1),
    (240.0, 240.0, 7000.0, 3.2),
]


# ---------------------------------------------------------------------------
# Helper: 2-D Gaussian
# ---------------------------------------------------------------------------
def _gaussian_2d(
    shape: tuple[int, int],
    x0: float,
    y0: float,
    peak: float,
    sigma: float,
) -> np.ndarray:
    """Generate a 2-D Gaussian on a grid of the given *shape*."""
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    return peak * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma**2))


# ---------------------------------------------------------------------------
# Helper: build a WCS header
# ---------------------------------------------------------------------------
def _make_wcs_header(nx: int, ny: int) -> fits.Header:
    """Return a FITS header with a valid TAN WCS (~1 arcsec/pixel)."""
    w = WCS(naxis=2)
    w.wcs.crpix = [nx / 2.0, ny / 2.0]
    w.wcs.crval = [_CENTER_RA, _CENTER_DEC]
    w.wcs.cdelt = [-_PIXEL_SCALE, _PIXEL_SCALE]  # RA axis is inverted
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cunit = ["deg", "deg"]
    return w.to_header()


# ---------------------------------------------------------------------------
# Helper: build the image data array
# ---------------------------------------------------------------------------
def _make_image_data(rng: np.random.Generator) -> np.ndarray:
    """Return a float32 image with background noise and injected sources."""
    data = rng.normal(
        _BACKGROUND_MEAN, _BACKGROUND_STD, (_IMAGE_SIZE, _IMAGE_SIZE)
    ).astype(np.float32)
    for x0, y0, peak, sigma in _INJECTED_SOURCES:
        data += _gaussian_2d((_IMAGE_SIZE, _IMAGE_SIZE), x0, y0, peak, sigma).astype(
            np.float32
        )
    return data


# ===================================================================
# Fixtures
# ===================================================================


@dataclass
class InjectedSources:
    """Container for injected source metadata."""

    x: np.ndarray
    y: np.ndarray
    peak_flux: np.ndarray
    sigma: np.ndarray
    sky_coords: SkyCoord


@pytest.fixture()
def known_sources() -> InjectedSources:
    """Return pixel positions, fluxes, and sky coordinates of injected sources."""
    xs = np.array([s[0] for s in _INJECTED_SOURCES])
    ys = np.array([s[1] for s in _INJECTED_SOURCES])
    peaks = np.array([s[2] for s in _INJECTED_SOURCES])
    sigmas = np.array([s[3] for s in _INJECTED_SOURCES])

    # Compute sky positions from the WCS
    wcs_header = _make_wcs_header(_IMAGE_SIZE, _IMAGE_SIZE)
    wcs = WCS(wcs_header)
    sky = wcs.pixel_to_world(xs, ys)

    return InjectedSources(x=xs, y=ys, peak_flux=peaks, sigma=sigmas, sky_coords=sky)


@pytest.fixture()
def mock_fits_image(tmp_path: Path) -> Path:
    """Create a synthetic FITS image with WCS and injected point sources.

    Returns the path to the written FITS file.
    """
    rng = np.random.default_rng(_RNG_SEED)
    data = _make_image_data(rng)
    header = _make_wcs_header(_IMAGE_SIZE, _IMAGE_SIZE)

    # Observation metadata
    header["FILTER"] = "V"
    header["GAIN"] = 1.5
    header["RDNOISE"] = 5.0
    header["EXPTIME"] = 30.0
    header["TELESCOP"] = "TestScope"
    header["MJD-OBS"] = 60500.0
    header["JD"] = 60500.0 + 2400000.5

    hdu = fits.PrimaryHDU(data=data, header=header)
    path = tmp_path / "test_image.fits"
    hdu.writeto(path, overwrite=True)
    return path


@pytest.fixture()
def mock_fits_image_no_wcs(tmp_path: Path) -> Path:
    """Create a synthetic FITS image *without* WCS headers.

    Returns the path to the written FITS file.
    """
    rng = np.random.default_rng(_RNG_SEED)
    data = _make_image_data(rng)

    header = fits.Header()
    header["FILTER"] = "V"
    header["GAIN"] = 1.5
    header["RDNOISE"] = 5.0
    header["EXPTIME"] = 30.0
    header["TELESCOP"] = "TestScope"
    header["MJD-OBS"] = 60500.0
    header["JD"] = 60500.0 + 2400000.5

    hdu = fits.PrimaryHDU(data=data, header=header)
    path = tmp_path / "test_image_nowcs.fits"
    hdu.writeto(path, overwrite=True)
    return path


@pytest.fixture()
def soap_config() -> SOAPConfig:
    """Return the default SOAPConfig loaded from the package TOML files."""
    return load_config()


@pytest.fixture()
def photometry_result() -> PhotometryResult:
    """Return a PhotometryResult pre-populated with 5 realistic measurements."""
    result = PhotometryResult(observation_id=12345)

    measurements = [
        {
            "image_file": "frame_001.fits",
            "telescope": "TestScope",
            "filter": "V",
            "exptime": 30.0,
            "mjd": 60500.0,
            "jd": 60500.0 + 2400000.5,
            "x_pix": 128.3,
            "y_pix": 127.9,
            "ra": 51.950,
            "dec": 74.660,
            "flux": 45000.0,
            "flux_err": 250.0,
            "snr": 180.0,
            "ins_mag": -11.63,
            "ins_mag_err": 0.006,
            "calibrated_mag": 13.42,
            "calibrated_mag_err": 0.015,
            "zeropoint": 25.05,
            "zeropoint_err": 0.012,
            "aperture_radius": 7.5,
            "fwhm": 3.0,
            "n_cal_stars": 18,
            "is_forced": False,
            "flag": 0,
        },
        {
            "image_file": "frame_002.fits",
            "telescope": "TestScope",
            "filter": "V",
            "exptime": 30.0,
            "mjd": 60500.01,
            "jd": 60500.01 + 2400000.5,
            "x_pix": 128.5,
            "y_pix": 128.1,
            "ra": 51.950,
            "dec": 74.660,
            "flux": 44200.0,
            "flux_err": 260.0,
            "snr": 170.0,
            "ins_mag": -11.61,
            "ins_mag_err": 0.006,
            "calibrated_mag": 13.44,
            "calibrated_mag_err": 0.016,
            "zeropoint": 25.05,
            "zeropoint_err": 0.013,
            "aperture_radius": 7.5,
            "fwhm": 3.1,
            "n_cal_stars": 17,
            "is_forced": False,
            "flag": 0,
        },
        {
            "image_file": "frame_003.fits",
            "telescope": "TestScope",
            "filter": "B",
            "exptime": 45.0,
            "mjd": 60500.02,
            "jd": 60500.02 + 2400000.5,
            "x_pix": 128.1,
            "y_pix": 128.4,
            "ra": 51.951,
            "dec": 74.660,
            "flux": 32000.0,
            "flux_err": 300.0,
            "snr": 106.7,
            "ins_mag": -11.26,
            "ins_mag_err": 0.010,
            "calibrated_mag": 14.01,
            "calibrated_mag_err": 0.022,
            "zeropoint": 25.27,
            "zeropoint_err": 0.018,
            "aperture_radius": 7.0,
            "fwhm": 2.8,
            "n_cal_stars": 15,
            "is_forced": False,
            "flag": 0,
        },
        {
            "image_file": "frame_004.fits",
            "telescope": "TestScope",
            "filter": "R",
            "exptime": 20.0,
            "mjd": 60500.03,
            "jd": 60500.03 + 2400000.5,
            "x_pix": 128.6,
            "y_pix": 127.7,
            "ra": 51.949,
            "dec": 74.660,
            "flux": 58000.0,
            "flux_err": 220.0,
            "snr": 263.6,
            "ins_mag": -11.91,
            "ins_mag_err": 0.004,
            "calibrated_mag": 12.88,
            "calibrated_mag_err": 0.012,
            "zeropoint": 24.79,
            "zeropoint_err": 0.010,
            "aperture_radius": 7.5,
            "fwhm": 3.0,
            "n_cal_stars": 20,
            "is_forced": False,
            "flag": 0,
        },
        {
            "image_file": "frame_005.fits",
            "telescope": "TestScope",
            "filter": "V",
            "exptime": 30.0,
            "mjd": 60500.04,
            "jd": 60500.04 + 2400000.5,
            "x_pix": 128.2,
            "y_pix": 128.0,
            "ra": 51.950,
            "dec": 74.660,
            "flux": 800.0,
            "flux_err": 250.0,
            "snr": 3.2,
            "ins_mag": -7.26,
            "ins_mag_err": 0.339,
            "calibrated_mag": 17.79,
            "calibrated_mag_err": 0.340,
            "zeropoint": 25.05,
            "zeropoint_err": 0.012,
            "aperture_radius": 7.5,
            "fwhm": 3.0,
            "n_cal_stars": 18,
            "is_forced": True,
            "flag": 0,
        },
    ]

    for m in measurements:
        result.add_measurement(**m)

    return result


@pytest.fixture()
def mock_reference_catalog(known_sources: InjectedSources) -> pd.DataFrame:
    """Return a mock reference catalog DataFrame overlapping injected sources.

    Uses the first 8 injected source positions with realistic catalog
    magnitudes in the 12-16 mag range.
    """
    rng = np.random.default_rng(_RNG_SEED + 1)
    n = 8  # use first 8 sources

    ra = known_sources.sky_coords.ra.deg[:n]
    dec = known_sources.sky_coords.dec.deg[:n]

    # Generate realistic magnitudes
    v_mag = rng.uniform(12.0, 16.0, n)
    b_mag = v_mag + rng.uniform(0.3, 1.0, n)  # B - V colour index
    r_mag = v_mag - rng.uniform(0.2, 0.6, n)  # V - R colour index

    return pd.DataFrame(
        {
            "RAJ2000": ra,
            "DEJ2000": dec,
            "Vmag": np.round(v_mag, 3),
            "e_Vmag": np.round(rng.uniform(0.01, 0.05, n), 3),
            "Bmag": np.round(b_mag, 3),
            "e_Bmag": np.round(rng.uniform(0.01, 0.06, n), 3),
            "Rmag": np.round(r_mag, 3),
            "e_Rmag": np.round(rng.uniform(0.01, 0.05, n), 3),
        }
    )


# ---------------------------------------------------------------------------
# Fixture aliases for backwards compatibility
# ---------------------------------------------------------------------------
@pytest.fixture
def default_config(soap_config):
    """Alias for soap_config."""
    return soap_config


@pytest.fixture
def sample_result(photometry_result):
    """Alias for photometry_result."""
    return photometry_result


@pytest.fixture
def mock_fits_no_wcs(mock_fits_image_no_wcs):
    """Alias for mock_fits_image_no_wcs."""
    return mock_fits_image_no_wcs
