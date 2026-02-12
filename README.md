# SOAP

**Skynet Observation Aperture Photometry** — a field-wide photometry pipeline for [Skynet](https://skynet.unc.edu) observations with automated source extraction, catalog cross-matching, and magnitude calibration.

## Features

- **sep-based source extraction** with automatic FWHM estimation and configurable aperture selection (FWHM-scaled, optimal, or fixed)
- **Photometric calibration** via Vizier catalog queries (APASS, SkyMapper), Jordi+2006 filter transformations, and inverse-variance weighted zeropoint computation
- **CCD error model** propagating Poisson noise, read noise, background, and zeropoint uncertainty
- **Field-wide pipeline** — extracts and calibrates all sources per image; single-target lightcurves are a post-processing step
- **Pluggable backends** — swap calibration or astrometry implementations via Python Protocols
- **TOML configuration** for filters, catalogs, and pipeline parameters
- **Multiple export formats** — CSV, ECSV, Parquet, JSON, GCN circular format

## Installation

Requires Python 3.12+.

```bash
git clone https://github.com/dschlekat/soap.git
cd soap
uv sync
```

Set your Skynet API token:

```bash
export SKYNET_API_TOKEN="your-token-here"
```

## Usage

### Field-wide photometry

```python
from skynetsoap import Soap

s = Soap(observation_id=11920699, verbose=True)
s.download(after="2025-01-12")
result = s.run()
result.to_csv("all_sources.csv")
```

### Single-target extraction

```python
from astropy.coordinates import SkyCoord

target = SkyCoord("12:49:37.598", "-63:32:09.8", unit=("hourangle", "deg"))
target_result = result.extract_target(target, radius_arcsec=3.0)
target_result.to_csv("target.csv")
```

### Plotting

```python
s.plot(units="calibrated_mag", show=True)
```

### Export formats

```python
result.to_csv("output.csv")
result.to_ecsv("output.ecsv")
result.to_parquet("output.parquet")
result.to_gcn("gcn_table.txt", start_time=60400.0)
```

## Configuration

Default parameters are in `skynetsoap/config/defaults.toml`. Override with a custom TOML file:

```python
s = Soap(observation_id=12345, config_path="my_config.toml")
```

Or pass overrides directly:

```python
from skynetsoap import Soap, SOAPConfig
from skynetsoap.config import load_config

cfg = load_config(overrides={"aperture": {"mode": "optimal"}, "calibration": {"sigma_clip": 2.5}})
s = Soap(observation_id=12345, config=cfg)
```

## Package Structure

```
skynetsoap/
├── soap.py              # Pipeline orchestrator
├── config/              # TOML configuration + loader
├── core/                # FITSImage, PhotometryResult, coordinates, CCD errors
├── extraction/          # sep background, source extraction, aperture photometry
├── calibration/         # Vizier catalogs, filter transforms, zeropoint
├── astrometry/          # WCS validation, astrometry.net solver
├── io/                  # Skynet API, plotting, table export, caching
└── utils/               # Logging, date filtering
```

## License

MIT
