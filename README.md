# SOAP

**Science Observation Aperture Photometry** — a field-wide photometry pipeline for [Skynet](https://skynet.unc.edu) observations with automated source extraction, catalog cross-matching, and magnitude calibration.

## Features

- **sep-based source extraction** with automatic FWHM estimation and configurable aperture selection (FWHM-scaled, optimal, fixed, or multi-aperture)
- **Multi-aperture photometry** with curve-of-growth analysis for optimal aperture selection
- **Forced photometry** at user-specified sky positions (for transients, variables, or non-detections)
- **Limiting magnitude** calculation for all measurements (5-sigma detection threshold)
- **Photometric calibration** via Vizier catalog queries (APASS, PanSTARRS, SkyMapper), Jordi+2006 filter transformations, and inverse-variance weighted zeropoint computation
- **CCD error model** propagating Poisson noise, read noise, background, and zeropoint uncertainty
- **Field-wide pipeline** — extracts and calibrates all sources per image; single-target lightcurves are a post-processing step
- **Debugging utilities** — multi-panel diagnostic plots, intermediate product saving
- **Smart caching** — reuses downloaded images and results across runs
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

### Multi-aperture photometry

Test multiple aperture radii to find the optimal aperture for each source:

```python
from skynetsoap import Soap, load_config

# Configure multi-aperture mode
cfg = load_config(overrides={
    "aperture": {
        "mode": "multi",
        "radii": [3.0, 5.0, 7.0, 10.0, 15.0],  # Test these radii
        "keep_all": False  # Auto-select best aperture (1 row per source)
    }
})

s = Soap(observation_id=12345, config=cfg)
result = s.run()

# Or keep all apertures for curve-of-growth analysis
cfg_cog = load_config(overrides={"aperture": {"mode": "multi", "keep_all": True}})
s_cog = Soap(observation_id=12345, config=cfg_cog)
result_cog = s_cog.run()  # Returns N rows per source (one per aperture)

# Filter to specific aperture
result_ap3 = result_cog.filter_by_aperture(aperture_id=2)  # 7.0 pixel radius
```

### Forced photometry

Measure flux at specified positions, even if no source is detected:

```python
from astropy.coordinates import SkyCoord

# Define positions (e.g., expected transient location)
positions = [
    SkyCoord("12:34:56.78", "+45:12:34.5", unit=("hourangle", "deg")),
    SkyCoord("01:23:45.67", "-10:20:30.4", unit=("hourangle", "deg")),
]

# Run with forced photometry
result = s.run(forced_positions=positions)

# Extract forced measurements
target = result.extract_target(positions[0], forced_photometry=True, snr_threshold=3.0)

# Check limiting magnitude for non-detections
print(f"Limiting magnitude: {target.table['limiting_mag'][0]:.2f} mag")
```

### Debugging

Generate diagnostic plots for individual images:

```python
# Create multi-panel debug plot
s.debug_image("soap_images/12345/r67890.fits", show=True)

# Enable automatic debug plots during pipeline run
cfg_debug = load_config(overrides={"debug": {"enabled": True}})
s_debug = Soap(observation_id=12345, config=cfg_debug)
result = s_debug.run()  # Saves debug plots to soap_debug/12345/
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
