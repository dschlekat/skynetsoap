# Skynet SOAP

**Skynet Science Observation Aperture Photometry** — a field-wide photometry pipeline for [Skynet](https://skynet.unc.edu) observations with automated source extraction, catalog cross-matching, and magnitude calibration.

## Features

- **sep-based source extraction** with automatic FWHM estimation and configurable aperture selection (FWHM-scaled, optimal, fixed, or multi-aperture)
- **Multi-aperture photometry** with curve-of-growth analysis for optimal aperture selection
- **Forced photometry** at user-specified sky positions (for transients, variables, or non-detections)
- **Limiting magnitude** calculation for all measurements (5-sigma detection threshold), with a robust blank-sky sampling method to handle crowded fields and extended sources
- **Photometric calibration** via Vizier catalog queries (APASS, PanSTARRS, SkyMapper), Jordi+2006 filter transformations, and inverse-variance weighted zeropoint computation
- **Sensor error model** propagating Poisson noise, read noise, background, and zeropoint uncertainty
- **Field-wide pipeline** — extracts and calibrates all sources per image; single-target lightcurves are a post-processing step
- **Debugging utilities** — multi-panel diagnostic plots, intermediate product saving
- **Smart caching** — reuses downloaded images and results across runs
- **Pluggable backends** — swap calibration or astrometry implementations via the config system
- **TOML configuration** for filters, catalogs, and pipeline parameters
- **Multiple export formats** — CSV, ECSV, Parquet, JSON, GCN circular format

## Installation

Requires:

- Python 3.12+
- Installation of the [`skynetapi`](https://github.com/astrodyl/skynetapi) Python package
- A Skynet API token

Clone the repository:

```bash
git clone https://github.com/dschlekat/soap.git
cd soap
```

### Option 1: uv (recommended)

Install `uv`:

- Official install guide: <https://docs.astral.sh/uv/getting-started/installation/>
- GitHub releases: <https://github.com/astral-sh/uv/releases>

Install project dependencies:

```bash
uv sync
```

### Option 2: native Python venv + pip

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the package and dependencies:

```bash
python -m pip install --upgrade pip
pip install -e .
```

Set your Skynet API token:

```bash
export SKYNET_API_TOKEN="your-token-here"
```

## Usage

### Quick start

```python
from astropy.coordinates import SkyCoord
from skynetsoap import Soap

s = Soap(observation_id=11920699, verbose=True)
s.download(after="2025-01-12")
result = s.run()
result.to_csv("all_sources.csv")

target = SkyCoord("12:49:37.598", "-63:32:09.8", unit=("hourangle", "deg"))
forced = s.run(forced_positions=[target])
target_result = forced.extract_target(target, forced_photometry=True)
target_result.to_csv("target_forced.csv")
```

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

### Cache management

Inspect and clean per-observation cache files:

```python
from skynetsoap import Soap

s = Soap(observation_id=12345)
print(s.cache_info())

# Clear only downloaded FITS files for this observation
s.clear_cache(images=True, results=False, confirm=False)

# Static cleanup for any observation ID
Soap.cleanup_observation(12345, images=True, results=True, confirm=False)
```

Enforce a disk budget across multiple observation runs:

```python
from skynetsoap import Soap

stats = Soap.prune_cache(
    max_total_size_mb=2048,  # keep total SOAP cache under 2 GB
    keep_recent=2,           # always keep the 2 most recent observations
    confirm=False,
)
print(stats)
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

### Magnitude systems (AB/Vega)

Each calibrated row includes a `cal_mag_system` column (`"AB"`, `"Vega"`, or `"Unknown"`).

To automatically convert Vega-based calibrated results to AB during pipeline execution:

```python
# One-off per run
result = s.run(convert_vega_to_ab=True)

# Or set as a config default
cfg = load_config(overrides={"calibration": {"convert_vega_to_ab": True}})
s = Soap(observation_id=12345, config=cfg)
result = s.run()
```

AB-Vega offsets and filter/system mappings are centrally defined in `skynetsoap/config/filters.toml` under:

- `photometry.band_mag_system`
- `photometry.ab_minus_vega_offsets`

These offsets are intentionally not runtime-overridable; update them in `filters.toml` when adopting newer literature values.

Enable robust blank-sky limiting-magnitude sampling (optional):

```python
cfg = load_config(
    overrides={
        "limiting_mag": {
            "method": "robust",
            "robust": {
                "n_samples": 2000,
                "mask_dilate_pixels": 3,
                "edge_buffer_pixels": 25,
                "sigma_estimator": "mad",
                "max_draws_multiplier": 20,
                # random_seed > 0 for deterministic sampling, <= 0 for random
                "random_seed": 123,
            },
        }
    }
)
```

In robust mode, the limiting-magnitude aperture radius is taken from the
pipeline photometry aperture for each measurement.

## Package Structure

``` text
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
