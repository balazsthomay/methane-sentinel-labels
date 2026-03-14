# methane-sentinel-labels

Cross-sensor methane plume label dataset generator. Takes confirmed methane point-source detections from [Carbon Mapper](https://carbonmapper.org/) and matches them to temporally close [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) imagery, producing georeferenced patches with verified emission labels.

The output is a curated dataset for validating synthetic training data used in satellite-based methane detection — bridging the gap between simulated plumes and real-world observations.

## Why this matters

Methane emissions from oil and gas infrastructure are [massively underreported](https://www.iea.net/reports/global-methane-tracker-2024), with industry estimates falling ~70% below actual measurements. Satellite-based detection using Sentinel-2's SWIR bands (B11 at 1610nm, B12 at 2190nm) enables facility-level monitoring at global scale with a 5-day revisit cadence.

Training detectors on synthetic plumes (via radiative transfer simulation) is effective but requires real confirmed-plume imagery for validation. This pipeline produces exactly that: Sentinel-2 patches centered on Carbon Mapper's confirmed detections, with emission rates, temporal proximity, and cloud-free quality metadata.

## Full-scale results

The pipeline processed the full Carbon Mapper catalog:

| Metric | Value |
|---|---|
| Detections ingested | 500 |
| Patches extracted | 615 |
| Unique detections matched | 337 |
| Unique Sentinel-2 scenes | 230 |
| Median time delta | 48.7 hours |
| Emission rate range | 98 – 9,168 kg/hr |
| Median emission rate | 779 kg/hr |
| Mean cloud-free fraction | 59.8% |
| Dataset size | 464 MB |

## Installation

Requires Python >= 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/thomaybalazs/methane-sentinel-labels.git
cd methane-sentinel-labels
uv sync
```

AWS credentials are needed for S3 access to Sentinel-2 COGs (free, no special permissions required — just a valid AWS account):

```bash
aws configure  # or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
```

Carbon Mapper API credentials are optional (the plumes endpoint is public):

```bash
export CM_EMAIL="your@email.com"
export CM_PASSWORD="your-password"
```

## Usage

### Full pipeline

```bash
# Run everything: ingest → match → extract → assemble → visualize
uv run methane-sentinel-labels run --output-dir output --limit 20

# Full-scale (no limit, takes several hours)
uv run methane-sentinel-labels run --output-dir output
```

### Individual steps

```bash
# Fetch detections from Carbon Mapper
uv run methane-sentinel-labels ingest --output-dir output

# Match detections to Sentinel-2 scenes
uv run methane-sentinel-labels match --output-dir output
```

### Configuration flags

| Flag | Default | Description |
|---|---|---|
| `--output-dir` | `output` | Output directory |
| `--limit` | None | Max detections to process |
| `--max-time-delta` | 120.0 | Max hours between detection and Sentinel-2 acquisition |
| `--max-cloud-cover` | 30.0 | Max scene-level cloud cover (%) |
| `--patch-size` | 2560.0 | Patch half-size in meters (default: 5.12 km patches) |
| `--min-cloud-free` | 0.5 | Min patch-level cloud-free fraction |
| `-v` | off | Verbose (debug) logging |

### Resumability

The pipeline automatically skips patches that already exist on disk. If a run is interrupted (e.g., by an S3 timeout), simply re-run the same command — it will pick up where it left off.

## Output structure

```
output/
├── detections.parquet      # Raw Carbon Mapper detections
├── manifest.parquet        # Patch metadata (615 rows × 10 columns)
├── summary.json            # Aggregate statistics
├── patches/                # Multi-band GeoTIFF patches
│   ├── {detection_id}_{scene_id}.tif
│   └── ...
└── viz/                    # Side-by-side RGB + SWIR visualizations
    ├── {detection_id}_{scene_id}.png
    └── ...
```

### Manifest schema

| Column | Type | Description |
|---|---|---|
| `detection_source_id` | string | Carbon Mapper plume ID |
| `scene_id` | string | Sentinel-2 scene identifier |
| `patch_path` | string | Relative path to GeoTIFF |
| `latitude` | float | Detection latitude (WGS84) |
| `longitude` | float | Detection longitude (WGS84) |
| `emission_rate_kg_hr` | float | Emission rate from Carbon Mapper (kg/hr) |
| `time_delta_hours` | float | Hours between detection and S2 acquisition |
| `cloud_free_fraction` | float | Patch-level cloud-free fraction (SCL-based) |
| `crs` | string | UTM CRS (e.g., EPSG:32613) |
| `bbox` | tuple | Bounding box in UTM coordinates |

### GeoTIFF patches

Each patch is a 256 x 256 pixel, 6-band GeoTIFF at 20m resolution (5.12 km x 5.12 km):

| Band | Sentinel-2 | Wavelength | Resolution | Use |
|---|---|---|---|---|
| B02 | Blue | 490 nm | 10m → 20m | RGB composite |
| B03 | Green | 560 nm | 10m → 20m | RGB composite |
| B04 | Red | 665 nm | 10m → 20m | RGB composite |
| B8A | NIR | 865 nm | 20m | SWIR false-color |
| B11 | SWIR-1 | 1610 nm | 20m | Methane absorption |
| B12 | SWIR-2 | 2190 nm | 20m | Methane absorption |

The detection point is always at the center of the patch.

### Visualizations

Side-by-side composites with the detection marked as a red crosshair:
- **Left**: RGB true color (B04, B03, B02)
- **Right**: SWIR false color (B12, B11, B8A) — highlights methane-sensitive wavelengths

## Architecture

```
Carbon Mapper API ──[A: Ingest]──→ Detection[]
                                       │
Earth Search STAC ──[B: Match]────→ SceneMatch[]
                                       │
Sentinel-2 COGs  ──[C: Extract]──→ PatchRecord[] + GeoTIFFs
                                       │
                   [D: Assemble]──→ manifest.parquet + summary.json
                                       │
                   [E: Visualize]─→ RGB + SWIR PNGs
```

Each stage filters progressively: A deduplicates detections, B filters by time delta and cloud cover, C filters by patch-level cloud-free fraction (SCL mask).

**Data sources:**
- [Carbon Mapper API](https://api.carbonmapper.org/) — confirmed CH4 point-source detections with emission rates
- [Element 84 Earth Search](https://earth-search.aws.element84.com/v1) — Sentinel-2 L2A STAC catalog (free, no auth)
- Sentinel-2 COGs on S3 (`us-west-2`) — cloud-optimized GeoTIFFs via `rasterio`

## Testing

```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=methane_sentinel_labels --cov-report=term-missing

# Integration tests (hits real APIs)
uv run pytest -m integration
```

75 tests, 88% coverage.

## License and data attribution

- **Sentinel-2 imagery**: [Copernicus Sentinel Data](https://scihub.copernicus.eu/twiki/pub/SciHubWebPortal/TermsConditions/Sentinel_Data_Terms_and_Conditions.pdf) — free for all uses including redistribution
- **Carbon Mapper detections**: referenced as metadata (source IDs, emission rates); raw data not redistributed
- **This dataset**: contains Sentinel-2 patches with derived metadata from Carbon Mapper labels
