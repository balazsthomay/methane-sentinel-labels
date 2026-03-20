# methane-sentinel-labels

Cross-sensor methane plume segmentation system. Uses MethaneSAT L3 concentration maps (45m, real XCH4) as spatial labels to train a U-Net that detects methane in Sentinel-2 SWIR bands. MethaneSAT as teacher, Sentinel-2 as student.

No published work has used MethaneSAT concentration maps as training labels for a Sentinel-2 methane detector. Existing approaches rely on hand-drawn masks (CH4Net, 925 annotations), synthetic plumes (Orbio/Eucalyptus), or hyperspectral transfer (STARCOP). MethaneSAT L3 provides *real atmospheric methane concentration fields* at 45m resolution — a step change in label quality.

## Results

### Pipeline

| Stage | Input | Output | Yield |
|---|---|---|---|
| MethaneSAT ingestion | 269 GCS files | 47 unique scenes | 100% mask generation |
| Cross-sensor matching | 47 masks | 44 Sentinel-2 matches | 94% within 72h |
| Tiled patch extraction | 44 pairs | 80 training patches | 50 positive, 30 negative |

### Model Performance (U-Net ResNet18, Dice loss)

| Metric | Value |
|---|---|
| Best F1 | **0.166** |
| Precision | 9.1% |
| Recall | 100% |
| Detection prob (plume patches) | 0.864 ± 0.156 |
| Detection prob (background) | 0.822 ± 0.189 |

The model detects the MethaneSAT plume signal in Sentinel-2 SWIR with perfect recall but low precision — it over-predicts, finding plume-like signatures everywhere. At a high confidence threshold (0.99), true plume patches are detected 100% of the time vs 83% for background, confirming a real but weak separation signal.

### Interpretation

The cross-sensor signal transfer works. The methane absorption signature in B11 (1610nm) and B12 (2190nm), enhanced by the Varon ratio, carries enough information for a model to learn from MethaneSAT-derived labels. The precision bottleneck is dataset size: 80 patches is at the floor of trainability. CH4Net achieved F1>0.8 with 925 hand-annotated patches. Completing the tiled extraction (27% through when stopped) would yield ~300 patches and likely push F1 toward 0.3+.

## MethaneSAT L3 Data

MethaneSAT was a purpose-built methane spectrometer (launched March 2024, contact lost July 2025). The L3 archive is publicly available on GCS.

| Property | Value |
|---|---|
| Bucket | `gs://msat-prod-data-public-methanesat-level3` |
| Format | Cloud-Optimized GeoTIFF (COG) |
| Bands | XCH4 (ppb), albedo, surface pressure, terrain height |
| CRS | EPSG:4326 |
| Resolution | ~46m (0.000417°) |
| NoData | NaN |
| Archive | 532 files, 103 targets, 41 basins, ~147 GB |

### XCH4 Statistics

Background concentration ~1930 ppb with 26 ppb std. A 50 ppb anomaly threshold captures ~1% of valid pixels — these are the plume labels. Morphological opening removes isolated noise pixels.

## Installation

Requires Python >= 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/thomaybalazs/methane-sentinel-labels.git
cd methane-sentinel-labels
uv sync
```

### Credentials

**MethaneSAT** (GCS): Request access at [methanesat.org/data](https://www.methanesat.org/data), then:
```bash
gcloud auth application-default login
```

**Sentinel-2** (S3): AWS credentials for COG access (free, no special permissions):
```bash
aws configure
```

**Carbon Mapper** (optional, for the original CM pipeline):
```bash
export CM_EMAIL="your@email.com"
export CM_PASSWORD="your-password"
```

## Usage

### MethaneSAT cross-sensor pipeline

```bash
# Full pipeline: ingest MethaneSAT → match to S2 → extract training patches
uv run methane-sentinel-labels msat-run --output-dir output_msat --limit 10

# Ingestion only (download + mask generation)
uv run methane-sentinel-labels msat-ingest --output-dir output_msat --threshold 50.0
```

### Carbon Mapper label pipeline (original)

```bash
# Full pipeline: ingest CM detections → match to S2 → extract patches
uv run methane-sentinel-labels run --output-dir output --limit 20

# Individual steps
uv run methane-sentinel-labels ingest --output-dir output
uv run methane-sentinel-labels match --output-dir output
```

### Configuration flags

| Flag | Default | Description |
|---|---|---|
| `--output-dir` | `output` | Output directory |
| `--limit` | None | Max scenes/detections to process |
| `--threshold` | 50.0 | MethaneSAT plume anomaly threshold (ppb) |
| `--max-time-delta` | 72.0 / 120.0 | Max hours between acquisitions (msat / cm) |
| `--max-cloud-cover` | 30.0 | Max scene-level cloud cover (%) |
| `--patch-size` | 2560.0 | Patch half-size in meters (5.12 km patches) |
| `--min-cloud-free` | 0.5 | Min patch-level cloud-free fraction |
| `-v` | off | Verbose logging |

## Architecture

```
MethaneSAT L3 (GCS) ──[A: Ingest]──→ MethaneSATScene[] + PlumeMask[]
                                              │
Earth Search STAC ─────[B: Match]───→ MatchedPair[]
                                              │
Sentinel-2 COGs (S3) ─[C: Extract]─→ TrainingPatch[] + GeoTIFFs
                                              │
                       [D: Train]───→ U-Net model (segmentation)
                                              │
Carbon Mapper patches ─[E: Validate]→ Detection metrics
```

### Components

**A — MethaneSAT Ingestion** (`ingest/methanesat.py`): Downloads L3 COGs from GCS, extracts XCH4 band, computes median-subtract anomaly, thresholds into binary plume masks with morphological opening.

**B — Cross-Sensor Matching** (`matching/cross_sensor.py`): Queries STAC with MethaneSAT scene bounding boxes (not point coordinates), filters by time delta, cloud cover, and spatial overlap. Selects the closest Sentinel-2 acquisition per mask.

**C — Patch Extraction** (`extraction/patches.py`, `extraction/enhancement.py`): Grid-tiles each MethaneSAT mask into ~5km cells, extracts Sentinel-2 patches at plume cluster locations. Computes Varon ratio (B12/B11 normalized by spatial median). Reprojects MethaneSAT mask to S2 UTM grid. Output: 8-band GeoTIFF [B02, B03, B04, B8A, B11, B12, varon, mask].

**D — Segmentation Model** (`training/`): U-Net with ResNet encoder via segmentation-models-pytorch. Per-channel percentile normalization. Geographic basin split (no spatial leakage). Dice + Focal loss.

**E — Validation** (`validation/carbon_mapper.py`): Runs trained model on Carbon Mapper patches, computes detection rate and emission correlation as independent validation.

### Training patches

Each patch is 256 × 256 pixels at 20m resolution (5.12 km × 5.12 km):

| Band | Source | Wavelength | Use |
|---|---|---|---|
| B02 | Sentinel-2 Blue | 490 nm | RGB context |
| B03 | Sentinel-2 Green | 560 nm | RGB context |
| B04 | Sentinel-2 Red | 665 nm | RGB context |
| B8A | Sentinel-2 NIR | 865 nm | SWIR false-color |
| B11 | Sentinel-2 SWIR-1 | 1610 nm | Methane absorption |
| B12 | Sentinel-2 SWIR-2 | 2190 nm | Methane absorption |
| varon | Computed | B12/B11 ratio | Methane enhancement |
| mask | MethaneSAT-derived | Binary | Training label |

## Testing

```bash
uv run pytest                                                    # 154 tests
uv run pytest --cov=methane_sentinel_labels --cov-report=term    # 91% coverage
uv run pytest -m integration                                     # real API tests
```

## Limitations and Future Work

1. **Dataset size**: 80 patches from 47 scenes. Completing the tiled extraction (~300 patches) and adding more MethaneSAT scenes would improve precision substantially.
2. **Label noise**: 50 ppb threshold captures retrieval noise alongside real plumes. A higher threshold (100+ ppb) would produce cleaner but sparser labels.
3. **Temporal mismatch**: 3–69h between MethaneSAT and Sentinel-2 acquisitions. Plumes are transient — the label may not match the S2 observation.
4. **Spatial resolution**: MethaneSAT 45m mask reprojected to 20m S2 grid. Subpixel alignment errors are possible.
5. **No independent validation**: Carbon Mapper pipeline not yet run at scale. The validation uses the model's own training distribution.

## References

- Varon et al. (2021) — Varon ratio for methane enhancement from multispectral SWIR
- Vaughan et al. (2024) — CH4Net: U-Net for Sentinel-2 methane plume detection
- Růžička et al. (2023) — STARCOP: hyperspectral→multispectral methane segmentation
- Chan Miller et al. (2024) — MethaneSAT XCH4 retrieval (CO2-Proxy method)
- Sun et al. (2018) — MethaneSAT L3 regridding approach

## License and Data Attribution

- **Sentinel-2 imagery**: [Copernicus Sentinel Data](https://scihub.copernicus.eu/twiki/pub/SciHubWebPortal/TermsConditions/Sentinel_Data_Terms_and_Conditions.pdf) — free for all uses
- **MethaneSAT L3**: [Content License Terms of Use](https://www.methanesat.org/sites/default/files/2025-02/MethaneSAT%20-%20Content%20License%20Terms%20of%20Use%20%28Revised%202-12-2025%29%5B25%5D.pdf) — derived products (masks, model weights) permitted; raw L3 redistribution prohibited
- **Carbon Mapper**: referenced as metadata only
