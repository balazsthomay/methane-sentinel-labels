# Findings

## Phase 0: MethaneSAT L3 Data Inspection (2026-03-18)

### Access
- Bucket: `gs://msat-prod-data-public-methanesat-level3`
- Requires authentication (GCS application default credentials)
- GDAL access via `/vsigs/` with `GOOGLE_APPLICATION_CREDENTIALS` env var
- Python GCS client needs explicit `project=` parameter

### Format
- **Cloud-Optimized GeoTIFF (COG)**
- Path pattern: `cog_gee/t{target_id}/{YYYYMMDD}/{collection_id}/{processing_id}/45m/MSAT_L3_45m_COG_GEE_{collection_id}_{processing_id}_{version}_{timestamp}.tif`
- 532 TIF files across 103 targets, ~147 GB total
- Individual files: 150–720 MB

### Bands (4 × float32)
| Band | Description | Units |
|------|-------------|-------|
| 1 | XCH4 — column-averaged dry-air CH4 mole fraction | ppb |
| 2 | albedo — clear-sky surface albedo at 1622nm | unitless [0,1] |
| 3 | surface pressure | hPa |
| 4 | terrain height | km |

### Spatial Properties
- **CRS**: EPSG:4326 (WGS84 geographic)
- **Resolution**: ~0.000417° ≈ 46.4m at equator (matches spec's "45m")
- **NoData**: NaN (float32)
- **Coverage**: 14.2% valid pixels in sample scene (rest NaN — swath gaps)

### XCH4 Statistics (sample scene t110/20250117)
- Range: 1412–2516 ppb
- Mean: 1927 ppb, Median: 1930 ppb, Std: 26 ppb
- P1/P99: 1849/1982 ppb

### Anomaly Analysis (median-subtract)
- Std: 26 ppb (consistent with L2 precision of 22–35 ppb)
- P95: +30 ppb, P99: +51 ppb
- Pixels > 50 ppb anomaly: 74,365 (1.1% of valid)
- Pixels > 100 ppb anomaly: 7,982 (0.12%)
- Pixels > 200 ppb anomaly: 1,018 (0.015%)

### Key Metadata Tags
- `target_id`, `collection_id`, `processing_id`
- `time_coverage_start`, `time_coverage_end` (ISO 8601)
- `platform`: "MethaneSAT"

### Decision: Threshold
A 50 ppb anomaly threshold captures ~1% of valid pixels — reasonable for plume detection. Will make configurable (default 50 ppb). Morphological opening needed to remove noise at this threshold.

### Decision: Dependencies
COG/GeoTIFF format → use existing `rasterio`. No need for `xarray`/`netCDF4`/`zarr`. Add `scipy` for morphological filtering only.

### Bug Fix: Bbox Ordering
MethaneSAT COGs have inverted Y-axis in rasterio bounds (`bottom > top` in lat). Fixed `parse_scene` to normalize bbox to `(lon_min, lat_min, lon_max, lat_max)` order required by STAC API.

## End-to-End Pipeline Validation (2026-03-19)

### Run: 3 scenes from target t100 (San Joaquin Valley, CA)

| Scene | Plume Pixels | Plume % | Best S2 Match | Δt (hours) | Cloud % |
|-------|-------------|---------|---------------|-----------|---------|
| 01460640 | 342,664 | 2.0% | S2B_11SLU_20240909 | 51.4h | 6.2% |
| 02F00640 | 729,846 | 3.6% | S2B_11SLU_20241019 | 27.2h | 0.2% |
| 03900640 | 706,295 | 4.0% | S2A_11SLT_20241103 | 27.3h | 0.6% |

### Key Results
- **Mask generation**: 3/3 scenes produced usable plume masks at 50 ppb threshold
- **Cross-sensor matching**: 3/3 masks matched to Sentinel-2 within 72h
- **Time deltas**: 27–51h (median ~27h — excellent)
- **Cloud cover**: 0.2–6.2% (very clean scenes)
- **Training patch**: First patch extracted successfully — 1,059 plume pixels (1.6% of 256×256 patch)
- **Decision point passed**: All 3 scenes viable → proceed to scale up
