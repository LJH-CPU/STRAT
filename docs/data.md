# Data availability

## What is *not* in this repository

The raw and processed trajectory data are **not distributed** with this code
release:

- raw GPS manifests / GPX files (`data/`, `data_20/`)
- crawler session files (`cookies.json`) — removed for security reasons
- cleaned / clustered trajectory CSVs (`data-project/cleaned_labeled_data/`,
  `data-project/segment_features/`)
- DEM caches (`data-project/dem_cache/`)

These artifacts contain personally identifying movement traces and, in the case
of the crawler session files, authentication credentials. They are therefore
excluded by `.gitignore` and must not be committed.

Only the curated, aggregated experimental outputs under `final_results/`
(JSON/CSV/figures) and the source code are shipped.

## Obtaining the data

The scenic-area trajectory dataset used in the paper is available from the
corresponding author on reasonable request, subject to the terms of the
original data provider. Please refer to the paper's *Data Availability*
statement.

## Reproducing the pipeline with your own data

The code expects cleaned per-scenic-area CSV files under
`data-project/cleaned_labeled_data/` (see `config.py` → `CLEANED_DIR`), named
`<scene>_cleaned.csv`. Each file contains GPS points with, at minimum,
timestamp, latitude, longitude and a track identifier. See
`data-project/clean_scenery_pipeline.py` and `data-project/clean_all.py` for the
cleaning/segmentation steps.

Terrain features are derived from SRTM DEM tiles, which are downloaded on
demand by `data-project/dem_lookup.py` and cached under
`data-project/dem_cache/`.

POI-based semantic features require an AMap (Gaode) API key, supplied via the
command line rather than hard-coded:

```bash
python poi/fetch.py --key <YOUR_AMAP_KEY>
```

Never commit API keys or `cookies.json`; the patterns are listed in
`.gitignore`.

## Directory layout expected at runtime

```
data-project/
├── cleaned_labeled_data/   # <scene>_cleaned.csv      (not shipped)
├── segment_features/       # derived features         (not shipped)
├── dem_cache/              # SRTM tiles               (downloaded on demand)
└── data/                   # POI GeoJSON / raw exports (not shipped)
```
