# Expedition Clustering

## Table of Contents
- [Overview](#overview)
- [Capabilities](#capabilities)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Data Workflow](#data-workflow)
- [Project Structure](#project-structure)
- [Development](#development)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview
Expedition Clustering groups California Academy of Sciences botany specimens into inferred expeditions using spatiotemporal DBSCAN. The package pairs a reproducible pipeline with database helpers, plotting utilities, and command-line tooling so museum staff and collaborators can cluster full database exports or smaller slices for validation.

## Capabilities
- Spatiotemporal DBSCAN pipeline with safeguards to keep clusters spatially and temporally connected after per-location time slicing.
- Ready-to-run sklearn pipeline factory: `create_pipeline(e_dist, e_days)` yields a transformer that returns a DataFrame with `spatiotemporal_cluster_id`.
- Database ingestion helpers: `DatabaseConfig`, `load_core_tables`, and `build_clean_dataframe` replicate the cleaning steps used in the exploratory notebooks.
- Command-line entry point `expedition-cluster` for end-to-end clustering from a MySQL export.
- Plotting utilities for quick QA (`plot_geographical_positions`, `plot_geographical_heatmap`, `plot_time_histogram`) and notebooks for deeper inspection.

## Requirements
- Python 3.10+
- Docker with Docker Compose (for the bundled MySQL configuration)
- Access to the CAS botany MySQL dump used in the notebooks (not distributed with this repository)

## Setup
1) Clone the repository
```bash
git clone https://github.com/calacademy-research/expedition-clustering.git
cd expedition-clustering
```

2) Install dependencies (pick one)
```bash
# Recommended
uv sync --dev

# Or with pip
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

3) Configure and start MySQL via Docker
- Copy `.env.example` to `.env` and set:
  - `MYSQL_ROOT_PASSWORD`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`
  - `MYSQL_PORT` (host port to expose 3306)
  - `SQL_DUMP_PATH` (path to your dump, e.g., `./data/cas-db.sql.gz`)
- Start the stack (seeds the DB once per fresh volume):
```bash
docker-compose down -v   # only when you want to reseed
docker-compose up -d
```

## Usage

### Command line
Run the full pipeline directly against the database:
```bash
uv run expedition-cluster \
  --e-dist 10 \
  --e-days 7 \
  --limit 50000 \
  --output data/clustered_expeditions.csv
```
Flags of note: `--include-centroids` fills missing locality coordinates with geography centroids; `--log-level DEBUG` surfaces validation details.

### Python API
```python
from expedition_clustering import (
    DatabaseConfig,
    build_clean_dataframe,
    create_pipeline,
    load_core_tables,
)

config = DatabaseConfig()  # defaults match docker-compose.yml
tables = load_core_tables(
    config,
    limit=5000,           # cap rows per table for a quick dry run
    related_only=True,
    primary_table="collectionobject",
)

clean_df = build_clean_dataframe(tables)
pipeline = create_pipeline(e_dist=10, e_days=7)
clustered = pipeline.fit_transform(clean_df)
clustered.to_csv("data/clustered_expeditions.csv", index=False)
print("Expeditions:", clustered["spatiotemporal_cluster_id"].nunique())
```
If the clean DataFrame is empty, raise the `limit`, switch `primary_table="collectingevent"`, or set `related_only=False`/`filter_related=False` to include more records.

## Data Workflow
1. Pull source tables from MySQL with `load_core_tables` (optionally limited for smoke tests).
2. Merge and clean with `build_clean_dataframe`, which normalizes column names, converts dates, and prefers precise locality coordinates over centroids.
3. Build and run the clustering pipeline via `create_pipeline(e_dist, e_days)`. The output includes spatial, temporal, and combined cluster IDs plus validation to guard against disconnected clusters.
4. Use plotting helpers or the notebooks for visual QA of spatial and temporal patterns.

## Project Structure
```
expedition-clustering/
├── docker-compose.yml             # MySQL/PMA stack (reads credentials/dump path from .env)
├── expedition_clustering/
│   ├── cli.py                     # expedition-cluster entry point
│   ├── data.py                    # Database connectors and table loaders
│   ├── pipeline.py                # Spatiotemporal DBSCAN pipeline components
│   ├── plotting.py                # Mapping and histogram utilities
│   └── preprocessing.py           # Merge/clean helpers for core tables
├── notebooks/                     # EDA, manual labeling, and algorithm walkthroughs
├── pyproject.toml                 # Packaging and tooling configuration
├── requirements.txt
└── data/                          # Local datasets and outputs (git-ignored)
```

## Development
- Lint/format: `uv run ruff check` and `uv run ruff format --check` (or `ruff format` to apply fixes).
- Scripts and notebooks: run with `uv run python ...` to ensure the environment matches the declared dependencies.
- Pre-commit hooks: install once with `pre-commit install`, then commits will run lint/format checks automatically.

## License
MIT License. See `LICENSE` for details.

## Acknowledgments
Thanks to the California Academy of Sciences botany collections team for access to the source data and domain guidance, and to the open-source community whose tools underpin this work.
