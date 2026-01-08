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

3) Prepare and start MySQL via Docker
- Copy `.env.example` to `.env` and set:
  - `MYSQL_ROOT_PASSWORD`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`
  - `MYSQL_PORT` (host port to expose 3306)
  - `SQL_DUMP_PATH` (path to your SQL dump, e.g., `./data/CASBotanybackup.sql.gz`)

- **Option A: Direct import with on-the-fly cleaning (simpler)**
  ```bash
  # Point .env to your raw dump
  # SQL_DUMP_PATH=./data/CASBotanybackup.sql.gz

  docker compose down -v
  docker compose up -d
  ```
  Note: First-time database initialization takes **20-30 minutes** for a 500MB compressed dump (~2-4GB uncompressed, ~20M lines). The init script automatically strips DEFINER clauses during import.

- **Option B: Pre-clean the dump for faster restarts (recommended for development)**
  ```bash
  # Clean your dump once (takes ~5 minutes)
  ./scripts/clean_dump.sh ./data/raw-dump.sql.gz ./data/cleaned-dump.sql.gz

  # Update .env to point to the cleaned dump
  # SQL_DUMP_PATH=./data/cleaned-dump.sql.gz

  docker compose down -v
  docker compose up -d
  ```
  Using a pre-cleaned dump reduces initialization time to **~10-15 minutes**.

## Usage

### Command line
The `expedition-cluster` command runs the full spatiotemporal DBSCAN pipeline directly against your MySQL database. It loads specimen data, applies cleaning and preprocessing, clusters specimens into expeditions, and outputs the results to a CSV file.

#### Basic usage:
```bash
uv run expedition-cluster \
  --e-dist 10 \
  --e-days 7 \
  --output data/clustered_expeditions.csv
```

#### Key options:

**Clustering parameters:**
- `--e-dist DISTANCE` - Spatial epsilon in kilometers. Maximum distance between specimens to be in the same cluster (default: 10.0)
- `--e-days DAYS` - Temporal epsilon in days. Maximum time gap between specimens to be in the same cluster (default: 7.0)

**Database connection:**
- `--host HOST` - MySQL host (default: localhost)
- `--port PORT` - MySQL port (default: 3306)
- `--user USER` - MySQL user (default: myuser)
- `--password PASSWORD` - MySQL password
- `--database DATABASE` - Database name (default: exped_cluster_db)

**Data filtering:**
- `--limit N` - Process only first N specimens (useful for testing)
- `--include-centroids` - Include specimens with only geography centroids (less precise). By default, only specimens with precise locality coordinates are used.

**Output:**
- `--output PATH` - Output CSV file path (default: data/clustered_expeditions.csv)
- `--redact` - Apply IPT redaction rules (mask locality text and remove coordinates) before writing output
- `--drop-redacted` - Drop records flagged for redaction instead of masking fields
- `--log-level LEVEL` - Logging verbosity: DEBUG, INFO, WARNING, ERROR (default: INFO)

#### Example with common options:
```bash
# Test run with 5000 specimens at higher verbosity
uv run expedition-cluster \
  --e-dist 15 \
  --e-days 10 \
  --limit 5000 \
  --log-level DEBUG \
  --output data/test_clusters.csv

# Production run with centroids included
uv run expedition-cluster \
  --e-dist 10 \
  --e-days 7 \
  --include-centroids \
  --output data/expeditions_full.csv

# Produce a redacted output suitable for sharing
uv run expedition-cluster \
  --e-dist 10 \
  --e-days 7 \
  --redact \
  --output data/clustered_expeditions.csv

# Drop redacted records entirely
uv run expedition-cluster \
  --e-dist 10 \
  --e-days 7 \
  --drop-redacted \
  --output data/clustered_expeditions_redacted.csv

# Verify redaction against IPT flags
uv run expedition-cluster verify-redaction \
  --input data/clustered_expeditions_redacted.csv

# Verify that redacted records were dropped entirely
uv run expedition-cluster verify-redaction \
  --input data/clustered_expeditions_redacted.csv \
  --expect-dropped
```

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
To apply IPT redaction rules on an output CSV:
```python
from expedition_clustering import (
    DatabaseConfig,
    redact_clustered_csv,
    verify_redacted_csv,
    verify_redacted_csv_drop,
)

config = DatabaseConfig()  # defaults match docker-compose.yml
redact_clustered_csv("data/clustered_expeditions.csv", config=config)
redact_clustered_csv(
    "data/clustered_expeditions.csv",
    output_path="data/clustered_expeditions_redacted.csv",
    config=config,
    drop_redacted=True,
)
result = verify_redacted_csv("data/clustered_expeditions_redacted.csv", config=config)
print("Redaction OK?", result.ok)
drop_result = verify_redacted_csv_drop("data/clustered_expeditions_redacted.csv", config=config)
print("Drop Redaction OK?", drop_result.ok)
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
├── scripts/
│   ├── clean_dump.sh              # Pre-process SQL dumps to remove DEFINER clauses
│   └── init-db.sh                 # Docker entrypoint script for database initialization
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
