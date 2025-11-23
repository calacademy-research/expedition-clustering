# Expedition Clustering

## Table of Contents
- [Overview](#overview)
- [Project Background](#project-background)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview  
The **Expedition Clustering** project aims to analyze and organize botanical specimen collection data by identifying and grouping individual specimens into their respective expeditions. While our dataset contains extensive information on over a million specimens, their association with specific collection expeditions is often unclear. Understanding these connections will improve data organization, enable better summarization, and support the development of interactive, narrative-based tools for both scientists and educators.

By clustering specimens based on collection patterns, locations, collector information, and dates, this project will provide deeper insights into historical and modern botanical expeditions. These insights will facilitate research, highlight under-sampled regions, and help build engaging visualizations for storytelling and scientific analysis.

---

## Project Background  
Botanical collections serve as invaluable records of biodiversity, helping scientists track species distributions, ecological changes, and conservation needs. However, these collections are often cataloged at the specimen level, making it difficult to reconstruct the expeditions that gathered them. Expeditions—organized efforts by researchers, institutions, and explorers—play a crucial role in shaping our understanding of plant biodiversity. By identifying expedition clusters, we can:

- Reconstruct historical collection efforts and their geographic scope.  
- Detect biases in collection efforts and identify under-sampled regions.  
- Enhance database organization by grouping specimens within meaningful contexts.  
- Enable interactive tools that allow scientists, educators, and the public to explore past expeditions through maps and narratives.  

This project leverages data science, natural language processing, and clustering techniques to infer expedition groupings from collection metadata, contributing to both research and public engagement with botanical history.

---

## Features  
✅ **Spatiotemporal Clustering** – Groups specimens into expeditions based on collection locations and dates.  
✅ **Machine Learning Integration** – Applies clustering algorithms and NLP techniques for automated classification.  
✅ **Data Visualization** – Generates interactive maps and plots for understanding collection trends.  
✅ **Reproducible Workflow** – Supports automation and scalability through Python scripts and Jupyter notebooks.  
✅ **Open Source** – Designed to be extensible for research and educational purposes.  

---

## Installation

### Prerequisites
- Python 3.10+
- Docker (for running MySQL database)

### Quick Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/expedition-clustering.git
   cd expedition-clustering
   ```

2. **Install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .  # Install the package in editable mode
   ```

3. **Start the database:**
   ```bash
   docker-compose up -d
   # Optional: Load your database backup
   # docker exec -i exped_cluster_mysql_container mysql -u root -prootpassword exped_cluster_db < data/backup.sql
   ```

4. **Install the package in editable mode** (or use `uv sync --dev`):
   ```bash
   pip install -e .
   ```


## Usage

### Quick Start (CLI)

The CLI orchestrates every step: download a slice of the database, clean it, and run the spatiotemporal clustering pipeline. The script lives in `scripts/run_pipeline.py`, with a convenience wrapper `scripts/run_pipeline.sh`.

```bash
# Small dry-run on 10k specimens (random sample)
scripts/run_pipeline.sh --table-limit 10000 --sample 10000 --log-level INFO

# Full run with custom epsilons and output path
python scripts/run_pipeline.py \
  --e-dist 12 \
  --e-days 5 \
  --table-limit 200000 \
  --sample 0 \
  --primary-table collectionobject \
  --output data/clustered_expeditions.csv

# View all options
python scripts/run_pipeline.py --help
```

**Frequently used flags**

| Flag | Purpose |
| --- | --- |
| `--host/--port/--user/--password/--database` | Database connection settings (defaults match `docker-compose.yml`). |
| `--table-limit` | Limits the number of rows fetched per table (prevents pulling all million records during experimentation). |
| `--sample` | Randomly subsample after cleaning; set to `0` or omit to process everything fetched. |
| `--fetch-related-only / --no-fetch-related-only` | Whether to load only rows related to the limited slice (default: on). |
| `--primary-table {collectionobject, collectingevent}` | Choose which table to anchor the slice; `collectionobject` ensures we only pull specimens with valid `CollectingEventID`s. |
| `--no-filter-related` | Disable the second filtering pass that keeps only rows referenced by the loaded events. |
| `--e-dist`, `--e-days` | Spatial/temporal epsilons for DBSCAN. |
| `--log-level DEBUG` | Emits per-table timings, row counts, and warnings (e.g., when related fetches fall back to limited mode). |
| `--output` | Target CSV file (default `data/clustered_output.csv`). |

If the CLI reports that the clean dataframe is empty, try increasing `--table-limit`, switching `--primary-table collectingevent`, disabling related-only mode (`--no-fetch-related-only`), or turning off filtering (`--no-filter-related`) so all specimens of interest make it into the batch.

### Jupyter Notebooks (Advanced Exploration)

For exploratory analysis and parameter tuning, use the notebooks in order:

1. [0_table_eda.ipynb](notebooks/0_table_eda.ipynb) - Explore database structure
2. [1_manual_cluster_labeling.ipynb](notebooks/1_manual_cluster_labeling.ipynb) - Manual validation
3. [2_spatiotemporal_clustering_algorithm.ipynb](notebooks/2_spatiotemporal_clustering_algorithm.ipynb) - Main clustering
4. [3_secondary_clustering.ipynb](notebooks/3_secondary_clustering.ipynb) - Refinement
5. [4_determination.ipynb](notebooks/4_determination.ipynb) - Final analysis

### Python Package (Programmatic Usage)

All functionality is available as importable helpers. Typical workflow:

```python
from expedition_clustering import (
    DatabaseConfig,
    build_clean_dataframe,
    create_pipeline,
    load_core_tables,
)

config = DatabaseConfig()
tables = load_core_tables(
    config,
    limit=25000,
    related_only=True,
    primary_table="collectionobject",
)
clean_df = build_clean_dataframe(tables)
pipeline = create_pipeline(e_dist=10, e_days=7)
clustered = pipeline.fit_transform(clean_df)
print("Expeditions:", clustered["spatiotemporal_cluster_id"].nunique())
```

For visual inspection, the package also provides `plot_geographical_positions`, `plot_geographical_heatmap`, and `plot_time_histogram`.

### Development (uv + Ruff)

```bash
uv sync --dev
uv run ruff check
uv run ruff format --check  # or just `ruff format`
```

The CLI emits detailed logs so you can smoke-test changes without a dedicated test suite.

## Project Structure

```
expedition-clustering/
├── GETTING_STARTED.md              # ⭐ 3-step quick start
├── QUICKSTART.md                   # Detailed beginner guide
├── README.md                       # Complete documentation
│
├── expedition_clustering/          # Python package
│   ├── __init__.py                # Package exports (core API)
│   ├── cli.py                     # Command-line tool
│   ├── pipeline.py                # Clustering pipeline (DBSCAN)
│   ├── preprocessing.py           # Data cleaning utilities
│   ├── data.py                    # Database connection utilities
│   └── plotting.py                # Visualization functions
│
├── notebooks/                      # Jupyter notebooks for exploration
│   ├── 0_table_eda.ipynb          # Database exploration
│   ├── 1_manual_cluster_labeling.ipynb  # Validation
│   ├── 2_spatiotemporal_clustering_algorithm.ipynb  # Main analysis
│   ├── 3_secondary_clustering.ipynb     # Refinement
│   └── 4_determination.ipynb            # Final analysis
│
├── data/                           # Data files (git-ignored)
│   └── *.csv                      # Input/output CSV files
│
├── docker-compose.yml             # MySQL database setup
├── requirements.txt               # Python dependencies
└── pyproject.toml                 # Package configuration
```

**Key Components:**
- **Command-line tool**: `expedition-cluster` (installed with package)
- **Python package**: `expedition_clustering` - Import clustering, visualization, and data utilities
- **Notebooks**: Interactive exploration and analysis
- **Documentation**: GETTING_STARTED.md → QUICKSTART.md → README.md (increasing detail)  
 
---  
 
## Contributing  
We welcome contributions! To contribute:  
1. Fork the repository.  
2. Create a new branch (`git checkout -b feature-branch`).  
3. Commit your changes (`git commit -m "Add feature"`) and push (`git push origin feature-branch`).  
4. Open a Pull Request.  
 
---  
 
## License  
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  
 
---  
 
## Acknowledgments  
🔬 Special thanks to **CalAcademy's botany collections team** for providing access to over a million collection records.  
🌍 Thanks to the **open-source community** for developing powerful data science tools that make this research possible.  
📚 Inspired by previous works in **botanical data clustering and machine learning applications in biodiversity research**.  
 
---  
 
Happy exploring! 🚀🌿  
