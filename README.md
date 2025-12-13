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
✅ **Spatiotemporal DBSCAN** – Haversine spatial DBSCAN + per-spatial temporal DBSCAN with a safety reconnect to prevent spatially disconnected clusters after time-slicing.  
✅ **End-to-end Pipeline** – `create_pipeline(e_dist, e_days)` returns a ready-to-run sklearn pipeline.  
✅ **Database Helpers** – `DatabaseConfig`, `load_core_tables`, and preprocessing utilities mirror the notebook cleaning steps.  
✅ **Visualizations** – Mapping and temporal histogram helpers for inspection.  
✅ **uv-first workflow** – `uv run expedition-cluster ...` CLI plus notebooks for deeper exploration.  

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

### Quick Start (Programmatic)

1) **Install & start DB**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
uv pip install -e .
docker-compose up -d
```

2) **Cluster a slice**
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
    limit=5000,              # cap rows per table for a quick test
    related_only=True,
    primary_table="collectionobject",
)

clean_df = build_clean_dataframe(tables)
pipeline = create_pipeline(e_dist=10, e_days=7)
clustered = pipeline.fit_transform(clean_df)
clustered.to_csv("data/clustered_expeditions.csv", index=False)
print("Expeditions:", clustered["spatiotemporal_cluster_id"].nunique())
```

3) **Scale up** by raising `limit` (or disabling related-only filtering) once the small run succeeds.

If `clean_df` is empty, increase `limit`, switch `primary_table="collectingevent"`, or call `load_core_tables(..., related_only=False)` / `build_clean_dataframe(..., filter_related=False)` to include more rows.

### Jupyter Notebooks (Advanced Exploration)

Key notebooks for debugging/inspection:
- `notebooks/debug_dbscan.ipynb`: step-through of spatial DBSCAN → temporal DBSCAN → reconnect → validation.
- `notebooks/0_table_eda.ipynb`: Explore database structure.
- `notebooks/1_manual_cluster_labeling.ipynb`: Manual validation.

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

Use `uv run python ...` for running your own scripts during development; the pipeline emits detailed logs for quick sanity checks.

## Project Structure

```
expedition-clustering/
├── README.md
├── pyproject.toml                 # Package metadata + dev tooling
├── requirements.txt               # Dependency list (mirrors pyproject)
├── docker-compose.yml             # MySQL database setup
├── expedition_clustering/         # Python package
│   ├── __init__.py
│   ├── data.py                    # DB helpers
│   ├── preprocessing.py           # Data merging/cleaning
│   ├── pipeline.py                # Spatiotemporal DBSCAN pipeline
│   └── plotting.py                # Visualization helpers
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── 0_table_eda.ipynb
│   ├── 1_manual_cluster_labeling.ipynb
│   ├── 2_spatiotemporal_clustering_algorithm.ipynb
│   ├── 3_secondary_clustering.ipynb
│   └── 4_determination.ipynb
└── data/                          # Local datasets and outputs (git-ignored)
```

**Key Components:**
- Python package: `expedition_clustering` (clustering, preprocessing, plotting, DB helpers)
- Notebooks: interactive exploration and analysis
 
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
