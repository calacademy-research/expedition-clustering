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
Ensure you have the following installed:  
- Python 3.8+  
- Docker (for containerized execution)  

### Setup  
Clone the repository:
```bash
git clone https://github.com/yourusername/expedition-clustering.git
cd expedition-clustering
```

Create a virtual environment and install dependencies:
```python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
pip install -r requirements.txt
```

(Optional) Start the database container using Docker:

```bash
docker-compose up -d
docker exec -i exped_cluster_mysql_container mysql -u root -prootpassword exped_cluster_db < ./data/{unzipped botany backup filename}
```


## Usage  
 
### Running the Clustering Pipeline  
Currently all clustering functions must be called from within the Jupyter Notebooks. Programmatic pipeline scripts are under development.
 
### Running Jupyter Notebooks  
The notebooks are meant to be run in order of ascending prepending integer. Running them in sequence is crucial as they depend on artifacts saved between notebooks.

### Command-Line Runner

After installing the package, you can execute the full pipeline without a notebook:

```bash
python scripts/run_pipeline.py --e-dist 12 --e-days 5 --output data/clustered.csv
```

or via the shell wrapper:

```bash
scripts/run_pipeline.sh --sample 50000
```

Use `--host`, `--user`, `--password`, and related flags to override database credentials. `--sample` limits the number of rows passed to the clustering step for quicker experiments, `--table-limit` constrains how many rows are fetched from each MySQL table, and `--tables` lets you control which tables are loaded. The CLI fetches only rows related to the limited data slice and further filters them before merging; toggle this behavior with `--fetch-related-only/--no-fetch-related-only`, pick whether to anchor the slice on collection objects or collecting events via `--primary-table`, and control the join filtering with `--no-filter-related`. Add `--log-level DEBUG` to see detailed progress logs (per-table timings, row counts, warnings when a related table returns zero rows, etc.). If the clean dataframe ends up empty, increase `--table-limit`, adjust `--primary-table`, disable related filtering, or load full tables so the clustering pipeline has data to work with.


## Python Package

Install the reusable module with `pip install -e .` and mirror the notebook workflow in regular scripts:

```python
from expedition_clustering import (
    DatabaseConfig,
    build_clean_dataframe,
    create_pipeline,
    load_core_tables,
)

config = DatabaseConfig()
tables = load_core_tables(config)
clean_df = build_clean_dataframe(tables)
pipeline = create_pipeline(e_dist=10, e_days=7)
clustered = pipeline.fit_transform(clean_df)
```

This encapsulates the preparation logic from `0_table_eda.ipynb` and exposes the clustering helpers without copying code across notebooks.

### Development (uv + Ruff)

Install runtime and development dependencies with [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync --dev
```

Run Ruff linting/formatting via:

```bash
uv run ruff check
uv run ruff format --check  # or `ruff format` to auto-format
```
 
---  
 
## Project Structure  
 
The repository is organized as follows:  
 
```plaintext  
EXPEDITION-CLUSTERING/  
│── data/                      # Contains raw and processed dataset files  
│   ├── CASBotanybackup2025-01-23.sql        # SQL database backup  
│   ├── CASBotanybackup2025-01-23.sql.gz  ‼️  # Compressed SQL backup  
│   ├── clean_df.csv                         # Cleaned dataset after preprocessing  
│   ├── cluster_summary_stats.csv            # Summary statistics of clustered data  
│   ├── full_df.csv                          # Full dataset before processing  
│   ├── full_processed_df.csv                # Fully processed dataset after clustering  
│   ├── labeled_clean_df.csv                 # Clean dataset with assigned cluster labels  
│   ├── processed_df.csv                      # Intermediate processed dataset  
│  
│── docs/                      # Documentation and notes  
│   ├── exped_clust_notes.pages  # Notes related to expedition clustering  
│  
│── notebooks/                  # Jupyter notebooks for analysis and clustering  
│   ├── 0_table_eda.ipynb                        # Exploratory data analysis of specimen collections  
│   ├── 1_manual_cluster_labeling.ipynb         # Manual cluster labeling and validation  
│   ├── 2_spatiotemporal_clustering_algorithm.ipynb  # Primary clustering algorithm based on space-time data  
│   ├── 3_secondary_clustering.ipynb            # Secondary clustering refinements and validation  
│   ├── 4_determination.ipynb                    # Final decision-making on clusters  
│  
│── .gitignore                # Specifies files and folders to ignore in version control  
│── cluster_pipeline.py       # Python script implementing the clustering pipeline  
│── docker-compose.yml        # Configuration for running services in Docker  
│── plotting.py               # Script for visualizing clustering results  
│── README.md                 # Project documentation  
│── requirements.txt          # Dependencies required for running the project  
```  
 
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























# Expedition Clustering

## Project Overview  
The **Expedition Clustering** project aims to analyze and organize botanical specimen collection data by identifying and grouping individual specimens into their respective expeditions. While our dataset contains extensive information on over a million specimens, their association with specific collection expeditions is often unclear. Understanding these connections will improve data organization, enable better summarization, and support the development of interactive, narrative-based tools for both scientists and educators.

By clustering specimens based on collection patterns, locations, collector information, and dates, this project will provide deeper insights into historical and modern botanical expeditions. These insights will facilitate research, highlight under-sampled regions, and help build engaging visualizations for storytelling and scientific analysis.

---

## Project Background  
Botanical collections serve as invaluable records of biodiversity, helping scientists track species distributions, ecological changes, and conservation needs. However, these collections are often cataloged at the specimen level, making it difficult to reconstruct the expeditions that gathered them. Expeditions—organized efforts by researchers, institutions, and explorers—play a crucial role in shaping our understanding of plant biodiversity. By identifying expedition clusters, we can:

- Reconstruct historical collection efforts and their geographic scope.  
- Detect biases in collection efforts and identify under-sampled regions.  
- Enhance database organization by grouping specimens within meaningful contexts.  
- Enable interactive tools that allow scientists, educators, and the public to explore past expeditions through maps and narratives.  

This project leverages data science, natural language processing, and clustering techniques to infer expedition groupings 
from collection metadata, contributing to both research and public engagement with botanical history.

---

## Project Structure

This repository is established as a data science exploratory effort. As such, it consists of a simple flat package layout with **data/**, **docs/**, and **notebooks/** subdirectories, and a few python modules containing pipeline and plotting functions. 

The repository is organized as follows:

```plaintext
EXPEDITION-CLUSTERING/
│── data/                      # Contains raw and processed dataset files. The contents of this directory are hidden from git
│   ├── **CASBotanybackup2025-01-23.sql.gz**     #  ‼️ Compressed SQL backup: upload your own version locally ‼️
│
│── docs/                      # Documentation and notes
│   ├── exped_clust_notes.pages  # Notes related to expedition clustering
│
│── notebooks/                  # Jupyter notebooks for analysis and clustering
│   ├── 0_table_eda.ipynb                        # Exploratory data analysis of specimen collections
│   ├── 1_manual_cluster_labeling.ipynb         # Manual cluster labeling and validation
│   ├── 2_spatiotemporal_clustering_algorithm.ipynb  # Primary clustering algorithm based on space-time data
│   ├── 3_secondary_clustering.ipynb            # Secondary clustering refinements and validation
│   ├── 4_determination.ipynb                    # Final decision-making on clusters
│
│── .gitignore                # Specifies files and folders to ignore in version control
│── cluster_pipeline.py       # Python script implementing the clustering pipeline
│── docker-compose.yml        # Configuration for running services in Docker
│── plotting.py               # Script for visualizing clustering results
│── README.md                 # Project documentation
│── requirements.txt          # Dependencies required for running the project

```
---

## Requirements

The project dependencies are listed in the [`requirements.txt`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fdangause%2FDesktop%2Fcalacademy%2Fexpedition-clustering%2Frequirements.txt%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/dangause/Desktop/calacademy/expedition-clustering/requirements.txt") file. To install the required packages, run:

```sh
pip install -r requirements.txt
