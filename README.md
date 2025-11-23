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

4. **Verify installation:**
   ```bash
   expedition-cluster --limit 1000
   ```


## Usage

### Quick Start: Run from Command Line

After installation, use the `expedition-cluster` command:

```bash
# Basic usage (clusters all specimens with default parameters)
expedition-cluster

# Specify clustering parameters
expedition-cluster --e-dist 15 --e-days 10

# Test with a sample
expedition-cluster --limit 10000 --output data/test_output.csv

# Full parameter list
expedition-cluster --help
```

**Key Parameters:**
- `--e-dist`: Spatial epsilon in kilometers (default: 10) - maximum distance between specimens in the same cluster
- `--e-days`: Temporal epsilon in days (default: 7) - maximum time gap between specimens in the same cluster
- `--limit`: Number of specimens to process (useful for testing)
- `--batch-size`: Specimens per batch (default: 50000) - lower if running out of memory
- `--no-batch`: Disable batch processing (only for small datasets <100K)
- `--output`: Output CSV file path (default: `data/clustered_expeditions.csv`)
- `--host`, `--user`, `--password`, `--database`: Database connection settings

**Memory Management:**

For large datasets (>100K specimens), **spatial-aware batch processing** is automatically enabled:
- **Prevents out-of-memory crashes**: Processes data in manageable chunks
- **Smart batching**: Uses spatial pre-clustering to avoid splitting expeditions across batches
- **Default batch size**: 50K specimens (~2-4 GB RAM per batch, configurable with `--batch-size`)
- **Expedition integrity**: Specimens within your `--e-dist` are guaranteed to stay in the same batch
- See [MEMORY_OPTIMIZATION.md](MEMORY_OPTIMIZATION.md) for algorithm details

**Example Output:**
```
============================================================
Clustering Results:
  Total specimens: 149622
  Total expeditions (clusters): 21511
  Average expedition size: 6.96 specimens
  Processed in 3 batches
============================================================
```

### Jupyter Notebooks (Advanced Exploration)

For exploratory analysis and parameter tuning, use the notebooks in order:

1. [0_table_eda.ipynb](notebooks/0_table_eda.ipynb) - Explore database structure
2. [1_manual_cluster_labeling.ipynb](notebooks/1_manual_cluster_labeling.ipynb) - Manual validation
3. [2_spatiotemporal_clustering_algorithm.ipynb](notebooks/2_spatiotemporal_clustering_algorithm.ipynb) - Main clustering
4. [3_secondary_clustering.ipynb](notebooks/3_secondary_clustering.ipynb) - Refinement
5. [4_determination.ipynb](notebooks/4_determination.ipynb) - Final analysis

### Python Package (Programmatic Usage)

Use the expedition_clustering package in your own scripts:

```python
from expedition_clustering import create_pipeline, plot_geographical_positions
import pandas as pd
import pymysql

# Connect and load data
conn = pymysql.connect(host='localhost', user='myuser', password='mypassword',
                       database='exped_cluster_db')

query = """
    SELECT ce.CollectingEventID as collectingeventid,
           ce.StartDate as startdate,
           l.Latitude1 as latitude1,
           l.Longitude1 as longitude1
    FROM collectingevent ce
    INNER JOIN collectionobject co ON ce.CollectingEventID = co.CollectingEventID
    LEFT JOIN locality l ON ce.LocalityID = l.LocalityID
    WHERE ce.StartDate IS NOT NULL AND l.Latitude1 IS NOT NULL
    LIMIT 10000
"""

df = pd.read_sql_query(query, conn)
df.columns = df.columns.str.lower()
df['startdate'] = pd.to_datetime(df['startdate'])

# Run clustering
pipeline = create_pipeline(e_dist=10, e_days=7)
clustered = pipeline.fit_transform(df)

print(f"Created {clustered['spatiotemporal_cluster_id'].nunique()} expeditions")

# Visualize results (optional)
plot_geographical_positions(
    clustered,
    lat_col='latitude1',
    lon_col='longitude1',
    datetime_col='startdate'
)
```

**Available imports:**
- `create_pipeline` - Build clustering pipeline
- `plot_geographical_positions`, `plot_geographical_heatmap`, `plot_time_histogram` - Visualization
- `DatabaseConfig`, `load_core_tables`, `build_clean_dataframe` - Advanced database utilities

For more advanced usage, see the [notebooks](notebooks/).

---

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
