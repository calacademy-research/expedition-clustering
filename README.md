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
   python cluster.py --limit 1000
   ```


## Usage

### Quick Start: Run from Command Line

The simplest way to run the clustering pipeline is using the [cluster.py](cluster.py) script:

```bash
# Basic usage (clusters all specimens with default parameters)
python cluster.py

# Specify clustering parameters
python cluster.py --e-dist 15 --e-days 10

# Test with a sample
python cluster.py --limit 10000 --output data/test_output.csv

# Full parameter list
python cluster.py --help
```

**Key Parameters:**
- `--e-dist`: Spatial epsilon in kilometers (default: 10) - maximum distance between specimens in the same cluster
- `--e-days`: Temporal epsilon in days (default: 7) - maximum time gap between specimens in the same cluster
- `--limit`: Number of specimens to process (useful for testing)
- `--output`: Output CSV file path (default: `data/clustered_expeditions.csv`)
- `--host`, `--user`, `--password`, `--database`: Database connection settings

**Example Output:**
```
============================================================
Clustering Results:
  Total specimens: 5000
  Total expeditions (clusters): 2161
  Average expedition size: 2.31 specimens
  Median expedition size: 1 specimens
  Largest expedition: 111 specimens
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

Install the package and use it in your own scripts:

```bash
pip install -e .
```

```python
import pandas as pd
import pymysql
from expedition_clustering import create_pipeline

# Connect to database and load data
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

print(f"Created {clustered['spatiotemporal_cluster_id'].nunique()} expedition clusters")
```

For more advanced usage with the built-in data loading utilities, see the [notebooks](notebooks/).

---

## Project Structure

```
expedition-clustering/
├── cluster.py                      # ⭐ Main CLI tool (start here!)
├── QUICKSTART.md                   # Quick start guide
├── README.md                       # Full documentation
│
├── expedition_clustering/          # Python package
│   ├── __init__.py                # Package exports
│   ├── pipeline.py                # Clustering pipeline (DBSCAN)
│   ├── preprocessing.py           # Data cleaning utilities
│   └── data.py                    # Database connection utilities
│
├── notebooks/                      # Jupyter notebooks for exploration
│   ├── 0_table_eda.ipynb          # Database exploration
│   ├── 1_manual_cluster_labeling.ipynb  # Validation
│   ├── 2_spatiotemporal_clustering_algorithm.ipynb  # Main analysis
│   ├── 3_secondary_clustering.ipynb     # Refinement
│   └── 4_determination.ipynb            # Final analysis
│
├── plotting.py                     # Visualization utilities
├── data/                           # Data files (git-ignored)
│   ├── clustered_expeditions.csv  # Output from cluster.py
│   └── *.csv                      # Intermediate/processed data
│
├── docker-compose.yml             # MySQL database setup
├── requirements.txt               # Python dependencies
└── pyproject.toml                 # Package configuration
```

**Key Files:**
- **[cluster.py](cluster.py)**: Single-command clustering tool - this is what you'll use most
- **[QUICKSTART.md](QUICKSTART.md)**: Step-by-step beginner guide
- **[expedition_clustering/](expedition_clustering/)**: Reusable Python package for custom workflows
- **[notebooks/](notebooks/)**: Interactive analysis and parameter tuning  
 
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
