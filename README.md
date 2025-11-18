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

For more advanced usage with the built-in data loading utilities, see the [notebooks](notebooks/) and [test_simple.py](test_simple.py).

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
