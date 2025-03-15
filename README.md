# Clustering Pipeline Project

This project is designed to perform spatiotemporal clustering on botanical data from the CAS Botany Backup database. The project includes data processing, clustering algorithms, and visualization of the results.

## Project Structure

Workspace
.DS_Store .gitignore cluster_pipeline.py data/ .DS_Store CASBotanybackup2025-01-23.sql clean_df.csv cluster_summary_stats.csv full_df.csv full_processed_df.csv labeled_clean_df.csv processed_df.csv docker-compose.yml docs/ exped_clust_notes.pages notebooks/ 0_table_eda.ipynb 1_manual_cluster_labeling.ipynb 2_spatiotemporal_clustering_algorithm.ipynb 3_secondary_clustering.ipynb 4_determination.ipynb plotting.py README.md requirements.txt



### Key Files and Directories

- [`cluster_pipeline.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fdangause%2FDesktop%2Fcalacademy%2Fexpedition-clustering%2Fcluster_pipeline.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/dangause/Desktop/calacademy/expedition-clustering/cluster_pipeline.py"): Main script for processing data and running clustering algorithms.
- [`data/`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fdangause%2FDesktop%2Fcalacademy%2Fexpedition-clustering%2Fdata%2F%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/dangause/Desktop/calacademy/expedition-clustering/data/"): Directory containing raw and processed data files.
- [`docker-compose.yml`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fdangause%2FDesktop%2Fcalacademy%2Fexpedition-clustering%2Fdocker-compose.yml%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/dangause/Desktop/calacademy/expedition-clustering/docker-compose.yml"): Docker configuration for setting up the project environment.
- [`docs/`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fdangause%2FDesktop%2Fcalacademy%2Fexpedition-clustering%2Fdocs%2F%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/dangause/Desktop/calacademy/expedition-clustering/docs/"): Documentation and notes related to the project.
- [`notebooks/`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fdangause%2FDesktop%2Fcalacademy%2Fexpedition-clustering%2Fnotebooks%2F%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/dangause/Desktop/calacademy/expedition-clustering/notebooks/"): Jupyter notebooks for exploratory data analysis (EDA) and clustering steps.
- [`plotting.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fdangause%2FDesktop%2Fcalacademy%2Fexpedition-clustering%2Fplotting.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/dangause/Desktop/calacademy/expedition-clustering/plotting.py"): Script for generating visualizations of the clustering results.
- [`requirements.txt`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fdangause%2FDesktop%2Fcalacademy%2Fexpedition-clustering%2Frequirements.txt%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/dangause/Desktop/calacademy/expedition-clustering/requirements.txt"): List of Python dependencies required for the project.

## Data Processing

The data processing pipeline is implemented in [`cluster_pipeline.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fdangause%2FDesktop%2Fcalacademy%2Fexpedition-clustering%2Fcluster_pipeline.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/dangause/Desktop/calacademy/expedition-clustering/cluster_pipeline.py"). The script reads raw data from the [`data/`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fdangause%2FDesktop%2Fcalacademy%2Fexpedition-clustering%2Fdata%2F%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/dangause/Desktop/calacademy/expedition-clustering/data/") directory, cleans it, and prepares it for clustering. The processed data is saved back to the [`data/`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fdangause%2FDesktop%2Fcalacademy%2Fexpedition-clustering%2Fdata%2F%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/dangause/Desktop/calacademy/expedition-clustering/data/") directory.

### Processed Data

- [`clean_df.csv`](command:_github.copilot.openSymbolFromReferences?%5B%22clean_df.csv%22%2C%5B%7B%22uri%22%3A%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22%2FUsers%2Fdangause%2FDesktop%2Fcalacademy%2Fexpedition-clustering%2Fnotebooks%2F0_table_eda.ipynb%22%2C%22external%22%3A%22file%3A%2F%2F%2FUsers%2Fdangause%2FDesktop%2Fcalacademy%2Fexpedition-clustering%2Fnotebooks%2F0_table_eda.ipynb%22%2C%22path%22%3A%22%2FUsers%2Fdangause%2FDesktop%2Fcalacademy%2Fexpedition-clustering%2Fnotebooks%2F0_table_eda.ipynb%22%2C%22scheme%22%3A%22file%22%7D%2C%22pos%22%3A%7B%22line%22%3A6359%2C%22character%22%3A5%7D%7D%5D%5D "Go to definition"): Cleaned data frame.
- `cluster_summary_stats.csv`: Summary statistics of the clusters.
- `full_df.csv`: Full data frame before processing.
- `full_processed_df.csv`: Fully processed data frame.
- `labeled_clean_df.csv`: Cleaned data frame with cluster labels.
- `processed_df.csv`: Processed data frame.

## Clustering

The clustering process is divided into several steps, each documented in a separate Jupyter notebook in the [`notebooks/`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fdangause%2FDesktop%2Fcalacademy%2Fexpedition-clustering%2Fnotebooks%2F%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/dangause/Desktop/calacademy/expedition-clustering/notebooks/") directory:

1. [`0_table_eda.ipynb`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fdangause%2FDesktop%2Fcalacademy%2Fexpedition-clustering%2Fnotebooks%2F0_table_eda.ipynb%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/dangause/Desktop/calacademy/expedition-clustering/notebooks/0_table_eda.ipynb"): Exploratory data analysis of the tables in the CAS Botany Backup database.
2. `1_manual_cluster_labeling.ipynb`: Manual labeling of clusters.
3. `2_spatiotemporal_clustering_algorithm.ipynb`: Implementation of the spatiotemporal clustering algorithm.
4. `3_secondary_clustering.ipynb`: Secondary clustering steps.
5. `4_determination.ipynb`: Determination of final clusters.

## Visualization

The [`plotting.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fdangause%2FDesktop%2Fcalacademy%2Fexpedition-clustering%2Fplotting.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/dangause/Desktop/calacademy/expedition-clustering/plotting.py") script generates visualizations of the clustering results. These visualizations help in understanding the spatial and temporal distribution of the clusters.

## Requirements

The project dependencies are listed in the [`requirements.txt`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fdangause%2FDesktop%2Fcalacademy%2Fexpedition-clustering%2Frequirements.txt%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/Users/dangause/Desktop/calacademy/expedition-clustering/requirements.txt") file. To install the required packages, run:

```sh
pip install -r requirements.txt