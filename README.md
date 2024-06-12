# Clustering Expeditions with DBSCAN

This project aims to cluster CAS expeditions across the continental US by way of DBSCAN clustering. Within the Botany database, we've found that we can recreate entire expeditions by looking at the locality of a specimen's collection site as well as the start and end date of a collection event, clustering on these points, and then using convex hulls to create a polygon representation of an expedition's area of coverage. 

## What is DBSCAN?

DBSCAN, short for Density-Based Spatial Clustering of Applications with Noise, is a very popular clustering algorithm that can identify clusters, or groups, in data with varying shapes and sizes. This is what makes it versatile across many domains. Unlike traditional clustering algorithms like K-Means, DBSCAN doesn't require you to specify the number of clusters in advance, making it particularly useful for large, complex, or unstructured data.

### How DBSCAN Works
DBSCAN relies on two key parameters: eps (epsilon) and min_samples.

* eps (epsilon): This is the maximum distance between two points for them to be considered as part of the same neighborhood.
* min_samples: This is the minimum number of points required to form a dense region (i.e., a cluster).

The algorithm works as follows:

* Identify Core Points: Points that have at least min_samples points within a distance of eps are considered core points.
* Form Clusters: A cluster is formed by core points and all points (core or non-core) that are reachable from these core points. Reachability means being within eps distance from a core point.
* Identify Noise: Points that are not reachable from any core points are classified as noise.

## The Data

The data used in this project comes from the Botany database. Specifically, we join the CollectingEvent and Locality tables on the LocalityID column.

* From CollectingEvent, we use the StartDate and EndDate columns. This allows us to cluster on temporal data.
* From Locality, we use the LocalityName, Latitude1, and Longitude1 columns. LocalityName provides a short text description of where the collecting event occurred, e.g. "Snowball Creek Valley, on private property, abt. 10km north of Grand Forks". Latitude1 and Longitude1 provide the actual lat/long coordinates of where the collecting event occurred.

## Data Preprocessing

Firstly, collecting events dated prior to the year 1677 were filtered out (very few events match this criteria), as dates prior to then are not supported by PANDAS datetime. Next, we compute a Levenshtein distance matrix to encode the string distances between the different textual locality data. This is by far the most computationally expensive portion of the project, taking several minutes to run even on a small subset of 200 data points, and much much longer on the full dataset of over 200k points. 

We then perform multidimensional scaling, or MDS, on the Levenshtein matrix to reduce the dimensionality of the matrix down to 2 dimensions. Doing this allows for faster computations during the clustering step, while preserving much of the information in the original matrix. We then drop NA values and normalize all variables using the Sklearn StandardScaler. 

## Clustering

We have yet to analyze clustering results on all the features (lat/long + text + date), and thus we don't yet know a suitable value or range for the epsilon parameter. Clustering only on latitude and longitude, we found that an epsilon value of between 0.01 and 0.04 achieves good results. For min_samples_per_cluster, a value between 3 and 10 is suitable regardless of data size. As an example, we've shown the results restricted to the Western US clustered on just lat and long. Here, we use an epsilon value of 0.04, with a minimum of 10 samples per cluster. The code can be found [here](https://github.com/calacademy-research/expedition-clustering/blob/main/DBSCAN_hulls_latlong.ipynb). <img width="882" alt="png1" src="https://github.com/calacademy-research/expedition-clustering/assets/51836467/9c778f26-d11f-4b52-b334-ac651fba2bf6">


## Next Steps

The most obvious next step would be to optimize the computation of the Levenshtein matrix, or to go with an alternative method for representing locality text. One recommendation would be to use a [BERT](https://huggingface.co/google-bert/bert-base-uncased) model, or another such encoder-only transformer, to create vector embeddings and compute similarites that way. Unlike Levenshtein, BERT computations can be sent to the GPU, significantly speeding up processing time. Additionally, parameter tuning for optimal values of eps and min_samples_per_cluster would be needed.
