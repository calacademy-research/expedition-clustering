# Clustering Expeditions with DBSCAN

This project aims to cluster CAS expeditions across the continental US by way of DBSCAN clustering. Within the Botany database, we've found that we can recreate entire expeditions by looking at the locality of a specimen's collection site as well as the start and end date of a collection event, clustering on these points, and then using convex hulls to create a polygon representation of an expedition's area of coverage. 

## What is DBSCAN?

DBSCAN, short for Density-Based Spatial Clustering of Applications with Noise, is a very popular clustering algorithm that can identify clusters, or groups, in data with varying shapes and sizes. This is what makes it versatile across many domains. Unlike traditional clustering algorithms like K-Means, DBSCAN doesn't require you to specify the number of clusters in advance, making it particularly useful for large, complex, or unstructured data.

### How DBSCAN Works
DBSCAN relies on two key parameters: eps (epsilon) and min_samples.

eps (epsilon): This is the maximum distance between two points for them to be considered as part of the same neighborhood.
min_samples: This is the minimum number of points required to form a dense region (i.e., a cluster).

The algorithm works as follows:

Identify Core Points: Points that have at least min_samples points within a distance of eps are considered core points.
Form Clusters: A cluster is formed by core points and all points (core or non-core) that are reachable from these core points. Reachability means being within eps distance from a core point.
Identify Noise: Points that are not reachable from any core points are classified as noise.

## The Data

