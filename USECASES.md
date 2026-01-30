# Expedition Clustering - Use Cases

This document explains how to cluster specimens into expeditions for each CAS collection.

## Prerequisites

```bash
cd /Users/joe/expedition-clustering
source .venv/bin/activate
```

## Available Collections

| Collection | Description | Specimens | Location |
|------------|-------------|-----------|----------|
| botany | Botanical specimens | ~1.3M | `/Users/joe/collections_explorer/incoming_data/botany` |
| ich | Ichthyology (fish) | ~220K | `/Users/joe/collections_explorer/incoming_data/ich` |
| iz | Invertebrate Zoology | ~271K | `/Users/joe/collections_explorer/incoming_data/iz` |
| mam | Mammalogy | ~39K | `/Users/joe/collections_explorer/incoming_data/mam` |
| orn | Ornithology (birds) | ~102K | `/Users/joe/collections_explorer/incoming_data/orn` |
| orn-en | Ornithology Eggs & Nests | ~11K | `/Users/joe/collections_explorer/incoming_data/orn-en` |

## Use Case 1: Cluster a Single Collection

Generate `clustered_expeditions.csv` for one collection:

```bash
# Botany
python scripts/cluster_csv.py \
    /Users/joe/collections_explorer/incoming_data/botany

# Ichthyology
python scripts/cluster_csv.py \
    /Users/joe/collections_explorer/incoming_data/ich

# Invertebrate Zoology
python scripts/cluster_csv.py \
    /Users/joe/collections_explorer/incoming_data/iz

# Mammalogy
python scripts/cluster_csv.py \
    /Users/joe/collections_explorer/incoming_data/mam

# Ornithology
python scripts/cluster_csv.py \
    /Users/joe/collections_explorer/incoming_data/orn

# Ornithology Eggs & Nests
python scripts/cluster_csv.py \
    /Users/joe/collections_explorer/incoming_data/orn-en
```

Output is saved as `clustered_expeditions.csv` in each collection's directory.

## Use Case 2: Customize Clustering Parameters

Adjust spatial and temporal thresholds:

```bash
python scripts/cluster_csv.py \
    /Users/joe/collections_explorer/incoming_data/botany \
    --e-dist 15 \
    --e-days 14
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--e-dist` | 10 | Spatial distance in kilometers |
| `--e-days` | 7 | Temporal window in days |
| `--limit` | None | Limit rows (for testing) |

## Use Case 3: Combine Multiple Collections

Find cross-collection expeditions (specimens collected together across disciplines):

```bash
# Fish and invertebrates
python scripts/cluster_csv.py \
    /Users/joe/collections_explorer/incoming_data/ich \
    /Users/joe/collections_explorer/incoming_data/iz \
    -o /Users/joe/collections_explorer/incoming_data/fish_and_inverts_clustered.csv

# All vertebrates
python scripts/cluster_csv.py \
    /Users/joe/collections_explorer/incoming_data/ich \
    /Users/joe/collections_explorer/incoming_data/mam \
    /Users/joe/collections_explorer/incoming_data/orn \
    -o /Users/joe/collections_explorer/incoming_data/vertebrates_clustered.csv

# Plants and animals from the same expeditions
python scripts/cluster_csv.py \
    /Users/joe/collections_explorer/incoming_data/botany \
    /Users/joe/collections_explorer/incoming_data/orn \
    /Users/joe/collections_explorer/incoming_data/mam \
    -o /Users/joe/collections_explorer/incoming_data/plants_and_animals_clustered.csv
```

## Use Case 4: Cluster All Collections Together

Find all cross-collection expeditions at once:

```bash
python scripts/cluster_csv.py \
    --all \
    -o /Users/joe/collections_explorer/incoming_data/all_collections_clustered.csv
```

## Output Columns

The `clustered_expeditions.csv` files contain:

| Column | Description |
|--------|-------------|
| `catalognumber` | Specimen identifier |
| `latitude1`, `longitude1` | Coordinates |
| `startdate` | Collection date |
| `collector` | Collector name(s) |
| `spatiotemporal_cluster_id` | Expedition ID |
| `collection` | Source collection name |

For multi-collection outputs:

| Column | Description |
|--------|-------------|
| `collections_in_expedition` | All collections in this cluster |
| `collection_count` | Number of collections |
| `is_multi_collection` | True if expedition spans collections |

## Clustering Results Summary

Current clustering results (default parameters: 10km, 7 days):

| Collection | Specimens Clustered | Expeditions Found | Avg Size | Largest |
|------------|--------------------:|------------------:|---------:|--------:|
| botany | ~1.1M | ~120K | ~9 | ~5K |
| ich | 130,755 | 19,931 | 6.6 | 3,855 |
| iz | 228,331 | 30,597 | 7.5 | 2,885 |
| mam | 31,429 | 10,329 | 3.0 | 513 |
| orn | 68,455 | 15,285 | 4.5 | 1,904 |
| orn-en | 545 | 313 | 1.7 | 75 |

## Batch Regeneration

To regenerate all collections:

```bash
cd /Users/joe/expedition-clustering
source .venv/bin/activate

for collection in botany ich iz mam orn orn-en; do
    echo "Processing $collection..."
    python scripts/cluster_csv.py \
        /Users/joe/collections_explorer/incoming_data/$collection
done
```
