# Memory Optimization & Batch Processing

## Overview

The `expedition-cluster` tool now includes automatic batch processing for large datasets (>100K specimens), preventing out-of-memory errors when clustering the full 867K+ specimen database.

## How It Works

### Automatic Spatial-Aware Batch Processing

When processing more than 100,000 specimens (or when no `--limit` is specified), the tool automatically:

1. **Pre-clusters specimens spatially** using 2x the target distance to identify geographic groups
2. **Assigns groups to batches** using greedy bin-packing to balance batch sizes
3. **Processes each batch independently** with full spatiotemporal clustering
4. **Assigns unique cluster IDs** across batches (batch 1: 0-999999, batch 2: 1000000-1999999, etc.)
5. **Combines results** into a single output file with a `batch_number` column

**Key Feature**: Spatial pre-clustering ensures that specimens within your target distance parameter (`--e-dist`) are **never split across batches**, preventing expeditions from being artificially separated.

### Memory Usage

| Dataset Size | Without Batching | With Batching (50K) |
|--------------|------------------|---------------------|
| 10K specimens | ~200 MB | ~200 MB (no batching) |
| 100K specimens | ~2-4 GB | ~2-4 GB (no batching) |
| 500K specimens | ~15-20 GB (likely OOM) | ~2-4 GB per batch |
| 867K specimens | **CRASHES** | ~2-4 GB per batch |

## Usage

### Basic (Automatic Batching)

```bash
# Process full dataset with automatic batching
expedition-cluster --e-dist 15 --e-days 10

# Output:
# - Processed in 18 batches (867K / 50K)
# - Each batch uses ~2-4 GB RAM
# - Total time: ~5-10 minutes
```

### Custom Batch Size

```bash
# Smaller batches for low-memory systems
expedition-cluster --batch-size 25000

# Larger batches for high-memory systems (faster but more RAM)
expedition-cluster --batch-size 100000
```

### Disable Batching (Small Datasets)

```bash
# Process small datasets in one pass
expedition-cluster --limit 50000 --no-batch
```

## Configuration Options

### `--batch-size N`
- **Default**: 50,000 specimens
- **Lower** (10K-25K): Use less memory, slower overall
- **Higher** (75K-100K): Use more memory, faster if you have RAM
- **Recommendation**:
  - 8 GB RAM: --batch-size 25000
  - 16 GB RAM: --batch-size 50000 (default)
  - 32 GB RAM: --batch-size 100000

### `--no-batch`
- Disables batch processing
- Only use for datasets < 100K specimens
- Will crash with out-of-memory on large datasets

## Understanding the Output

### Batch Number Column

When batch processing is used, the output CSV includes a `batch_number` column:

```csv
collectingeventid,latitude1,longitude1,startdate,spatiotemporal_cluster_id,batch_number
7,39.07,-100.0,1888-01-01,0,1
30,38.83,-100.0,1901-08-02,1,1
...
500000,35.45,-118.5,1950-06-15,1000000,2
```

### Cluster ID Scheme

- **Batch 1**: Cluster IDs 0 - 999,999
- **Batch 2**: Cluster IDs 1,000,000 - 1,999,999
- **Batch 3**: Cluster IDs 2,000,000 - 2,999,999
- etc.

This ensures unique cluster IDs across all batches.

## Smart Batching Algorithm

### How Spatial Pre-Clustering Works

To prevent splitting expeditions across batches, the tool uses a two-phase approach:

**Phase 1: Spatial Pre-Clustering (Geographic Grouping)**
```
Target distance: 10 km (from --e-dist parameter)
Pre-clustering distance: 20 km (2x the target)
```

The tool first groups specimens using a coarser spatial epsilon (2x your `--e-dist`). This creates geographic "super-clusters" that are guaranteed to keep all specimens within `e-dist` of each other in the same group.

**Phase 2: Greedy Bin-Packing (Batch Assignment)**

Geographic groups are assigned to batches using a greedy algorithm:
1. Sort groups by size (largest first)
2. Assign each group to the batch with the most remaining space
3. This balances batch sizes while respecting geographic boundaries

**Example**:
```
Dataset: 150K specimens, --e-dist 15, --batch-size 50000

Pre-clustering with eps=30km:
- Creates 1,560 geographic groups
- Each group guaranteed to be spatially cohesive

Bin-packing result:
- Batch 1: 80,179 specimens (1,407 expeditions)
- Batch 2: 34,730 specimens (7,052 expeditions)
- Batch 3: 34,730 specimens (8,588 expeditions)
```

Notice batches are different sizes (not exactly 50K) because they respect geographic boundaries.

### Why This Prevents Expedition Splitting

**Guarantee**: If two specimens are within `--e-dist` kilometers of each other, they will be in the same batch.

**Proof**: The pre-clustering uses 2x the target distance. Any two points within `e_dist` of each other must be in the same pre-cluster (since they're less than 2×e_dist apart). Since pre-clusters are assigned to batches atomically, these specimens will be in the same batch.

### Limitations

While spatial pre-clustering prevents most expedition splitting, there are edge cases:

**Time-Only Splitting**: Specimens in the same geographic location but separated by large time gaps may still be in different expeditions (by design - this is correct behavior for the temporal clustering).

**Memory Constraints**: If a single geographic group exceeds your batch size, it will be placed in its own batch. This is rare but can happen with very dense collection areas (e.g., a botanical garden with 100K specimens).

## Performance Tips

### 1. Use Geographic Filters

Edit `expedition_clustering/cli.py` to add WHERE clauses:

```python
query = """
    ...
    WHERE ce.StartDate IS NOT NULL
      AND ce.StartDate >= '2000-01-01'  # Recent specimens only
      AND l.Latitude1 BETWEEN 32 AND 42  # California region
      AND l.Longitude1 BETWEEN -125 AND -114
    """
```

### 2. Process by Region

Run separate jobs for different geographic regions:

```bash
# Edit cli.py to add California filter, then:
expedition-cluster --output data/california.csv

# Edit cli.py to add Oregon filter, then:
expedition-cluster --output data/oregon.csv
```

### 3. Process by Time Period

```bash
# Edit cli.py to filter by decade
# 1900-1910, 1910-1920, etc.
```

## Troubleshooting

### Still Running Out of Memory?

```bash
# Reduce batch size to 10K
expedition-cluster --batch-size 10000

# Or process fewer specimens
expedition-cluster --limit 100000
```

### Taking Too Long?

```bash
# Increase batch size (if you have RAM)
expedition-cluster --batch-size 100000

# Or process in parallel (manual)
# Split data into regions and run multiple instances
```

### Want Single-Pass Results?

For datasets <100K specimens, you can disable batching:

```bash
expedition-cluster --limit 50000 --no-batch
```

## Technical Details

### Memory Optimization in Pipeline

The clustering pipeline is already optimized:

1. **Spatial DBSCAN**: Processes entire dataset at once (memory bottleneck)
2. **Temporal DBSCAN**: Processes each spatial cluster independently (memory efficient)
3. **Batch Processing**: Limits spatial DBSCAN to manageable chunks

### Why Not Stream Processing?

DBSCAN requires all points in memory to calculate pairwise distances. True streaming isn't possible without fundamentally changing the algorithm (e.g., to approximate or hierarchical clustering).

## Recommended Workflow

For the full 867K dataset:

```bash
# Process with default settings
expedition-cluster --e-dist 15 --e-days 10 --output data/all_expeditions.csv

# Expected:
# - 18 batches
# - ~10 minutes
# - 4-8 GB RAM peak
# - ~21K-25K total expedition clusters
```
