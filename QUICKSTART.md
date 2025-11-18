# Quick Start Guide

## TL;DR

Run expedition clustering in one command:

```bash
python cluster.py --limit 10000
```

## What This Does

The expedition clustering pipeline:

1. **Connects** to your MySQL database containing botanical specimen data
2. **Loads** specimen records with collection dates and locations
3. **Clusters** specimens into expeditions using:
   - **Spatial DBSCAN**: Groups specimens within `--e-dist` kilometers (default: 10km)
   - **Temporal DBSCAN**: Within each spatial cluster, splits by time gaps > `--e-days` (default: 7 days)
4. **Saves** results to a CSV file with a unique `spatiotemporal_cluster_id` for each expedition

## Prerequisites

1. **Database**: MySQL running with botany data (via Docker):
   ```bash
   docker-compose up -d
   ```

2. **Python Environment**: Install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

## Basic Usage

### Test with a sample
```bash
python cluster.py --limit 5000 --output data/test.csv
```

### Adjust clustering sensitivity
```bash
# Tighter clusters (smaller distances, shorter time gaps)
python cluster.py --e-dist 5 --e-days 3

# Looser clusters (larger distances, longer time gaps)
python cluster.py --e-dist 20 --e-days 14
```

### Process full dataset
```bash
python cluster.py --output data/all_expeditions.csv
```
**Warning**: Processing 1M+ specimens may take 10-30 minutes depending on your hardware.

## Understanding Parameters

### `--e-dist` (Spatial Epsilon)
- **What it does**: Maximum distance in kilometers between specimens in the same spatial cluster
- **Lower values**: More, smaller clusters (stricter geographic grouping)
- **Higher values**: Fewer, larger clusters (specimens further apart can be in same expedition)
- **Recommended**: 5-20 km depending on typical expedition travel distances

### `--e-days` (Temporal Epsilon)
- **What it does**: Maximum gap in days between collection dates within a spatial cluster
- **Lower values**: Splits multi-day expeditions into separate temporal groups
- **Higher values**: Combines specimens collected weeks apart into same expedition
- **Recommended**: 3-14 days depending on typical expedition duration

## Output Format

The output CSV includes all input columns plus three clustering columns:

| Column | Description |
|--------|-------------|
| `spatial_cluster_id` | Geographic cluster assignment |
| `temporal_cluster_id` | Temporal sub-cluster within each spatial cluster |
| `spatiotemporal_cluster_id` | **Final unique expedition ID** (use this!) |

### Example Output
```csv
collectingeventid,collectionobjectid,latitude1,longitude1,startdate,spatiotemporal_cluster_id
1001,50234,37.7749,-122.4194,2020-05-15,42
1002,50235,37.7751,-122.4190,2020-05-16,42
1003,50236,37.7748,-122.4195,2020-05-17,42
1004,50237,34.0522,-118.2437,2020-05-15,43
```

Specimens 1001-1003 are in expedition #42 (same location, consecutive days).
Specimen 1004 is in expedition #43 (different location, same date).

## Troubleshooting

### "No data loaded from database"
- **Check database**: `docker-compose ps` (should show running containers)
- **Test connection**: `docker exec -it exped_cluster_mysql_container mysql -u myuser -pmypassword -e "SELECT COUNT(*) FROM exped_cluster_db.collectingevent"`

### "Clean dataframe is empty"
- Increase `--limit` to get more candidate specimens
- Check that your database has specimens with both dates AND coordinates

### Memory issues
- Use `--limit` to process data in batches
- Process smaller geographic regions separately (modify SQL in `cluster.py`)

## Next Steps

1. **Analyze results**: Open the output CSV in Python/R/Excel to explore cluster sizes and distributions
2. **Visualize**: Use [plotting.py](plotting.py) to create maps of expedition routes
3. **Tune parameters**: Experiment with different `--e-dist` and `--e-days` values
4. **Validate**: Compare clusters against known expeditions (see [notebooks](notebooks/))

## Advanced: Customize the SQL Query

Edit [cluster.py](cluster.py) line 100-127 to:
- Filter by date range: `AND ce.StartDate BETWEEN '2020-01-01' AND '2023-12-31'`
- Filter by region: `AND l.Latitude1 BETWEEN 30 AND 50`
- Add collector info: `SELECT ... co.Collectors as collectors`

## Getting Help

- See full documentation: [README.md](README.md)
- Check notebooks for examples: [notebooks/](notebooks/)
- View test scripts: [test_simple.py](test_simple.py)
- Report issues: https://github.com/yourusername/expedition-clustering/issues
