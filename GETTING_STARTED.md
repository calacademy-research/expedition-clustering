# Getting Started with Expedition Clustering

## What This Does

Groups botanical specimens into expeditions based on when and where they were collected.

## Quick Start (3 Steps)

### 1. Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Start Database
```bash
docker-compose up -d
```

### 3. Run Clustering
```bash
# Test with 5,000 specimens
expedition-cluster --limit 5000

# Full dataset (may take 10-30 minutes)
expedition-cluster
```

## Output

Results are saved to `data/clustered_expeditions.csv` with a unique `spatiotemporal_cluster_id` for each expedition.

## What's Next?

- **Adjust parameters**: `expedition-cluster --e-dist 15 --e-days 10`
- **Read more**: See [QUICKSTART.md](QUICKSTART.md) for detailed guide
- **Full docs**: See [README.md](README.md) for complete documentation
- **Explore data**: Open Jupyter notebooks in [notebooks/](notebooks/)

## Troubleshooting

**Database connection error?**
```bash
docker-compose up -d
docker-compose ps  # Verify MySQL is running
```

**No output data?**
- Use `--limit 10000` to process a smaller sample first
- Check that your database has specimens with dates and coordinates

## File Reference

- `expedition-cluster command` - Main clustering script ⭐
- `expedition_clustering/` - Python package
- `notebooks/` - Jupyter notebooks for analysis
- `data/` - Input/output data (git-ignored)
