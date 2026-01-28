#!/usr/bin/env python3
"""
Cluster specimens from CSV files into expeditions.

Usage:
    # Single collection
    python scripts/cluster_csv.py INPUT_CSV OUTPUT_CSV
    python scripts/cluster_csv.py INPUT_CSV OUTPUT_CSV --e-dist 15 --e-days 14

    # Multiple collections
    python scripts/cluster_csv.py collection1 collection2 collection3 -o OUTPUT_CSV

    # All collections in a directory
    python scripts/cluster_csv.py --all --incoming-data-dir /path/to/incoming_data -o OUTPUT_CSV
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

# Direct imports to avoid cartopy dependency in __init__.py
_pkg_dir = Path(__file__).parent.parent / "expedition_clustering"

_spec = importlib.util.spec_from_file_location("csv_source", _pkg_dir / "csv_source.py")
_csv_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_csv_module)
load_collection_csv = _csv_module.load_collection_csv
list_available_collections = _csv_module.list_available_collections

_spec = importlib.util.spec_from_file_location("pipeline", _pkg_dir / "pipeline.py")
_pipeline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pipeline_module)
create_pipeline = _pipeline_module.create_pipeline

DEFAULT_INCOMING_DATA_DIR = Path("/Users/joe/collections_explorer/incoming_data")


def load_multiple_collections(
    inputs: list[Path],
    limit: int | None = None,
) -> pd.DataFrame:
    """Load and combine data from multiple collections."""
    all_dfs = []

    for input_path in inputs:
        collection_name = input_path.name if input_path.is_dir() else input_path.stem
        print(f"Loading {collection_name}...")

        df = load_collection_csv(input_path, limit=limit)
        df["collection"] = collection_name
        all_dfs.append(df)
        print(f"  Loaded {len(df)} rows from {collection_name}")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined total: {len(combined)} rows from {len(inputs)} collections")
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Cluster specimens from CSV files into expeditions"
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="*",
        help="Path(s) to input CSV file(s) or collection directory(ies)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Path for output CSV file",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all collections in the incoming-data-dir",
    )
    parser.add_argument(
        "--incoming-data-dir",
        type=Path,
        default=DEFAULT_INCOMING_DATA_DIR,
        help=f"Directory containing collections (default: {DEFAULT_INCOMING_DATA_DIR})",
    )
    parser.add_argument(
        "--e-dist",
        type=float,
        default=10.0,
        help="Spatial epsilon in kilometers (default: 10.0)",
    )
    parser.add_argument(
        "--e-days",
        type=float,
        default=7.0,
        help="Temporal epsilon in days (default: 7.0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows to process per collection",
    )

    args = parser.parse_args()

    # Determine inputs
    if args.all:
        # Load all collections from incoming_data_dir
        collections = list_available_collections(args.incoming_data_dir)
        # Exclude PortalFiles (duplicate of orn)
        collections = [c for c in collections if c != "PortalFiles"]
        inputs = [args.incoming_data_dir / c for c in collections]
        print(f"Processing all {len(inputs)} collections: {', '.join(collections)}")
    elif args.input:
        inputs = args.input
    else:
        parser.error("Must specify input path(s) or --all")

    # Determine output
    if args.output:
        output = args.output
    elif len(inputs) == 1:
        # Default: same directory as input
        if inputs[0].is_dir():
            output = inputs[0] / "clustered_expeditions.csv"
        else:
            output = inputs[0].parent / "clustered_expeditions.csv"
    else:
        # Multiple inputs require explicit output
        parser.error("Must specify --output (-o) when using multiple inputs or --all")

    # Load data
    if len(inputs) == 1:
        print(f"Loading data from {inputs[0]}...")
        df = load_collection_csv(inputs[0], limit=args.limit)
        df["collection"] = inputs[0].name if inputs[0].is_dir() else inputs[0].stem
        print(f"  Loaded {len(df)} rows")
    else:
        df = load_multiple_collections(inputs, limit=args.limit)

    # Filter to valid records
    valid_mask = df["latitude1"].notna() & df["longitude1"].notna() & df["startdate"].notna()
    df_valid = df[valid_mask].copy()
    dropped = len(df) - len(df_valid)

    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing coordinates or dates")

    if df_valid.empty:
        print("Error: No valid data to cluster")
        sys.exit(1)

    # Run clustering
    print(f"Clustering {len(df_valid)} specimens (e_dist={args.e_dist}km, e_days={args.e_days} days)...")
    pipeline = create_pipeline(e_dist=args.e_dist, e_days=args.e_days)
    clustered = pipeline.fit_transform(df_valid)

    # Identify multi-collection expeditions
    if "collection" in clustered.columns:
        # Count collections per cluster
        cluster_collections = clustered.groupby("spatiotemporal_cluster_id")["collection"].apply(
            lambda x: x.unique().tolist()
        )
        cluster_collection_counts = cluster_collections.apply(len)

        # Map back to each row
        clustered["collections_in_expedition"] = clustered["spatiotemporal_cluster_id"].map(
            lambda x: ", ".join(sorted(cluster_collections[x]))
        )
        clustered["collection_count"] = clustered["spatiotemporal_cluster_id"].map(cluster_collection_counts)
        clustered["is_multi_collection"] = clustered["collection_count"] > 1

    # Report results
    num_clusters = clustered["spatiotemporal_cluster_id"].nunique()
    cluster_sizes = clustered.groupby("spatiotemporal_cluster_id").size()

    print(f"\nResults:")
    print(f"  Specimens: {len(clustered)}")
    print(f"  Clusters: {num_clusters}")
    print(f"  Avg size: {len(clustered) / num_clusters:.1f}")
    print(f"  Largest: {cluster_sizes.max()}")

    # Multi-collection expedition stats
    if "is_multi_collection" in clustered.columns and clustered["collection"].nunique() > 1:
        multi_coll_clusters = clustered[clustered["is_multi_collection"]]["spatiotemporal_cluster_id"].nunique()
        multi_coll_specimens = clustered["is_multi_collection"].sum()
        print(f"\nCross-collection expeditions:")
        print(f"  Multi-collection clusters: {multi_coll_clusters} ({100*multi_coll_clusters/num_clusters:.1f}%)")
        print(f"  Specimens in multi-collection clusters: {multi_coll_specimens} ({100*multi_coll_specimens/len(clustered):.1f}%)")

        # Show breakdown by collection combination
        combo_counts = clustered[clustered["is_multi_collection"]].groupby("collections_in_expedition").agg(
            clusters=("spatiotemporal_cluster_id", "nunique"),
            specimens=("spatiotemporal_cluster_id", "count")
        ).sort_values("specimens", ascending=False)

        if len(combo_counts) > 0:
            print(f"\n  Collection combinations:")
            for combo, row in combo_counts.head(10).iterrows():
                print(f"    {combo}: {row['clusters']} clusters, {row['specimens']} specimens")

    # Per-collection breakdown if multiple collections
    if "collection" in clustered.columns and clustered["collection"].nunique() > 1:
        print(f"\nPer-collection breakdown:")
        for coll in sorted(clustered["collection"].unique()):
            coll_df = clustered[clustered["collection"] == coll]
            coll_clusters = coll_df["spatiotemporal_cluster_id"].nunique()
            print(f"  {coll}: {len(coll_df)} specimens in {coll_clusters} clusters")

    # Save output
    output.parent.mkdir(parents=True, exist_ok=True)
    clustered.to_csv(output, index=False)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
