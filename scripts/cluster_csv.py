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

_spec = importlib.util.spec_from_file_location("geo_classify", _pkg_dir / "geo_classify.py")
_geo_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_geo_module)
classify_expeditions = _geo_module.classify_expeditions

DEFAULT_INCOMING_DATA_DIR = Path("/Users/joe/collections_explorer/incoming_data")

# Collection-specific spatial clustering distances (in km)
# Initial DBSCAN spatial epsilon - kept tight, merge stage handles broader reach.
# Based on tuning results from collections_explorer (384 param combos per collection).
COLLECTION_E_DIST = {
    "ich": 15.0,   # Fish expeditions have clean patterns, 15km is optimal
    "orn": 15.0,   # Was 100km; merge stage now handles island hopping at 15km
}
DEFAULT_E_DIST = 10.0

# Collection-specific temporal clustering (in days)
# Initial DBSCAN temporal epsilon - tight initial clusters, merge handles reconnection.
# Tuning showed max_gap_days=2-3 is optimal for most collections.
COLLECTION_E_DAYS = {
    "orn": 2.0,        # Was 60; tuning: 2 (collector-aware prevents chaining)
    "iz": 3.0,         # Was 7; tuning: 3
    "mam": 5.0,        # Tuning: 5 (trap checking has multi-day gaps)
    "ent_types": 4.0,  # Tuning: 4
}
DEFAULT_E_DAYS = 3.0   # Was 14; tuning: most collections optimal at 2-3

# Collections that require collector-aware clustering
# When enabled, specimens from different collectors are NEVER merged,
# even if collected at the same place/time.
COLLECTION_COLLECTOR_AWARE = {
    "orn": True,     # Prevents multi-decade chaining
    "botany": True,  # Worst coherence (0.547); biggest improvement opportunity
    "herp": True,    # Tuning uses stage3, implying collector patterns matter
}
DEFAULT_COLLECTOR_AWARE = False

# ---- Merge stage parameters ----
# After initial tight DBSCAN clustering, the merge stage combines nearby
# expeditions using weighted scoring (temporal + spatial + collector overlap + vessel).
# Parameters from tuning results across all 11 collections.

# Maximum temporal gap (days) between expedition end/start to consider merging
COLLECTION_MERGE_GAP_DAYS = {
    "antweb": 2, "ent": 2, "orn-en": 2, "herp": 2, "ich": 2, "geo": 2,
    "iz": 4, "orn": 4,
    # botany(3), ent_types(3), mam(3) match default
}
DEFAULT_MERGE_GAP_DAYS = 3

# Maximum spatial distance (km) between expedition centroids to consider merging
COLLECTION_MERGE_DISTANCE_KM = {
    "antweb": 10, "ent": 10, "mam": 10, "orn-en": 10,
    "orn": 15, "ent_types": 20,
    # botany(30), geo(30), herp(30), ich(30), iz(30) match default
}
DEFAULT_MERGE_DISTANCE_KM = 30

# Minimum merge score (0-1) required to execute a merge
COLLECTION_MERGE_THRESHOLD = {
    "antweb": 0.5, "ent": 0.5, "ent_types": 0.5, "geo": 0.5, "orn-en": 0.5,
    "ich": 0.7, "botany": 0.7, "iz": 0.7, "mam": 0.7,
    # herp(0.6), orn(0.6) match default
}
DEFAULT_MERGE_THRESHOLD = 0.6


def get_collection_e_dist(collection_name: str) -> float:
    """
    Get the optimal spatial clustering distance for a collection.

    Args:
        collection_name: Collection code (e.g., 'ich', 'iz', 'herp')

    Returns:
        Spatial epsilon in kilometers

    """
    return COLLECTION_E_DIST.get(collection_name, DEFAULT_E_DIST)


def get_collection_e_days(collection_name: str) -> float:
    """
    Get the optimal temporal clustering window for a collection.

    Args:
        collection_name: Collection code (e.g., 'orn', 'mam')

    Returns:
        Temporal epsilon in days

    """
    return COLLECTION_E_DAYS.get(collection_name, DEFAULT_E_DAYS)


def get_collection_collector_aware(collection_name: str) -> bool:
    """
    Check if a collection should use collector-aware clustering.

    When enabled, specimens from different collectors are never merged
    into the same expedition, preventing multi-decade chaining.

    Args:
        collection_name: Collection code (e.g., 'orn', 'mam')

    Returns:
        True if collector-aware clustering should be used

    """
    return COLLECTION_COLLECTOR_AWARE.get(collection_name, DEFAULT_COLLECTOR_AWARE)


def get_collection_merge_gap_days(collection_name: str) -> int:
    """Get the optimal merge gap days for a collection."""
    return COLLECTION_MERGE_GAP_DAYS.get(collection_name, DEFAULT_MERGE_GAP_DAYS)


def get_collection_merge_distance_km(collection_name: str) -> float:
    """Get the optimal merge distance for a collection."""
    return COLLECTION_MERGE_DISTANCE_KM.get(collection_name, DEFAULT_MERGE_DISTANCE_KM)


def get_collection_merge_threshold(collection_name: str) -> float:
    """Get the optimal merge threshold for a collection."""
    return COLLECTION_MERGE_THRESHOLD.get(collection_name, DEFAULT_MERGE_THRESHOLD)


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
        default=None,
        help="Spatial epsilon in kilometers (default: auto per collection, ich=15, others=10)",
    )
    parser.add_argument(
        "--e-days",
        type=float,
        default=None,
        help="Temporal epsilon in days (default: auto per collection, orn=2, mam=5, others=3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows to process per collection",
    )
    parser.add_argument(
        "--min-specimens",
        type=int,
        default=20,
        help="Minimum specimens per cluster (clusters with fewer are excluded, default: 20)",
    )
    parser.add_argument(
        "--collector-aware",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        help="Enable collector-aware clustering (default: auto per collection, orn=True, others=False). "
             "When enabled, specimens from different collectors are never merged into the same expedition.",
    )
    parser.add_argument(
        "--collector-alias-file",
        type=Path,
        default=None,
        help="Path to CSV file with collector disambiguation mappings (normalized_name,canonical_id). "
             "If provided, uses disambiguated collector IDs for partitioning instead of normalized names.",
    )
    parser.add_argument(
        "--merge-gap-days",
        type=int,
        default=None,
        help="Max temporal gap (days) between expeditions to consider merging (default: auto per collection)",
    )
    parser.add_argument(
        "--merge-distance-km",
        type=float,
        default=None,
        help="Max spatial distance (km) between expedition centroids to merge (default: auto per collection)",
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=None,
        help="Minimum merge score (0-1) required to execute a merge (default: auto per collection)",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Disable the merge stage (only run initial DBSCAN clustering)",
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

    # Determine e_dist and e_days: use explicit args if provided, otherwise auto-detect from collection
    collection_name = None
    if len(inputs) == 1:
        collection_name = inputs[0].name if inputs[0].is_dir() else inputs[0].stem

    if args.e_dist is not None:
        e_dist = args.e_dist
    elif collection_name:
        e_dist = get_collection_e_dist(collection_name)
    else:
        e_dist = DEFAULT_E_DIST

    if args.e_days is not None:
        e_days = args.e_days
    elif collection_name:
        e_days = get_collection_e_days(collection_name)
    else:
        e_days = DEFAULT_E_DAYS

    # Determine collector_aware setting
    if args.collector_aware is not None:
        collector_aware = args.collector_aware
    elif collection_name:
        collector_aware = get_collection_collector_aware(collection_name)
    else:
        collector_aware = DEFAULT_COLLECTOR_AWARE

    # Determine merge parameters
    enable_merge = not args.no_merge

    if args.merge_gap_days is not None:
        merge_gap_days = args.merge_gap_days
    elif collection_name:
        merge_gap_days = get_collection_merge_gap_days(collection_name)
    else:
        merge_gap_days = DEFAULT_MERGE_GAP_DAYS

    if args.merge_distance_km is not None:
        merge_distance_km = args.merge_distance_km
    elif collection_name:
        merge_distance_km = get_collection_merge_distance_km(collection_name)
    else:
        merge_distance_km = DEFAULT_MERGE_DISTANCE_KM

    if args.merge_threshold is not None:
        merge_threshold = args.merge_threshold
    elif collection_name:
        merge_threshold = get_collection_merge_threshold(collection_name)
    else:
        merge_threshold = DEFAULT_MERGE_THRESHOLD

    if collection_name:
        print(f"  Using collection-specific params for {collection_name}: "
              f"e_dist={e_dist}km, e_days={e_days} days, collector_aware={collector_aware}")
        if enable_merge:
            print(f"  Merge params: gap={merge_gap_days}d, dist={merge_distance_km}km, threshold={merge_threshold}")

    # Run clustering
    collector_str = ", collector-aware" if collector_aware else ""
    merge_str = ", merge disabled" if not enable_merge else ""
    alias_str = ", using alias file" if args.collector_alias_file else ""
    print(f"Clustering {len(df_valid)} specimens (e_dist={e_dist}km, e_days={e_days} days{collector_str}{merge_str}{alias_str})...")
    pipeline = create_pipeline(
        e_dist=e_dist,
        e_days=e_days,
        collector_aware=collector_aware,
        collector_alias_file=args.collector_alias_file,
        merge_gap_days=merge_gap_days,
        merge_distance_km=merge_distance_km,
        merge_threshold=merge_threshold,
        enable_merge=enable_merge,
    )
    clustered = pipeline.fit_transform(df_valid)

    # Filter out clusters with fewer than min_specimens
    if args.min_specimens > 1:
        cluster_sizes = clustered.groupby("spatiotemporal_cluster_id").size()
        valid_clusters = cluster_sizes[cluster_sizes >= args.min_specimens].index
        excluded_clusters = len(cluster_sizes) - len(valid_clusters)
        excluded_specimens = len(clustered) - clustered[clustered["spatiotemporal_cluster_id"].isin(valid_clusters)].shape[0]

        if excluded_clusters > 0:
            print(f"  Excluded {excluded_clusters} clusters with <{args.min_specimens} specimens ({excluded_specimens} specimens)")

        clustered = clustered[clustered["spatiotemporal_cluster_id"].isin(valid_clusters)].copy()
        print(f"  Retained {len(valid_clusters)} clusters ({len(clustered)} specimens)")

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

    # Add geographic classification to expeditions
    print("Adding geographic classification...")
    clustered = classify_expeditions(
        clustered,
        cluster_col="spatiotemporal_cluster_id",
        lat_col="latitude1",
        lng_col="longitude1",
    )

    # Report results
    num_clusters = clustered["spatiotemporal_cluster_id"].nunique()
    cluster_sizes = clustered.groupby("spatiotemporal_cluster_id").size()

    print("\nResults:")
    print(f"  Specimens: {len(clustered)}")
    print(f"  Clusters: {num_clusters}")
    print(f"  Avg size: {len(clustered) / num_clusters:.1f}")
    print(f"  Largest: {cluster_sizes.max()}")

    # Multi-collection expedition stats
    if "is_multi_collection" in clustered.columns and len(clustered["collection"].dropna().unique()) > 1:
        multi_coll_clusters = clustered[clustered["is_multi_collection"]]["spatiotemporal_cluster_id"].nunique()
        multi_coll_specimens = clustered["is_multi_collection"].sum()
        print("\nCross-collection expeditions:")
        print(f"  Multi-collection clusters: {multi_coll_clusters} ({100*multi_coll_clusters/num_clusters:.1f}%)")
        print(f"  Specimens in multi-collection clusters: {multi_coll_specimens} ({100*multi_coll_specimens/len(clustered):.1f}%)")

        # Show breakdown by collection combination
        combo_counts = clustered[clustered["is_multi_collection"]].groupby("collections_in_expedition").agg(
            clusters=("spatiotemporal_cluster_id", "nunique"),
            specimens=("spatiotemporal_cluster_id", "count")
        ).sort_values("specimens", ascending=False)

        if len(combo_counts) > 0:
            print("\n  Collection combinations:")
            for combo, row in combo_counts.head(10).iterrows():
                print(f"    {combo}: {row['clusters']} clusters, {row['specimens']} specimens")

    # Per-collection breakdown if multiple collections
    if "collection" in clustered.columns and len(clustered["collection"].dropna().unique()) > 1:
        print("\nPer-collection breakdown:")
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
