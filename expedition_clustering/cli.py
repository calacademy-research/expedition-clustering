#!/usr/bin/env python3
"""
Command-line tool to run expedition clustering on the CAS Botany database.

Features:
- Automatic batch processing for large datasets
- Memory-efficient clustering
- Progress tracking

Usage:
    expedition-cluster --output clustered_data.csv
    expedition-cluster --e-dist 15 --e-days 10 --batch-size 50000
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql
from sklearn.cluster import DBSCAN

from expedition_clustering import create_pipeline


def create_spatial_batches(df, batch_size, e_dist, logger):
    """
    Create batches using spatial pre-clustering to avoid splitting expeditions.

    This uses coarse spatial clustering (2x the target e_dist) to group specimens
    that are geographically close, then assigns these groups to batches.

    Returns:
        List of DataFrames, one per batch
    """
    logger.info("Creating spatial-aware batches to avoid splitting expeditions...")
    logger.info("Pre-clustering with eps=%.1f km (2x target distance)...", e_dist * 2)

    # Use coarse spatial clustering to identify geographic groups
    # We use 2x the target e_dist to ensure specimens within e_dist stay together
    coords = np.radians(df[['latitude1', 'longitude1']].values)
    coarse_eps = (e_dist * 2) / 6371  # Convert km to radians (Earth radius = 6371 km)

    # DBSCAN with larger epsilon to create geographic super-clusters
    spatial_pre_cluster = DBSCAN(eps=coarse_eps, min_samples=1, metric='haversine')
    df['_spatial_group'] = spatial_pre_cluster.fit_predict(coords)

    num_groups = df['_spatial_group'].nunique()
    logger.info(f"Identified {num_groups} geographic groups")

    # Sort by spatial group to keep groups together
    df_sorted = df.sort_values('_spatial_group').reset_index(drop=True)

    # Assign groups to batches, trying to balance batch sizes
    group_sizes = df_sorted.groupby('_spatial_group').size().sort_values(ascending=False)

    # Greedy bin packing: assign each group to the batch with the most space
    batch_assignments = {}
    batch_sizes = {}
    num_batches = max(1, (len(df) + batch_size - 1) // batch_size)

    for batch_num in range(num_batches):
        batch_sizes[batch_num] = 0

    for group_id, size in group_sizes.items():
        # Find batch with most remaining space
        available_space = {b: batch_size - batch_sizes[b] for b in batch_sizes}
        target_batch = max(available_space, key=available_space.get)

        batch_assignments[group_id] = target_batch
        batch_sizes[target_batch] += size

    # Assign batch numbers to specimens
    df_sorted['_batch_num'] = df_sorted['_spatial_group'].map(batch_assignments)

    # Create batch DataFrames
    batches = []
    for batch_num in sorted(df_sorted['_batch_num'].unique()):
        batch_df = df_sorted[df_sorted['_batch_num'] == batch_num].copy()
        batch_df = batch_df.drop(columns=['_spatial_group', '_batch_num'])
        batches.append(batch_df)
        logger.info(f"Batch {batch_num + 1}: {len(batch_df)} specimens")

    return batches


def process_batch(df_batch, pipeline, logger, batch_num=None):
    """Process a single batch of specimens through the clustering pipeline."""
    batch_desc = f" (batch {batch_num})" if batch_num is not None else ""
    logger.info(f"Processing {len(df_batch)} specimens{batch_desc}...")

    try:
        clustered = pipeline.fit_transform(df_batch)
        return clustered
    except MemoryError:
        logger.error(f"Out of memory processing batch{batch_desc}. Try reducing --batch-size.")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Cluster botanical specimens into expeditions using spatiotemporal DBSCAN",
        epilog="For large datasets (>100K specimens), batch processing is automatically enabled."
    )

    # Database connection
    parser.add_argument("--host", default="localhost", help="MySQL host (default: localhost)")
    parser.add_argument("--port", type=int, default=3306, help="MySQL port (default: 3306)")
    parser.add_argument("--user", default="myuser", help="MySQL user (default: myuser)")
    parser.add_argument("--password", default="mypassword", help="MySQL password")
    parser.add_argument(
        "--database",
        default="exped_cluster_db",
        help="Database name (default: exped_cluster_db)",
    )

    # Clustering parameters
    parser.add_argument(
        "--e-dist",
        type=float,
        default=10.0,
        help="Spatial epsilon in kilometers - max distance between specimens in same cluster (default: 10.0)",
    )
    parser.add_argument(
        "--e-days",
        type=float,
        default=7.0,
        help="Temporal epsilon in days - max time gap between specimens in same cluster (default: 7.0)",
    )

    # Data options
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of specimens to process (for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Number of specimens to process per batch (default: 50000). Lower this if running out of memory.",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable batch processing (not recommended for datasets >100K specimens)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/clustered_expeditions.csv"),
        help="Output CSV file path (default: data/clustered_expeditions.csv)",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        # Connect to database
        logger.info("Connecting to MySQL database at %s:%s...", args.host, args.port)
        conn = pymysql.connect(
            host=args.host,
            user=args.user,
            password=args.password,
            database=args.database,
            port=args.port,
        )

        # Build query
        query = """
        SELECT
            ce.CollectingEventID as collectingeventid,
            ce.StartDate as startdate,
            ce.EndDate as enddate,
            ce.Remarks as remarks,
            ce.LocalityID as localityid,
            co.CollectionObjectID as collectionobjectid,
            co.Text1 as text1,
            l.MinElevation as minelevation,
            l.MaxElevation as maxelevation,
            l.ElevationAccuracy as elevationaccuracy,
            l.Latitude1 as latitude1,
            l.Longitude1 as longitude1,
            l.LocalityName as localityname,
            l.NamedPlace as namedplace,
            l.GeographyID as geographyid,
            g.CentroidLat as centroidlat,
            g.CentroidLon as centroidlon,
            g.CommonName as commonname,
            g.FullName as fullname,
            g.Name as name
        FROM collectingevent ce
        INNER JOIN collectionobject co ON ce.CollectingEventID = co.CollectingEventID
        LEFT JOIN locality l ON ce.LocalityID = l.LocalityID
        LEFT JOIN geography g ON l.GeographyID = g.GeographyID
        WHERE ce.StartDate IS NOT NULL
          AND (l.Latitude1 IS NOT NULL OR g.CentroidLat IS NOT NULL)
          AND (l.Longitude1 IS NOT NULL OR g.CentroidLon IS NOT NULL)
        """

        if args.limit:
            query += f"\nLIMIT {args.limit}"
            logger.info("Loading up to %d specimens from database...", args.limit)
        else:
            logger.info("Loading all specimens from database (this may take a while)...")

        # Load data
        df = pd.read_sql_query(query, conn)
        conn.close()

        logger.info("Loaded %d rows", len(df))

        if df.empty:
            logger.error("No data loaded from database!")
            sys.exit(1)

        # Normalize column names
        df.columns = df.columns.str.lower()

        # Convert dates
        df['startdate'] = pd.to_datetime(df['startdate'], errors='coerce')
        df['enddate'] = pd.to_datetime(df['enddate'], errors='coerce')

        # Fill in missing precise coordinates with geography centroids
        df['latitude1'] = df['latitude1'].fillna(df['centroidlat'])
        df['longitude1'] = df['longitude1'].fillna(df['centroidlon'])

        # Drop rows without coordinates or dates
        initial_count = len(df)
        df = df[df['latitude1'].notna() & df['longitude1'].notna() & df['startdate'].notna()]
        dropped = initial_count - len(df)

        if dropped > 0:
            logger.warning("Dropped %d rows with missing data after loading", dropped)

        if df.empty:
            logger.error("No valid data remaining after cleaning!")
            sys.exit(1)

        total_specimens = len(df)
        logger.info("Processing %d specimens...", total_specimens)

        # Determine if batch processing is needed
        use_batching = not args.no_batch and (total_specimens > 100000 or args.limit is None)

        if use_batching:
            logger.info("Using spatial-aware batch processing (target batch size: %d specimens)", args.batch_size)
            logger.info("Batches are created using spatial pre-clustering to avoid splitting expeditions.")

            # Create spatial-aware batches
            batches = create_spatial_batches(df, args.batch_size, args.e_dist, logger)
            num_batches = len(batches)

            # Process each batch
            all_results = []
            for batch_num, batch_df in enumerate(batches, start=1):
                logger.info(f"\n{'='*60}")
                logger.info(f"Batch {batch_num}/{num_batches}")
                logger.info(f"{'='*60}")

                # Create pipeline for this batch
                pipeline = create_pipeline(e_dist=args.e_dist, e_days=args.e_days)

                # Process batch
                clustered_batch = process_batch(batch_df, pipeline, logger, batch_num)

                # Add batch number to cluster IDs to make them unique across batches
                if 'spatiotemporal_cluster_id' in clustered_batch.columns:
                    clustered_batch['spatiotemporal_cluster_id'] += (batch_num - 1) * 1000000
                    clustered_batch['batch_number'] = batch_num

                all_results.append(clustered_batch)

                # Report batch results
                batch_clusters = clustered_batch['spatiotemporal_cluster_id'].nunique()
                logger.info(f"Batch {batch_num} complete: {len(clustered_batch)} specimens → {batch_clusters} expeditions")

            # Combine all batches
            logger.info("\nCombining all batches...")
            clustered = pd.concat(all_results, ignore_index=True)

        else:
            # Single-batch processing
            logger.info("Processing entire dataset in a single pass...")
            logger.info("Running spatiotemporal clustering (e_dist=%skm, e_days=%s days)...",
                       args.e_dist, args.e_days)

            pipeline = create_pipeline(e_dist=args.e_dist, e_days=args.e_days)
            clustered = process_batch(df, pipeline, logger)

        # Report results
        num_clusters = clustered['spatiotemporal_cluster_id'].nunique()
        avg_size = len(clustered) / num_clusters
        cluster_sizes = clustered.groupby('spatiotemporal_cluster_id').size()

        logger.info("=" * 60)
        logger.info("Clustering Results:")
        logger.info("  Total specimens: %d", len(clustered))
        logger.info("  Total expeditions (clusters): %d", num_clusters)
        logger.info("  Average expedition size: %.2f specimens", avg_size)
        logger.info("  Median expedition size: %.0f specimens", cluster_sizes.median())
        logger.info("  Largest expedition: %d specimens", cluster_sizes.max())
        logger.info("  Smallest expedition: %d specimens", cluster_sizes.min())
        if use_batching:
            logger.info("  Processed in %d batches", num_batches)
        logger.info("=" * 60)

        # Save results
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        clustered.to_csv(output_path, index=False)
        logger.info("Results saved to: %s", output_path)

        # Show sample
        logger.info("\nSample of results (first 10 rows):")
        sample_cols = ['collectingeventid', 'collectionobjectid', 'latitude1', 'longitude1',
                      'startdate', 'spatiotemporal_cluster_id']
        if 'batch_number' in clustered.columns:
            sample_cols.append('batch_number')
        print(clustered[sample_cols].head(10).to_string(index=False))

        logger.info("\n✓ Clustering completed successfully!")

    except pymysql.Error as e:
        logger.error("Database error: %s", e)
        logger.error("Make sure the database is running: docker-compose up -d")
        sys.exit(1)

    except MemoryError as e:
        logger.error("Out of memory! Try one of these solutions:")
        logger.error("  1. Reduce --batch-size (current: %d)", args.batch_size)
        logger.error("  2. Use --limit to process fewer specimens")
        logger.error("  3. Add geographic/temporal filters to the query")
        logger.error("  4. Use a machine with more RAM")
        sys.exit(1)

    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
