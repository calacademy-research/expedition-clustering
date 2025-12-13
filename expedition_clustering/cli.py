#!/usr/bin/env python3
"""
Command-line tool to run expedition clustering on the CAS Botany database.

Features:
- Full-dataset spatiotemporal clustering
- Progress tracking

Usage:
    expedition-cluster --output clustered_data.csv
    expedition-cluster --e-dist 15 --e-days 10
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import pymysql

from expedition_clustering import create_pipeline


def process_batch(df_batch, pipeline, logger):
    """Process a single batch of specimens through the clustering pipeline."""
    logger.info("Processing %s specimens...", len(df_batch))

    try:
        return pipeline.fit_transform(df_batch)
    except MemoryError:
        logger.exception("Out of memory processing dataset. Consider reducing --limit.")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Cluster botanical specimens into expeditions using spatiotemporal DBSCAN"
    )

    # Database connection
    parser.add_argument("--host", default="localhost", help="MySQL host (default: localhost)")
    parser.add_argument("--port", type=int, default=3306, help="MySQL port (default: 3306)")
    parser.add_argument("--user", default="myuser", help="MySQL user (default: myuser)")
    parser.add_argument("--password", default="examplepw", help="MySQL password")
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
        "--include-centroids",
        action="store_true",
        help="Include specimens using geography centroids when precise coordinates are missing (may include inaccurate data)",
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

        # Build query with coordinate filtering based on --include-centroids flag
        if args.include_centroids:
            # Include specimens with geography centroids when precise coords are missing
            logger.info("Including specimens with geography centroids (may contain inaccurate data)")
            coord_filter = """
          AND (l.Latitude1 IS NOT NULL OR g.CentroidLat IS NOT NULL)
          AND (l.Longitude1 IS NOT NULL OR g.CentroidLon IS NOT NULL)"""
        else:
            # Default: only use precise locality coordinates
            logger.info("Using only precise locality coordinates (excluding geography centroids)")
            coord_filter = """
          AND l.Latitude1 IS NOT NULL
          AND l.Longitude1 IS NOT NULL"""

        query = f"""  # noqa: S608
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
        WHERE ce.StartDate IS NOT NULL{coord_filter}
        """

        if args.limit:
            query += f"\nLIMIT {args.limit}"
            logger.info("Loading up to %d specimens from database...", args.limit)
        else:
            logger.info("Loading all specimens from database (this may take a while)...")

        # Load data
        records_df = pd.read_sql_query(query, conn)
        conn.close()

        logger.info("Loaded %d rows", len(records_df))

        if records_df.empty:
            logger.error("No data loaded from database!")
            sys.exit(1)

        # Normalize column names
        records_df.columns = records_df.columns.str.lower()

        # Convert dates
        records_df["startdate"] = pd.to_datetime(records_df["startdate"], errors="coerce")
        records_df["enddate"] = pd.to_datetime(records_df["enddate"], errors="coerce")

        # Fill in missing precise coordinates with geography centroids (only if --include-centroids)
        if args.include_centroids:
            records_df["latitude1"] = records_df["latitude1"].fillna(records_df["centroidlat"])
            records_df["longitude1"] = records_df["longitude1"].fillna(records_df["centroidlon"])
            logger.info(
                "Filled missing coordinates with %d geography centroids",
                records_df["latitude1"].notna().sum()
                - (records_df["latitude1"].notna() & records_df["centroidlat"].isna()).sum(),
            )

        # Drop rows without coordinates or dates
        initial_count = len(records_df)
        records_df = records_df[
            records_df["latitude1"].notna() & records_df["longitude1"].notna() & records_df["startdate"].notna()
        ]
        dropped = initial_count - len(records_df)

        if dropped > 0:
            logger.warning("Dropped %d rows with missing data after loading", dropped)

        if records_df.empty:
            logger.error("No valid data remaining after cleaning!")
            sys.exit(1)

        total_specimens = len(records_df)
        logger.info("Processing %d specimens...", total_specimens)

        logger.info("Processing entire dataset in a single pass...")
        logger.info(
            "Running spatiotemporal clustering (e_dist=%skm, e_days=%s days)...",
            args.e_dist,
            args.e_days,
        )

        pipeline = create_pipeline(e_dist=args.e_dist, e_days=args.e_days)
        clustered = process_batch(records_df, pipeline, logger)

        # Report results
        num_clusters = clustered["spatiotemporal_cluster_id"].nunique()
        avg_size = len(clustered) / num_clusters
        cluster_sizes = clustered.groupby("spatiotemporal_cluster_id").size()

        logger.info("=" * 60)
        logger.info("Clustering Results:")
        logger.info("  Total specimens: %d", len(clustered))
        logger.info("  Total expeditions (clusters): %d", num_clusters)
        logger.info("  Average expedition size: %.2f specimens", avg_size)
        logger.info("  Median expedition size: %.0f specimens", cluster_sizes.median())
        logger.info("  Largest expedition: %d specimens", cluster_sizes.max())
        logger.info("  Smallest expedition: %d specimens", cluster_sizes.min())
        logger.info("=" * 60)

        # Save results
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        clustered.to_csv(output_path, index=False)
        logger.info("Results saved to: %s", output_path)

        # Show sample
        logger.info("\nSample of results (first 10 rows):")
        sample_cols = [
            "collectingeventid",
            "collectionobjectid",
            "latitude1",
            "longitude1",
            "startdate",
            "spatiotemporal_cluster_id",
        ]
        logger.info("\n%s", clustered[sample_cols].head(10).to_string(index=False))

        logger.info("\n✓ Clustering completed successfully!")

    except pymysql.Error:
        logger.exception("Database error. Make sure the database is running: docker-compose up -d")
        sys.exit(1)

    except MemoryError:
        logger.exception("Out of memory! Try one of these solutions:")
        logger.error("  1. Use --limit to process fewer specimens")
        logger.error("  2. Add geographic/temporal filters to the query")
        logger.error("  3. Use a machine with more RAM")
        sys.exit(1)

    except Exception:
        logger.exception("Error during clustering run")
        sys.exit(1)


if __name__ == "__main__":
    main()
