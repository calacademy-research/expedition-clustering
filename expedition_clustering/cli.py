#!/usr/bin/env python3
"""
Command-line tool to run expedition clustering on the CAS Botany database.

Features:
- Full-dataset spatiotemporal clustering
- Progress tracking

Usage:
    expedition-cluster --output clustered_data.csv
    expedition-cluster --e-dist 15 --e-days 10
    expedition-cluster verify-redaction --input data/clustered_expeditions_redacted.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import pymysql

from expedition_clustering import create_pipeline
from expedition_clustering.csv_source import list_available_collections, load_collection_csv
from expedition_clustering.data import DatabaseConfig
from expedition_clustering.redaction import (
    DEFAULT_COORDINATE_COLUMNS,
    DEFAULT_LOCALITY_TEXT_COLUMNS,
    fetch_redaction_flags,
    redact_clustered_dataframe,
    verify_redacted_csv,
    verify_redacted_csv_drop,
)

# Default path to incoming_data directory
DEFAULT_INCOMING_DATA_DIR = Path("/Users/joe/collections_explorer/incoming_data")


def process_batch(df_batch, pipeline, logger):
    """Process a single batch of specimens through the clustering pipeline."""
    logger.info("Processing %s specimens...", len(df_batch))

    try:
        return pipeline.fit_transform(df_batch)
    except MemoryError:
        logger.exception("Out of memory processing dataset. Consider reducing --limit.")
        raise


def _add_db_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="localhost", help="MySQL host (default: localhost)")
    parser.add_argument("--port", type=int, default=3306, help="MySQL port (default: 3306)")
    parser.add_argument("--user", default="myuser", help="MySQL user (default: myuser)")
    parser.add_argument("--password", default="mypassword", help="MySQL password")
    parser.add_argument(
        "--database",
        default="exped_cluster_db",
        help="Database name (default: exped_cluster_db)",
    )


def _add_log_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )


def _add_csv_args(parser: argparse.ArgumentParser) -> None:
    """Add CSV data source arguments."""
    csv_group = parser.add_argument_group("CSV data source (alternative to database)")
    csv_group.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to a CSV file or collection directory (e.g., incoming_data/botany)",
    )
    csv_group.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection name to load from incoming_data directory (e.g., botany, ich, iz)",
    )
    csv_group.add_argument(
        "--incoming-data-dir",
        type=Path,
        default=DEFAULT_INCOMING_DATA_DIR,
        help=f"Path to incoming_data directory (default: {DEFAULT_INCOMING_DATA_DIR})",
    )
    csv_group.add_argument(
        "--list-collections",
        action="store_true",
        help="List available collections and exit",
    )


def _add_cluster_args(parser: argparse.ArgumentParser) -> None:
    _add_db_args(parser)
    _add_csv_args(parser)
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
    redaction_group = parser.add_mutually_exclusive_group()
    redaction_group.add_argument(
        "--redact",
        action="store_true",
        help="Apply IPT redaction rules (mask locality text and remove coordinates) before writing output",
    )
    redaction_group.add_argument(
        "--drop-redacted",
        action="store_true",
        help="Drop records flagged for redaction instead of masking fields",
    )
    _add_log_args(parser)


def _add_verify_args(parser: argparse.ArgumentParser) -> None:
    _add_db_args(parser)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Redacted clustered CSV file to verify",
    )
    parser.add_argument(
        "--expect-dropped",
        action="store_true",
        help="Expect redacted records to be dropped rather than masked",
    )
    parser.add_argument(
        "--id-column",
        default="collectionobjectid",
        help="Column name used to join redaction flags (default: collectionobjectid)",
    )
    parser.add_argument(
        "--redacted-placeholder",
        default="*",
        help="Placeholder used for redacted locality text (default: *)",
    )
    parser.add_argument(
        "--max-offenders",
        type=int,
        default=10,
        help="Max offender rows to display when verification fails (default: 10)",
    )
    _add_log_args(parser)


def _inject_default_command(argv: list[str]) -> list[str]:
    if not argv or argv[0].startswith("-"):
        return ["cluster", *argv]
    return argv


def _load_from_csv(args: argparse.Namespace, logger: logging.Logger) -> pd.DataFrame:
    """Load data from CSV source."""
    # Determine CSV path
    if args.csv:
        csv_path = args.csv
        logger.info("Loading data from CSV: %s", csv_path)
    elif args.collection:
        csv_path = args.incoming_data_dir / args.collection
        logger.info("Loading collection '%s' from %s", args.collection, args.incoming_data_dir)
    else:
        raise ValueError("Must specify --csv or --collection for CSV data source")

    # Load and transform CSV data
    records_df = load_collection_csv(
        csv_path,
        limit=args.limit,
        include_centroids=args.include_centroids,
        logger=logger,
    )

    return records_df


def _load_from_database(args: argparse.Namespace, logger: logging.Logger) -> pd.DataFrame:
    """Load data from MySQL database."""
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
        logger.info("Including specimens with geography centroids (may contain inaccurate data)")
        coord_filter = """
      AND (l.Latitude1 IS NOT NULL OR g.CentroidLat IS NOT NULL)
      AND (l.Longitude1 IS NOT NULL OR g.CentroidLon IS NOT NULL)"""
    else:
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

    records_df = pd.read_sql_query(query, conn)
    conn.close()

    logger.info("Loaded %d rows", len(records_df))

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

    return records_df


def run_cluster(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)

    # Handle --list-collections
    if getattr(args, "list_collections", False):
        collections = list_available_collections(args.incoming_data_dir)
        if collections:
            print("Available collections:")
            for name in collections:
                print(f"  {name}")
        else:
            print(f"No collections found in {args.incoming_data_dir}")
        return

    try:
        # Determine data source: CSV or database
        use_csv = args.csv is not None or args.collection is not None

        if use_csv:
            records_df = _load_from_csv(args, logger)
        else:
            records_df = _load_from_database(args, logger)

        if records_df.empty:
            logger.error("No data loaded!")
            sys.exit(1)

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

        if args.redact or args.drop_redacted:
            if use_csv:
                logger.warning("Redaction from CSV uses embedded flags only (no database lookup)")
                # For CSV, we can only use the redaction flags in the CSV itself
                # Check if redaction columns exist
                redact_cols = ["yesno2", "co_yesno2", "tx_yesno2"]
                has_redact_flags = any(col in clustered.columns for col in redact_cols)
                if has_redact_flags:
                    # Build is_redacted flag from CSV columns
                    is_redacted = pd.Series(False, index=clustered.index)
                    for col in redact_cols:
                        if col in clustered.columns:
                            is_redacted |= clustered[col].fillna(0).astype(bool)
                    flags = clustered[["collectionobjectid"]].copy()
                    flags["is_redacted"] = is_redacted.astype(int)
                else:
                    logger.warning("No redaction flags found in CSV data, skipping redaction")
                    flags = None
            else:
                if args.drop_redacted:
                    logger.info("Dropping redacted rows using IPT flags...")
                else:
                    logger.info("Applying IPT redaction rules to clustered output...")
                config = DatabaseConfig(
                    host=args.host,
                    user=args.user,
                    password=args.password,
                    database=args.database,
                    port=args.port,
                )
                flags = fetch_redaction_flags(
                    config,
                    clustered["collectionobjectid"].dropna().unique().tolist(),
                    logger=logger,
                )

            if flags is not None:
                total_before_redaction = len(clustered)
                clustered, redacted_rows = redact_clustered_dataframe(
                    clustered,
                    flags,
                    drop_redacted=args.drop_redacted,
                )
                percent = (redacted_rows / total_before_redaction * 100) if total_before_redaction else 0
                if args.drop_redacted:
                    logger.info("Dropped %d redacted rows (%.2f%% of clustered rows).", redacted_rows, percent)
                else:
                    logger.info("Redacted %d rows (%.2f%% of clustered rows).", redacted_rows, percent)

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

    except FileNotFoundError as e:
        logger.exception("File not found: %s", e)
        sys.exit(1)

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


def run_verify_redaction(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)
    try:
        config = DatabaseConfig(
            host=args.host,
            user=args.user,
            password=args.password,
            database=args.database,
            port=args.port,
        )
        if args.expect_dropped:
            result = verify_redacted_csv_drop(
                args.input,
                config=config,
                id_column=args.id_column,
                logger=logger,
            )
            logger.info("Flagged rows present: %d", result.flagged_rows)

            if not result.ok:
                logger.error("Redaction drop verification failed.")
                if args.max_offenders > 0 and not result.offenders.empty:
                    show_columns = [args.id_column]
                    show_columns.extend(
                        [
                            column
                            for column in (*DEFAULT_LOCALITY_TEXT_COLUMNS, *DEFAULT_COORDINATE_COLUMNS)
                            if column in result.offenders.columns
                        ]
                    )
                    logger.error(
                        "Sample offenders (up to %d rows):\n%s",
                        args.max_offenders,
                        result.offenders[show_columns].head(args.max_offenders).to_string(index=False),
                    )
                sys.exit(2)

            logger.info("\n✓ Redaction drop verification passed.")
        else:
            result = verify_redacted_csv(
                args.input,
                config=config,
                id_column=args.id_column,
                redacted_placeholder=args.redacted_placeholder,
                logger=logger,
            )
            logger.info("Flagged rows: %d", result.flagged_rows)
            logger.info("Bad locality rows: %d", result.bad_locality_rows)
            logger.info("Bad coordinate rows: %d", result.bad_coordinate_rows)

            if not result.ok:
                logger.error("Redaction verification failed.")
                if args.max_offenders > 0 and not result.offenders.empty:
                    show_columns = [args.id_column]
                    show_columns.extend(
                        [
                            column
                            for column in (*DEFAULT_LOCALITY_TEXT_COLUMNS, *DEFAULT_COORDINATE_COLUMNS)
                            if column in result.offenders.columns
                        ]
                    )
                    logger.error(
                        "Sample offenders (up to %d rows):\n%s",
                        args.max_offenders,
                        result.offenders[show_columns].head(args.max_offenders).to_string(index=False),
                    )
                sys.exit(2)

            logger.info("\n✓ Redaction verification passed.")

    except pymysql.Error:
        logger.exception("Database error. Make sure the database is running: docker-compose up -d")
        sys.exit(1)

    except Exception:
        logger.exception("Error during redaction verification")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Cluster botanical specimens into expeditions using spatiotemporal DBSCAN"
    )
    subparsers = parser.add_subparsers(dest="command")

    cluster_parser = subparsers.add_parser("cluster", help="Run the clustering pipeline")
    _add_cluster_args(cluster_parser)
    cluster_parser.set_defaults(command="cluster")

    verify_parser = subparsers.add_parser("verify-redaction", help="Verify redaction on a clustered CSV")
    _add_verify_args(verify_parser)
    verify_parser.set_defaults(command="verify-redaction")

    argv = _inject_default_command(sys.argv[1:])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.command == "verify-redaction":
        run_verify_redaction(args)
    else:
        run_cluster(args)


if __name__ == "__main__":
    main()
