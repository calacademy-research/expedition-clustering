#!/usr/bin/env python3
"""
Quick test script to verify the pipeline works end-to-end.
Run with: python test_pipeline.py
"""

import logging
from expedition_clustering import (
    DatabaseConfig,
    build_clean_dataframe,
    create_pipeline,
    load_core_tables,
)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting pipeline test...")

    # Configure database
    config = DatabaseConfig(
        host="localhost",
        user="myuser",
        password="mypassword",
        database="exped_cluster_db",
        port=3306,
    )

    # Load tables with a small limit for testing
    # Use related_only to ensure we get matching rows
    logger.info("Loading tables (limit=10000 for testing, with related_only=True)...")
    tables = load_core_tables(
        config,
        tables=["collectingevent", "collectionobject", "locality", "geography"],
        limit=10000,
        logger=logger,
        related_only=True,
        primary_table="collectingevent",  # Start from collecting events
    )

    # Build clean dataframe
    logger.info("Building clean dataframe...")
    logger.info(f"Tables loaded: {list(tables.keys())}")
    for name, df in tables.items():
        logger.info(f"  {name}: {len(df)} rows, columns: {list(df.columns)[:5]}...")

    clean_df = build_clean_dataframe(tables, filter_related=False, logger=logger)
    logger.info(f"Clean dataframe shape: {clean_df.shape}")
    logger.info(f"Clean dataframe columns: {list(clean_df.columns)}")

    if clean_df.empty:
        logger.error("Clean dataframe is empty! Cannot proceed.")
        return

    # Create and run pipeline
    logger.info("Creating clustering pipeline (e_dist=10km, e_days=7)...")
    pipeline = create_pipeline(e_dist=10, e_days=7)

    logger.info("Running clustering pipeline...")
    clustered_df = pipeline.fit_transform(clean_df)

    logger.info(f"Clustered dataframe shape: {clustered_df.shape}")
    logger.info(f"Number of unique clusters: {clustered_df['spatiotemporal_cluster_id'].nunique()}")
    logger.info(f"Sample of results:\n{clustered_df[['collectingeventid', 'latitude1', 'longitude1', 'startdate', 'spatial_cluster_id', 'temporal_cluster_id', 'spatiotemporal_cluster_id']].head(10)}")

    logger.info("✓ Pipeline test completed successfully!")

if __name__ == "__main__":
    main()
