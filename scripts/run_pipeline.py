#!/usr/bin/env python3
"""
Command-line entry point for the expedition clustering pipeline.

This mirrors the workflow described in the notebooks and README:
1. Connect to the CAS Botany MySQL dump (running via docker-compose).
2. Load the core tables and build the cleaned dataframe.
3. Fit the spatiotemporal DBSCAN pipeline and persist the augmented dataframe.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from expedition_clustering import (
    DatabaseConfig,
    build_clean_dataframe,
    create_pipeline,
    load_core_tables,
)

DEFAULT_TABLES = ("collectingevent", "collectionobject", "locality", "geography")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the expedition clustering pipeline.")
    parser.add_argument("--host", default="localhost", help="MySQL host (default: localhost)")
    parser.add_argument("--port", type=int, default=3306, help="MySQL port (default: 3306)")
    parser.add_argument("--user", default="myuser", help="MySQL user (default: myuser)")
    parser.add_argument("--password", default="mypassword", help="MySQL password")
    parser.add_argument(
        "--database",
        default="exped_cluster_db",
        help="Database name (default: exped_cluster_db)",
    )
    parser.add_argument(
        "--e-dist",
        type=float,
        default=10.0,
        help="Spatial DBSCAN epsilon in kilometers (default: 10.0)",
    )
    parser.add_argument(
        "--e-days",
        type=float,
        default=7.0,
        help="Temporal DBSCAN epsilon in days (default: 7.0)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional number of rows to randomly sample (after cleaning) before clustering.",
    )
    parser.add_argument(
        "--table-limit",
        type=int,
        default=None,
        help="Limit the number of rows fetched from each source table to reduce memory usage.",
    )
    parser.add_argument(
        "--tables",
        type=str,
        default=",".join(DEFAULT_TABLES),
        help="Comma-separated list of tables to load (default: collectingevent,collectionobject,locality,geography).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/clustered_output.csv"),
        help="Where to write the clustered dataframe (default: data/clustered_output.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling (only when --sample is set).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, etc.).",
    )
    parser.add_argument(
        "--fetch-related-only",
        dest="fetch_related_only",
        action="store_true",
        default=True,
        help="Fetch only rows related to the limited collecting events (default: on).",
    )
    parser.add_argument(
        "--no-fetch-related-only",
        dest="fetch_related_only",
        action="store_false",
        help="Disable related-only fetching.",
    )
    parser.add_argument(
        "--related-chunk-size",
        type=int,
        default=2000,
        help="Chunk size for related-only SQL fetches (default: 2000).",
    )
    parser.add_argument(
        "--no-filter-related",
        action="store_true",
        help="Disable filtering collection objects/localities/geographies to only those referenced by the loaded collecting events.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DatabaseConfig:
    return DatabaseConfig(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
    )


def maybe_sample(df: pd.DataFrame, sample_size: Optional[int], seed: int) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(df):
        return df
    return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("expedition_clustering.runner")
    run_start = time.perf_counter()

    config = build_config(args)
    table_list = [
        table.strip() for table in args.tables.split(",") if table.strip()
    ] or list(DEFAULT_TABLES)
    logger.info(
        "Loading tables: %s%s",
        ", ".join(table_list),
        f" (limit={args.table_limit})" if args.table_limit else "",
    )
    tables = load_core_tables(
        config,
        tables=table_list,
        limit=args.table_limit,
        logger=logger,
        related_only=args.fetch_related_only,
        related_chunk_size=args.related_chunk_size,
    )
    logger.info("Building clean dataframe...")
    clean_df = build_clean_dataframe(
        tables,
        filter_related=not args.no_filter_related,
        logger=logger,
    )
    logger.info("Clean dataframe rows: %s", len(clean_df))

    clean_df = maybe_sample(clean_df, args.sample, args.seed)
    logger.info("Rows passed to clustering: %s", len(clean_df))

    logger.info("Creating pipeline (e_dist=%s, e_days=%s)...", args.e_dist, args.e_days)
    pipeline = create_pipeline(e_dist=args.e_dist, e_days=args.e_days)
    clustered = pipeline.fit_transform(clean_df)

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clustered.to_csv(output_path, index=False)
    logger.info("Clustered dataframe written to %s", output_path)
    logger.info("Pipeline completed in %.2fs", time.perf_counter() - run_start)


if __name__ == "__main__":
    main()
