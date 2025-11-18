"""
Utilities for retrieving source tables from the CAS Botany MySQL dump.

The exploratory notebooks (`notebooks/0_table_eda.ipynb` and
`notebooks/1_manual_cluster_labeling.ipynb`) relied on ad-hoc `pd.read_sql`
calls.  This module promotes that workflow into reusable helpers that can be
shared across scripts, notebooks, and the command-line.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence

import pandas as pd
import pymysql


@dataclass
class DatabaseConfig:
    """
    Lightweight container describing how to connect to the expedition database.

    The defaults match the docker-compose configuration committed in this repo,
    so creating an instance without arguments is enough for local development.
    """

    host: str = "localhost"
    user: str = "myuser"
    password: str = "mypassword"
    database: str = "exped_cluster_db"
    port: int = 3306
    charset: str = "utf8mb4"

    def connect(self) -> pymysql.connections.Connection:
        """Open a new connection using the stored credentials."""
        return pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
            port=self.port,
            charset=self.charset,
            cursorclass=pymysql.cursors.DictCursor,
        )


@contextmanager
def db_connection(config: DatabaseConfig) -> Iterator[pymysql.connections.Connection]:
    """Context manager that ensures connections are closed."""
    connection = config.connect()
    try:
        yield connection
    finally:
        connection.close()


def fetch_table(
    config: DatabaseConfig,
    table_name: str,
    columns: Iterable[str] | str = "*",
    where: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch a table (or subset of columns) into a pandas DataFrame.

    Parameters
    ----------
    config:
        Database connection configuration.
    table_name:
        Name of the table to query.
    columns:
        Iterable of column names or `"*"` for all columns.
    where:
        Optional SQL WHERE clause (without the `WHERE` keyword).
    limit:
        Optional LIMIT clause to restrict the number of returned rows.
    """

    if isinstance(columns, str):
        column_expr = columns
    else:
        column_expr = ", ".join(columns)

    query = f"SELECT {column_expr} FROM {table_name}"
    if where:
        query += f" WHERE {where}"
    if limit:
        query += f" LIMIT {limit}"

    with db_connection(config) as conn:
        return pd.read_sql_query(query, conn)


def fetch_table_by_ids(
    config: DatabaseConfig,
    table_name: str,
    id_column: str,
    ids: Sequence,
    columns: Iterable[str] | str = "*",
    chunk_size: int = 1000,
) -> pd.DataFrame:
    """
    Fetch rows whose primary/foreign keys are contained within ``ids``.
    """

    clean_ids = [i for i in ids if pd.notna(i)]
    if not clean_ids:
        return pd.DataFrame()

    frames = []
    with db_connection(config) as conn:
        column_expr = columns if isinstance(columns, str) else ", ".join(columns)
        for start in range(0, len(clean_ids), chunk_size):
            chunk = clean_ids[start : start + chunk_size]
            placeholders = ", ".join(["%s"] * len(chunk))
            query = (
                f"SELECT {column_expr} FROM {table_name} "
                f"WHERE {id_column} IN ({placeholders})"
            )
            frames.append(pd.read_sql_query(query, conn, params=chunk))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


DEFAULT_TABLES = (
    "collectingevent",
    "collectingeventattribute",
    "collectionobject",
    "collectionobjectattachment",
    "attachment",
    "locality",
    "localitydetail",
    "geography",
    "geocoorddetail",
)


def load_core_tables(
    config: DatabaseConfig,
    tables: Iterable[str] = DEFAULT_TABLES,
    limit: int | None = None,
    table_limits: Mapping[str, int] | None = None,
    logger: Optional[logging.Logger] = None,
    related_only: bool = False,
    related_chunk_size: int = 2000,
    primary_table: str = "collectionobject",
) -> Dict[str, pd.DataFrame]:
    """
    Load the tables explored in the notebooks into memory.

    Returns a mapping keyed by table name, which can then be fed directly into
    the preprocessing helpers when building the clean dataset.

    Parameters
    ----------
    limit:
        Optional global LIMIT applied to each table.
    table_limits:
        Optional per-table overrides. When provided, this dictionary takes
        precedence over the global ``limit`` for matching table names.
    """

    table_list = list(tables)
    data: Dict[str, pd.DataFrame] = {}

    def fetch_with_logging(
        table_name: str, table_limit: Optional[int], where: Optional[str] = None
    ) -> pd.DataFrame:
        if logger:
            msg = f"Fetching {table_name}"
            if table_limit:
                msg += f" (limit={table_limit})"
            logger.info("%s...", msg)
        start = time.perf_counter()
        df = fetch_table(config, table_name, limit=table_limit, where=where)
        if logger:
            logger.info(
                "Finished %s in %.2fs (rows=%s)",
                table_name,
                time.perf_counter() - start,
                len(df),
            )
        return df

    def related_fetch(table_name: str, id_column: str, ids: Sequence) -> pd.DataFrame:
        if logger:
            logger.info(
                "Fetching %s rows for %s %s IDs...",
                table_name,
                len(ids),
                id_column,
            )
        start = time.perf_counter()
        df = fetch_table_by_ids(
            config,
            table_name,
            id_column,
            ids,
            chunk_size=related_chunk_size,
        )
        if logger:
            logger.info(
                "Finished %s in %.2fs (rows=%s)",
                table_name,
                time.perf_counter() - start,
                len(df),
            )
        if df.empty:
            if logger:
                logger.warning(
                    "%s returned 0 related rows; falling back to limited fetch.",
                    table_name,
                )
            return fetch_with_logging(
                table_name, table_limits.get(table_name) if table_limits else limit
            )
        return df

    if related_only:
        primary_table = primary_table.lower()
        if primary_table not in {"collectionobject", "collectingevent"}:
            raise ValueError("primary_table must be 'collectionobject' or 'collectingevent'")
        if "collectingevent" not in table_list:
            raise ValueError("collectingevent must be included when related_only=True")
        if primary_table == "collectionobject":
            object_limit = table_limits.get("collectionobject") if table_limits else limit
            objects_df = fetch_with_logging(
                "collectionobject", object_limit, where="CollectingEventID IS NOT NULL"
            )
            if objects_df.empty:
                raise ValueError(
                    "No collectionobject rows with CollectingEventID were retrieved. "
                    "Increase --table-limit or disable related-only fetching."
                )
            data["collectionobject"] = objects_df
            event_ids = objects_df["CollectingEventID"].dropna().unique().tolist()
            locality_ids: list = []
            if "collectingevent" in table_list:
                event_df = related_fetch("collectingevent", "CollectingEventID", event_ids)
                data["collectingevent"] = event_df
                locality_ids = event_df["LocalityID"].dropna().unique().tolist()
            if "locality" in table_list:
                locality_df = related_fetch("locality", "LocalityID", locality_ids)
                data["locality"] = locality_df
                geography_ids = locality_df["GeographyID"].dropna().unique().tolist()
            else:
                geography_ids = []
            if "geography" in table_list:
                data["geography"] = related_fetch("geography", "GeographyID", geography_ids)
        else:
            event_limit = table_limits.get("collectingevent") if table_limits else limit
            events_df = fetch_with_logging("collectingevent", event_limit)
            data["collectingevent"] = events_df

            event_ids = events_df["CollectingEventID"].dropna().unique().tolist()
            locality_ids = events_df["LocalityID"].dropna().unique().tolist()

            if "collectionobject" in table_list:
                data["collectionobject"] = related_fetch(
                    "collectionobject", "CollectingEventID", event_ids
                )
            if "locality" in table_list:
                data["locality"] = related_fetch("locality", "LocalityID", locality_ids)
            if "geography" in table_list:
                geography_ids = (
                    data["locality"]["GeographyID"].dropna().unique().tolist()
                    if "locality" in data
                    else []
                )
                data["geography"] = related_fetch("geography", "GeographyID", geography_ids)

        for table in table_list:
            if table in data:
                continue
            data[table] = fetch_with_logging(
                table, table_limits.get(table) if table_limits else limit
            )
        return data

    for table in table_list:
        data[table] = fetch_with_logging(
            table, table_limits.get(table) if table_limits else limit
        )
    return data
