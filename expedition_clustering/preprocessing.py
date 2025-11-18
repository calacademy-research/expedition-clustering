"""
Data wrangling helpers extracted from the exploratory notebooks.

The functions below mirror the `0_table_eda.ipynb` workflow that:
1. Loads the CollectingEvent parent table.
2. Joins CollectionObject details and the Locality/Geography lookups.
3. Drops rows without spatial or temporal coverage.
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

COLLECTING_EVENT_COLUMNS: Sequence[str] = [
    "CollectingEventID",
    "StartDate",
    "EndDate",
    "Remarks",
    "LocalityID",
]

COLLECTION_OBJECT_COLUMNS: Sequence[str] = [
    "CollectionObjectID",
    "CollectingEventID",
    "Text1",
]

LOCALITY_COLUMNS: Sequence[str] = [
    "LocalityID",
    "MinElevation",
    "MaxElevation",
    "ElevationAccuracy",
    "Latitude1",
    "Longitude1",
    "LocalityName",
    "NamedPlace",
    "GeographyID",
]

GEOGRAPHY_COLUMNS: Sequence[str] = [
    "GeographyID",
    "CentroidLat",
    "CentroidLon",
    "CommonName",
    "FullName",
    "Name",
]


def merge_core_tables(
    collecting_event_df: pd.DataFrame,
    collection_object_df: pd.DataFrame,
    locality_df: pd.DataFrame,
    geography_df: pd.DataFrame,
    *,
    filter_related: bool = True,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Reproduce the chained merge logic from `0_table_eda.ipynb`.

    This keeps only the columns that proved useful in downstream notebooks,
    making the resulting DataFrame lean enough for re-use in scripts/tests.
    """

    event_df = collecting_event_df.reindex(columns=COLLECTING_EVENT_COLUMNS).copy()
    object_df = collection_object_df.reindex(columns=COLLECTION_OBJECT_COLUMNS).copy()
    locality_df = locality_df.reindex(columns=LOCALITY_COLUMNS).copy()
    geography_df = geography_df.reindex(columns=GEOGRAPHY_COLUMNS).copy()

    for df, column in (
        (event_df, "CollectingEventID"),
        (object_df, "CollectingEventID"),
        (event_df, "LocalityID"),
        (locality_df, "LocalityID"),
        (locality_df, "GeographyID"),
        (geography_df, "GeographyID"),
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if filter_related:
        event_ids = event_df["CollectingEventID"].dropna().unique()
        object_df = object_df[object_df["CollectingEventID"].isin(event_ids)]
        locality_ids = event_df["LocalityID"].dropna().unique()
        locality_df = locality_df[locality_df["LocalityID"].isin(locality_ids)]
        geography_ids = locality_df["GeographyID"].dropna().unique()
        geography_df = geography_df[geography_df["GeographyID"].isin(geography_ids)]
        if logger:
            logger.debug(
                "Filtered tables -> events=%s, objects=%s, localities=%s, geographies=%s",
                len(event_df),
                len(object_df),
                len(locality_df),
                len(geography_df),
            )

    full_df = event_df.merge(
        object_df,
        on="CollectingEventID",
        how="inner",
    )

    full_df = full_df.merge(
        locality_df[LOCALITY_COLUMNS], on="LocalityID", how="left"
    )
    full_df = full_df.merge(
        geography_df[GEOGRAPHY_COLUMNS], on="GeographyID", how="left"
    )
    if logger:
        logger.debug("Merged dataframe rows=%s", len(full_df))
    return full_df


def clean_for_clustering(
    merged_df: pd.DataFrame,
    drop_missing_spatial: bool = True,
    drop_missing_start_date: bool = True,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Apply the spatial/temporal filters that produced `clean_df` in EDA.

    Parameters
    ----------
    merged_df:
        Output of :func:`merge_core_tables`.
    drop_missing_spatial:
        If True (default), drop rows missing both `Latitude1` and `CentroidLat`.
    drop_missing_start_date:
        If True (default), drop rows missing `StartDate`.
    """

    df = merged_df.copy()
    for column in ("StartDate", "EndDate"):
        df[column] = pd.to_datetime(df[column], errors="coerce")

    if drop_missing_spatial:
        df = df[~df[["Latitude1", "CentroidLat"]].isna().all(axis=1)]

    if drop_missing_start_date:
        df = df[df["StartDate"].notna()]

    df = df.reset_index(drop=True)
    df["spatial_flag"] = np.where(df["Latitude1"].notna(), 1, 0)
    if logger:
        logger.debug("Clean dataframe rows=%s (spatial flag counts=%s)", len(df), df["spatial_flag"].value_counts(dropna=False).to_dict())
    return df


def build_clean_dataframe(
    tables: Mapping[str, pd.DataFrame],
    drop_missing_spatial: bool = True,
    drop_missing_start_date: bool = True,
    filter_related: bool = True,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper that expects a dictionary returned by `load_core_tables`.

    Example
    -------
    ```python
    config = DatabaseConfig()
    tables = load_core_tables(config)
    clean_df = build_clean_dataframe(tables)
    ```
    """

    required_keys = {
        "collectingevent",
        "collectionobject",
        "locality",
        "geography",
    }
    missing = required_keys - tables.keys()
    if missing:
        raise KeyError(f"Missing required tables: {', '.join(sorted(missing))}")

    merged_df = merge_core_tables(
        tables["collectingevent"],
        tables["collectionobject"],
        tables["locality"],
        tables["geography"],
        filter_related=filter_related,
        logger=logger,
    )

    return clean_for_clustering(
        merged_df,
        drop_missing_spatial=drop_missing_spatial,
        drop_missing_start_date=drop_missing_start_date,
        logger=logger,
    )
