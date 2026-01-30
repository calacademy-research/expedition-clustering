"""
CSV data source for expedition clustering.

This module provides utilities for loading specimen data from CSV files
exported from the collections portal (incoming_data format) as an alternative
to the MySQL database source.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

# Mapping from incoming_data CSV columns to the expected pipeline schema
CSV_COLUMN_MAPPING = {
    "spid": "spid",  # Keep original for reference
    "catalogNumber": "catalognumber",
    "latitude1": "latitude1",
    "longitude1": "longitude1",
    "localityName": "localityname",
    "minElevation": "minelevation",
    "maxElevation": "maxelevation",
    "remarks": "remarks",
    "text1": "text1",
    "text2": "text2",
    "collectors": "collectors",
    "stationFieldNumber": "stationfieldnumber",
    "Continent": "continent",
    "Country": "country",
    "State": "state",
    "County": "county",
    "Town": "town",
    "Family": "family",
    "Genus": "genus",
    "Species": "species",
    "fullName": "fullname",
    "yesNo2": "yesno2",
    "co_yesNo2": "co_yesno2",
}


def load_csv_data(
    csv_path: Path | str,
    limit: int | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Load specimen data from a portal CSV file.

    Parameters
    ----------
    csv_path:
        Path to the PortalData.csv file.
    limit:
        Optional limit on number of rows to load.
    logger:
        Optional logger for status messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns mapped to the expected pipeline schema.
    """
    csv_path = Path(csv_path)

    if logger:
        logger.info("Loading CSV data from %s...", csv_path)

    # Read CSV - the portal exports use standard CSV format
    nrows = limit if limit else None
    df = pd.read_csv(csv_path, nrows=nrows, low_memory=False)

    if logger:
        logger.info("Loaded %d rows from CSV", len(df))

    return df


def _detect_date_format(df: pd.DataFrame, date_col: str) -> str:
    """
    Detect whether a date column contains ISO strings or year integers.

    Returns
    -------
    str
        'iso' if dates are ISO strings like '2007-06-20'
        'components' if dates are year integers with separate month/day columns
    """
    if date_col not in df.columns:
        return "components"

    # Get first non-null value
    sample = df[date_col].dropna().head(10)
    if sample.empty:
        return "components"

    # Check if values are strings containing dashes (ISO format)
    first_val = sample.iloc[0]
    if isinstance(first_val, str) and "-" in first_val:
        return "iso"

    return "components"


def transform_csv_to_pipeline_format(
    df: pd.DataFrame,
    include_centroids: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Transform CSV data to match the expected pipeline input format.

    The pipeline expects columns like:
    - collectingeventid, collectionobjectid
    - startdate (datetime), enddate (datetime)
    - latitude1, longitude1
    - localityname, remarks, text1
    - minelevation, maxelevation

    Handles two date formats:
    - ISO strings: '2007-06-20' in startDate column (mam, orn collections)
    - Components: year in startDate, month in ce_startDate, day in ce_startDate1 (botany, ich, iz)

    Parameters
    ----------
    df:
        Raw DataFrame from load_csv_data.
    include_centroids:
        Not used for CSV source (centroids not available), kept for API compatibility.
    logger:
        Optional logger for status messages.

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame ready for the clustering pipeline.
    """
    result = pd.DataFrame()

    # Generate synthetic IDs - use catalogNumber if available, otherwise row index
    if "catalogNumber" in df.columns:
        result["collectionobjectid"] = pd.to_numeric(df["catalogNumber"], errors="coerce")
        # Fill any NaN with a unique negative value based on index
        mask = result["collectionobjectid"].isna()
        result.loc[mask, "collectionobjectid"] = -1 * (df.index[mask] + 1)
        result["collectionobjectid"] = result["collectionobjectid"].astype(int)
    else:
        result["collectionobjectid"] = df.index + 1

    # Keep spid for reference
    if "spid" in df.columns:
        result["spid"] = df["spid"]

    # Generate synthetic collectingeventid
    # Group by collector + date + locality to create pseudo-events
    # For simplicity, use the same as collectionobjectid (each specimen = 1 event)
    result["collectingeventid"] = result["collectionobjectid"]

    # Detect date format and build datetime accordingly
    date_format = _detect_date_format(df, "startDate")
    if logger:
        logger.info("  Date format detected: %s", date_format)

    if date_format == "iso":
        # Direct parsing of ISO date strings (mam, orn collections)
        result["startdate"] = pd.to_datetime(df.get("startDate"), errors="coerce")
        result["enddate"] = pd.to_datetime(df.get("endDate"), errors="coerce")
    else:
        # Build datetime from year/month/day components (botany, ich, iz collections)
        result["startdate"] = _build_datetime(df, "startDate", "ce_startDate", "ce_startDate1")
        result["enddate"] = _build_datetime(df, "endDate", "ce_endDate", "ce_endDate1")

    # Copy coordinates
    result["latitude1"] = pd.to_numeric(df.get("latitude1"), errors="coerce")
    result["longitude1"] = pd.to_numeric(df.get("longitude1"), errors="coerce")

    # CSV doesn't have centroid fallbacks, set to NaN
    result["centroidlat"] = pd.NA
    result["centroidlon"] = pd.NA

    # Copy elevation
    result["minelevation"] = pd.to_numeric(df.get("minElevation"), errors="coerce")
    result["maxelevation"] = pd.to_numeric(df.get("maxElevation"), errors="coerce")
    result["elevationaccuracy"] = pd.NA  # Not in CSV

    # Copy text fields
    result["localityname"] = df.get("localityName", "")
    result["namedplace"] = df.get("Town", "")  # Use Town as namedplace
    result["remarks"] = df.get("remarks", "")
    result["text1"] = df.get("text1", "")

    # Geography - use the hierarchy from CSV
    result["commonname"] = df.get("County", "")
    result["fullname"] = _build_fullname(df)
    result["name"] = df.get("State", "")
    result["country"] = df.get("Country", "")
    result["continent"] = df.get("Continent", "")

    # Collectors - needed for expedition summaries (Bug #5 fix)
    result["collectors"] = df.get("collectors", "")

    # Placeholder for geographyid and localityid (not available in CSV)
    result["geographyid"] = pd.NA
    result["localityid"] = pd.NA

    # Copy redaction flags if present
    for col in ["yesNo2", "co_yesNo2", "tx_yesNo2"]:
        if col in df.columns:
            result[col.lower()] = df[col]

    if logger:
        logger.info("Transformed data: %d rows", len(result))
        valid_coords = result["latitude1"].notna() & result["longitude1"].notna()
        valid_dates = result["startdate"].notna()
        logger.info(
            "  Valid coordinates: %d (%.1f%%)",
            valid_coords.sum(),
            100 * valid_coords.sum() / len(result) if len(result) > 0 else 0,
        )
        logger.info(
            "  Valid start dates: %d (%.1f%%)",
            valid_dates.sum(),
            100 * valid_dates.sum() / len(result) if len(result) > 0 else 0,
        )

    return result


def _build_datetime(
    df: pd.DataFrame,
    year_col: str,
    month_col: str,
    day_col: str,
) -> pd.Series:
    """Build datetime from separate year/month/day columns."""
    # Handle missing columns by creating Series of NaN/default values
    if year_col in df.columns:
        years = pd.to_numeric(df[year_col], errors="coerce")
    else:
        years = pd.Series(pd.NA, index=df.index)

    if month_col in df.columns:
        months = pd.to_numeric(df[month_col], errors="coerce").fillna(1).astype(int)
    else:
        months = pd.Series(1, index=df.index)

    if day_col in df.columns:
        days = pd.to_numeric(df[day_col], errors="coerce").fillna(1).astype(int)
    else:
        days = pd.Series(1, index=df.index)

    # Clamp values to valid ranges
    months = months.clip(1, 12)
    days = days.clip(1, 31)

    # Build datetime strings and parse
    date_strings = pd.Series(index=df.index, dtype=object)
    valid_mask = years.notna() & (years > 0)

    date_strings[valid_mask] = (
        years[valid_mask].astype(int).astype(str)
        + "-"
        + months[valid_mask].astype(str).str.zfill(2)
        + "-"
        + days[valid_mask].astype(str).str.zfill(2)
    )

    return pd.to_datetime(date_strings, errors="coerce")


def _build_fullname(df: pd.DataFrame) -> pd.Series:
    """Build full geographic name from hierarchy columns."""
    parts = []
    for col in ["Continent", "Country", "State", "County"]:
        if col in df.columns:
            parts.append(df[col].fillna(""))

    if not parts:
        return pd.Series("", index=df.index)

    # Join non-empty parts with ", "
    result = parts[0]
    for part in parts[1:]:
        # Only add separator if both are non-empty
        result = result.str.cat(part, sep=", ", na_rep="")

    # Clean up multiple commas from empty values
    result = result.str.replace(r",\s*,", ",", regex=True)
    result = result.str.replace(r"^,\s*", "", regex=True)
    result = result.str.replace(r",\s*$", "", regex=True)

    return result


def load_collection_csv(
    collection_path: Path | str,
    limit: int | None = None,
    include_centroids: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Load and transform data from a collection's PortalData.csv.

    This is the main entry point for CSV-based data loading.

    Parameters
    ----------
    collection_path:
        Path to the collection directory (e.g., incoming_data/botany)
        or directly to a CSV file.
    limit:
        Optional limit on number of rows.
    include_centroids:
        Not used for CSV source, kept for API compatibility.
    logger:
        Optional logger.

    Returns
    -------
    pd.DataFrame
        DataFrame ready for the clustering pipeline.
    """
    collection_path = Path(collection_path)

    # If path is a directory, look for PortalData.csv
    if collection_path.is_dir():
        csv_path = collection_path / "PortalData.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"No PortalData.csv found in {collection_path}")
    else:
        csv_path = collection_path

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load raw data
    raw_df = load_csv_data(csv_path, limit=limit, logger=logger)

    # Transform to pipeline format
    return transform_csv_to_pipeline_format(
        raw_df,
        include_centroids=include_centroids,
        logger=logger,
    )


def list_available_collections(incoming_data_path: Path | str) -> list[str]:
    """
    List available collections in the incoming_data directory.

    Parameters
    ----------
    incoming_data_path:
        Path to the incoming_data directory.

    Returns
    -------
    list[str]
        List of collection names that have PortalData.csv files.
    """
    incoming_data_path = Path(incoming_data_path)
    collections = []

    for item in incoming_data_path.iterdir():
        if item.is_dir() and (item / "PortalData.csv").exists():
            collections.append(item.name)

    return sorted(collections)
