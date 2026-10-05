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
    # The month column is what the components layout adds. Looking at the
    # first value instead (bug #427) misreads an ISO file whose first date is
    # year-only ("1911") as components, and the herp converter now writes the
    # components layout, whose year column holds no dash either.
    if "ce_startDate" in df.columns:
        return "components"
    if date_col in df.columns:
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

    # The specimen's identity is its spid, the qualified catalog identifier
    # ("urn:catalog:CAS:SUR:6788"). A bare catalog number is not unique inside
    # one PortalData.csv: herp holds the HERP, SUA and SUR catalogs and ich two
    # catalogs whose numbers overlap, and keying on the number made the
    # Preprocessor drop every later row sharing a number as a duplicate
    # (bug #426: 46,169 herp rows, 69,708 ich rows on 2026-10-04). The Lens
    # maps clustered rows back to specimens by spid too, so a row without one
    # cannot be placed and is refused rather than given a made-up key.
    if "spid" not in df.columns:
        msg = "PortalData.csv has no spid column: spid is the specimen key for clustering (EXP-CLUSTER-ID-1)"
        raise ValueError(msg)
    spids = df["spid"].fillna("").astype(str).str.strip()
    blank = int((spids == "").sum())
    if blank:
        msg = f"{blank} PortalData.csv row(s) have no spid; spid is the specimen key for clustering (EXP-CLUSTER-ID-1)"
        raise ValueError(msg)

    # collectionobjectid is carried for reference only (the numeric catalog
    # number); nothing joins or deduplicates on it.
    if "catalogNumber" in df.columns:
        result["collectionobjectid"] = pd.to_numeric(df["catalogNumber"], errors="coerce")
        # Fill any NaN with a unique negative value based on index
        mask = result["collectionobjectid"].isna()
        result.loc[mask, "collectionobjectid"] = -1 * (df.index[mask] + 1)
        result["collectionobjectid"] = result["collectionobjectid"].astype(int)
    else:
        result["collectionobjectid"] = df.index + 1

    result["spid"] = spids

    # Each specimen is its own collecting event, keyed by its spid; the
    # Preprocessor's duplicate drop therefore only removes a specimen listed
    # twice.
    result["collectingeventid"] = spids

    # Detect date format and build datetime accordingly
    date_format = _detect_date_format(df, "startDate")
    if logger:
        logger.info("  Date format detected: %s", date_format)

    # date_precision (bug #427, EXP-DATE-2): 1 = day, 2 = month, 3 = year,
    # Specify's scale. A date known only to the month or the year is padded to
    # the first day (the sort key the clustering needs), and the precision
    # records that the padding happened so the pipeline never treats it as
    # that day.
    if date_format == "iso":
        # ISO strings "YYYY-MM-DD", "YYYY-MM" or "YYYY" (antweb)
        result["startdate"], result["date_precision"] = _parse_iso_with_precision(df.get("startDate"), df.index)
        result["enddate"], _ = _parse_iso_with_precision(df.get("endDate"), df.index)
    else:
        # Build datetime from year/month/day components (botany, ich, iz, herp collections)
        result["startdate"] = _build_datetime(df, "startDate", "ce_startDate", "ce_startDate1")
        result["enddate"] = _build_datetime(df, "endDate", "ce_endDate", "ce_endDate1")
        result["date_precision"] = _components_precision(df, "startDate", "ce_startDate", "ce_startDate1")
    # A row without a usable start date has no precision either.
    result.loc[result["startdate"].isna(), "date_precision"] = pd.NA

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
    years = pd.to_numeric(df[year_col], errors="coerce") if year_col in df.columns else pd.Series(pd.NA, index=df.index)

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


_ISO_DAY = r"^\d{4}-\d{2}-\d{2}$"
_ISO_MONTH = r"^\d{4}-\d{2}$"
_ISO_YEAR = r"^\d{4}$"


def _parse_iso_with_precision(values: pd.Series | None, index: pd.Index) -> tuple[pd.Series, pd.Series]:
    """
    Parse ISO date strings that may stop at the month or the year.

    Inputs:  a column of strings ("2001-03-04", "1911-04", "1911") or None.
    Output:  (dates, precisions): the date padded to the first day of what the
             string names, and 1 / 2 / 3 for day / month / year. A value of
             any other shape gives NaT and <NA>.
    """
    if values is None:
        return pd.Series(pd.NaT, index=index, dtype="datetime64[ns]"), pd.Series(pd.NA, index=index, dtype="Int64")
    text = values.fillna("").astype(str).str.strip()
    precision = pd.Series(pd.NA, index=index, dtype="Int64")
    padded = pd.Series("", index=index, dtype=object)
    for pattern, level, suffix in ((_ISO_DAY, 1, ""), (_ISO_MONTH, 2, "-01"), (_ISO_YEAR, 3, "-01-01")):
        mask = text.str.match(pattern)
        precision[mask] = level
        padded[mask] = text[mask] + suffix
    dates = pd.to_datetime(padded, format="%Y-%m-%d", errors="coerce")
    precision[dates.isna()] = pd.NA
    return dates, precision


def _components_precision(df: pd.DataFrame, year_col: str, month_col: str, day_col: str) -> pd.Series:
    """
    Date precision from separate year / month / day columns.

    Inputs:  the raw frame and the three column names.
    Output:  Int64 series: 1 when the day is known, 2 when only the month is,
             3 when only the year is, <NA> without a year. The same rule as the
             Lens loader (portal.etl.date_utils.precision_from_components).
    """
    def numeric(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(float("nan"), index=df.index)
        return pd.to_numeric(df[col], errors="coerce")

    years, months, days = numeric(year_col), numeric(month_col), numeric(day_col)
    precision = pd.Series(1, index=df.index, dtype="Int64")
    precision[days.isna()] = 2
    precision[months.isna()] = 3
    precision[years.isna() | (years <= 0)] = pd.NA
    return precision


def _build_fullname(df: pd.DataFrame) -> pd.Series:
    """Build full geographic name from hierarchy columns."""
    parts = [df[col].fillna("") for col in ["Continent", "Country", "State", "County"] if col in df.columns]

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
    return result.str.replace(r",\s*$", "", regex=True)


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
    return sorted(
        item.name for item in incoming_data_path.iterdir() if item.is_dir() and (item / "PortalData.csv").exists()
    )
