"""Apply IPT-style redaction rules to clustered expedition outputs."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from expedition_clustering.data import DatabaseConfig, db_connection

DEFAULT_LOCALITY_TEXT_COLUMNS: tuple[str, ...] = (
    "localityname",
    "namedplace",
    "locality",
    "municipality",
    "localityremarks",
    "verbatimlatitude",
    "verbatimlongitude",
)

DEFAULT_COORDINATE_COLUMNS: tuple[str, ...] = (
    "latitude1",
    "longitude1",
    "centroidlat",
    "centroidlon",
)


@dataclass(frozen=True)
class RedactionVerificationResult:
    flagged_rows: int
    bad_locality_rows: int
    bad_coordinate_rows: int
    offenders: pd.DataFrame

    @property
    def ok(self) -> bool:
        return self.bad_locality_rows == 0 and self.bad_coordinate_rows == 0


@dataclass(frozen=True)
class RedactionDropVerificationResult:
    flagged_rows: int
    offenders: pd.DataFrame

    @property
    def ok(self) -> bool:
        return self.flagged_rows == 0


def fetch_redaction_flags(
    config: DatabaseConfig,
    collection_object_ids: Sequence[int | float | str],
    *,
    chunk_size: int = 1000,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Fetch redaction flags for the provided CollectionObject IDs.

    Returns a DataFrame with columns: collectionobjectid, is_redacted.
    """
    clean_ids = pd.to_numeric(pd.Series(collection_object_ids), errors="coerce").dropna().unique().tolist()
    if not clean_ids:
        return pd.DataFrame(columns=["collectionobjectid", "is_redacted"])

    if logger:
        logger.info("Fetching redaction flags for %s collection objects...", len(clean_ids))

    frames: list[pd.DataFrame] = []
    with db_connection(config) as conn:
        for start in range(0, len(clean_ids), chunk_size):
            chunk = clean_ids[start : start + chunk_size]
            placeholders = ", ".join(["%s"] * len(chunk))
            query = f"""  # noqa: S608
            SELECT
                co.CollectionObjectID AS collectionobjectid,
                MAX(CASE WHEN co.YesNo2 = 1 OR vt.RedactLocality = 1 THEN 1 ELSE 0 END) AS is_redacted
            FROM collectionobject co
            LEFT JOIN determination d
                ON co.CollectionObjectID = d.CollectionObjectID AND d.IsCurrent = 1
            LEFT JOIN vtaxon2 vt
                ON d.TaxonID = vt.TaxonID
            WHERE co.CollectionObjectID IN ({placeholders})
            GROUP BY co.CollectionObjectID
            """
            with conn.cursor() as cursor:
                cursor.execute(query, chunk)
                rows = cursor.fetchall()
                if not rows:
                    continue
                if isinstance(rows[0], dict):
                    frame = pd.DataFrame.from_records(rows)
                else:
                    columns = [desc[0] for desc in cursor.description]
                    frame = pd.DataFrame.from_records(rows, columns=columns)
                frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["collectionobjectid", "is_redacted"])
    combined = pd.concat(frames, ignore_index=True)
    combined["collectionobjectid"] = pd.to_numeric(combined["collectionobjectid"], errors="coerce")
    combined["is_redacted"] = pd.to_numeric(combined["is_redacted"], errors="coerce").fillna(0).astype(int)
    return combined


def verify_redaction(
    df: pd.DataFrame,
    redaction_flags: pd.DataFrame,
    *,
    id_column: str = "collectionobjectid",
    locality_columns: Sequence[str] = DEFAULT_LOCALITY_TEXT_COLUMNS,
    coordinate_columns: Sequence[str] = DEFAULT_COORDINATE_COLUMNS,
    redacted_placeholder: str = "*",
) -> RedactionVerificationResult:
    """
    Verify that all flagged records are properly redacted in a DataFrame.
    """
    if id_column not in df.columns:
        raise KeyError(f"Missing required column: {id_column}")
    if redaction_flags.empty:
        return RedactionVerificationResult(0, 0, 0, pd.DataFrame())

    flags = redaction_flags[[id_column, "is_redacted"]].copy()
    flags[id_column] = pd.to_numeric(flags[id_column], errors="coerce")
    flags["is_redacted"] = pd.to_numeric(flags["is_redacted"], errors="coerce").fillna(0).astype(int)
    flags = flags.dropna(subset=[id_column]).drop_duplicates(subset=[id_column])

    merged = df.merge(flags, on=id_column, how="left")
    merged["is_redacted"] = pd.to_numeric(merged["is_redacted"], errors="coerce").fillna(0).astype(int)

    flagged = merged[merged["is_redacted"] == 1]
    flagged_rows = len(flagged)

    loc_cols = [column for column in locality_columns if column in merged.columns]
    coord_cols = [column for column in coordinate_columns if column in merged.columns]

    bad_locality = flagged[~flagged[loc_cols].eq(redacted_placeholder).all(axis=1)] if loc_cols else flagged.iloc[0:0]
    bad_coords = flagged[flagged[coord_cols].notna().any(axis=1)] if coord_cols else flagged.iloc[0:0]

    offenders = (
        pd.concat([bad_locality, bad_coords], ignore_index=False)
        .drop_duplicates(subset=[id_column])
        .reset_index(drop=True)
    )
    return RedactionVerificationResult(
        flagged_rows=flagged_rows,
        bad_locality_rows=len(bad_locality),
        bad_coordinate_rows=len(bad_coords),
        offenders=offenders,
    )


def verify_redaction_drop(
    df: pd.DataFrame,
    redaction_flags: pd.DataFrame,
    *,
    id_column: str = "collectionobjectid",
) -> RedactionDropVerificationResult:
    """
    Verify that no redacted records remain in a DataFrame.
    """
    if id_column not in df.columns:
        raise KeyError(f"Missing required column: {id_column}")
    if redaction_flags.empty:
        return RedactionDropVerificationResult(0, pd.DataFrame())

    flags = redaction_flags[[id_column, "is_redacted"]].copy()
    flags[id_column] = pd.to_numeric(flags[id_column], errors="coerce")
    flags["is_redacted"] = pd.to_numeric(flags["is_redacted"], errors="coerce").fillna(0).astype(int)
    flags = flags.dropna(subset=[id_column]).drop_duplicates(subset=[id_column])

    merged = df.merge(flags, on=id_column, how="left")
    merged["is_redacted"] = pd.to_numeric(merged["is_redacted"], errors="coerce").fillna(0).astype(int)

    offenders = merged[merged["is_redacted"] == 1]
    return RedactionDropVerificationResult(len(offenders), offenders.reset_index(drop=True))


def redact_clustered_dataframe(
    df: pd.DataFrame,
    redaction_flags: pd.DataFrame,
    *,
    id_column: str = "collectionobjectid",
    locality_columns: Sequence[str] = DEFAULT_LOCALITY_TEXT_COLUMNS,
    coordinate_columns: Sequence[str] = DEFAULT_COORDINATE_COLUMNS,
    redacted_placeholder: str = "*",
    add_redaction_column: bool = False,
    drop_redacted: bool = False,
) -> tuple[pd.DataFrame, int]:
    """
    Apply IPT redaction rules to a clustered DataFrame.

    When drop_redacted is True, flagged records are removed instead of masked.

    Returns:
        (output_df, redacted_row_count)

    """
    if id_column not in df.columns:
        raise KeyError(f"Missing required column: {id_column}")

    redacted_df = df.copy()
    if "is_redacted" in redacted_df.columns:
        redacted_df = redacted_df.drop(columns=["is_redacted"])

    if redaction_flags.empty:
        return redacted_df, 0

    flags = redaction_flags[[id_column, "is_redacted"]].copy()
    flags[id_column] = pd.to_numeric(flags[id_column], errors="coerce")
    flags["is_redacted"] = pd.to_numeric(flags["is_redacted"], errors="coerce").fillna(0).astype(int)
    redacted_df[id_column] = pd.to_numeric(redacted_df[id_column], errors="coerce")
    flags = flags.dropna(subset=[id_column]).drop_duplicates(subset=[id_column])

    merged = redacted_df.merge(flags, on=id_column, how="left")
    redacted_mask = merged["is_redacted"].fillna(0).astype(bool)

    redacted_rows = int(redacted_mask.sum())

    if drop_redacted:
        merged = merged.loc[~redacted_mask].copy()
    else:
        for column in locality_columns:
            if column in merged.columns:
                merged.loc[redacted_mask, column] = redacted_placeholder

        for column in coordinate_columns:
            if column in merged.columns:
                merged.loc[redacted_mask, column] = pd.NA

    if not add_redaction_column:
        merged = merged.drop(columns=["is_redacted"])

    return merged, redacted_rows


def verify_redacted_csv(
    input_path: str | Path,
    *,
    config: DatabaseConfig | None = None,
    id_column: str = "collectionobjectid",
    locality_columns: Sequence[str] = DEFAULT_LOCALITY_TEXT_COLUMNS,
    coordinate_columns: Sequence[str] = DEFAULT_COORDINATE_COLUMNS,
    redacted_placeholder: str = "*",
    chunk_size: int = 1000,
    logger: logging.Logger | None = None,
) -> RedactionVerificationResult:
    """
    Verify that a redacted clustered CSV honors IPT redaction rules.
    """
    input_path = Path(input_path)
    config = config or DatabaseConfig()

    clustered_df = pd.read_csv(input_path)
    if id_column not in clustered_df.columns:
        raise KeyError(f"Missing required column: {id_column}")

    flags = fetch_redaction_flags(
        config,
        clustered_df[id_column].dropna().unique().tolist(),
        chunk_size=chunk_size,
        logger=logger,
    )
    return verify_redaction(
        clustered_df,
        flags,
        id_column=id_column,
        locality_columns=locality_columns,
        coordinate_columns=coordinate_columns,
        redacted_placeholder=redacted_placeholder,
    )


def verify_redacted_csv_drop(
    input_path: str | Path,
    *,
    config: DatabaseConfig | None = None,
    id_column: str = "collectionobjectid",
    chunk_size: int = 1000,
    logger: logging.Logger | None = None,
) -> RedactionDropVerificationResult:
    """
    Verify that a redacted clustered CSV contains no redacted records.
    """
    input_path = Path(input_path)
    config = config or DatabaseConfig()

    clustered_df = pd.read_csv(input_path)
    if id_column not in clustered_df.columns:
        raise KeyError(f"Missing required column: {id_column}")

    flags = fetch_redaction_flags(
        config,
        clustered_df[id_column].dropna().unique().tolist(),
        chunk_size=chunk_size,
        logger=logger,
    )
    return verify_redaction_drop(
        clustered_df,
        flags,
        id_column=id_column,
    )


def redact_clustered_csv(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    config: DatabaseConfig | None = None,
    id_column: str = "collectionobjectid",
    locality_columns: Sequence[str] = DEFAULT_LOCALITY_TEXT_COLUMNS,
    coordinate_columns: Sequence[str] = DEFAULT_COORDINATE_COLUMNS,
    redacted_placeholder: str = "*",
    chunk_size: int = 1000,
    add_redaction_column: bool = False,
    drop_redacted: bool = False,
    logger: logging.Logger | None = None,
) -> tuple[Path, int]:
    """
    Load a clustered CSV, apply IPT redaction rules, and write the redacted CSV.

    When drop_redacted is True, flagged records are removed instead of masked.

    Returns:
        (output_path, redacted_row_count)

    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path
    config = config or DatabaseConfig()

    clustered_df = pd.read_csv(input_path)
    if id_column not in clustered_df.columns:
        raise KeyError(f"Missing required column: {id_column}")

    flags = fetch_redaction_flags(
        config,
        clustered_df[id_column].dropna().unique().tolist(),
        chunk_size=chunk_size,
        logger=logger,
    )
    redacted_df, redacted_rows = redact_clustered_dataframe(
        clustered_df,
        flags,
        id_column=id_column,
        locality_columns=locality_columns,
        coordinate_columns=coordinate_columns,
        redacted_placeholder=redacted_placeholder,
        add_redaction_column=add_redaction_column,
        drop_redacted=drop_redacted,
    )

    redacted_df.to_csv(output_path, index=False)
    if logger:
        percent = (redacted_rows / len(clustered_df) * 100) if len(clustered_df) else 0
        if drop_redacted:
            logger.info("Dropped %d redacted rows (%.2f%% of output).", redacted_rows, percent)
        else:
            logger.info("Redacted %d rows (%.2f%% of output).", redacted_rows, percent)
    return output_path, redacted_rows
