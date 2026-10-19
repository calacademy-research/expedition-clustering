#!/usr/bin/env python3
"""
Every specimen of every Specify collecting trip, in the clustering's columns
(CAS Lens bug #1397, EXP-TRIP-ALL-1).

An imported expedition on CAS Lens is a Specify CollectingTrip. Its members
are the specimens whose collecting event names the trip
(specimen_trip_mapping.csv: spid, collecting_trip_id), not the rows a
cluster kept: cluster_csv.py drops rows without coordinates or a date before
clustering and drops clusters under --min-specimens after it, and building
the trips from clustered_expeditions.csv lost those members (casiz trip 1486:
109 specimens, 57 on the page). This script writes the trips' rows
independently of the clustering, so the cutoff keeps applying to the
generated clusters and never to a trip.

Usage:
    python scripts/trip_members.py INPUT --trip-mapping specimen_trip_mapping.csv -o trip_members.csv

Inputs:
    INPUT           a collection directory holding PortalData.csv, or the CSV
                    itself; the same input cluster_csv.py reads (CAS Lens gives
                    both the merged clustering input, so latitude1/longitude1
                    are the row's effective coordinate).
    --trip-mapping  CSV with columns spid, collecting_trip_id (integers); a
                    header-only file means the collection has no trips.
    -o / --output   the file to write.
Output:
    one row per input row whose spid is in the mapping, rows without
    coordinates or a date included (their cells empty), in
    transform_csv_to_pipeline_format's columns plus collecting_trip_id and the
    geo_* classification of the trip (computed from the trip's located rows,
    as cluster_csv.py classifies a cluster; empty for a trip with none).
    One summary line on stdout:
    "trip members: N rows of T trips; M mapping spids are not in the input;
    U without coordinates or a date".
Raises / exits non-zero:
    a mapping without the spid / collecting_trip_id columns, a trip id that is
    not an integer (the spids named), an input that cannot be read. Nothing is
    written then.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

# Direct imports, as cluster_csv.py does, to avoid cartopy in __init__.py.
_pkg_dir = Path(__file__).parent.parent / "expedition_clustering"
_spec = importlib.util.spec_from_file_location("csv_source", _pkg_dir / "csv_source.py")
_csv_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_csv_module)
_spec = importlib.util.spec_from_file_location("geo_classify", _pkg_dir / "geo_classify.py")
_geo_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_geo_module)

TRIP_COL = "collecting_trip_id"


def read_trip_mapping(path: Path) -> pd.Series:
    """
    The spid -> collecting_trip_id mapping.

    Input:  the specimen_trip_mapping.csv path.
    Output: a Series of int trip ids indexed by spid (stripped strings); rows
            with an empty spid or trip id carry no link and are left out.
    Raises: ValueError naming the file when a column is missing, or naming the
            first spids whose trip id is not an integer.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in ("spid", TRIP_COL) if c not in frame.columns]
    if missing:
        msg = f"{path}: no {', '.join(missing)} column (expected spid, {TRIP_COL})"
        raise ValueError(msg)
    frame["spid"] = frame["spid"].str.strip()
    frame[TRIP_COL] = frame[TRIP_COL].str.strip()
    frame = frame[(frame["spid"] != "") & (frame[TRIP_COL] != "")]
    ids = pd.to_numeric(frame[TRIP_COL], errors="coerce")
    bad = frame.loc[ids.isna() | (ids != ids.round()), "spid"].tolist()
    if bad:
        msg = f"{path}: {len(bad)} {TRIP_COL} value(s) are not integers; first spids: {', '.join(bad[:3])}"
        raise ValueError(msg)
    return pd.Series(ids.astype("int64").to_numpy(), index=frame["spid"].to_numpy(), name=TRIP_COL)


def trip_members(df: pd.DataFrame, mapping: pd.Series) -> tuple[pd.DataFrame, int]:
    """
    The input rows that belong to a trip, with the trip id and its geo classification.

    Inputs: df -- transform_csv_to_pipeline_format's frame (every row, none
            filtered); mapping -- read_trip_mapping's Series.
    Output: (members, unmatched): members holds every df row whose spid is in
            the mapping, in input order, plus collecting_trip_id and the geo_*
            columns of its trip; unmatched counts mapping spids no df row has.
    Transformation: the trip's classification is classify_expeditions on the
            trip's located rows with the trip id as the cluster; a trip
            without one located row gets empty geo_* cells.
    """
    members = df[df["spid"].isin(mapping.index)].copy()
    members[TRIP_COL] = members["spid"].map(mapping).astype("int64")
    unmatched = int((~mapping.index.isin(df["spid"])).sum())

    located = members[members["latitude1"].notna() & members["longitude1"].notna()]
    if located.empty:
        geo = pd.DataFrame(columns=[TRIP_COL])
    else:
        classified = _geo_module.classify_expeditions(
            located, cluster_col=TRIP_COL, lat_col="latitude1", lng_col="longitude1",
        )
        geo_cols = [c for c in classified.columns if c.startswith("geo_")]
        geo = classified[[TRIP_COL, *geo_cols]].drop_duplicates(TRIP_COL)
    members = members.merge(geo, on=TRIP_COL, how="left")
    return members, unmatched


def main() -> int:
    parser = argparse.ArgumentParser(description="Every specimen of every Specify collecting trip (CAS Lens #1397)")
    parser.add_argument("input", type=Path, help="collection directory with PortalData.csv, or the CSV")
    parser.add_argument("--trip-mapping", type=Path, required=True, help="specimen_trip_mapping.csv (spid, collecting_trip_id)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="the trip_members.csv to write")
    args = parser.parse_args()

    try:
        mapping = read_trip_mapping(args.trip_mapping)
    except ValueError as e:
        print(f"trip_members.py: {e}", file=sys.stderr)
        return 1
    df = _csv_module.load_collection_csv(args.input)
    members, unmatched = trip_members(df, mapping)
    unplaced = int((members["latitude1"].isna() | members["longitude1"].isna() | members["startdate"].isna()).sum())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    members.to_csv(args.output, index=False)
    print(
        f"trip members: {len(members)} rows of {members[TRIP_COL].nunique()} trips; "
        f"{unmatched} mapping spids are not in the input; {unplaced} without coordinates or a date"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
