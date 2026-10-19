"""
CAS Lens bug #1397: ``scripts/trip_members.py`` writes every specimen of every
Specify collecting trip, whatever the clustering made of it.

An imported expedition (a Specify CollectingTrip) used to be assembled from
clustered_expeditions.csv, so it held only the members that survived
``cluster_csv.py --min-specimens`` (default 20) and had coordinates and a
date: CollectingTrip 1486 (casiz) holds 109 specimens and its page showed 57.
A trip's membership is its trip id (specimen_trip_mapping.csv), so the trip
file is the mapping joined to the clustering input, independent of the
clusters; ``cluster_csv.py`` keeps its cutoff for everything else.

Requirements: EXP-TRIP-ALL-1, EXP-TRIP-ALL-2 (cas-lens
docs/requirements/expedition-requirements.md).
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
_TRIPS = _ROOT / "scripts" / "trip_members.py"
_CLUSTER = _ROOT / "scripts" / "cluster_csv.py"


def _row(spid, *, lat=None, lng=None, year=1976, month=8, day=10, collectors="Grantz, A."):
    return {
        "spid": spid, "catalogNumber": str(abs(hash(spid)) % 100000), "collectors": collectors,
        "startDate": "" if year is None else year,
        "ce_startDate": "" if year is None else month,
        "ce_startDate1": "" if year is None else day,
        "localityName": "off coast of Alaska", "Country": "United States",
        "latitude1": "" if lat is None else lat, "longitude1": "" if lng is None else lng,
    }


def _input(tmp_path):
    """
    Trip 1486: 25 rows at one station (a cluster over the cutoff), 5 rows far
    away (a cluster under it), 2 rows without coordinates, 1 without a date.
    Not in a trip: 3 rows at a third place (a cluster under the cutoff).
    """
    rows = [_row(f"big-{n}", lat=71.0 + n * 1e-4, lng=-156.0) for n in range(25)]
    rows += [_row(f"small-{n}", lat=69.5 + n * 1e-4, lng=-163.0, day=20) for n in range(5)]
    rows += [_row(f"nocoord-{n}") for n in range(2)]
    rows += [_row("nodate-0", lat=37.7, lng=-122.3, year=None)]
    rows += [_row(f"other-{n}", lat=10.0, lng=10.0, collectors="Someone") for n in range(3)]
    coll = tmp_path / "iz"
    coll.mkdir()
    pd.DataFrame(rows).to_csv(coll / "PortalData.csv", index=False)
    members = [r["spid"] for r in rows if not r["spid"].startswith("other-")]
    mapping = tmp_path / "specimen_trip_mapping.csv"
    pd.DataFrame(
        {"spid": [*members, "not-published-0"], "collecting_trip_id": [1486] * len(members) + [1486]}
    ).to_csv(mapping, index=False)
    return coll, mapping, members


def _run(script, *argv):
    return subprocess.run(  # noqa: S603 - our own script, fixed argv
        [sys.executable, str(script), *map(str, argv)], capture_output=True, text=True, check=False,
    )


def test_every_trip_member_is_written_whatever_its_cluster(tmp_path):
    coll, mapping, members = _input(tmp_path)
    out = tmp_path / "trip_members.csv"
    run = _run(_TRIPS, coll, "--trip-mapping", mapping, "-o", out)
    assert run.returncode == 0, run.stderr + run.stdout
    got = pd.read_csv(out, dtype={"spid": str})
    assert sorted(got["spid"]) == sorted(members)  # 33: none lost to the cutoff or a missing value
    assert set(got["collecting_trip_id"]) == {1486}
    # Rows without coordinates or a date are members; their cells are empty.
    nocoord = got[got["spid"].str.startswith("nocoord-")]
    assert nocoord["latitude1"].isna().all() and nocoord["longitude1"].isna().all()
    nodate = got[got["spid"] == "nodate-0"]
    assert nodate["startdate"].isna().all() and nodate["date_precision"].isna().all()
    # The columns the Lens loader reads beside clustered_expeditions.csv's.
    for col in ("collectionobjectid", "startdate", "enddate", "date_precision", "latitude1",
                "longitude1", "localityname", "collectors", "country", "geo_region"):
        assert col in got.columns, col
    # The trip is classified from its located members (one classification per trip).
    assert got["geo_region"].nunique(dropna=False) == 1
    assert "33 rows of 1 trips" in run.stdout
    assert "1 mapping spids are not in the input" in run.stdout
    assert "3 without coordinates or a date" in run.stdout


def test_the_cutoff_still_applies_to_the_clusters(tmp_path):
    coll, _mapping, _members = _input(tmp_path)
    out = tmp_path / "clustered_expeditions.csv"
    run = _run(_CLUSTER, coll, "-o", out, "--min-specimens", "20")
    assert run.returncode == 0, run.stderr + run.stdout
    got = pd.read_csv(out, dtype={"spid": str})
    assert sorted(got["spid"]) == sorted(f"big-{n}" for n in range(25))


def test_a_header_only_mapping_writes_a_header_only_file(tmp_path):
    coll, _mapping, _members = _input(tmp_path)
    empty = tmp_path / "empty_mapping.csv"
    empty.write_text("spid,collecting_trip_id\n")
    out = tmp_path / "trip_members.csv"
    run = _run(_TRIPS, coll, "--trip-mapping", empty, "-o", out)
    assert run.returncode == 0, run.stderr + run.stdout
    got = pd.read_csv(out)
    assert len(got) == 0 and "collecting_trip_id" in got.columns
    assert "0 rows of 0 trips" in run.stdout


def test_a_mapping_without_its_columns_fails_by_name(tmp_path):
    coll, _mapping, _members = _input(tmp_path)
    bad = tmp_path / "bad.csv"
    bad.write_text("spid,trip\nbig-0,1\n")
    run = _run(_TRIPS, coll, "--trip-mapping", bad, "-o", tmp_path / "out.csv")
    assert run.returncode != 0
    assert "collecting_trip_id" in run.stderr
    assert not (tmp_path / "out.csv").exists()


def test_a_trip_id_that_is_not_an_integer_fails_by_name(tmp_path):
    coll, _mapping, _members = _input(tmp_path)
    bad = tmp_path / "bad.csv"
    bad.write_text("spid,collecting_trip_id\nbig-0,x1\n")
    run = _run(_TRIPS, coll, "--trip-mapping", bad, "-o", tmp_path / "out.csv")
    assert run.returncode != 0
    assert "big-0" in run.stderr
