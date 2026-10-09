"""
Feature #922 (CAS Lens): ``cluster_csv.py --collector-sequences`` end to end.

On a merged clustering input (CAS Lens #918: latitude1 / longitude1 are the
row's effective coordinate, coordinate_source names it), the run keeps every
spatial cluster as it is, adds the sequence rows the merge policy places,
writes the sequence columns, collector_sequences.csv and
collector_sequences_summary.txt beside the output, and prints the log line.

Requirements: EXP-SEQ-2, EXP-SEQ-4 (cas-lens docs/requirements/expedition-requirements.md),
PFV-385 (cas-lens docs/requirements/pipeline-failure-visibility.md).
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
_SCRIPT = _ROOT / "scripts" / "cluster_csv.py"


def _row(spid, collectors, number, day, *, lat=None, lng=None, source="", year=1953, month=4):
    return {
        "spid": spid, "catalogNumber": spid.rsplit("-", 1)[-1], "collectors": collectors,
        "stationFieldNumber": str(number), "startDate": year, "ce_startDate": month, "ce_startDate1": day,
        "verbatimDate": "", "localityName": "somewhere",
        "latitude1": "" if lat is None else lat, "longitude1": "" if lng is None else lng,
        "coordinate_source": source,
    }


def _input(tmp_path, *, with_source=True):
    rows = []
    # Thomas: numbers 1-30 located at one place on 4 April (one spatial cluster),
    # 31-40 the same day, not located: attached by collector number.
    rows += [_row(f"thomas-{n}", "Thomas, J. H.", n, 4, lat=37.40 + n * 1e-4, lng=-122.10,
                  source="specify" if n % 2 else "engine") for n in range(1, 31)]
    rows += [_row(f"thomas-{n}", "Thomas, John H.", n, 4) for n in range(31, 41)]
    # Breedlove: 25 unlocated numbers on one day, no cluster: a sequence-only expedition.
    rows += [_row(f"breedlove-{n}", "Breedlove, D. E.", 33000 + n, 9) for n in range(25)]
    # s.n. rows and a plot code: excluded by rule.
    rows += [_row(f"pollard-{n}", "Pollard, Henry M.", "s.n.", 5) for n in range(3)]
    rows += [_row("mozingo-1", "Mozingo, H. N.", "10/32b", 6)]
    coll = tmp_path / "botany"
    coll.mkdir()
    frame = pd.DataFrame(rows)
    if not with_source:
        frame = frame.drop(columns=["coordinate_source"])
    frame.to_csv(coll / "PortalData.csv", index=False)
    return coll


def _run(coll, out, *flags):
    return subprocess.run(  # noqa: S603 - our own script, fixed argv
        [sys.executable, str(_SCRIPT), *flags, str(coll), "-o", str(out)],
        capture_output=True, text=True, check=False,
    )


def test_sequences_join_the_output(tmp_path):
    coll = _input(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = out_dir / "clustered_expeditions.csv"
    result = _run(coll, out, "--collector-sequences")
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("collector sequences: 65 rows in sequences, 10 attached to clusters, "
            "1 new sequence-only expeditions, 4 rows excluded by rule") in result.stdout
    assert ("collector sequences excluded: s.n. 3, unparsed code 1, no day-precise date 0, before 1890 0, "
            "non-personal collector 0, numbers do not follow dates 0, spread over 500 km 0") in result.stdout
    assert ("collector sequences by coordinate: in sequences specify 15 / engine 15 / none 35; "
            "attached specify 0 / engine 0 / none 10") in result.stdout

    got = pd.read_csv(out, keep_default_na=False, dtype=str).set_index("spid")
    # The spatial cluster is unchanged and the ten unlocated Thomas rows join it.
    thomas_cluster = got.loc["thomas-1", "spatiotemporal_cluster_id"]
    for n in range(1, 41):
        assert got.loc[f"thomas-{n}", "spatiotemporal_cluster_id"] == thomas_cluster
    assert {got.loc[f"thomas-{n}", "cluster_join"] for n in range(1, 31)} == {"spatial"}
    assert {got.loc[f"thomas-{n}", "cluster_join"] for n in range(31, 41)} == {"collector_number"}
    assert got.loc["thomas-35", "latitude1"] == ""
    assert got.loc["thomas-35", "coordinate_source"] == ""
    assert got.loc["thomas-35", "collector_number"] == "35"
    # Breedlove's sequence is its own expedition, numbered above the spatial ids.
    breedlove = {got.loc[f"breedlove-{n}", "spatiotemporal_cluster_id"] for n in range(25)}
    assert len(breedlove) == 1
    assert int(next(iter(breedlove))) > int(thomas_cluster)
    assert got.loc["breedlove-3", "collector_number"] == "33003"
    assert got.loc["breedlove-3", "date_precision"] == "1"
    # Excluded rows are in no expedition.
    assert "pollard-0" not in got.index
    assert "mozingo-1" not in got.index
    # Integer columns stay integers after the added rows.
    assert got.loc["thomas-1", "spatial_cluster_id"].isdigit()
    assert got.loc["thomas-1", "collector_sequence_id"].isdigit()

    seqs = pd.read_csv(out_dir / "collector_sequences.csv", keep_default_na=False)
    assert sorted(seqs["disposition"]) == ["attached", "sequence-only"]
    summary = dict(line.split("=", 1) for line in (out_dir / "collector_sequences_summary.txt").read_text().split())
    assert summary["rows_in_sequences"] == "65"
    assert summary["attached_rows"] == "10"
    assert summary["attached_rows.none"] == "10"
    assert summary["sequence_only_expeditions"] == "1"
    assert summary["sequence_only_rows"] == "25"
    assert summary["excluded_rows"] == "4"
    assert summary["excluded.sn"] == "3"
    assert summary["excluded.unparsed_code"] == "1"
    assert summary["rows_in_sequences.engine"] == "15"


def test_without_the_flag_nothing_changes(tmp_path):
    coll = _input(tmp_path)
    out = tmp_path / "clustered_expeditions.csv"
    result = _run(coll, out)
    assert result.returncode == 0, result.stdout + result.stderr
    got = pd.read_csv(out, keep_default_na=False, dtype=str)
    assert "cluster_join" not in got.columns
    assert "collector sequences" not in result.stdout
    assert not (tmp_path / "collector_sequences_summary.txt").exists()


def test_a_plain_portal_csv_is_refused(tmp_path):
    coll = _input(tmp_path, with_source=False)
    result = _run(coll, tmp_path / "clustered_expeditions.csv", "--collector-sequences")
    assert result.returncode != 0
    assert "coordinate_source" in result.stderr
