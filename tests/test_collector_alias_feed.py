"""
Feature #378 (CAS Lens): Lens's name spellings feed the clustering.

CAS Lens writes a spelling map each week (alias as written, the person's
slug) from its people / collector_aliases tables. scripts/collector_alias_file.py
turns it into the alias file cluster_csv.py reads, keying each spelling with
this repo's own normaliser, so "H. Heath" and "Harold Heath" of one Lens person
share a canonical id. A key two people hold is left out and counted, never
given to one of them. The alias file is then used by the collector partition
and by the merge stage's collector overlap.

Both orders of a "Last, First" name are looked up, as Lens resolves a name
(NAME-ORDER-1): "Thomas, J. H." finds the alias "J. H. Thomas"; two orders
naming two people name nobody.

Requirements: EXP-ALIAS-4 .. EXP-ALIAS-6 (cas-lens docs/requirements/expedition-requirements.md).
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from expedition_clustering.pipeline import (
    CollectorPartitioning,
    MergeExpeditions,
    alias_coverage,
    alias_table_from_spellings,
    collector_key,
    load_collector_aliases,
)

_ROOT = Path(__file__).parent.parent
_SCRIPT = _ROOT / "scripts" / "collector_alias_file.py"


# ─── load_collector_aliases: an alias file that maps nothing is an error ────


def test_header_only_alias_file_raises(tmp_path):
    empty = tmp_path / "aliases.csv"
    empty.write_text("normalized_name,canonical_id\n")
    with pytest.raises(ValueError, match="no rows"):
        load_collector_aliases(empty)


def test_zero_byte_alias_file_raises(tmp_path):
    empty = tmp_path / "aliases.csv"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_collector_aliases(empty)


def test_key_with_two_ids_raises(tmp_path):
    bad = tmp_path / "aliases.csv"
    bad.write_text("normalized_name,canonical_id\nf baker,@f-baker\nf baker,@f-baker-2\n")
    with pytest.raises(ValueError, match="f baker"):
        load_collector_aliases(bad)


def test_blank_id_raises(tmp_path):
    bad = tmp_path / "aliases.csv"
    bad.write_text("normalized_name,canonical_id\nf baker,\n")
    with pytest.raises(ValueError, match="blank"):
        load_collector_aliases(bad)


# ─── alias_table_from_spellings ─────────────────────────────────────────────


def test_spellings_of_one_person_share_an_id():
    table = alias_table_from_spellings([
        ("H. Heath", "h-heath"), ("Harold Heath", "h-heath"), ("H.Heath", "h-heath"),
    ])
    assert table.aliases == {"h heath": "@h-heath", "harold heath": "@h-heath", "hheath": "@h-heath"}
    assert table.spellings == 3
    assert table.people == 1
    assert table.shared_keys == {}


def test_key_two_people_hold_is_left_out_and_counted():
    # "F. Baker" (f-baker) and "F Baker" (f-baker-2) are one key here: neither gets it.
    table = alias_table_from_spellings([
        ("F. Baker", "f-baker"), ("Fred Baker", "f-baker"), ("F Baker", "f-baker-2"),
    ])
    assert table.aliases == {"fred baker": "@f-baker"}
    assert table.shared_keys == {"f baker": ["@f-baker", "@f-baker-2"]}


def test_unknown_and_blank_spellings_are_skipped():
    table = alias_table_from_spellings([("", "x"), ("unknown", "x"), ("...", "x"), ("A. B. Cee", "x")])
    assert table.aliases == {"a b cee": "@x"}
    assert table.skipped == 3


def test_blank_slug_raises():
    with pytest.raises(ValueError, match="slug"):
        alias_table_from_spellings([("A. B. Cee", "")])


# ─── scripts/collector_alias_file.py ────────────────────────────────────────


def _script(*args):
    return subprocess.run(  # noqa: S603 - our own script, fixed argv
        [sys.executable, str(_SCRIPT), *map(str, args)], capture_output=True, text=True, check=False,
    )


def test_script_writes_the_alias_file(tmp_path):
    spellings = tmp_path / "collector_spelling_map.csv"
    spellings.write_text(
        "alias,person_id,slug,canonical_name\n"
        "H. Heath,1,h-heath,H. Heath\n"
        "Harold Heath,1,h-heath,H. Heath\n"
        "F. Baker,2,f-baker,Fred Baker\n"
        "F Baker,3,f-baker-2,F. Baker\n"
    )
    out = tmp_path / "collector_aliases.csv"
    result = _script(spellings, "-o", out)
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("collector aliases: 4 spellings, 2 keys for 1 people; "
            "1 keys held by two or more people left out, 0 spellings skipped") in result.stdout
    assert load_collector_aliases(out) == {"h heath": "@h-heath", "harold heath": "@h-heath"}


def test_script_refuses_a_missing_spelling_map(tmp_path):
    result = _script(tmp_path / "nope.csv", "-o", tmp_path / "out.csv")
    assert result.returncode != 0
    assert "nope.csv" in result.stderr
    assert not (tmp_path / "out.csv").exists()


def test_script_refuses_an_empty_spelling_map(tmp_path):
    spellings = tmp_path / "collector_spelling_map.csv"
    spellings.write_text("alias,person_id,slug,canonical_name\n")
    result = _script(spellings, "-o", tmp_path / "out.csv")
    assert result.returncode != 0
    assert "no rows" in result.stderr
    assert not (tmp_path / "out.csv").exists()


def test_script_refuses_missing_columns(tmp_path):
    spellings = tmp_path / "collector_spelling_map.csv"
    spellings.write_text("name,id\nH. Heath,1\n")
    result = _script(spellings, "-o", tmp_path / "out.csv")
    assert result.returncode != 0
    assert "alias" in result.stderr
    assert "slug" in result.stderr


# ─── The partition and the merge stage read the aliases ─────────────────────


def _alias_file(tmp_path, rows):
    path = tmp_path / "aliases.csv"
    pd.DataFrame(rows, columns=["normalized_name", "canonical_id"]).to_csv(path, index=False)
    return path


def test_partition_puts_two_spellings_of_one_person_together(tmp_path):
    aliases = _alias_file(tmp_path, [("h heath", "@h-heath"), ("harold heath", "@h-heath")])
    df = pd.DataFrame({
        "collectors": ["H. Heath", "Harold Heath", "E.C. Starks"],
        "startdate": pd.to_datetime(["1911-05-28"] * 3),
        "date_precision": [1, 1, 1],
    })
    step = CollectorPartitioning(enabled=True, alias_file=aliases).fit(df)
    out = step.transform(df)
    assert out.loc[0, "collector_partition"] == out.loc[1, "collector_partition"]
    assert out.loc[0, "collector_partition"] != out.loc[2, "collector_partition"]
    assert out.loc[0, "collector_group"] == "@h-heath"


def _merge_df(spelling_a, spelling_b):
    df = pd.DataFrame([
        {"latitude1": 0.0, "longitude1": 0.0, "startdate": "1953-05-30",
         "collectors": spelling_a, "spatiotemporal_cluster_id": 0},
        {"latitude1": 0.001, "longitude1": 0.001, "startdate": "1953-05-30",
         "collectors": spelling_a, "spatiotemporal_cluster_id": 0},
        {"latitude1": 0.15, "longitude1": 0.15, "startdate": "1953-05-31",
         "collectors": spelling_b, "spatiotemporal_cluster_id": 1},
        {"latitude1": 0.151, "longitude1": 0.151, "startdate": "1953-05-31",
         "collectors": spelling_b, "spatiotemporal_cluster_id": 1},
    ])
    df["startdate"] = pd.to_datetime(df["startdate"])
    df["spatial_cluster_id"] = df["spatiotemporal_cluster_id"]
    df["temporal_cluster_id"] = df["spatiotemporal_cluster_id"]
    df["collectingeventid"] = range(len(df))
    df["date_precision"] = 1
    return df


def _merger(alias_file=None):
    # Threshold between the score of the pair without a shared collector and with one.
    return MergeExpeditions(max_merge_gap_days=3, max_merge_distance_km=30,
                            merge_threshold=0.4, max_gap_days=3, alias_file=alias_file)


def test_merge_stage_counts_two_spellings_of_one_person_as_one_collector(tmp_path):
    df = _merge_df("Thomas, John H.", "Thomas, J. H.")
    # Without the aliases the two spellings share no collector: no merge.
    assert _merger().fit(df).transform(df)["spatiotemporal_cluster_id"].nunique() == 2
    aliases = _alias_file(tmp_path, [("thomas john h", "@john-h-thomas"), ("thomas j h", "@john-h-thomas")])
    merged = _merger(aliases).fit(df).transform(df)
    assert len(set(merged["spatiotemporal_cluster_id"])) == 1


def test_merge_stage_raises_for_a_missing_alias_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        _merger(tmp_path / "nope.csv").fit(None)


# ─── How much of a collection the aliases reach (the run's log line) ───────


def test_alias_coverage_counts_names_and_rows():
    collectors = pd.Series(["H. Heath", "H. Heath", "F. Baker and W.M. Mann", "Nobody, A.", None])
    cov = alias_coverage(collectors, {"h heath": "@h-heath", "wm mann": "@w-m-mann"})
    # names: h heath, f baker, wm mann, nobody a -> 2 resolve; rows with at least one: 3 of 4 with a collector
    assert (cov.names_resolved, cov.names, cov.rows_resolved, cov.rows) == (2, 4, 3, 4)
    assert cov.line() == "collector aliases: 2 of 4 collector names resolve to a CAS Lens person (3 of 4 rows)"


def test_cluster_csv_prints_the_alias_coverage(tmp_path):
    coll = tmp_path / "herp"
    coll.mkdir()
    rows = [{"spid": f"h-{i}", "catalogNumber": str(i), "collectors": "H. Heath" if i % 2 else "Harold Heath",
             "startDate": "1911-05-28", "verbatimDate": "", "localityName": "Natal",
             "latitude1": -5.79 + i * 1e-4, "longitude1": -35.21} for i in range(30)]
    pd.DataFrame(rows).to_csv(coll / "PortalData.csv", index=False)
    aliases = _alias_file(tmp_path, [("h heath", "@h-heath"), ("harold heath", "@h-heath")])
    out = tmp_path / "out.csv"
    result = subprocess.run(  # noqa: S603 - our own script, fixed argv
        [sys.executable, str(_ROOT / "scripts" / "cluster_csv.py"), str(coll), "-o", str(out),
         "--collector-alias-file", str(aliases)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "collector aliases: 2 of 2 collector names resolve to a CAS Lens person (30 of 30 rows)" in result.stdout
    got = pd.read_csv(out)
    # herp is collector-aware: the two spellings are one partition, one expedition.
    assert len(set(got["spatiotemporal_cluster_id"])) == 1
    assert set(got["collector_group"]) == {"@h-heath"}


# ─── Both orders of a name (CAS Lens NAME-ORDER-1) ─────────────────────────


def test_last_first_finds_the_first_last_alias():
    aliases = {"j h thomas": "@j-h-thomas"}
    # Lens holds "J. H. Thomas"; botany writes "Thomas, J. H.": one person.
    assert collector_key("Thomas, J. H.", aliases) == "@j-h-thomas"
    assert collector_key("J. H. Thomas", aliases) == "@j-h-thomas"


def test_two_orders_naming_two_people_map_to_nobody():
    aliases = {"baker f": "@f-baker-2", "f baker": "@f-baker"}
    # Lens: "Baker, F." and "F. Baker" are two records, so the name is ambiguous.
    assert collector_key("Baker, F.", aliases) == "baker f"


def test_coverage_counts_a_name_found_in_the_other_order():
    collectors = pd.Series(["Thomas, J. H.", "Nobody, A."])
    cov = alias_coverage(collectors, {"j h thomas": "@j-h-thomas"})
    assert (cov.names_resolved, cov.names) == (1, 2)
