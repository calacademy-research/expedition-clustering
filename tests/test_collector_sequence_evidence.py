"""
Feature #922 (CAS Lens): sequence rows without coordinates handed to the
georeference engine as evidence.

For each row of a sequence that passed every rule, with no effective
coordinate, whose sequence has a located row: the located row nearest in
collector number (then day, then spid), its coordinate and source, and the
gaps. Never a coordinate of the row itself.

Requirement: EXP-SEQ-7 (cas-lens docs/requirements/expedition-requirements.md).
"""

import datetime as dt

import pandas as pd

from expedition_clustering.collector_sequences import EVIDENCE_COLUMNS, build_sequences, evidence_rows, prepare_rows

D = dt.date(1953, 5, 23)


def _row(spid, collectors, number, date, *, lat=None, lng=None, source=""):
    return {
        "spid": spid, "collectors": collectors, "stationFieldNumber": str(number),
        "startDate": str(date.year), "ce_startDate": str(date.month), "ce_startDate1": str(date.day),
        "verbatimDate": "",
        "latitude1": "" if lat is None else str(lat), "longitude1": "" if lng is None else str(lng),
        "coordinate_source": source,
    }


def _evidence(rows):
    return evidence_rows(build_sequences(prepare_rows(pd.DataFrame(rows)))).set_index("spid")


def test_each_unlocated_row_gets_its_nearest_located_neighbour():
    rows = [
        _row("n1", "Thomas, J. H.", 3160, D, lat=34.1, lng=-118.2, source="specify"),
        _row("n2", "Thomas, J. H.", 3161, D),
        _row("n4", "Thomas, J. H.", 3163, D + dt.timedelta(days=1)),
        _row("n5", "Thomas, J. H.", 3164, D + dt.timedelta(days=1), lat=34.3, lng=-118.4, source="engine"),
    ]
    ev = _evidence(rows)
    assert list(ev.reset_index().columns) == list(EVIDENCE_COLUMNS)
    assert sorted(ev.index) == ["n2", "n4"]
    assert (ev.loc["n2", "neighbour_spid"], ev.loc["n2", "neighbour_number_gap"]) == ("n1", 1)
    assert (ev.loc["n2", "evidence_lat"], ev.loc["n2", "evidence_lng"]) == (34.1, -118.2)
    assert ev.loc["n2", "neighbour_source"] == "specify"
    assert (ev.loc["n4", "neighbour_spid"], ev.loc["n4", "neighbour_source"]) == ("n5", "engine")
    assert ev.loc["n4", "neighbour_day_gap"] == 0
    assert ev.loc["n4", "date"] == "1953-05-24"
    assert ev.loc["n2", "collector_number"] == 3161


def test_a_sequence_without_located_rows_gives_no_evidence():
    rows = [_row(f"u{n}", "Breedlove, D. E.", 33000 + n, D) for n in range(5)]
    assert _evidence(rows).empty


def test_rows_outside_a_sequence_give_no_evidence():
    rows = [
        _row("a", "Smith, A.", 10, D, lat=1.0, lng=1.0, source="specify"),
        _row("b", "Smith, A.", 50, D),  # gap 40: not linked
        _row("c", "Pollard, H. M.", "s.n.", D),
    ]
    assert _evidence(rows).empty
