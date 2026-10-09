"""
Feature #922 (CAS Lens): collector-number sequences against the spatial clusters.

The merge policy (cas-lens docs/requirements/bugs/922/PLAN.md §3, approved):
a clustered row never moves and no two clusters merge; a sequence row in no
cluster joins the cluster of the sequence's clustered row nearest in number
(then day, then the lower cluster id); a sequence with no clustered row and at
least min_specimens rows is its own expedition; a smaller one joins nothing.

Requirement: EXP-SEQ-4 (cas-lens docs/requirements/expedition-requirements.md).
"""

import datetime as dt

import pandas as pd

from expedition_clustering.collector_sequences import (
    DISPOSITION_ATTACHED,
    DISPOSITION_BELOW_FLOOR,
    DISPOSITION_IN_CLUSTER,
    DISPOSITION_SEQUENCE_ONLY,
    DISPOSITION_SPREAD,
    JOIN_NUMBER,
    build_sequences,
    merge_with_clusters,
    prepare_rows,
)

D = dt.date(1953, 4, 1)


def _row(spid, collectors, number, date, *, lat=None, lng=None):
    return {
        "spid": spid, "collectors": collectors, "stationFieldNumber": str(number),
        "startDate": str(date.year), "ce_startDate": str(date.month), "ce_startDate1": str(date.day),
        "verbatimDate": "",
        "latitude1": "" if lat is None else str(lat), "longitude1": "" if lng is None else str(lng),
        "coordinate_source": "" if lat is None else "specify",
    }


def _merge(rows, clusters, min_specimens=20):
    result = build_sequences(prepare_rows(pd.DataFrame(rows)))
    return merge_with_clusters(result, pd.Series(clusters, dtype="int64"), min_specimens)


def _assigned(merge):
    a = merge.assignments
    return dict(zip(a["spid"], a["spatiotemporal_cluster_id"], strict=True))


def test_attach_unlocated_rows_to_the_cluster_of_their_sequence():
    rows = [_row(f"r{n}", "Thomas, J. H.", n, D, lat=37.0, lng=-122.0) for n in range(1, 6)]
    rows += [_row(f"r{n}", "Thomas, John H.", n, D) for n in range(6, 11)]
    merge = _merge(rows, {f"r{n}": 7 for n in range(1, 6)})
    assert _assigned(merge) == {f"r{n}": 7 for n in range(6, 11)}
    assert set(merge.assignments["cluster_join"]) == {JOIN_NUMBER}
    assert merge.sequences["disposition"].tolist() == [DISPOSITION_ATTACHED]
    assert merge.sequence_only_expeditions == 0


def test_extend_a_cluster_by_a_day():
    rows = [_row(f"r{n}", "Thomas, J. H.", n, D, lat=37.0, lng=-122.0) for n in range(1, 6)]
    rows.append(_row("next-day", "Thomas, J. H.", 6, D + dt.timedelta(days=1)))
    merge = _merge(rows, {f"r{n}": 3 for n in range(1, 6)})
    assert _assigned(merge) == {"next-day": 3}


def test_a_sequence_spanning_two_clusters_never_splits_or_merges_them():
    rows = [_row(f"a{n}", "Howell, J. T.", n, D, lat=38.0, lng=-122.0) for n in range(1, 5)]
    rows += [_row("u5", "Howell, J. T.", 5, D), _row("u6", "Howell, J. T.", 6, D)]
    rows += [_row(f"b{n}", "Howell, J. T.", n, D, lat=38.5, lng=-122.5) for n in range(7, 11)]
    clusters = {**{f"a{n}": 1 for n in range(1, 5)}, **{f"b{n}": 2 for n in range(7, 11)}}
    merge = _merge(rows, clusters)
    # Each side's un-clustered row goes to its nearest numbered neighbour's cluster.
    assert _assigned(merge) == {"u5": 1, "u6": 2}
    # No clustered row is reassigned.
    assert not set(merge.assignments["spid"]) & set(clusters)
    assert merge.sequences["cluster_ids"].tolist() == ["1 2"]


def test_ties_go_to_the_nearest_day_then_the_lower_cluster():
    rows = [
        _row("c4", "Wiggins, I. L.", 4, D, lat=30.0, lng=-115.0),
        _row("c6", "Wiggins, I. L.", 6, D + dt.timedelta(days=1), lat=30.1, lng=-115.1),
        _row("u5", "Wiggins, I. L.", 5, D + dt.timedelta(days=1)),
        _row("x4", "Ferris, R. S.", 4, D, lat=36.0, lng=-121.0),
        _row("x6", "Ferris, R. S.", 6, D, lat=36.1, lng=-121.1),
        _row("y5", "Ferris, R. S.", 5, D),
    ]
    merge = _merge(rows, {"c4": 9, "c6": 4, "x4": 9, "x6": 4})
    assert _assigned(merge) == {"u5": 4, "y5": 4}


def test_a_sequence_without_clustered_rows_stands_alone_above_the_floor():
    rows = [_row(f"s{n}", "Breedlove, D. E.", 33000 + n, D) for n in range(25)]
    rows += [_row(f"t{n}", "Raven, P. H.", 100 + n, D) for n in range(22)]
    rows += [_row("k1", "Keck, D. D.", 1, D, lat=37.0, lng=-121.0)]
    merge = _merge(rows, {"k1": 41})
    got = _assigned(merge)
    assert set(got) == {f"s{n}" for n in range(25)} | {f"t{n}" for n in range(22)}
    new_ids = set(got.values())
    assert len(new_ids) == 2
    assert min(new_ids) > 41
    assert len({got[f"s{n}"] for n in range(25)}) == 1
    assert merge.sequence_only_expeditions == 2
    assert merge.sequence_only_rows == 47
    assert set(merge.sequences["disposition"]) == {DISPOSITION_SEQUENCE_ONLY}


def test_a_small_sequence_without_clustered_rows_joins_nothing():
    rows = [_row(f"s{n}", "Breedlove, D. E.", 33000 + n, D) for n in range(5)]
    merge = _merge(rows, {"other": 1})
    assert merge.assignments.empty
    assert merge.sequences["disposition"].tolist() == [DISPOSITION_BELOW_FLOOR]


def test_a_sequence_wholly_in_a_cluster_adds_nothing():
    rows = [_row(f"r{n}", "Thomas, J. H.", n, D, lat=37.0, lng=-122.0) for n in range(1, 6)]
    merge = _merge(rows, {f"r{n}": 5 for n in range(1, 6)})
    assert merge.assignments.empty
    assert merge.sequences["disposition"].tolist() == [DISPOSITION_IN_CLUSTER]


def test_a_spread_cut_sequence_attaches_nothing():
    rows = [
        _row("w1", "Far, A.", 1, D, lat=0, lng=0),
        _row("w2", "Far, A.", 2, D, lat=0, lng=0),
        _row("w3", "Far, A.", 3, D, lat=0, lng=10),
        _row("w4", "Far, A.", 4, D, lat=0, lng=10),
        _row("w5", "Far, A.", 5, D),
    ]
    merge = _merge(rows, {"w1": 1, "w2": 1})
    assert merge.assignments.empty
    assert merge.sequences["disposition"].tolist() == [DISPOSITION_SPREAD]


def test_attached_counts_by_coordinate_source():
    rows = [_row(f"r{n}", "Thomas, J. H.", n, D, lat=37.0, lng=-122.0) for n in range(1, 4)]
    rows.append(_row("e4", "Thomas, J. H.", 4, D, lat=37.1, lng=-122.1) | {"coordinate_source": "engine"})
    rows.append(_row("n5", "Thomas, J. H.", 5, D))
    rows.append(_row("s6", "Thomas, J. H.", 6, D, lat=37.2, lng=-122.2))  # Specify, small cluster dropped
    merge = _merge(rows, {f"r{n}": 2 for n in range(1, 4)})
    assert merge.attached_by_source() == {"specify": 1, "engine": 1, "none": 1}
