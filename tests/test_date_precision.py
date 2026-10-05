"""
Bug #427 (CAS Lens): a collecting date keeps its precision through clustering.

A year-only date ("1911") was padded to 1 January and clustered as that day,
so 94 herp specimens Heath collected somewhere in 1911 became one expedition
"1911-01-01, 1 day", and a year-only record could chain with specimens really
collected on 1-3 January. Every row now carries date_precision (1 day,
2 month, 3 year — Specify's scale): the padded date stays the sort key, but
rows of different precision never share a cluster or merge, so year-only
records group only with other year-only records of the same year and place.
The precision is written to clustered_expeditions.csv for the Lens.

Requirement: EXP-DATE-2 (cas-lens docs/requirements/expedition-requirements.md).
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from expedition_clustering.pipeline import MergeExpeditions, Preprocessor, create_pipeline

_csv_source_path = Path(__file__).parent.parent / "expedition_clustering" / "csv_source.py"
_spec = importlib.util.spec_from_file_location("csv_source", _csv_source_path)
_csv_source = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_csv_source)
transform_csv_to_pipeline_format = _csv_source.transform_csv_to_pipeline_format


def _components(rows):
    """Build PortalData rows in the year / month / day layout (botany, herp)."""
    return pd.DataFrame(
        [
            {
                "spid": f"urn:catalog:CAS:SUR:{i}",
                "catalogNumber": str(i),
                "latitude1": -3.728,
                "longitude1": -38.526,
                "startDate": year,
                "ce_startDate": month,
                "ce_startDate1": day,
                "collectors": "H. Heath",
            }
            for i, (year, month, day) in enumerate(rows, start=1)
        ]
    )


def test_components_layout_records_precision():
    df = transform_csv_to_pipeline_format(_components([("1911", "", ""), ("1911", "4", ""), ("1911", "4", "29")]))
    assert df["date_precision"].tolist() == [3, 2, 1]
    assert df["startdate"].dt.strftime("%Y-%m-%d").tolist() == ["1911-01-01", "1911-04-01", "1911-04-29"]


def test_iso_layout_records_precision():
    raw = pd.DataFrame(
        {
            "spid": ["a:1", "a:2", "a:3"],
            "latitude1": [1.0, 1.0, 1.0],
            "longitude1": [1.0, 1.0, 1.0],
            "startDate": ["2001-03-04", "1911", "1911-04"],
        }
    )
    df = transform_csv_to_pipeline_format(raw)
    assert df["date_precision"].tolist() == [1, 3, 2]
    assert df["startdate"].dt.strftime("%Y-%m-%d").tolist() == ["2001-03-04", "1911-01-01", "1911-04-01"]


def test_year_only_specimens_do_not_join_a_day_on_new_years():
    # 6 year-only "1911" specimens and 6 collected on 2 January 1911, same place.
    raw = _components([("1911", "", "")] * 6 + [("1911", "1", "2")] * 6)
    df = transform_csv_to_pipeline_format(raw)
    clustered = create_pipeline(e_dist=10, e_days=3).fit_transform(df)

    by_precision = clustered.groupby("date_precision")["spatiotemporal_cluster_id"].agg(lambda s: set(s))
    assert len(by_precision[3]) == 1
    assert len(by_precision[1]) == 1
    assert by_precision[3].isdisjoint(by_precision[1])
    assert "date_precision" in clustered.columns


def test_year_only_specimens_of_two_years_are_two_expeditions():
    raw = _components([("1911", "", "")] * 4 + [("1912", "", "")] * 4)
    clustered = create_pipeline(e_dist=10, e_days=3).fit_transform(transform_csv_to_pipeline_format(raw))
    assert clustered["spatiotemporal_cluster_id"].nunique() == 2


def test_merge_never_joins_two_precisions():
    df = pd.DataFrame(
        {
            "collectingeventid": [1, 2, 3, 4],
            "latitude1": [0.0, 0.0, 0.01, 0.01],
            "longitude1": [0.0, 0.0, 0.01, 0.01],
            "startdate": pd.to_datetime(["1911-01-01", "1911-01-01", "1911-01-01", "1911-01-01"]),
            "date_precision": [3, 3, 1, 1],
            "collectors": ["H. Heath"] * 4,
            "spatiotemporal_cluster_id": [10, 10, 20, 20],
        }
    )
    merged = MergeExpeditions(max_merge_gap_days=3, max_merge_distance_km=30, merge_threshold=0.1).transform(df)
    assert merged["spatiotemporal_cluster_id"].nunique() == 2


def test_preprocessor_requires_date_precision():
    df = pd.DataFrame({"collectingeventid": [1], "latitude1": [0.0], "longitude1": [0.0], "startdate": ["1911-01-01"]})
    with pytest.raises(ValueError, match="date_precision"):
        Preprocessor().transform(df)


def test_preprocessor_refuses_an_unknown_precision():
    df = pd.DataFrame(
        {
            "collectingeventid": [1, 2],
            "latitude1": [0.0, 0.0],
            "longitude1": [0.0, 0.0],
            "startdate": ["1911-01-01", "1911-01-01"],
            "date_precision": [3, None],
        }
    )
    with pytest.raises(ValueError, match="date_precision"):
        Preprocessor().transform(df)
