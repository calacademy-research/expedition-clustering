"""
Bug #426 (CAS Lens): a specimen is keyed on its qualified catalog identity.

transform_csv_to_pipeline_format made every row's collectingeventid its bare
numeric catalog number, and the Preprocessor drops duplicate
collectingeventids. Herp's PortalData.csv holds three catalogs (HERP, SUA,
SUR) whose numbers overlap, so every SUA/SUR row whose number a HERP row also
used was dropped before clustering (46,169 rows on 2026-10-04; ich lost
69,708 the same way). The key is now the row's spid
(urn:catalog:CAS:SUR:6788), which the Lens also uses to map rows back.

Requirement: EXP-CLUSTER-ID-1 (cas-lens docs/requirements/expedition-requirements.md).
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from expedition_clustering.pipeline import create_pipeline

_csv_source_path = Path(__file__).parent.parent / "expedition_clustering" / "csv_source.py"
_spec = importlib.util.spec_from_file_location("csv_source", _csv_source_path)
_csv_source = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_csv_source)
transform_csv_to_pipeline_format = _csv_source.transform_csv_to_pipeline_format


def _herp_rows(n: int, catalog: str, number_start: int, lat: float, lng: float, day: str) -> list[dict]:
    """Build n herp-shaped PortalData rows of one catalog at one place and day."""
    return [
        {
            "spid": f"urn:catalog:CAS:{catalog}:{number_start + i}",
            "catalogNumber": str(number_start + i),
            "latitude1": lat,
            "longitude1": lng,
            "startDate": day,
            "collectors": "H. Heath",
        }
        for i in range(n)
    ]


def test_overlapping_catalog_numbers_are_different_specimens():
    # HERP 6788..6792 in California; SUR 6788..6792 in Fortaleza, Brazil.
    raw = pd.DataFrame(
        _herp_rows(5, "HERP", 6788, 34.878, -118.296, "1905-04-24")
        + _herp_rows(5, "SUR", 6788, -3.728, -38.526, "1911-06-02")
    )
    df = transform_csv_to_pipeline_format(raw)
    assert df["collectingeventid"].is_unique
    assert set(df["collectingeventid"]) == set(raw["spid"])

    clustered = create_pipeline(e_dist=10, e_days=3).fit_transform(df)
    assert sorted(clustered["spid"]) == sorted(raw["spid"])
    sur = clustered[clustered["spid"].str.contains(":SUR:")]
    assert len(sur) == 5
    assert len(set(sur["spatiotemporal_cluster_id"])) == 1


def test_a_row_without_spid_is_refused():
    raw = pd.DataFrame(_herp_rows(2, "SUR", 1, -3.7, -38.5, "1911-06-02"))
    raw.loc[1, "spid"] = ""
    with pytest.raises(ValueError, match="spid"):
        transform_csv_to_pipeline_format(raw)


def test_a_csv_without_the_spid_column_is_refused():
    raw = pd.DataFrame(_herp_rows(2, "SUR", 1, -3.7, -38.5, "1911-06-02")).drop(columns=["spid"])
    with pytest.raises(ValueError, match="spid"):
        transform_csv_to_pipeline_format(raw)
