import pandas as pd
import pytest

from expedition_clustering.preprocessing import (
    build_clean_dataframe,
    clean_for_clustering,
    merge_core_tables,
)


def test_merge_core_tables_filters_unrelated_rows():
    collectingevent = pd.DataFrame(
        {
            "CollectingEventID": [1, 2],
            "StartDate": ["2020-01-01", "2020-01-02"],
            "EndDate": ["2020-01-01", "2020-01-03"],
            "Remarks": ["a", "b"],
            "LocalityID": [10, 20],
        }
    )
    collectionobject = pd.DataFrame(
        {
            "CollectionObjectID": [100, 200, 300],
            "CollectingEventID": [1, 2, 99],  # 99 should be filtered out
            "Text1": ["x", "y", "z"],
        }
    )
    locality = pd.DataFrame(
        {
            "LocalityID": [10, 20, 30],  # 30 should be filtered out
            "Latitude1": [1.0, 2.0, 3.0],
            "Longitude1": [4.0, 5.0, 6.0],
            "GeographyID": [1000, 2000, 3000],
        }
    )
    geography = pd.DataFrame(
        {
            "GeographyID": [1000, 2000, 4000],  # 4000 should be filtered out
            "CentroidLat": [10.0, 20.0, 40.0],
            "CentroidLon": [15.0, 25.0, 45.0],
        }
    )

    merged = merge_core_tables(
        collectingevent,
        collectionobject,
        locality,
        geography,
        filter_related=True,
    )

    # Two events with matching collection objects should survive; unrelated IDs are dropped.
    assert len(merged) == 2
    assert set(merged["CollectingEventID"]) == {1, 2}
    assert merged["GeographyID"].notna().all()


def test_clean_for_clustering_prefers_precise_coordinates_and_drops_missing():
    merged = pd.DataFrame(
        {
            "CollectingEventID": [1, 2, 3],
            "StartDate": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "EndDate": [None, None, None],
            "Latitude1": [1.0, None, None],
            "Longitude1": [2.0, None, None],
            "CentroidLat": [10.0, 11.0, None],
            "CentroidLon": [20.0, 21.0, None],
            "GeographyID": [100, 101, 102],
            "LocalityID": [10, 11, 12],
        }
    )

    clean = clean_for_clustering(merged, drop_missing_spatial=True, drop_missing_start_date=True)

    # Row 1 uses precise coordinates, row 2 gains centroid coordinates, row 3 drops for missing date/coords.
    assert len(clean) == 2
    assert clean.loc[0, "latitude1"] == 1.0
    assert clean.loc[1, "latitude1"] == 11.0
    assert clean.columns.str.islower().all()


def test_build_clean_dataframe_requires_core_tables():
    tables = {
        "collectingevent": pd.DataFrame(),
        "collectionobject": pd.DataFrame(),
        # missing locality and geography
    }
    with pytest.raises(KeyError):
        build_clean_dataframe(tables)
