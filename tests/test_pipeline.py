import pandas as pd
import pytest

from expedition_clustering.pipeline import (
    MergeExpeditions,
    Preprocessor,
    ValidateSpatiotemporalConnectivity,
    create_pipeline,
    haversine_km,
)


def test_preprocessor_drops_invalid_and_duplicate_rows():
    input_df = pd.DataFrame(
        {
            "collectingeventid": [1, 1, 2, 3, 4, 5],
            "latitude1": [0.0, 0.0, 95.0, 10.0, -20.0, 12.0],
            "longitude1": [0.0, 0.0, 50.0, 190.0, 200.0, 12.0],
            "startdate": [
                "2020-01-01",  # valid
                "2020-01-01",  # duplicate collectingeventid
                "2020-01-01",  # invalid latitude
                "1700-01-01",  # before minimum year
                "2021-06-01",  # invalid longitude
                "2022-03-15",  # valid
            ],
        }
    )

    processed = Preprocessor().transform(input_df)

    assert len(processed) == 2  # duplicate removed and invalid rows dropped
    assert set(processed["collectingeventid"]) == {1, 5}
    assert processed["latitude1"].between(-90, 90).all()
    assert processed["longitude1"].between(-180, 180).all()


def test_create_pipeline_clusters_and_labels_output():
    sample_df = pd.DataFrame(
        {
            "collectingeventid": [1, 2, 3, 4],
            "latitude1": [0.0, 0.001, 10.0, 10.001],
            "longitude1": [0.0, 0.001, 20.0, 20.001],
            "startdate": ["2020-01-01", "2020-01-02", "2020-02-01", "2020-02-02"],
        }
    )

    pipeline = create_pipeline(e_dist=1, e_days=5)
    clustered = pipeline.fit_transform(sample_df)

    assert "spatiotemporal_cluster_id" in clustered.columns
    assert clustered["spatiotemporal_cluster_id"].nunique() == 2
    counts = clustered["spatiotemporal_cluster_id"].value_counts().sort_index().tolist()
    assert counts == [2, 2]


def test_validate_spatiotemporal_connectivity_raises_for_disconnected_clusters():
    disconnected = pd.DataFrame(
        {
            "collectingeventid": [1, 2],
            "latitude1": [0.0, 0.0],
            "longitude1": [0.0, 2.0],  # ~222 km apart
            "startdate": ["2020-01-01", "2020-01-01"],
            "spatial_cluster_id": [0, 0],
            "temporal_cluster_id": [0, 0],
            "spatiotemporal_cluster_id": [0, 0],
        }
    )

    validator = ValidateSpatiotemporalConnectivity(e_dist=1, e_days=1)
    with pytest.raises(ValueError, match="spatially disconnected"):
        validator.transform(disconnected)


def _make_merge_df(records):
    """Helper to build a DataFrame suitable for MergeExpeditions."""
    df = pd.DataFrame(records)
    df["startdate"] = pd.to_datetime(df["startdate"])
    for col in ("spatial_cluster_id", "temporal_cluster_id"):
        if col not in df.columns:
            df[col] = df["spatiotemporal_cluster_id"]
    if "collectingeventid" not in df.columns:
        df["collectingeventid"] = range(len(df))
    return df


def test_merge_combines_nearby_clusters():
    """Two clusters close in space/time with same collector should merge."""
    df = _make_merge_df([
        {"latitude1": 0.0, "longitude1": 0.0, "startdate": "2020-01-01",
         "collectors": "Smith, J.", "spatiotemporal_cluster_id": 0},
        {"latitude1": 0.001, "longitude1": 0.001, "startdate": "2020-01-02",
         "collectors": "Smith, J.", "spatiotemporal_cluster_id": 0},
        {"latitude1": 0.01, "longitude1": 0.01, "startdate": "2020-01-04",
         "collectors": "Smith, J.", "spatiotemporal_cluster_id": 1},
        {"latitude1": 0.011, "longitude1": 0.011, "startdate": "2020-01-05",
         "collectors": "Smith, J.", "spatiotemporal_cluster_id": 1},
    ])
    merger = MergeExpeditions(
        max_merge_gap_days=5,
        max_merge_distance_km=30,
        merge_threshold=0.3,
        max_gap_days=5,
    )
    result = merger.transform(df)
    assert result["spatiotemporal_cluster_id"].nunique() == 1


def test_merge_does_not_merge_distant_clusters():
    """Two clusters far apart should not merge despite same collector."""
    df = _make_merge_df([
        {"latitude1": 0.0, "longitude1": 0.0, "startdate": "2020-01-01",
         "collectors": "Smith, J.", "spatiotemporal_cluster_id": 0},
        {"latitude1": 0.001, "longitude1": 0.001, "startdate": "2020-01-02",
         "collectors": "Smith, J.", "spatiotemporal_cluster_id": 0},
        {"latitude1": 10.0, "longitude1": 10.0, "startdate": "2020-01-04",
         "collectors": "Smith, J.", "spatiotemporal_cluster_id": 1},
        {"latitude1": 10.001, "longitude1": 10.001, "startdate": "2020-01-05",
         "collectors": "Smith, J.", "spatiotemporal_cluster_id": 1},
    ])
    merger = MergeExpeditions(
        max_merge_gap_days=5,
        max_merge_distance_km=30,
        merge_threshold=0.3,
        max_gap_days=5,
    )
    result = merger.transform(df)
    # ~1570 km apart, should not merge
    assert result["spatiotemporal_cluster_id"].nunique() == 2


def test_merge_respects_threshold():
    """Clusters with different collectors should score low and not merge at high threshold."""
    df = _make_merge_df([
        {"latitude1": 0.0, "longitude1": 0.0, "startdate": "2020-01-01",
         "collectors": "Smith, J.", "spatiotemporal_cluster_id": 0},
        {"latitude1": 0.001, "longitude1": 0.001, "startdate": "2020-01-02",
         "collectors": "Smith, J.", "spatiotemporal_cluster_id": 0},
        {"latitude1": 0.01, "longitude1": 0.01, "startdate": "2020-01-04",
         "collectors": "Jones, B.", "spatiotemporal_cluster_id": 1},
        {"latitude1": 0.011, "longitude1": 0.011, "startdate": "2020-01-05",
         "collectors": "Jones, B.", "spatiotemporal_cluster_id": 1},
    ])
    merger = MergeExpeditions(
        max_merge_gap_days=5,
        max_merge_distance_km=30,
        merge_threshold=0.9,  # Very high threshold
        max_gap_days=5,
    )
    result = merger.transform(df)
    # Different collectors + high threshold = no merge
    assert result["spatiotemporal_cluster_id"].nunique() == 2


def test_merge_validates_internal_gaps():
    """Merge should be rejected if it creates large internal gaps."""
    df = _make_merge_df([
        {"latitude1": 0.0, "longitude1": 0.0, "startdate": "2020-01-01",
         "collectors": "Smith, J.", "spatiotemporal_cluster_id": 0},
        {"latitude1": 0.001, "longitude1": 0.001, "startdate": "2020-01-02",
         "collectors": "Smith, J.", "spatiotemporal_cluster_id": 0},
        # 20-day gap within cluster 0
        {"latitude1": 0.002, "longitude1": 0.002, "startdate": "2020-01-22",
         "collectors": "Smith, J.", "spatiotemporal_cluster_id": 0},
        {"latitude1": 0.01, "longitude1": 0.01, "startdate": "2020-01-24",
         "collectors": "Smith, J.", "spatiotemporal_cluster_id": 1},
    ])
    merger = MergeExpeditions(
        max_merge_gap_days=5,
        max_merge_distance_km=30,
        merge_threshold=0.3,
        max_gap_days=3,  # Internal gap must be <= 3 days
    )
    result = merger.transform(df)
    # Cluster 0 already has a 20-day internal gap; merge would keep it, so rejected
    assert result["spatiotemporal_cluster_id"].nunique() == 2


def test_merge_does_not_cross_year_boundary():
    """Expeditions in different years should not be merged."""
    df = _make_merge_df([
        {"latitude1": 0.0, "longitude1": 0.0, "startdate": "2019-12-30",
         "collectors": "Smith, J.", "spatiotemporal_cluster_id": 0},
        {"latitude1": 0.001, "longitude1": 0.001, "startdate": "2019-12-31",
         "collectors": "Smith, J.", "spatiotemporal_cluster_id": 0},
        {"latitude1": 0.01, "longitude1": 0.01, "startdate": "2020-01-02",
         "collectors": "Smith, J.", "spatiotemporal_cluster_id": 1},
        {"latitude1": 0.011, "longitude1": 0.011, "startdate": "2020-01-03",
         "collectors": "Smith, J.", "spatiotemporal_cluster_id": 1},
    ])
    merger = MergeExpeditions(
        max_merge_gap_days=5,
        max_merge_distance_km=30,
        merge_threshold=0.3,
        max_gap_days=5,
    )
    result = merger.transform(df)
    # Different years -> not considered as candidates
    assert result["spatiotemporal_cluster_id"].nunique() == 2


def test_create_pipeline_with_merge():
    """Full pipeline with merge stage enabled should work end-to-end."""
    sample_df = pd.DataFrame({
        "collectingeventid": [1, 2, 3, 4],
        "latitude1": [0.0, 0.001, 10.0, 10.001],
        "longitude1": [0.0, 0.001, 20.0, 20.001],
        "startdate": ["2020-01-01", "2020-01-02", "2020-02-01", "2020-02-02"],
    })
    pipeline = create_pipeline(e_dist=1, e_days=5, enable_merge=True)
    clustered = pipeline.fit_transform(sample_df)
    assert "spatiotemporal_cluster_id" in clustered.columns
    # Two distant clusters should remain separate even with merge
    assert clustered["spatiotemporal_cluster_id"].nunique() == 2


def test_create_pipeline_no_merge():
    """Pipeline with merge disabled should still produce valid output."""
    sample_df = pd.DataFrame({
        "collectingeventid": [1, 2, 3, 4],
        "latitude1": [0.0, 0.001, 10.0, 10.001],
        "longitude1": [0.0, 0.001, 20.0, 20.001],
        "startdate": ["2020-01-01", "2020-01-02", "2020-02-01", "2020-02-02"],
    })
    pipeline = create_pipeline(e_dist=1, e_days=5, enable_merge=False)
    clustered = pipeline.fit_transform(sample_df)
    assert "spatiotemporal_cluster_id" in clustered.columns
    assert clustered["spatiotemporal_cluster_id"].nunique() == 2


def test_haversine_km():
    """Sanity check haversine distance calculation."""
    # Same point
    assert haversine_km(0, 0, 0, 0) == 0.0
    # ~111km for 1 degree of latitude
    dist = haversine_km(0, 0, 1, 0)
    assert 110 < dist < 112
