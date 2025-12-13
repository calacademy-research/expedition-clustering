import pandas as pd
import pytest

from expedition_clustering.pipeline import (
    Preprocessor,
    ValidateSpatiotemporalConnectivity,
    create_pipeline,
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
