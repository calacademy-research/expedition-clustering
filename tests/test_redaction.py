import pandas as pd

from expedition_clustering.redaction import (
    redact_clustered_dataframe,
    verify_redaction,
    verify_redaction_drop,
)


def test_redact_clustered_dataframe_masks_locality_and_coordinates():
    sample_df = pd.DataFrame(
        {
            "collectionobjectid": [1, 2, 3],
            "localityname": ["Loc A", "Loc B", "Loc C"],
            "namedplace": ["Place A", "Place B", "Place C"],
            "latitude1": [1.0, 2.0, 3.0],
            "longitude1": [10.0, 20.0, 30.0],
            "centroidlat": [1.1, 2.1, 3.1],
            "centroidlon": [10.1, 20.1, 30.1],
            "text1": ["t1", "t2", "t3"],
            "remarks": ["r1", "r2", "r3"],
        }
    )
    flags = pd.DataFrame(
        {
            "collectionobjectid": [2],
            "is_redacted": [1],
        }
    )

    redacted_df, redacted_rows = redact_clustered_dataframe(sample_df, flags)

    assert redacted_rows == 1
    assert "is_redacted" not in redacted_df.columns

    redacted_row = redacted_df[redacted_df["collectionobjectid"] == 2].iloc[0]
    assert redacted_row["localityname"] == "*"
    assert redacted_row["namedplace"] == "*"
    assert pd.isna(redacted_row["latitude1"])
    assert pd.isna(redacted_row["longitude1"])
    assert pd.isna(redacted_row["centroidlat"])
    assert pd.isna(redacted_row["centroidlon"])
    assert redacted_row["text1"] == "t2"
    assert redacted_row["remarks"] == "r2"

    unredacted_row = redacted_df[redacted_df["collectionobjectid"] == 1].iloc[0]
    assert unredacted_row["localityname"] == "Loc A"
    assert unredacted_row["latitude1"] == 1.0


def test_redact_clustered_dataframe_drops_redacted_rows():
    sample_df = pd.DataFrame(
        {
            "collectionobjectid": [1, 2, 3],
            "localityname": ["Loc A", "Loc B", "Loc C"],
            "latitude1": [1.0, 2.0, 3.0],
            "longitude1": [10.0, 20.0, 30.0],
        }
    )
    flags = pd.DataFrame(
        {
            "collectionobjectid": [2],
            "is_redacted": [1],
        }
    )

    redacted_df, redacted_rows = redact_clustered_dataframe(sample_df, flags, drop_redacted=True)

    assert redacted_rows == 1
    assert len(redacted_df) == 2
    assert 2 not in set(redacted_df["collectionobjectid"])


def test_verify_redaction_detects_unmasked_fields():
    sample_df = pd.DataFrame(
        {
            "collectionobjectid": [10, 11],
            "localityname": ["*", "Loc B"],
            "latitude1": [pd.NA, 5.0],
            "longitude1": [pd.NA, 6.0],
        }
    )
    flags = pd.DataFrame(
        {
            "collectionobjectid": [10, 11],
            "is_redacted": [1, 1],
        }
    )

    result = verify_redaction(sample_df, flags)

    assert result.flagged_rows == 2
    assert result.bad_locality_rows == 1
    assert result.bad_coordinate_rows == 1
    assert not result.ok


def test_verify_redaction_drop_fails_when_flagged_rows_present():
    sample_df = pd.DataFrame(
        {
            "collectionobjectid": [10, 11],
            "localityname": ["Loc A", "Loc B"],
            "latitude1": [1.0, 2.0],
            "longitude1": [10.0, 20.0],
        }
    )
    flags = pd.DataFrame(
        {
            "collectionobjectid": [11],
            "is_redacted": [1],
        }
    )

    result = verify_redaction_drop(sample_df, flags)

    assert result.flagged_rows == 1
    assert not result.ok
