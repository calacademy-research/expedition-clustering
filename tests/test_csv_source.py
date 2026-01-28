"""Tests for CSV data source functionality."""

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest

# Direct import to avoid cartopy dependency from __init__.py
_csv_source_path = Path(__file__).parent.parent / "expedition_clustering" / "csv_source.py"
_spec = importlib.util.spec_from_file_location("csv_source", _csv_source_path)
_csv_source = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_csv_source)

# Import functions from the loaded module
_build_datetime = _csv_source._build_datetime
_build_fullname = _csv_source._build_fullname
list_available_collections = _csv_source.list_available_collections
load_collection_csv = _csv_source.load_collection_csv
load_csv_data = _csv_source.load_csv_data
transform_csv_to_pipeline_format = _csv_source.transform_csv_to_pipeline_format

# Test data directory (assumes running from project root)
INCOMING_DATA_DIR = Path("/Users/joe/collections_explorer/incoming_data")
BOTANY_DIR = INCOMING_DATA_DIR / "botany"


class TestListAvailableCollections:
    def test_lists_collections(self):
        """Test that we can list available collections."""
        if not INCOMING_DATA_DIR.exists():
            pytest.skip("incoming_data directory not available")

        collections = list_available_collections(INCOMING_DATA_DIR)
        assert isinstance(collections, list)
        assert len(collections) > 0
        assert "botany" in collections

    def test_collections_are_sorted(self):
        """Test that collections are returned sorted."""
        if not INCOMING_DATA_DIR.exists():
            pytest.skip("incoming_data directory not available")

        collections = list_available_collections(INCOMING_DATA_DIR)
        assert collections == sorted(collections)


class TestLoadCsvData:
    def test_loads_csv(self):
        """Test basic CSV loading."""
        if not BOTANY_DIR.exists():
            pytest.skip("botany data not available")

        df = load_csv_data(BOTANY_DIR / "PortalData.csv", limit=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "spid" in df.columns
        assert "catalogNumber" in df.columns
        assert "latitude1" in df.columns

    def test_limit_parameter(self):
        """Test that limit parameter works."""
        if not BOTANY_DIR.exists():
            pytest.skip("botany data not available")

        df5 = load_csv_data(BOTANY_DIR / "PortalData.csv", limit=5)
        df10 = load_csv_data(BOTANY_DIR / "PortalData.csv", limit=10)

        assert len(df5) == 5
        assert len(df10) == 10


class TestBuildDatetime:
    def test_builds_valid_date(self):
        """Test building datetime from year/month/day columns."""
        df = pd.DataFrame({
            "year": [2020, 2021, 2022],
            "month": [1, 6, 12],
            "day": [15, 20, 31],
        })

        result = _build_datetime(df, "year", "month", "day")

        assert pd.notna(result[0])
        assert result[0].year == 2020
        assert result[0].month == 1
        assert result[0].day == 15

    def test_handles_missing_year(self):
        """Test that missing year results in NaT."""
        df = pd.DataFrame({
            "year": [2020, None, 2022],
            "month": [1, 6, 12],
            "day": [15, 20, 31],
        })

        result = _build_datetime(df, "year", "month", "day")

        assert pd.notna(result[0])
        assert pd.isna(result[1])
        assert pd.notna(result[2])

    def test_handles_missing_month_day(self):
        """Test that missing month/day defaults to 1."""
        df = pd.DataFrame({
            "year": [2020],
            "month": [None],
            "day": [None],
        })

        result = _build_datetime(df, "year", "month", "day")

        assert pd.notna(result[0])
        assert result[0].month == 1
        assert result[0].day == 1


class TestBuildFullname:
    def test_builds_fullname(self):
        """Test building full geographic name."""
        df = pd.DataFrame({
            "Continent": ["North America"],
            "Country": ["United States"],
            "State": ["California"],
            "County": ["San Francisco"],
        })

        result = _build_fullname(df)

        assert "North America" in result[0]
        assert "United States" in result[0]
        assert "California" in result[0]
        assert "San Francisco" in result[0]

    def test_handles_missing_parts(self):
        """Test handling missing geographic parts."""
        df = pd.DataFrame({
            "Continent": ["Asia"],
            "Country": [None],
            "State": ["Yunnan"],
            "County": [None],
        })

        result = _build_fullname(df)

        assert "Asia" in result[0]
        assert "Yunnan" in result[0]


class TestTransformCsvToPipelineFormat:
    def test_transforms_data(self):
        """Test that transformation produces expected columns."""
        if not BOTANY_DIR.exists():
            pytest.skip("botany data not available")

        raw_df = load_csv_data(BOTANY_DIR / "PortalData.csv", limit=10)
        result = transform_csv_to_pipeline_format(raw_df)

        # Check required columns exist
        required_cols = [
            "collectionobjectid",
            "collectingeventid",
            "startdate",
            "latitude1",
            "longitude1",
            "localityname",
        ]
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_ids_are_numeric(self):
        """Test that IDs are numeric."""
        if not BOTANY_DIR.exists():
            pytest.skip("botany data not available")

        raw_df = load_csv_data(BOTANY_DIR / "PortalData.csv", limit=10)
        result = transform_csv_to_pipeline_format(raw_df)

        assert pd.api.types.is_integer_dtype(result["collectionobjectid"])

    def test_dates_are_datetime(self):
        """Test that dates are datetime type."""
        if not BOTANY_DIR.exists():
            pytest.skip("botany data not available")

        raw_df = load_csv_data(BOTANY_DIR / "PortalData.csv", limit=10)
        result = transform_csv_to_pipeline_format(raw_df)

        assert pd.api.types.is_datetime64_any_dtype(result["startdate"])


class TestLoadCollectionCsv:
    def test_loads_from_directory(self):
        """Test loading from collection directory."""
        if not BOTANY_DIR.exists():
            pytest.skip("botany data not available")

        df = load_collection_csv(BOTANY_DIR, limit=10)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "collectionobjectid" in df.columns
        assert "startdate" in df.columns

    def test_loads_from_csv_path(self):
        """Test loading directly from CSV file path."""
        if not BOTANY_DIR.exists():
            pytest.skip("botany data not available")

        csv_path = BOTANY_DIR / "PortalData.csv"
        df = load_collection_csv(csv_path, limit=10)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10

    def test_raises_for_missing_file(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            load_collection_csv("/nonexistent/path/to/data")

    def test_raises_for_missing_csv_in_directory(self):
        """Test that FileNotFoundError is raised when directory lacks CSV."""
        with pytest.raises(FileNotFoundError):
            load_collection_csv("/tmp")  # Directory exists but no PortalData.csv
