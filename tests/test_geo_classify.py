"""
Tests for geo_classify module.

Tests geographic classification functions for:
- Named regions (Galapagos, Micronesia, etc.)
- Islands vs mainland
- Köppen climate zones
- Biogeographic realms
- Latitude bands
- Elevation bands
- Biomes (taiga, tundra, desert, rainforest, etc.)
- Mountain ranges
- Coral reefs and mangroves
"""

import numpy as np
import pandas as pd
import pytest

from expedition_clustering.geo_classify import (
    # Main functions
    classify_coordinates,
    classify_dataframe,
    classify_expeditions,
    # Elevation functions
    get_elevation_band,
    is_high_altitude,
    is_above_treeline,
    _get_treeline_elevation,
    # Climate functions
    get_koppen_zone,
    get_latitude_band,
    # Biome functions
    get_biome,
    get_biogeographic_realm,
    # Feature detection
    is_island_location,
    get_mountain_range,
    is_coral_reef_region,
    is_mangrove_region,
    get_specific_desert,
    # Helper functions
    _point_in_bounds,
    _point_in_region_bounds,
    _is_desert_region,
    _is_steppe_region,
    _is_mediterranean_region,
    # Constants
    NAMED_REGIONS,
    ISLAND_GROUPS,
    DESERT_REGIONS,
    RAINFOREST_REGIONS,
    TAIGA_REGIONS,
    TUNDRA_REGIONS,
    MOUNTAIN_RANGES,
)


# ============================================================================
# Test Named Regions
# ============================================================================

class TestNamedRegions:
    """Tests for named region detection."""

    def test_galapagos_center(self):
        """Galapagos Islands center point."""
        result = classify_coordinates(-0.5, -90.5)
        assert result["region"] == "Galapagos"
        assert result["region_group"] == "Pacific Islands"
        assert result["is_island"] is True

    def test_galapagos_edge(self):
        """Galapagos Islands edge point."""
        result = classify_coordinates(-1.5, -91.5)
        assert result["region"] == "Galapagos"

    def test_hawaii(self):
        """Hawaii detection."""
        # Honolulu area
        result = classify_coordinates(21.3, -157.8)
        assert result["region"] == "Hawaii"
        assert result["region_group"] == "Pacific Islands"
        assert result["is_island"] is True

    def test_micronesia(self):
        """Micronesia detection."""
        # Pohnpei
        result = classify_coordinates(6.9, 158.2)
        assert result["region"] == "Micronesia"
        assert result["is_island"] is True

    def test_caribbean(self):
        """Caribbean detection."""
        # Jamaica
        result = classify_coordinates(18.1, -77.3)
        assert result["region"] == "Caribbean"
        assert result["region_group"] == "Caribbean Islands"
        assert result["is_island"] is True

    def test_madagascar(self):
        """Madagascar detection."""
        result = classify_coordinates(-18.9, 47.5)
        assert result["region"] == "Madagascar"
        assert result["region_group"] == "Indian Ocean Islands"

    def test_amazon_basin(self):
        """Amazon Basin detection."""
        # Manaus, Brazil
        result = classify_coordinates(-3.1, -60.0)
        assert result["region"] == "Amazon Basin"
        assert result["is_island"] is False

    def test_himalayas(self):
        """Himalayas detection."""
        # Everest area
        result = classify_coordinates(28.0, 86.9)
        assert result["region"] == "Himalayas"

    def test_sahara(self):
        """Sahara detection."""
        # Central Sahara
        result = classify_coordinates(25.0, 10.0)
        assert result["region"] == "Sahara"

    def test_arctic(self):
        """Arctic detection."""
        # North Pole area
        result = classify_coordinates(85.0, 0.0)
        assert result["region"] == "Arctic"
        assert result["latitude_band"] == "polar"

    def test_antarctic(self):
        """Antarctic detection."""
        result = classify_coordinates(-75.0, 0.0)
        assert result["region"] == "Antarctic"
        assert result["latitude_band"] == "polar"

    def test_no_named_region(self):
        """Point not in any named region."""
        # Middle of nowhere in North America
        result = classify_coordinates(45.0, -90.0)
        # Might be in a general region or None
        # Just verify it doesn't crash


# ============================================================================
# Test Elevation Classification
# ============================================================================

class TestElevationBands:
    """Tests for elevation band classification."""

    def test_coastal(self):
        """Coastal elevation (0-50m)."""
        assert get_elevation_band(0) == "coastal"
        assert get_elevation_band(25) == "coastal"
        assert get_elevation_band(49) == "coastal"

    def test_lowland(self):
        """Lowland elevation (50-500m)."""
        assert get_elevation_band(50) == "lowland"
        assert get_elevation_band(250) == "lowland"
        assert get_elevation_band(499) == "lowland"

    def test_submontane(self):
        """Submontane elevation (500-1500m)."""
        assert get_elevation_band(500) == "submontane"
        assert get_elevation_band(1000) == "submontane"
        assert get_elevation_band(1499) == "submontane"

    def test_montane(self):
        """Montane elevation (1500-2500m)."""
        assert get_elevation_band(1500) == "montane"
        assert get_elevation_band(2000) == "montane"
        assert get_elevation_band(2499) == "montane"

    def test_upper_montane(self):
        """Upper montane elevation (2500-3500m)."""
        assert get_elevation_band(2500) == "upper_montane"
        assert get_elevation_band(3000) == "upper_montane"
        assert get_elevation_band(3499) == "upper_montane"

    def test_alpine(self):
        """Alpine elevation (3500-4500m)."""
        assert get_elevation_band(3500) == "alpine"
        assert get_elevation_band(4000) == "alpine"
        assert get_elevation_band(4499) == "alpine"

    def test_nival(self):
        """Nival elevation (>4500m)."""
        assert get_elevation_band(4500) == "nival"
        assert get_elevation_band(5000) == "nival"
        assert get_elevation_band(8848) == "nival"  # Everest

    def test_below_sea_level(self):
        """Below sea level locations."""
        assert get_elevation_band(-10) == "below_sea_level"
        assert get_elevation_band(-430) == "below_sea_level"  # Dead Sea

    def test_none_elevation(self):
        """None elevation returns unknown."""
        assert get_elevation_band(None) == "unknown"

    def test_nan_elevation(self):
        """NaN elevation returns unknown."""
        assert get_elevation_band(float("nan")) == "unknown"


class TestHighAltitude:
    """Tests for high altitude detection."""

    def test_high_altitude_true(self):
        """Elevations above 2500m are high altitude."""
        assert is_high_altitude(2500) is True
        assert is_high_altitude(3000) is True
        assert is_high_altitude(5000) is True

    def test_high_altitude_false(self):
        """Elevations below 2500m are not high altitude."""
        assert is_high_altitude(2499) is False
        assert is_high_altitude(1000) is False
        assert is_high_altitude(0) is False

    def test_custom_threshold(self):
        """Custom threshold works."""
        assert is_high_altitude(1500, threshold_m=1000) is True
        assert is_high_altitude(1500, threshold_m=2000) is False

    def test_none_elevation(self):
        """None elevation returns False."""
        assert is_high_altitude(None) is False


class TestTreeline:
    """Tests for treeline estimation."""

    def test_tropical_treeline(self):
        """Tropical regions have high treelines."""
        # Equatorial
        assert _get_treeline_elevation(5) == 4000
        # Tropical
        assert _get_treeline_elevation(20) == 3800

    def test_temperate_treeline(self):
        """Temperate regions have moderate treelines."""
        assert _get_treeline_elevation(40) == 2500
        assert _get_treeline_elevation(50) == 1800

    def test_subarctic_treeline(self):
        """Subarctic regions have low treelines."""
        assert _get_treeline_elevation(60) == 1000

    def test_arctic_treeline(self):
        """Arctic has no treeline (0m)."""
        assert _get_treeline_elevation(70) == 0
        assert _get_treeline_elevation(80) == 0

    def test_above_treeline_tropical(self):
        """Tropical point above treeline."""
        # 4500m at equator is above treeline (4000m)
        assert is_above_treeline(4500, 5) is True
        # 3500m at equator is below treeline
        assert is_above_treeline(3500, 5) is False

    def test_above_treeline_temperate(self):
        """Temperate point above treeline."""
        # 3000m at 40°N is above treeline (2500m)
        assert is_above_treeline(3000, 40) is True
        # 2000m at 40°N is below treeline
        assert is_above_treeline(2000, 40) is False

    def test_above_treeline_arctic(self):
        """Arctic - any positive elevation is above treeline."""
        assert is_above_treeline(100, 70) is True
        assert is_above_treeline(1, 70) is True


# ============================================================================
# Test Climate Classification
# ============================================================================

class TestKoppenZones:
    """Tests for Köppen climate zone classification."""

    def test_tropical(self):
        """Tropical climate detection."""
        # Amazon
        code, name = get_koppen_zone(-3, -60)
        assert name == "Tropical"

    def test_desert(self):
        """Desert climate detection."""
        # Sahara
        code, name = get_koppen_zone(25, 10)
        assert name == "Desert"
        assert code == "BW"

    def test_mediterranean(self):
        """Mediterranean climate detection."""
        # Southern California
        code, name = get_koppen_zone(34, -118)
        assert name == "Mediterranean"
        assert code == "Cs"

    def test_oceanic(self):
        """Oceanic climate detection."""
        # Paris (45-50° range)
        code, name = get_koppen_zone(48.8, 2.3)
        assert name == "Oceanic"
        assert code == "Cfb"

    def test_continental(self):
        """Continental climate detection."""
        # Chicago area
        code, name = get_koppen_zone(42, -88)
        assert name == "Continental"
        assert code == "D"

    def test_subarctic(self):
        """Subarctic climate detection."""
        # Fairbanks, Alaska area
        code, name = get_koppen_zone(55, -100)
        assert name == "Subarctic"
        assert code == "Dfc"

    def test_polar(self):
        """Polar climate detection."""
        code, name = get_koppen_zone(80, 0)
        assert name == "Polar"
        assert code == "E"


class TestLatitudeBands:
    """Tests for latitude band classification."""

    def test_equatorial(self):
        """Equatorial band (0-10°)."""
        assert get_latitude_band(0) == "equatorial"
        assert get_latitude_band(5) == "equatorial"
        assert get_latitude_band(-5) == "equatorial"
        assert get_latitude_band(9.9) == "equatorial"

    def test_tropical(self):
        """Tropical band (10-23.5°)."""
        assert get_latitude_band(15) == "tropical"
        assert get_latitude_band(-15) == "tropical"
        assert get_latitude_band(23) == "tropical"

    def test_subtropical(self):
        """Subtropical band (23.5-35°)."""
        assert get_latitude_band(30) == "subtropical"
        assert get_latitude_band(-30) == "subtropical"

    def test_temperate(self):
        """Temperate band (35-50°)."""
        assert get_latitude_band(40) == "temperate"
        assert get_latitude_band(-40) == "temperate"

    def test_subarctic_subantarctic(self):
        """Subarctic/Subantarctic bands (50-66.5°)."""
        assert get_latitude_band(55) == "subarctic"
        assert get_latitude_band(-55) == "subantarctic"

    def test_polar(self):
        """Polar bands (>66.5°)."""
        assert get_latitude_band(70) == "polar"
        assert get_latitude_band(-70) == "polar"
        assert get_latitude_band(90) == "polar"


# ============================================================================
# Test Biome Classification
# ============================================================================

class TestBiomes:
    """Tests for biome classification."""

    def test_tundra_arctic(self):
        """Arctic tundra detection."""
        assert get_biome(70, 0) == "tundra"
        assert get_biome(75, -100) == "tundra"

    def test_tundra_antarctic(self):
        """Antarctic tundra detection."""
        assert get_biome(-70, 0) == "tundra"

    def test_taiga_canada(self):
        """Canadian boreal forest detection."""
        assert get_biome(55, -100) == "taiga"

    def test_taiga_siberia(self):
        """Siberian taiga detection."""
        assert get_biome(60, 100) == "taiga"

    def test_taiga_scandinavia(self):
        """Scandinavian taiga detection."""
        assert get_biome(62, 15) == "taiga"

    def test_desert_sahara(self):
        """Sahara desert detection."""
        assert get_biome(25, 10) == "desert"

    def test_desert_gobi(self):
        """Gobi desert detection."""
        assert get_biome(42, 105) == "desert"

    def test_desert_australian(self):
        """Australian outback detection."""
        assert get_biome(-25, 135) == "desert"

    def test_desert_sonoran(self):
        """Sonoran desert detection."""
        assert get_biome(32, -112) == "desert"

    def test_tropical_rainforest_amazon(self):
        """Amazon rainforest detection."""
        assert get_biome(-3, -60) == "tropical_rainforest"

    def test_tropical_rainforest_congo(self):
        """Congo rainforest detection."""
        assert get_biome(0, 20) == "tropical_rainforest"

    def test_tropical_savanna_africa(self):
        """African savanna detection."""
        assert get_biome(-10, 30) == "tropical_savanna"

    def test_tropical_savanna_cerrado(self):
        """Brazilian cerrado detection."""
        # Use a point that's clearly in cerrado, not overlapping with rainforest regions
        # Cerrado: lat -24 to -5, lng -60 to -41
        # Atlantic_Forest: lat -30 to -5, lng -55 to -35
        # Choose lng west of -55 to avoid Atlantic Forest overlap
        assert get_biome(-20, -58) == "tropical_savanna"

    def test_temperate_grassland_great_plains(self):
        """Great Plains steppe detection."""
        assert get_biome(40, -100) == "temperate_grassland"

    def test_temperate_grassland_pampas(self):
        """Pampas detection."""
        assert get_biome(-35, -60) == "temperate_grassland"

    def test_mediterranean_california(self):
        """California chaparral detection."""
        assert get_biome(35, -120) == "mediterranean"

    def test_mediterranean_basin(self):
        """Mediterranean Basin detection."""
        assert get_biome(37, 15) == "mediterranean"

    def test_temperate_forest_eastern_us(self):
        """Eastern US temperate forest detection."""
        assert get_biome(40, -80) == "temperate_forest"

    def test_temperate_forest_europe(self):
        """European temperate forest detection."""
        assert get_biome(50, 10) == "temperate_forest"

    def test_alpine_high_elevation(self):
        """Alpine biome at high elevation."""
        # Very high elevation should return alpine
        assert get_biome(30, 85, elevation_m=4500) == "alpine"

    def test_montane_moderate_elevation(self):
        """Montane biome at moderate high elevation in mountains."""
        # 3000m in the Himalayas
        assert get_biome(30, 85, elevation_m=3000) == "montane"


# ============================================================================
# Test Mountain Range Detection
# ============================================================================

class TestMountainRanges:
    """Tests for mountain range detection."""

    def test_himalayas(self):
        """Himalayas detection."""
        result = get_mountain_range(28, 86)
        assert result == "Himalayas"

    def test_andes(self):
        """Andes detection."""
        result = get_mountain_range(-15, -70)
        assert result == "Andes"

    def test_alps(self):
        """Alps detection."""
        result = get_mountain_range(46, 10)
        assert result == "Alps"

    def test_rockies(self):
        """Rocky Mountains detection."""
        result = get_mountain_range(40, -110)
        assert result == "Rocky Mountains"

    def test_appalachians(self):
        """Appalachians detection."""
        result = get_mountain_range(38, -80)
        assert result == "Appalachians"

    def test_no_mountain_range(self):
        """Point not in any mountain range."""
        result = get_mountain_range(0, 0)
        assert result is None


# ============================================================================
# Test Island Detection
# ============================================================================

class TestIslandDetection:
    """Tests for island detection."""

    def test_hawaii_is_island(self):
        """Hawaii is an island."""
        assert is_island_location(21, -157) is True

    def test_galapagos_is_island(self):
        """Galapagos is an island."""
        assert is_island_location(-0.5, -90.5) is True

    def test_madagascar_is_island(self):
        """Madagascar is an island."""
        assert is_island_location(-18, 47) is True

    def test_caribbean_is_island(self):
        """Caribbean locations are islands."""
        assert is_island_location(18, -77) is True  # Jamaica
        assert is_island_location(18, -65) is True  # Puerto Rico

    def test_pacific_islands(self):
        """Pacific island locations."""
        assert is_island_location(5, 160) is True  # Micronesia

    def test_mainland_not_island(self):
        """Mainland locations are not islands."""
        assert is_island_location(40, -100) is False  # Kansas
        assert is_island_location(50, 10) is False  # Germany


# ============================================================================
# Test Marine Features
# ============================================================================

class TestCoralReefs:
    """Tests for coral reef region detection."""

    def test_great_barrier_reef(self):
        """Great Barrier Reef detection."""
        assert is_coral_reef_region(-18, 147) is True

    def test_caribbean_reefs(self):
        """Caribbean coral reefs detection."""
        assert is_coral_reef_region(18, -75) is True

    def test_maldives_reefs(self):
        """Maldives coral reefs detection."""
        assert is_coral_reef_region(4, 73) is True

    def test_no_coral_reef(self):
        """Non-reef location."""
        assert is_coral_reef_region(50, 0) is False


class TestMangroves:
    """Tests for mangrove region detection."""

    def test_sundarbans(self):
        """Sundarbans mangrove detection."""
        assert is_mangrove_region(22, 89) is True

    def test_florida_mangroves(self):
        """Florida mangroves detection."""
        assert is_mangrove_region(25, -81) is True

    def test_no_mangroves(self):
        """Non-mangrove location."""
        assert is_mangrove_region(50, 0) is False


# ============================================================================
# Test Desert Detection
# ============================================================================

class TestDesertDetection:
    """Tests for specific desert detection."""

    def test_sahara(self):
        """Sahara desert detection."""
        assert get_specific_desert(25, 10) == "Sahara"

    def test_gobi(self):
        """Gobi desert detection."""
        assert get_specific_desert(42, 105) == "Gobi"

    def test_atacama(self):
        """Atacama desert detection."""
        assert get_specific_desert(-24, -70) == "Atacama"

    def test_sonoran(self):
        """Sonoran desert detection."""
        assert get_specific_desert(32, -112) == "Sonoran"

    def test_mojave(self):
        """Mojave desert detection."""
        # Use coordinates clearly within Mojave bounds (34-37°N, -117 to -114°W)
        assert get_specific_desert(36, -115) == "Mojave"

    def test_kalahari(self):
        """Kalahari desert detection."""
        assert get_specific_desert(-24, 22) == "Kalahari"

    def test_no_desert(self):
        """Non-desert location."""
        assert get_specific_desert(50, 0) is None


# ============================================================================
# Test Biogeographic Realms
# ============================================================================

class TestBiogeographicRealms:
    """Tests for biogeographic realm classification."""

    def test_nearctic(self):
        """Nearctic realm (North America)."""
        assert get_biogeographic_realm(40, -100) == "Nearctic"

    def test_neotropical(self):
        """Neotropical realm (South America)."""
        assert get_biogeographic_realm(-15, -60) == "Neotropical"

    def test_palearctic(self):
        """Palearctic realm (Europe/Asia)."""
        assert get_biogeographic_realm(50, 10) == "Palearctic"

    def test_afrotropic(self):
        """Afrotropic realm (Africa)."""
        assert get_biogeographic_realm(-10, 30) == "Afrotropic"

    def test_indomalayan(self):
        """Indomalayan realm (South Asia)."""
        assert get_biogeographic_realm(15, 100) == "Indomalayan"

    def test_australasian(self):
        """Australasian realm."""
        assert get_biogeographic_realm(-25, 135) == "Australasian"

    def test_oceanian(self):
        """Oceanian realm (Pacific Islands)."""
        assert get_biogeographic_realm(5, 160) == "Oceanian"

    def test_antarctic(self):
        """Antarctic realm."""
        assert get_biogeographic_realm(-70, 0) == "Antarctic"


# ============================================================================
# Test Main Classification Function
# ============================================================================

class TestClassifyCoordinates:
    """Tests for the main classify_coordinates function."""

    def test_all_fields_returned(self):
        """All expected fields are returned."""
        result = classify_coordinates(0, 0)
        expected_keys = [
            "region", "region_group", "is_island", "climate_zone",
            "climate_name", "biogeographic_realm", "latitude_band",
            "environment", "elevation_band", "is_high_altitude",
            "is_above_treeline", "biome", "mountain_range",
            "is_coral_reef_region", "is_mangrove_region", "desert_name"
        ]
        for key in expected_keys:
            assert key in result

    def test_none_coordinates(self):
        """None coordinates return default values."""
        result = classify_coordinates(None, None)
        assert result["region"] is None
        assert result["is_island"] is False
        assert result["climate_zone"] is None
        assert result["elevation_band"] == "unknown"
        assert result["biome"] == "unknown"

    def test_nan_coordinates(self):
        """NaN coordinates return default values."""
        result = classify_coordinates(float("nan"), float("nan"))
        assert result["region"] is None

    def test_with_elevation(self):
        """Classification with elevation data."""
        # High altitude in Himalayas
        result = classify_coordinates(28, 86, elevation_m=5000)
        assert result["elevation_band"] == "nival"
        assert result["is_high_altitude"] is True
        assert result["mountain_range"] == "Himalayas"

    def test_galapagos_full(self):
        """Full classification for Galapagos."""
        result = classify_coordinates(-0.5, -90.5, elevation_m=100)
        assert result["region"] == "Galapagos"
        assert result["region_group"] == "Pacific Islands"
        assert result["is_island"] is True
        assert result["latitude_band"] == "equatorial"
        assert result["elevation_band"] == "lowland"

    def test_sahara_full(self):
        """Full classification for Sahara."""
        result = classify_coordinates(25, 10, elevation_m=500)
        assert result["region"] == "Sahara"
        assert result["climate_name"] == "Desert"
        assert result["biome"] == "desert"
        assert result["desert_name"] == "Sahara"
        assert result["environment"] == "desert"

    def test_amazon_full(self):
        """Full classification for Amazon."""
        result = classify_coordinates(-3, -60, elevation_m=50)
        assert result["region"] == "Amazon Basin"
        assert result["climate_name"] == "Tropical"
        assert result["biome"] == "tropical_rainforest"
        assert result["latitude_band"] == "equatorial"


# ============================================================================
# Test DataFrame Classification
# ============================================================================

class TestClassifyDataframe:
    """Tests for classify_dataframe function."""

    def test_basic_dataframe(self):
        """Basic DataFrame classification."""
        df = pd.DataFrame({
            "latitude1": [0, 25, 40],
            "longitude1": [-90.5, 10, -100],
            "minelevation": [100, 500, 1000],
        })

        result = classify_dataframe(df)

        assert "geo_region" in result.columns
        assert "geo_biome" in result.columns
        assert "geo_elevation_band" in result.columns
        assert len(result) == 3

    def test_preserves_original_columns(self):
        """Original columns are preserved."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "latitude1": [0, 25, 40],
            "longitude1": [-90.5, 10, -100],
        })

        result = classify_dataframe(df, elevation_col=None)

        assert "id" in result.columns
        assert "latitude1" in result.columns

    def test_handles_missing_values(self):
        """Missing values are handled gracefully."""
        df = pd.DataFrame({
            "latitude1": [0, None, 40],
            "longitude1": [-90.5, 10, None],
        })

        result = classify_dataframe(df, elevation_col=None)

        # Rows with None should have default values
        assert result["geo_biome"].iloc[1] == "unknown"
        assert result["geo_biome"].iloc[2] == "unknown"


# ============================================================================
# Test Expedition Classification
# ============================================================================

class TestClassifyExpeditions:
    """Tests for classify_expeditions function."""

    def test_basic_expedition_classification(self):
        """Basic expedition classification."""
        df = pd.DataFrame({
            "spatiotemporal_cluster_id": [1, 1, 2, 2],
            "latitude1": [0, 0.1, 25, 25.1],
            "longitude1": [-90.5, -90.4, 10, 10.1],
            "minelevation": [100, 150, 500, 600],
        })

        result = classify_expeditions(df)

        assert "geo_region" in result.columns
        assert "geo_biome" in result.columns
        # All rows in same cluster should have same classification
        cluster1 = result[result["spatiotemporal_cluster_id"] == 1]
        assert len(cluster1["geo_region"].unique()) == 1

    def test_uses_centroid(self):
        """Classification uses cluster centroid."""
        # Two points that span a region boundary
        df = pd.DataFrame({
            "spatiotemporal_cluster_id": [1, 1],
            "latitude1": [-1, 1],  # Spans equator
            "longitude1": [-90.5, -90.5],
        })

        result = classify_expeditions(df, elevation_col=None)

        # Centroid should be at lat=0
        assert result["geo_latitude_band"].iloc[0] == "equatorial"

    def test_uses_median_elevation(self):
        """Classification uses median elevation."""
        df = pd.DataFrame({
            "spatiotemporal_cluster_id": [1, 1, 1],
            "latitude1": [28, 28, 28],
            "longitude1": [86, 86, 86],
            "minelevation": [1000, 3000, 5000],  # median = 3000
        })

        result = classify_expeditions(df)

        # Median elevation is 3000m, which is upper_montane
        assert result["geo_elevation_band"].iloc[0] == "upper_montane"


# ============================================================================
# Test Helper Functions
# ============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_point_in_bounds_normal(self):
        """Normal bounding box check."""
        bounds = (0, 10, 0, 10)
        assert _point_in_bounds(5, 5, bounds) is True
        assert _point_in_bounds(15, 5, bounds) is False
        assert _point_in_bounds(5, 15, bounds) is False

    def test_point_in_bounds_dateline(self):
        """Bounding box crossing dateline."""
        # Fiji bounds cross dateline: 177°E to 179°W (which is -179°)
        # This covers 177 to 180 and -180 to -179
        bounds = (-21.0, -12.0, 177.0, -179.0)
        assert _point_in_bounds(-15, 178, bounds) is True   # East of dateline
        assert _point_in_bounds(-15, 179.5, bounds) is True  # Near dateline
        assert _point_in_bounds(-15, -180, bounds) is True   # At dateline (west side)
        assert _point_in_bounds(-15, 170, bounds) is False   # Too far west
        assert _point_in_bounds(-15, -178, bounds) is False  # Outside: -178 is east of -179

    def test_point_in_region_bounds(self):
        """Region bounds dict check."""
        bounds = {"lat": (0, 10), "lng": (0, 10)}
        assert _point_in_region_bounds(5, 5, bounds) is True
        assert _point_in_region_bounds(15, 5, bounds) is False

    def test_is_desert_region(self):
        """Desert region check."""
        assert _is_desert_region(25, 10) is True  # Sahara
        assert _is_desert_region(42, 105) is True  # Gobi
        assert _is_desert_region(50, 0) is False  # Europe

    def test_is_steppe_region(self):
        """Steppe region check."""
        assert _is_steppe_region(45, 70) is True  # Kazakhstan
        assert _is_steppe_region(40, -100) is True  # Great Plains
        assert _is_steppe_region(0, 0) is False

    def test_is_mediterranean_region(self):
        """Mediterranean region check."""
        assert _is_mediterranean_region(37, 15) is True  # Mediterranean Basin
        assert _is_mediterranean_region(35, -120) is True  # California
        assert _is_mediterranean_region(50, 0) is False


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_exact_boundary_latitude(self):
        """Exact boundary latitude values."""
        # Tropic of Cancer
        assert get_latitude_band(23.5) == "subtropical"
        # Tropic of Capricorn
        assert get_latitude_band(-23.5) == "subtropical"
        # Arctic Circle
        assert get_latitude_band(66.5) == "polar"

    def test_extreme_elevations(self):
        """Extreme elevation values."""
        # Mount Everest
        assert get_elevation_band(8848) == "nival"
        # Mariana Trench (hypothetical land)
        assert get_elevation_band(-11000) == "below_sea_level"

    def test_poles(self):
        """Exact pole coordinates."""
        result_north = classify_coordinates(90, 0)
        assert result_north["latitude_band"] == "polar"

        result_south = classify_coordinates(-90, 0)
        assert result_south["latitude_band"] == "polar"

    def test_dateline_crossing(self):
        """Points near the international dateline."""
        # Fiji is at ~178°E to ~-178°W
        result = classify_coordinates(-17, 179)
        # Should still classify correctly


# ============================================================================
# Test Data Integrity
# ============================================================================

class TestDataIntegrity:
    """Tests for data structure integrity."""

    def test_all_named_regions_have_valid_bounds(self):
        """All named regions have valid 4-tuple bounds."""
        for name, bounds in NAMED_REGIONS.items():
            assert len(bounds) == 4, f"{name} has invalid bounds"
            min_lat, max_lat, min_lng, max_lng = bounds
            assert min_lat < max_lat, f"{name} has invalid latitude bounds"

    def test_all_island_groups_reference_valid_regions(self):
        """All island group members exist in NAMED_REGIONS."""
        for group, regions in ISLAND_GROUPS.items():
            for region in regions:
                assert region in NAMED_REGIONS, f"{region} in {group} not in NAMED_REGIONS"

    def test_all_biome_regions_have_valid_bounds(self):
        """All biome regions have valid lat/lng bounds."""
        for region_dict in [DESERT_REGIONS, RAINFOREST_REGIONS, TAIGA_REGIONS,
                           TUNDRA_REGIONS, MOUNTAIN_RANGES]:
            for name, bounds in region_dict.items():
                assert "lat" in bounds, f"{name} missing lat"
                assert "lng" in bounds, f"{name} missing lng"
                assert len(bounds["lat"]) == 2, f"{name} invalid lat bounds"
                assert len(bounds["lng"]) == 2, f"{name} invalid lng bounds"
