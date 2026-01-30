"""
Geographic classification for expedition clusters.

Classifies coordinates into:
- Named scientific regions (Galapagos, Micronesia, etc.)
- Island vs mainland
- Climate zones (Köppen simplified)
- Biogeographic realms
- Latitude bands
- Marine/terrestrial environment
- Elevation bands (alpine, montane, lowland, etc.)
- Biomes (taiga, tundra, rainforest, savanna, desert, etc.)
- Mountain ranges
- Coastal/marine features (coral reefs, mangroves, etc.)

Uses coordinate-based heuristics optimized for low error rates.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================================
# Named Scientific Regions - Bounding boxes for key research areas
# Format: (min_lat, max_lat, min_lng, max_lng)
# ============================================================================

NAMED_REGIONS = {
    # Pacific Islands
    "Galapagos": (-1.8, 1.0, -92.5, -89.0),
    "Hawaii": (18.5, 23.0, -161.0, -154.0),
    "Micronesia": (0.0, 15.0, 130.0, 170.0),
    "Melanesia": (-25.0, 0.0, 140.0, 180.0),
    "Polynesia": (-30.0, -5.0, -180.0, -120.0),
    "Guam": (13.2, 13.7, 144.6, 145.0),
    "Palau": (2.0, 8.5, 131.0, 135.5),
    "Fiji": (-21.0, -12.0, 177.0, -179.0),  # Crosses dateline
    "Samoa": (-15.0, -13.0, -173.0, -168.0),
    "Tonga": (-22.5, -15.0, -176.5, -173.5),
    "Solomon Islands": (-12.0, -5.0, 155.0, 170.0),
    "Vanuatu": (-21.0, -13.0, 166.0, 171.0),
    "New Caledonia": (-23.0, -19.0, 163.0, 169.0),
    "Marshall Islands": (4.0, 15.0, 160.0, 173.0),
    "Mariana Islands": (13.0, 21.0, 144.0, 146.5),
    "Caroline Islands": (5.0, 10.0, 135.0, 165.0),

    # Caribbean
    "Caribbean": (10.0, 27.0, -85.0, -60.0),
    "Greater Antilles": (17.5, 24.0, -85.0, -66.0),
    "Lesser Antilles": (10.0, 18.5, -65.0, -59.0),
    "Bahamas": (20.0, 27.5, -80.0, -72.0),

    # Atlantic Islands
    "Canary Islands": (27.5, 29.5, -18.5, -13.0),
    "Azores": (36.5, 40.0, -31.5, -25.0),
    "Madeira": (32.0, 33.5, -17.5, -16.0),
    "Cape Verde": (14.5, 17.5, -25.5, -22.5),
    "Bermuda": (32.0, 32.5, -65.0, -64.5),

    # Indian Ocean
    "Madagascar": (-26.0, -11.5, 43.0, 51.0),
    "Seychelles": (-10.5, -3.5, 46.0, 56.5),
    "Mauritius": (-20.6, -19.9, 57.2, 57.9),
    "Reunion": (-21.5, -20.8, 55.1, 55.9),
    "Maldives": (-1.0, 7.5, 72.5, 74.0),
    "Andaman Islands": (6.5, 14.0, 92.0, 94.5),
    "Sri Lanka": (5.9, 10.0, 79.5, 82.0),

    # Southeast Asia / Malay Archipelago
    "Philippines": (4.5, 21.5, 116.0, 127.0),
    "Borneo": (-4.5, 7.5, 108.5, 119.5),
    "Sulawesi": (-6.0, 2.0, 118.5, 125.5),
    "Java": (-8.8, -5.9, 105.0, 114.5),
    "Sumatra": (-6.0, 6.0, 95.0, 106.0),
    "New Guinea": (-11.0, 0.0, 130.0, 151.0),
    "Lesser Sunda Islands": (-11.0, -8.0, 115.0, 128.0),
    "Moluccas": (-8.5, 3.0, 124.0, 135.0),

    # Other Notable Islands
    "Taiwan": (21.5, 25.5, 119.5, 122.5),
    "Hainan": (18.0, 20.5, 108.5, 111.5),
    "Japan": (24.0, 46.0, 122.0, 146.0),
    "New Zealand": (-48.0, -34.0, 166.0, 179.0),
    "Tasmania": (-44.0, -40.0, 144.0, 149.0),
    "Iceland": (63.0, 66.5, -24.5, -13.0),
    "British Isles": (49.5, 61.0, -11.0, 2.0),
    "Corsica": (41.3, 43.1, 8.5, 9.6),
    "Sardinia": (38.8, 41.3, 8.1, 9.9),
    "Sicily": (36.6, 38.4, 12.3, 15.7),
    "Crete": (34.8, 35.7, 23.5, 26.4),
    "Cyprus": (34.5, 35.8, 32.2, 34.7),

    # Continental Regions of Interest
    "Baja California": (22.5, 32.5, -118.0, -109.0),
    "Central America": (7.0, 18.5, -92.5, -77.0),
    "Amazon Basin": (-20.0, 5.0, -75.0, -45.0),
    "Andes": (-55.0, 10.0, -80.0, -65.0),
    "Patagonia": (-55.0, -38.0, -76.0, -63.0),
    "Mediterranean Basin": (30.0, 46.0, -6.0, 36.0),
    "Sahara": (15.0, 35.0, -17.0, 35.0),
    "Congo Basin": (-10.0, 5.0, 10.0, 30.0),
    "East African Rift": (-15.0, 12.0, 29.0, 42.0),
    "Himalayas": (26.0, 36.0, 73.0, 95.0),
    "Southeast Asian Mainland": (5.0, 28.0, 92.0, 110.0),
    "Australian Outback": (-35.0, -18.0, 120.0, 145.0),
    "Great Barrier Reef": (-24.5, -10.5, 142.5, 154.0),

    # Polar/Subpolar
    "Arctic": (66.5, 90.0, -180.0, 180.0),
    "Antarctic": (-90.0, -60.0, -180.0, 180.0),
    "Subantarctic Islands": (-60.0, -45.0, -180.0, 180.0),
    "Alaska": (54.0, 72.0, -180.0, -130.0),
    "Greenland": (59.0, 84.0, -73.0, -11.0),
}

# Island groups for more specific classification
ISLAND_GROUPS = {
    "Pacific Islands": ["Galapagos", "Hawaii", "Micronesia", "Melanesia", "Polynesia",
                        "Guam", "Palau", "Fiji", "Samoa", "Tonga", "Solomon Islands",
                        "Vanuatu", "New Caledonia", "Marshall Islands", "Mariana Islands",
                        "Caroline Islands"],
    "Caribbean Islands": ["Caribbean", "Greater Antilles", "Lesser Antilles", "Bahamas"],
    "Atlantic Islands": ["Canary Islands", "Azores", "Madeira", "Cape Verde", "Bermuda"],
    "Indian Ocean Islands": ["Madagascar", "Seychelles", "Mauritius", "Reunion",
                             "Maldives", "Andaman Islands"],
    "Malay Archipelago": ["Philippines", "Borneo", "Sulawesi", "Java", "Sumatra",
                          "New Guinea", "Lesser Sunda Islands", "Moluccas"],
}


# ============================================================================
# Elevation Classification
# Thresholds based on ecological zonation standards
# ============================================================================

# Elevation band thresholds in meters
# These vary by latitude - tropical mountains have higher treelines
ELEVATION_BANDS = {
    "coastal": (None, 50),        # Sea level to 50m
    "lowland": (50, 500),         # 50-500m
    "submontane": (500, 1500),    # 500-1500m (lower montane)
    "montane": (1500, 2500),      # 1500-2500m (mid-montane)
    "upper_montane": (2500, 3500),  # 2500-3500m (upper montane/subalpine)
    "alpine": (3500, 4500),       # 3500-4500m (alpine)
    "nival": (4500, None),        # Above 4500m (permanent snow/ice zone)
}

# Latitude-adjusted treeline estimates (approximate elevation in meters)
# Treeline is higher in tropics, lower at high latitudes
def _get_treeline_elevation(lat: float) -> float:
    """
    Estimate treeline elevation based on latitude.

    Treeline varies from ~4500m at equator to ~0m at poles.
    This is a simplified model - actual treeline depends on
    local conditions, continentality, and precipitation.
    """
    abs_lat = abs(lat)

    if abs_lat < 10:
        return 4000  # Equatorial
    elif abs_lat < 23.5:
        return 3800  # Tropical
    elif abs_lat < 35:
        return 3200  # Subtropical
    elif abs_lat < 45:
        return 2500  # Temperate
    elif abs_lat < 55:
        return 1800  # Cool temperate
    elif abs_lat < 66.5:
        return 1000  # Subarctic
    else:
        return 0  # Arctic - no treeline


def get_elevation_band(
    elevation_m: float | None,
    lat: float | None = None,
) -> str:
    """
    Classify elevation into ecological bands.

    Parameters
    ----------
    elevation_m : float or None
        Elevation in meters
    lat : float or None
        Latitude for latitude-adjusted classification

    Returns
    -------
    str
        Elevation band name: coastal, lowland, submontane, montane,
        upper_montane, alpine, nival, or unknown
    """
    if elevation_m is None or pd.isna(elevation_m):
        return "unknown"

    # Handle negative elevations (below sea level - Dead Sea, Death Valley, etc.)
    if elevation_m < 0:
        return "below_sea_level"

    # Standard classification
    if elevation_m < 50:
        return "coastal"
    elif elevation_m < 500:
        return "lowland"
    elif elevation_m < 1500:
        return "submontane"
    elif elevation_m < 2500:
        return "montane"
    elif elevation_m < 3500:
        return "upper_montane"
    elif elevation_m < 4500:
        return "alpine"
    else:
        return "nival"


def is_high_altitude(
    elevation_m: float | None,
    lat: float | None = None,
    threshold_m: float = 2500,
) -> bool:
    """
    Determine if location is high altitude.

    Parameters
    ----------
    elevation_m : float or None
        Elevation in meters
    lat : float or None
        Latitude (for potential future latitude-adjusted thresholds)
    threshold_m : float
        Elevation threshold in meters (default 2500m)

    Returns
    -------
    bool
        True if elevation >= threshold
    """
    if elevation_m is None or pd.isna(elevation_m):
        return False
    return elevation_m >= threshold_m


def is_above_treeline(
    elevation_m: float | None,
    lat: float | None,
) -> bool:
    """
    Determine if location is above the treeline.

    Uses latitude-adjusted treeline estimates.

    Parameters
    ----------
    elevation_m : float or None
        Elevation in meters
    lat : float or None
        Latitude in decimal degrees

    Returns
    -------
    bool
        True if above estimated treeline
    """
    if elevation_m is None or pd.isna(elevation_m):
        return False
    if lat is None or pd.isna(lat):
        # Use conservative tropical treeline if no latitude
        return elevation_m >= 4000

    treeline = _get_treeline_elevation(lat)
    return elevation_m >= treeline


# ============================================================================
# Köppen Climate Zones - Simplified classification by latitude/longitude
# ============================================================================

def get_koppen_zone(lat: float, lng: float) -> tuple[str, str]:
    """
    Estimate Köppen climate zone from coordinates.

    Returns (zone_code, zone_name) tuple.

    This is a simplified estimation - for precise classification,
    use actual Köppen shapefiles.
    """
    abs_lat = abs(lat)

    # Polar
    if abs_lat >= 66.5:
        return ("E", "Polar")

    # Tropical (within ~23.5° of equator, but accounting for monsoons)
    if abs_lat <= 23.5:
        # Check for desert regions
        if _is_desert_region(lat, lng):
            return ("BW", "Desert")
        return ("A", "Tropical")

    # Subtropical/Mediterranean (23.5° - 35°)
    if abs_lat <= 35:
        if _is_desert_region(lat, lng):
            return ("BW", "Desert")
        if _is_steppe_region(lat, lng):
            return ("BS", "Steppe")
        if _is_mediterranean_region(lat, lng):
            return ("Cs", "Mediterranean")
        return ("C", "Subtropical")

    # Temperate (35° - 50°)
    if abs_lat <= 50:
        if _is_oceanic_climate(lat, lng):
            return ("Cfb", "Oceanic")
        if _is_continental_climate(lat, lng):
            return ("D", "Continental")
        return ("C", "Temperate")

    # Subarctic/Boreal (50° - 66.5°)
    if _is_oceanic_climate(lat, lng):
        return ("Cfc", "Subpolar Oceanic")
    return ("Dfc", "Subarctic")


# ============================================================================
# Biome Classification
# Based on coordinates and optionally elevation
# ============================================================================

# Major desert regions with bounding boxes
DESERT_REGIONS = {
    "Sahara": {"lat": (15, 35), "lng": (-17, 35)},
    "Arabian": {"lat": (12, 32), "lng": (35, 60)},
    "Thar": {"lat": (24, 30), "lng": (68, 76)},
    "Karakum": {"lat": (35, 42), "lng": (52, 65)},
    "Kyzylkum": {"lat": (38, 45), "lng": (58, 68)},
    "Gobi": {"lat": (38, 46), "lng": (90, 115)},
    "Taklamakan": {"lat": (36, 42), "lng": (76, 90)},
    "Australian_Interior": {"lat": (-30, -18), "lng": (120, 145)},
    "Simpson": {"lat": (-28, -23), "lng": (135, 142)},
    "Gibson": {"lat": (-26, -22), "lng": (122, 130)},
    "Great_Victoria": {"lat": (-32, -26), "lng": (123, 132)},
    "Atacama": {"lat": (-30, -18), "lng": (-72, -68)},
    "Sechura": {"lat": (-7, -4), "lng": (-81, -79)},
    "Patagonian": {"lat": (-52, -40), "lng": (-72, -65)},
    "Namib": {"lat": (-28, -15), "lng": (12, 17)},
    "Kalahari": {"lat": (-28, -18), "lng": (17, 26)},
    "Sonoran": {"lat": (27, 35), "lng": (-116, -109)},
    "Mojave": {"lat": (34, 37), "lng": (-117, -114)},
    "Chihuahuan": {"lat": (25, 33), "lng": (-108, -103)},
    "Great_Basin": {"lat": (36, 42), "lng": (-120, -111)},
    "Colorado_Plateau": {"lat": (35, 40), "lng": (-112, -107)},
}

# Tropical/subtropical rainforest regions
RAINFOREST_REGIONS = {
    "Amazon": {"lat": (-18, 8), "lng": (-78, -44)},
    "Congo": {"lat": (-8, 6), "lng": (9, 31)},
    "Southeast_Asian": {"lat": (-10, 20), "lng": (95, 140)},
    "Atlantic_Forest": {"lat": (-30, -5), "lng": (-55, -35)},
    "Guinean": {"lat": (4, 12), "lng": (-18, 12)},
    "Madagascar_East": {"lat": (-25, -12), "lng": (48, 51)},
    "Queensland": {"lat": (-20, -10), "lng": (144, 148)},
    "Central_American": {"lat": (7, 20), "lng": (-92, -77)},
    "Choco": {"lat": (0, 9), "lng": (-79, -76)},
}

# Taiga/Boreal forest regions (simplified - mainly latitude-based)
TAIGA_REGIONS = {
    "Canadian_Boreal": {"lat": (50, 68), "lng": (-140, -55)},
    "Scandinavian_Taiga": {"lat": (58, 70), "lng": (5, 30)},
    "Russian_Taiga": {"lat": (50, 72), "lng": (30, 180)},
    "Siberian_Taiga": {"lat": (50, 72), "lng": (60, 180)},
}

# Tundra regions
TUNDRA_REGIONS = {
    "Arctic_Tundra": {"lat": (66.5, 90), "lng": (-180, 180)},
    "Canadian_Arctic": {"lat": (60, 83), "lng": (-140, -60)},
    "Siberian_Tundra": {"lat": (65, 78), "lng": (60, 180)},
    "Scandinavian_Tundra": {"lat": (68, 71), "lng": (5, 32)},
    "Greenland_Tundra": {"lat": (60, 84), "lng": (-73, -11)},
    "Antarctic_Tundra": {"lat": (-90, -60), "lng": (-180, 180)},
}

# Savanna/grassland regions
SAVANNA_REGIONS = {
    "African_Savanna": {"lat": (-25, 15), "lng": (-18, 45)},
    "Cerrado": {"lat": (-24, -5), "lng": (-60, -41)},
    "Llanos": {"lat": (2, 10), "lng": (-74, -62)},
    "Australian_Savanna": {"lat": (-20, -10), "lng": (120, 150)},
    "Indian_Savanna": {"lat": (8, 25), "lng": (72, 88)},
}

# Steppe/grassland regions
STEPPE_REGIONS = {
    "Eurasian_Steppe": {"lat": (40, 55), "lng": (20, 120)},
    "Kazakh_Steppe": {"lat": (42, 55), "lng": (50, 90)},
    "Mongolian_Steppe": {"lat": (42, 52), "lng": (90, 120)},
    "Great_Plains": {"lat": (30, 50), "lng": (-110, -95)},
    "Pampas": {"lat": (-40, -28), "lng": (-65, -55)},
    "Patagonian_Steppe": {"lat": (-52, -38), "lng": (-72, -65)},
}

# Temperate forest regions
TEMPERATE_FOREST_REGIONS = {
    "Eastern_North_America": {"lat": (30, 48), "lng": (-95, -65)},
    "Western_Europe": {"lat": (42, 60), "lng": (-10, 25)},
    "East_Asia": {"lat": (25, 45), "lng": (100, 145)},
    "Southern_Chile": {"lat": (-55, -38), "lng": (-76, -70)},
    "New_Zealand": {"lat": (-48, -34), "lng": (166, 179)},
    "Tasmania": {"lat": (-44, -40), "lng": (144, 149)},
    "Southeast_Australia": {"lat": (-42, -30), "lng": (140, 154)},
}

# Mediterranean scrubland/chaparral regions
MEDITERRANEAN_REGIONS = {
    "Mediterranean_Basin": {"lat": (30, 45), "lng": (-10, 40)},
    "California_Chaparral": {"lat": (32, 42), "lng": (-125, -117)},
    "Chilean_Matorral": {"lat": (-40, -30), "lng": (-75, -70)},
    "Cape_Fynbos": {"lat": (-35, -31), "lng": (17, 26)},
    "Southwest_Australia": {"lat": (-38, -30), "lng": (114, 122)},
}

# Major mountain ranges for montane classification
MOUNTAIN_RANGES = {
    "Himalayas": {"lat": (26, 36), "lng": (73, 95)},
    "Karakoram": {"lat": (34, 37), "lng": (74, 78)},
    "Hindu_Kush": {"lat": (34, 37), "lng": (69, 74)},
    "Tibetan_Plateau": {"lat": (27, 40), "lng": (78, 103)},
    "Andes": {"lat": (-55, 10), "lng": (-80, -65)},
    "Rocky_Mountains": {"lat": (35, 60), "lng": (-125, -105)},
    "Alps": {"lat": (43, 48), "lng": (5, 17)},
    "Pyrenees": {"lat": (42, 43.5), "lng": (-2, 3)},
    "Carpathians": {"lat": (44, 50), "lng": (17, 27)},
    "Caucasus": {"lat": (41, 44), "lng": (39, 50)},
    "Urals": {"lat": (48, 68), "lng": (54, 62)},
    "Atlas": {"lat": (30, 36), "lng": (-10, 10)},
    "Ethiopian_Highlands": {"lat": (6, 15), "lng": (35, 43)},
    "East_African_Mountains": {"lat": (-5, 5), "lng": (29, 40)},
    "Drakensberg": {"lat": (-32, -28), "lng": (27, 32)},
    "Southern_Alps_NZ": {"lat": (-46, -42), "lng": (168, 172)},
    "Australian_Alps": {"lat": (-37.5, -35.5), "lng": (146, 149)},
    "Japanese_Alps": {"lat": (35, 37), "lng": (137, 139)},
    "Taiwan_Mountains": {"lat": (23, 25), "lng": (120, 122)},
    "New_Guinea_Highlands": {"lat": (-7, -4), "lng": (138, 148)},
    "Sierra_Nevada_US": {"lat": (35, 40), "lng": (-120, -117)},
    "Cascade_Range": {"lat": (40, 50), "lng": (-123, -120)},
    "Appalachians": {"lat": (33, 47), "lng": (-85, -75)},
    "Brooks_Range": {"lat": (66, 69), "lng": (-165, -142)},
    "Alaska_Range": {"lat": (61, 64), "lng": (-154, -143)},
    "Scandinavian_Mountains": {"lat": (58, 71), "lng": (5, 20)},
    "Altai": {"lat": (46, 52), "lng": (82, 100)},
    "Tien_Shan": {"lat": (38, 44), "lng": (70, 95)},
    "Pamir": {"lat": (37, 40), "lng": (68, 76)},
    "Kunlun": {"lat": (34, 37), "lng": (78, 97)},
}

# Coral reef regions
CORAL_REEF_REGIONS = {
    "Great_Barrier_Reef": {"lat": (-24.5, -10.5), "lng": (142.5, 154)},
    "Coral_Triangle": {"lat": (-12, 20), "lng": (95, 145)},
    "Caribbean_Reefs": {"lat": (10, 27), "lng": (-90, -60)},
    "Red_Sea_Reefs": {"lat": (12, 30), "lng": (32, 44)},
    "Maldives_Reefs": {"lat": (-1, 8), "lng": (72, 74)},
    "Mesoamerican_Reef": {"lat": (15, 22), "lng": (-89, -84)},
    "Florida_Keys": {"lat": (24, 26), "lng": (-82, -80)},
    "Hawaiian_Reefs": {"lat": (18.5, 23), "lng": (-161, -154)},
    "French_Polynesia_Reefs": {"lat": (-28, -8), "lng": (-155, -134)},
    "Micronesian_Reefs": {"lat": (0, 15), "lng": (130, 170)},
}

# Mangrove regions (tropical/subtropical coastlines)
MANGROVE_REGIONS = {
    "Sundarbans": {"lat": (21, 23), "lng": (88, 90)},
    "West_African_Mangroves": {"lat": (4, 14), "lng": (-18, 12)},
    "East_African_Mangroves": {"lat": (-12, 5), "lng": (38, 50)},
    "Southeast_Asian_Mangroves": {"lat": (-10, 20), "lng": (95, 145)},
    "Northern_Australia_Mangroves": {"lat": (-20, -10), "lng": (120, 150)},
    "Caribbean_Mangroves": {"lat": (8, 27), "lng": (-100, -60)},
    "Florida_Mangroves": {"lat": (24, 28), "lng": (-83, -80)},
    "Central_American_Mangroves": {"lat": (5, 22), "lng": (-95, -77)},
    "South_American_Mangroves": {"lat": (-5, 12), "lng": (-82, -35)},
    "Madagascar_Mangroves": {"lat": (-26, -12), "lng": (43, 50)},
}


def _is_desert_region(lat: float, lng: float) -> bool:
    """Check if coordinates fall in major desert regions."""
    for name, bounds in DESERT_REGIONS.items():
        if _point_in_region_bounds(lat, lng, bounds):
            return True
    return False


def _is_steppe_region(lat: float, lng: float) -> bool:
    """Check if coordinates fall in steppe/semi-arid regions."""
    for name, bounds in STEPPE_REGIONS.items():
        if _point_in_region_bounds(lat, lng, bounds):
            return True
    return False


def _is_mediterranean_region(lat: float, lng: float) -> bool:
    """Check if coordinates fall in Mediterranean climate regions."""
    for name, bounds in MEDITERRANEAN_REGIONS.items():
        if _point_in_region_bounds(lat, lng, bounds):
            return True
    return False


def _is_oceanic_climate(lat: float, lng: float) -> bool:
    """Check if coordinates fall in oceanic climate regions."""
    # Western Europe
    if 45 <= lat <= 60 and -10 <= lng <= 10:
        return True
    # Pacific Northwest
    if 42 <= lat <= 55 and -130 <= lng <= -120:
        return True
    # New Zealand
    if -48 <= lat <= -34 and 166 <= lng <= 179:
        return True
    # Southern Chile
    if -55 <= lat <= -40 and -76 <= lng <= -70:
        return True
    # Tasmania
    if -44 <= lat <= -40 and 144 <= lng <= 149:
        return True
    return False


def _is_continental_climate(lat: float, lng: float) -> bool:
    """Check if coordinates fall in continental climate regions."""
    # Interior North America
    if 35 <= lat <= 55 and -120 <= lng <= -70:
        return True
    # Eastern Europe / Western Russia
    if 45 <= lat <= 60 and 20 <= lng <= 60:
        return True
    # Northeast Asia
    if 35 <= lat <= 55 and 100 <= lng <= 140:
        return True
    return False


def _point_in_region_bounds(lat: float, lng: float, bounds: dict) -> bool:
    """Check if point is within region bounds dict with lat/lng tuples."""
    lat_range = bounds["lat"]
    lng_range = bounds["lng"]

    if not (lat_range[0] <= lat <= lat_range[1]):
        return False

    # Handle longitude wraparound
    if lng_range[0] <= lng_range[1]:
        return lng_range[0] <= lng <= lng_range[1]
    else:
        return lng >= lng_range[0] or lng <= lng_range[1]


def get_biome(
    lat: float,
    lng: float,
    elevation_m: float | None = None,
) -> str:
    """
    Classify coordinates into biome type.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees
    lng : float
        Longitude in decimal degrees
    elevation_m : float or None
        Elevation in meters (improves accuracy)

    Returns
    -------
    str
        Biome name: tundra, taiga, temperate_forest, temperate_grassland,
        desert, mediterranean, tropical_rainforest, tropical_savanna,
        montane, alpine, or unknown
    """
    abs_lat = abs(lat)

    # Check for alpine/montane based on elevation first
    if elevation_m is not None and not pd.isna(elevation_m):
        if elevation_m >= 4000:
            return "alpine"
        if elevation_m >= 2500:
            # Check if in mountain range
            for name, bounds in MOUNTAIN_RANGES.items():
                if _point_in_region_bounds(lat, lng, bounds):
                    return "montane"

    # Polar/Tundra (high latitudes)
    if abs_lat >= 66.5:
        return "tundra"

    # Check specific tundra regions
    for name, bounds in TUNDRA_REGIONS.items():
        if _point_in_region_bounds(lat, lng, bounds):
            return "tundra"

    # Taiga/Boreal (50-66.5° in continental regions)
    if 50 <= abs_lat < 66.5:
        for name, bounds in TAIGA_REGIONS.items():
            if _point_in_region_bounds(lat, lng, bounds):
                return "taiga"

    # Desert
    if _is_desert_region(lat, lng):
        return "desert"

    # Tropical rainforest
    for name, bounds in RAINFOREST_REGIONS.items():
        if _point_in_region_bounds(lat, lng, bounds):
            return "tropical_rainforest"

    # Tropical savanna
    for name, bounds in SAVANNA_REGIONS.items():
        if _point_in_region_bounds(lat, lng, bounds):
            return "tropical_savanna"

    # Mediterranean
    for name, bounds in MEDITERRANEAN_REGIONS.items():
        if _point_in_region_bounds(lat, lng, bounds):
            return "mediterranean"

    # Steppe/temperate grassland
    for name, bounds in STEPPE_REGIONS.items():
        if _point_in_region_bounds(lat, lng, bounds):
            return "temperate_grassland"

    # Temperate forest
    for name, bounds in TEMPERATE_FOREST_REGIONS.items():
        if _point_in_region_bounds(lat, lng, bounds):
            return "temperate_forest"

    # Default by latitude
    if abs_lat < 23.5:
        return "tropical"
    elif abs_lat < 35:
        return "subtropical"
    elif abs_lat < 50:
        return "temperate"
    elif abs_lat < 66.5:
        return "boreal"
    else:
        return "polar"


def get_mountain_range(lat: float, lng: float) -> str | None:
    """
    Identify if coordinates fall within a known mountain range.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees
    lng : float
        Longitude in decimal degrees

    Returns
    -------
    str or None
        Mountain range name or None if not in a known range
    """
    for name, bounds in MOUNTAIN_RANGES.items():
        if _point_in_region_bounds(lat, lng, bounds):
            return name.replace("_", " ")
    return None


def is_coral_reef_region(lat: float, lng: float) -> bool:
    """
    Check if coordinates are in a coral reef region.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees
    lng : float
        Longitude in decimal degrees

    Returns
    -------
    bool
        True if in a coral reef region
    """
    for name, bounds in CORAL_REEF_REGIONS.items():
        if _point_in_region_bounds(lat, lng, bounds):
            return True
    return False


def is_mangrove_region(lat: float, lng: float) -> bool:
    """
    Check if coordinates are in a mangrove region.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees
    lng : float
        Longitude in decimal degrees

    Returns
    -------
    bool
        True if in a mangrove region
    """
    for name, bounds in MANGROVE_REGIONS.items():
        if _point_in_region_bounds(lat, lng, bounds):
            return True
    return False


def get_specific_desert(lat: float, lng: float) -> str | None:
    """
    Identify specific desert if in a desert region.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees
    lng : float
        Longitude in decimal degrees

    Returns
    -------
    str or None
        Desert name or None if not in a desert
    """
    for name, bounds in DESERT_REGIONS.items():
        if _point_in_region_bounds(lat, lng, bounds):
            return name.replace("_", " ")
    return None


# ============================================================================
# Biogeographic Realms (Simplified)
# ============================================================================

BIOGEOGRAPHIC_REALMS = {
    "Nearctic": {"lat": (15, 90), "lng": (-170, -50)},
    "Neotropical": {"lat": (-60, 30), "lng": (-120, -30)},
    "Palearctic": {"lat": (20, 90), "lng": (-30, 180)},
    "Afrotropic": {"lat": (-40, 20), "lng": (-20, 55)},
    "Indomalayan": {"lat": (-15, 35), "lng": (60, 150)},
    "Australasian": {"lat": (-50, 0), "lng": (110, 180)},
    "Oceanian": {"lat": (-30, 30), "lng": (150, -100)},  # Pacific Islands
    "Antarctic": {"lat": (-90, -60), "lng": (-180, 180)},
}


def get_biogeographic_realm(lat: float, lng: float) -> str:
    """
    Determine biogeographic realm from coordinates.

    Uses simplified bounding boxes - for precise classification,
    use WWF ecoregion shapefiles.
    """
    # Handle Pacific Islands specially (Oceanian realm)
    if -30 <= lat <= 30:
        # Central/South Pacific
        if 150 <= lng <= 180 or -180 <= lng <= -100:
            return "Oceanian"

    # Antarctic
    if lat < -60:
        return "Antarctic"

    # Check other realms
    for realm, bounds in BIOGEOGRAPHIC_REALMS.items():
        if realm in ("Oceanian", "Antarctic"):
            continue
        lat_range = bounds["lat"]
        lng_range = bounds["lng"]

        if lat_range[0] <= lat <= lat_range[1]:
            # Handle longitude wraparound
            if lng_range[0] <= lng_range[1]:
                if lng_range[0] <= lng <= lng_range[1]:
                    return realm
            else:  # Crosses dateline
                if lng >= lng_range[0] or lng <= lng_range[1]:
                    return realm

    return "Unknown"


# ============================================================================
# Latitude Bands
# ============================================================================

def get_latitude_band(lat: float) -> str:
    """Classify latitude into climate bands."""
    abs_lat = abs(lat)

    if abs_lat < 10:
        return "equatorial"
    elif abs_lat < 23.5:
        return "tropical"
    elif abs_lat < 35:
        return "subtropical"
    elif abs_lat < 50:
        return "temperate"
    elif abs_lat < 66.5:
        return "subarctic" if lat > 0 else "subantarctic"
    else:
        return "polar"


# ============================================================================
# Island Detection
# ============================================================================

def is_island_location(lat: float, lng: float) -> bool:
    """
    Determine if coordinates are on an island.

    Uses known island regions - for precise classification,
    use Natural Earth data.
    """
    # First check if in known island regions
    for region_name, bounds in NAMED_REGIONS.items():
        if _point_in_bounds(lat, lng, bounds):
            # Check if this region is an island group
            for group_name, regions in ISLAND_GROUPS.items():
                if region_name in regions:
                    return True

    # Additional island indicators based on coordinates
    # Small Pacific islands
    if -30 <= lat <= 30 and (lng > 150 or lng < -120):
        return True

    # Caribbean
    if 10 <= lat <= 27 and -85 <= lng <= -60:
        return True

    return False


def _point_in_bounds(lat: float, lng: float, bounds: tuple) -> bool:
    """Check if point is within bounding box (min_lat, max_lat, min_lng, max_lng)."""
    min_lat, max_lat, min_lng, max_lng = bounds

    if not (min_lat <= lat <= max_lat):
        return False

    # Handle longitude wraparound
    if min_lng <= max_lng:
        return min_lng <= lng <= max_lng
    else:
        return lng >= min_lng or lng <= max_lng


# ============================================================================
# Main Classification Function
# ============================================================================

def classify_coordinates(
    lat: float | None,
    lng: float | None,
    elevation_m: float | None = None,
) -> dict[str, Any]:
    """
    Classify a single coordinate pair into geographic categories.

    Parameters
    ----------
    lat : float or None
        Latitude in decimal degrees
    lng : float or None
        Longitude in decimal degrees
    elevation_m : float or None
        Elevation in meters (optional, improves classification)

    Returns
    -------
    dict with keys:
        - region: Named scientific region or None
        - region_group: Broader region group (e.g., "Pacific Islands")
        - is_island: Boolean
        - climate_zone: Köppen zone code
        - climate_name: Köppen zone name
        - biogeographic_realm: WWF realm name
        - latitude_band: Latitude band name
        - environment: "marine_coastal", "terrestrial", or "unknown"
        - elevation_band: Elevation band name
        - is_high_altitude: Boolean (elevation >= 2500m)
        - is_above_treeline: Boolean
        - biome: Biome classification
        - mountain_range: Mountain range name or None
        - is_coral_reef_region: Boolean
        - is_mangrove_region: Boolean
        - desert_name: Specific desert name or None
    """
    result = {
        "region": None,
        "region_group": None,
        "is_island": False,
        "climate_zone": None,
        "climate_name": None,
        "biogeographic_realm": None,
        "latitude_band": None,
        "environment": "unknown",
        "elevation_band": "unknown",
        "is_high_altitude": False,
        "is_above_treeline": False,
        "biome": "unknown",
        "mountain_range": None,
        "is_coral_reef_region": False,
        "is_mangrove_region": False,
        "desert_name": None,
    }

    if lat is None or lng is None or pd.isna(lat) or pd.isna(lng):
        return result

    # Named region
    for region_name, bounds in NAMED_REGIONS.items():
        if _point_in_bounds(lat, lng, bounds):
            result["region"] = region_name
            # Find region group
            for group_name, regions in ISLAND_GROUPS.items():
                if region_name in regions:
                    result["region_group"] = group_name
                    break
            break

    # Island detection
    result["is_island"] = is_island_location(lat, lng)

    # Climate zone
    zone_code, zone_name = get_koppen_zone(lat, lng)
    result["climate_zone"] = zone_code
    result["climate_name"] = zone_name

    # Biogeographic realm
    result["biogeographic_realm"] = get_biogeographic_realm(lat, lng)

    # Latitude band
    result["latitude_band"] = get_latitude_band(lat)

    # Elevation-based classifications
    result["elevation_band"] = get_elevation_band(elevation_m, lat)
    result["is_high_altitude"] = is_high_altitude(elevation_m, lat)
    result["is_above_treeline"] = is_above_treeline(elevation_m, lat)

    # Biome
    result["biome"] = get_biome(lat, lng, elevation_m)

    # Mountain range
    result["mountain_range"] = get_mountain_range(lat, lng)

    # Marine features
    result["is_coral_reef_region"] = is_coral_reef_region(lat, lng)
    result["is_mangrove_region"] = is_mangrove_region(lat, lng)

    # Specific desert
    result["desert_name"] = get_specific_desert(lat, lng)

    # Environment (enhanced)
    if result["is_island"]:
        result["environment"] = "marine_coastal"
    elif result["is_coral_reef_region"] or result["is_mangrove_region"]:
        result["environment"] = "marine_coastal"
    elif result["is_above_treeline"]:
        result["environment"] = "alpine"
    elif result["biome"] in ("desert", "tundra"):
        result["environment"] = result["biome"]
    elif result["latitude_band"] in ("equatorial", "tropical"):
        result["environment"] = "terrestrial_tropical"
    else:
        result["environment"] = "terrestrial"

    return result


def classify_dataframe(
    df: pd.DataFrame,
    lat_col: str = "latitude1",
    lng_col: str = "longitude1",
    elevation_col: str | None = "minelevation",
) -> pd.DataFrame:
    """
    Add geographic classification columns to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with latitude and longitude columns
    lat_col : str
        Name of latitude column
    lng_col : str
        Name of longitude column
    elevation_col : str or None
        Name of elevation column (optional)

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added classification columns
    """
    logger.info(f"Classifying {len(df)} coordinates...")

    # Initialize new columns
    new_cols = {
        "geo_region": [],
        "geo_region_group": [],
        "geo_is_island": [],
        "geo_climate_zone": [],
        "geo_climate_name": [],
        "geo_realm": [],
        "geo_latitude_band": [],
        "geo_environment": [],
        "geo_elevation_band": [],
        "geo_is_high_altitude": [],
        "geo_is_above_treeline": [],
        "geo_biome": [],
        "geo_mountain_range": [],
        "geo_is_coral_reef_region": [],
        "geo_is_mangrove_region": [],
        "geo_desert_name": [],
    }

    for idx, row in df.iterrows():
        lat = row.get(lat_col)
        lng = row.get(lng_col)
        elevation = row.get(elevation_col) if elevation_col else None

        classification = classify_coordinates(lat, lng, elevation)

        new_cols["geo_region"].append(classification["region"])
        new_cols["geo_region_group"].append(classification["region_group"])
        new_cols["geo_is_island"].append(classification["is_island"])
        new_cols["geo_climate_zone"].append(classification["climate_zone"])
        new_cols["geo_climate_name"].append(classification["climate_name"])
        new_cols["geo_realm"].append(classification["biogeographic_realm"])
        new_cols["geo_latitude_band"].append(classification["latitude_band"])
        new_cols["geo_environment"].append(classification["environment"])
        new_cols["geo_elevation_band"].append(classification["elevation_band"])
        new_cols["geo_is_high_altitude"].append(classification["is_high_altitude"])
        new_cols["geo_is_above_treeline"].append(classification["is_above_treeline"])
        new_cols["geo_biome"].append(classification["biome"])
        new_cols["geo_mountain_range"].append(classification["mountain_range"])
        new_cols["geo_is_coral_reef_region"].append(classification["is_coral_reef_region"])
        new_cols["geo_is_mangrove_region"].append(classification["is_mangrove_region"])
        new_cols["geo_desert_name"].append(classification["desert_name"])

    # Add columns to dataframe
    for col_name, values in new_cols.items():
        df[col_name] = values

    # Log summary
    region_counts = df["geo_region"].value_counts()
    island_count = df["geo_is_island"].sum()
    high_alt_count = df["geo_is_high_altitude"].sum()
    logger.info(f"  Islands: {island_count} ({100*island_count/len(df):.1f}%)")
    logger.info(f"  High altitude: {high_alt_count} ({100*high_alt_count/len(df):.1f}%)")
    logger.info(f"  Top regions: {dict(region_counts.head(5))}")

    return df


def classify_expeditions(
    df: pd.DataFrame,
    cluster_col: str = "spatiotemporal_cluster_id",
    lat_col: str = "latitude1",
    lng_col: str = "longitude1",
    elevation_col: str | None = "minelevation",
) -> pd.DataFrame:
    """
    Add geographic classification based on expedition centroid.

    For each expedition cluster, computes the centroid and classifies it.
    Uses median elevation from specimens in the cluster.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cluster IDs and coordinates
    cluster_col : str
        Name of cluster ID column
    lat_col : str
        Name of latitude column
    lng_col : str
        Name of longitude column
    elevation_col : str or None
        Name of elevation column (optional)

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added classification columns
    """
    logger.info(f"Classifying {df[cluster_col].nunique()} expedition clusters...")

    # Compute centroid and median elevation for each cluster
    agg_dict = {
        lat_col: "mean",
        lng_col: "mean",
    }
    if elevation_col and elevation_col in df.columns:
        agg_dict[elevation_col] = "median"

    centroids = df.groupby(cluster_col).agg(agg_dict).reset_index()

    # Classify each centroid
    classifications = []
    for _, row in centroids.iterrows():
        elevation = row.get(elevation_col) if elevation_col else None
        classification = classify_coordinates(row[lat_col], row[lng_col], elevation)
        classification[cluster_col] = row[cluster_col]
        classifications.append(classification)

    class_df = pd.DataFrame(classifications)

    # Rename columns with geo_ prefix
    class_df = class_df.rename(columns={
        "region": "geo_region",
        "region_group": "geo_region_group",
        "is_island": "geo_is_island",
        "climate_zone": "geo_climate_zone",
        "climate_name": "geo_climate_name",
        "biogeographic_realm": "geo_realm",
        "latitude_band": "geo_latitude_band",
        "environment": "geo_environment",
        "elevation_band": "geo_elevation_band",
        "is_high_altitude": "geo_is_high_altitude",
        "is_above_treeline": "geo_is_above_treeline",
        "biome": "geo_biome",
        "mountain_range": "geo_mountain_range",
        "is_coral_reef_region": "geo_is_coral_reef_region",
        "is_mangrove_region": "geo_is_mangrove_region",
        "desert_name": "geo_desert_name",
    })

    # Merge back to original dataframe
    df = df.merge(class_df, on=cluster_col, how="left")

    # Log summary
    region_counts = class_df["geo_region"].value_counts()
    island_count = class_df["geo_is_island"].sum()
    high_alt_count = class_df["geo_is_high_altitude"].sum()
    total_clusters = len(class_df)
    logger.info(f"  Island expeditions: {island_count} ({100*island_count/total_clusters:.1f}%)")
    logger.info(f"  High altitude expeditions: {high_alt_count} ({100*high_alt_count/total_clusters:.1f}%)")
    if len(region_counts) > 0:
        logger.info(f"  Top regions: {dict(region_counts.head(5))}")

    return df
