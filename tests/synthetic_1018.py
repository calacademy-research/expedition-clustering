"""
CAS Lens #1018: a deterministic synthetic clustering input.

Shared by tests/test_pipeline_speed_1018.py and scripts/golden_1018.py (which
wrote tests/data/pipeline_1018_expected.csv with the engine before the #1018
speed-up, 628db2f). Built to reach every stage the speed-up rewrote:
collector partitions of one row and of many, identical points, spatial
clusters that split in time, chains that reconnect, nearby expeditions the
merge joins, and rows without a collector.

Inputs:  n_trips (trips per collector-year), seed.
Outputs: a DataFrame with collectingeventid, spid, collectors, latitude1,
         longitude1, startdate, date_precision.
"""

import numpy as np
import pandas as pd

COLLECTORS = [
    "A. Eastwood", "J. T. Howell", "Eastwood, Alice", "H. M. Hall", "W. L. Jepson",
    "P. H. Raven", "M. E. Jones", "L. Abrams", "C. F. Baker", "E. Lee", "",
]


def synthetic_input(n_trips: int = 6, seed: int = 1018) -> pd.DataFrame:
    """The input frame described in the module docstring."""
    rng = np.random.default_rng(seed)
    rows = []
    eid = 0
    for who in COLLECTORS:
        for year in (1901, 1902, 1950):
            for _ in range(n_trips):
                # A trip: a walk of stops a few km apart over a few days,
                # several specimens per stop, sometimes at one point.
                lat = rng.uniform(32.0, 42.0)
                lon = rng.uniform(-124.0, -114.0)
                day = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=int(rng.integers(0, 330)))
                for _stop in range(int(rng.integers(1, 7))):
                    lat += rng.normal(0, 0.05)
                    lon += rng.normal(0, 0.05)
                    day += pd.Timedelta(days=int(rng.integers(0, 7)))
                    same_point = rng.random() < 0.3
                    for _ in range(int(rng.integers(1, 5))):
                        jitter = 0.0 if same_point else rng.normal(0, 0.002)
                        eid += 1
                        rows.append({
                            "collectingeventid": eid,
                            "spid": eid,
                            "collectors": who,
                            "latitude1": round(lat + jitter, 6),
                            "longitude1": round(lon + jitter, 6),
                            "startdate": day.strftime("%Y-%m-%d"),
                            "date_precision": 1,
                        })
                # Now and then a second trip right after the first, 15-25 km
                # away: a merge candidate.
                if rng.random() < 0.25:
                    day += pd.Timedelta(days=int(rng.integers(1, 3)))
                    lat2, lon2 = lat + 0.15, lon + 0.1
                    for _ in range(int(rng.integers(1, 4))):
                        eid += 1
                        rows.append({
                            "collectingeventid": eid, "spid": eid, "collectors": who,
                            "latitude1": round(lat2, 6), "longitude1": round(lon2, 6),
                            "startdate": day.strftime("%Y-%m-%d"), "date_precision": 1,
                        })
    return pd.DataFrame(rows)


ID_COLUMNS = ["spatial_cluster_id", "temporal_cluster_id", "spatiotemporal_cluster_id"]
CONFIGS = {
    # The weekly stage's botany settings (collector-aware, merge on).
    "collector_merge": {"e_dist": 10.0, "e_days": 3.0, "collector_aware": True, "merge_gap_days": 3,
                        "merge_distance_km": 30, "merge_threshold": 0.7},
    # The settings of a collection that is not collector-aware (ich).
    "plain": {"e_dist": 10.0, "e_days": 3.0, "collector_aware": False},
}
