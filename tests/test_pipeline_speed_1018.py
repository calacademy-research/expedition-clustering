"""
CAS Lens #1018: the clustering speed-up gives the same ids as before.

The weekly botany clustering took 4 h 22 min because every per-group loop in
pipeline.py scanned and copied the whole frame once per group. The speed-up
(row positions per group, temporal DBSCAN for all groups at once, a label
cache, an optional process pool, kept row positions in the merge) must not
change a single id. These tests hold it to that:

  * the ids on the synthetic input (tests/synthetic_1018.py) equal those the
    engine gave before the speed-up (628db2f; tests/data/pipeline_1018_expected.csv,
    written by scripts/golden_1018.py), collector-aware with merge and plain;
  * n_jobs does not change the ids;
  * temporal_components equals sklearn DBSCAN run per group;
  * haversine_components equals sklearn DBSCAN run per group, serial and in
    a process pool, for one-row groups and groups of identical points too;
  * scripts/cluster_csv.py takes --jobs.

Requirement: EXP-PERF-1 (cas-lens docs/requirements/expedition-requirements.md).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import DBSCAN

from expedition_clustering.pipeline import create_pipeline, haversine_components, temporal_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_1018 import CONFIGS, ID_COLUMNS, synthetic_input

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / "tests" / "data" / "pipeline_1018_expected.csv"


def _ids(config: str, n_jobs: int = 1) -> pd.DataFrame:
    out = create_pipeline(**CONFIGS[config], n_jobs=n_jobs).fit_transform(synthetic_input())
    return out[["spid", *ID_COLUMNS]].sort_values("spid").reset_index(drop=True)


@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_ids_equal_the_engine_before_the_speed_up(config):
    expected = pd.read_csv(EXPECTED)
    expected = expected[expected["config"] == config].drop(columns="config").sort_values("spid").reset_index(drop=True)
    got = _ids(config)
    assert len(got) == len(expected) > 1000
    for col in ID_COLUMNS:
        assert (got[col].to_numpy() == expected[col].to_numpy()).all(), col


def test_the_synthetic_input_reaches_the_rewritten_stages():
    """The reference is only worth something if it exercises splits and merges."""
    expected = pd.read_csv(EXPECTED)
    e = expected[expected["config"] == "collector_merge"]
    split = e.groupby("spatial_cluster_id")["temporal_cluster_id"].nunique()
    assert (split > 1).sum() >= 50
    nomerge = create_pipeline(**CONFIGS["collector_merge"], enable_merge=False).fit_transform(synthetic_input())
    assert nomerge["spatiotemporal_cluster_id"].nunique() > e["spatiotemporal_cluster_id"].nunique()


def test_n_jobs_does_not_change_the_ids():
    serial = _ids("collector_merge", n_jobs=1)
    pooled = _ids("collector_merge", n_jobs=2)
    pd.testing.assert_frame_equal(serial, pooled)


def _per_group_temporal(groups, days, eps):
    labels = np.empty(len(days), dtype=np.int64)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        labels[idx] = DBSCAN(eps=eps, min_samples=1).fit_predict(days[idx].reshape(-1, 1))
    return labels


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_temporal_components_equal_dbscan_per_group(seed):
    rng = np.random.default_rng(seed)
    n = 3000
    groups = rng.integers(0, 120, n)
    days = rng.integers(0, 60, n).astype(float)  # many equal days and steps of exactly eps
    got = temporal_components(groups, days, 3.0)
    assert (got == _per_group_temporal(groups, days, 3.0)).all()


def test_temporal_components_of_nothing():
    assert len(temporal_components(np.zeros(0, dtype=np.int64), np.zeros(0), 3.0)) == 0


def _groups_for_haversine(rng):
    coords = np.radians(np.column_stack([rng.uniform(30, 40, 4000), rng.uniform(-125, -115, 4000)]))
    groups = []
    start = 0
    while start < len(coords):
        size = int(rng.integers(1, 40))
        groups.append(np.arange(start, min(start + size, len(coords))))
        start += size
    # A group of identical points.
    coords[groups[3]] = coords[groups[3][0]]
    return coords, groups


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_haversine_components_equal_dbscan_per_group(n_jobs):
    rng = np.random.default_rng(1018)
    coords, groups = _groups_for_haversine(rng)
    eps = 50.0 / 6371.0088
    got = haversine_components(coords, groups, eps, n_jobs=n_jobs, chunk=7)
    for idx, lab in zip(groups, got, strict=True):
        want = DBSCAN(eps=eps, min_samples=1, metric="haversine", algorithm="ball_tree").fit_predict(coords[idx])
        assert (lab == want).all()
    assert any(len(g) == 1 for g in groups)


def test_cluster_csv_takes_jobs():
    out = subprocess.run(  # noqa: S603 - our own script, fixed arguments
        [sys.executable, str(ROOT / "scripts" / "cluster_csv.py"), "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "--jobs" in out.stdout
