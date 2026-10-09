"""
Bug #918 (CAS Lens, EXP-CLUSTER-8): the clustering output names where each
row's coordinates came from.

The Lens weekly `clustering` stage hands cluster_csv.py a merged PortalData.csv:
Specify's coordinates where they exist, the locality engine's otherwise, with a
`coordinate_source` column (specify | engine | empty). The column is carried
through the transform and the pipeline into clustered_expeditions.csv so the
Lens can tell an expedition built on the engine's points from one built on
Specify's.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location("csv_source", _ROOT / "expedition_clustering" / "csv_source.py")
_csv_source = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_csv_source)
transform_csv_to_pipeline_format = _csv_source.transform_csv_to_pipeline_format


def _raw(n: int, source: list[str] | None) -> pd.DataFrame:
    df = pd.DataFrame({
        "spid": [f"urn:catalog:CAS:BOT:{i}" for i in range(n)],
        "catalogNumber": list(range(n)),
        "latitude1": [37.9] * n,
        "longitude1": [-122.6] * n,
        "startDate": [1950] * n,
        "ce_startDate": [6] * n,
        "ce_startDate1": [1 + (i % 2) for i in range(n)],
        "collectors": ["A. Eastwood"] * n,
        "localityName": ["Mt. Tamalpais"] * n,
    })
    if source is not None:
        df["coordinate_source"] = source
    return df


def test_transform_carries_coordinate_source():
    out = transform_csv_to_pipeline_format(_raw(3, ["specify", "engine", ""]))
    assert list(out["coordinate_source"].fillna("")) == ["specify", "engine", ""]


def test_transform_without_the_column_adds_none():
    out = transform_csv_to_pipeline_format(_raw(3, None))
    assert "coordinate_source" not in out.columns


def test_cluster_csv_output_carries_coordinate_source(tmp_path):
    coll = tmp_path / "botany"
    coll.mkdir()
    source = ["engine" if i % 3 == 0 else "specify" for i in range(30)]
    _raw(30, source).to_csv(coll / "PortalData.csv", index=False)
    out = tmp_path / "clustered_expeditions.csv"
    result = subprocess.run(  # noqa: S603 - our own script, fixed argv
        [sys.executable, str(_ROOT / "scripts" / "cluster_csv.py"), str(coll), "-o", str(out)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    clustered = pd.read_csv(out, keep_default_na=False)
    assert "coordinate_source" in clustered.columns
    by_spid = dict(zip(clustered["spid"], clustered["coordinate_source"], strict=True))
    assert by_spid == {f"urn:catalog:CAS:BOT:{i}": s for i, s in enumerate(source)}
