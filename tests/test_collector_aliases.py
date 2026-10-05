"""
Bug #430 (CAS Lens): the collector alias loader fails loudly.

load_collector_aliases(path) used to return {} when the file was missing, and
log a warning and return {} when its columns were wrong. A run given
--collector-alias-file then clustered without the spelling map and said
nothing, so its expeditions differed from what the caller asked for. Both
cases now raise, naming the file.

Requirement: EXP-ALIAS-1 (cas-lens docs/requirements/expedition-requirements.md).
"""

import pytest

from expedition_clustering.pipeline import CollectorPartitioning, load_collector_aliases


def test_missing_file_raises(tmp_path):
    missing = tmp_path / "aliases.csv"
    with pytest.raises(FileNotFoundError, match=r"aliases\.csv"):
        load_collector_aliases(missing)


def test_missing_columns_raise(tmp_path):
    bad = tmp_path / "aliases.csv"
    bad.write_text("name,id\nbeck r h,1\n")
    with pytest.raises(ValueError, match="normalized_name"):
        load_collector_aliases(bad)


def test_valid_file_loads(tmp_path):
    good = tmp_path / "aliases.csv"
    good.write_text("normalized_name,canonical_id\nbeck r h,7\nr h beck,7\n")
    assert load_collector_aliases(good) == {"beck r h": "7", "r h beck": "7"}


def test_partitioning_fit_raises_for_missing_file(tmp_path):
    step = CollectorPartitioning(enabled=True, alias_file=tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError):
        step.fit(None)
