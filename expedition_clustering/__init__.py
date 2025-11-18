"""
Top-level package for the Expedition Clustering toolkit.

This module exposes the most commonly used helpers for:
* Fetching tables from the CAS Botany MySQL backup (`data` module).
* Reproducing the cleaning and merging steps from the exploratory notebooks (`preprocessing` module).
* Building and tuning the spatiotemporal clustering pipeline (`pipeline` module).
"""

from .data import DatabaseConfig, fetch_table, load_core_tables
from .pipeline import (
    CombineClusters,
    Preprocessor,
    SpatialDBSCAN,
    TemporalDBSCAN,
    cluster_pipeline_scorer,
    create_pipeline,
    custom_cv_search,
    kfold_analysis,
    partial_ari_with_penalty,
    penalized_ari_scorer,
)
from .preprocessing import (
    build_clean_dataframe,
    clean_for_clustering,
    merge_core_tables,
)

__all__ = [
    "DatabaseConfig",
    "fetch_table",
    "load_core_tables",
    "build_clean_dataframe",
    "clean_for_clustering",
    "merge_core_tables",
    "CombineClusters",
    "Preprocessor",
    "SpatialDBSCAN",
    "TemporalDBSCAN",
    "cluster_pipeline_scorer",
    "create_pipeline",
    "custom_cv_search",
    "kfold_analysis",
    "partial_ari_with_penalty",
    "penalized_ari_scorer",
]

__version__ = "0.1.0"
