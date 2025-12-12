"""
Expedition Clustering - Group botanical specimens into collection expeditions.

Main functionality:
    create_pipeline: Build a spatiotemporal DBSCAN clustering pipeline

Visualization:
    plot_geographical_positions: Plot specimen locations on a map
    plot_geographical_heatmap: Create density heatmaps
    plot_time_histogram: Show temporal distribution

For advanced usage (notebooks):
    Database utilities: DatabaseConfig, fetch_table, load_core_tables
    Preprocessing: build_clean_dataframe, clean_for_clustering, merge_core_tables
    Pipeline components: SpatialDBSCAN, TemporalDBSCAN, Preprocessor, CombineClusters

Basic usage:
    >>> from expedition_clustering import create_pipeline
    >>> pipeline = create_pipeline(e_dist=10, e_days=7)
    >>> clustered_df = pipeline.fit_transform(clean_dataframe)
"""

# Core clustering functionality (most users only need this)
from .pipeline import create_pipeline

# Visualization functions
from .plotting import (
    plot_geographical_heatmap,
    plot_geographical_heatmap_by_day,
    plot_geographical_positions,
    plot_time_histogram,
)

# Database utilities (for advanced users and notebooks)
from .data import DatabaseConfig, fetch_table, load_core_tables

# Preprocessing utilities (for advanced users and notebooks)
from .preprocessing import (
    build_clean_dataframe,
    clean_for_clustering,
    merge_core_tables,
)

# Pipeline components (for customization)
from .pipeline import (
    CombineClusters,
    Preprocessor,
    SpatialDBSCAN,
    TemporalDBSCAN,
    SpatialReconnectWithinTemporal,
    ValidateSpatiotemporalConnectivity,
)

# Evaluation utilities (experimental, not recommended for production)
from .pipeline import (
    cluster_pipeline_scorer,
    custom_cv_search,
    kfold_analysis,
    partial_ari_with_penalty,
    penalized_ari_scorer,
)

__all__ = [
    # Core API
    "create_pipeline",
    # Visualization
    "plot_geographical_heatmap",
    "plot_geographical_heatmap_by_day",
    "plot_geographical_positions",
    "plot_time_histogram",
    # Database utilities
    "DatabaseConfig",
    "fetch_table",
    "load_core_tables",
    # Preprocessing
    "build_clean_dataframe",
    "clean_for_clustering",
    "merge_core_tables",
    # Pipeline components
    "CombineClusters",
    "Preprocessor",
    "SpatialDBSCAN",
    "TemporalDBSCAN",
    "SpatialReconnectWithinTemporal",
    "ValidateSpatiotemporalConnectivity",
    # Evaluation (experimental)
    "cluster_pipeline_scorer",
    "custom_cv_search",
    "kfold_analysis",
    "partial_ari_with_penalty",
    "penalized_ari_scorer",
]

__version__ = "0.1.0"
