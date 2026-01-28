"""
Expedition Clustering - Group botanical specimens into collection expeditions.

Main functionality:
    create_pipeline: Build a spatiotemporal DBSCAN clustering pipeline

Data sources:
    Database: DatabaseConfig, fetch_table, load_core_tables
    CSV: load_collection_csv, list_available_collections

Visualization:
    plot_geographical_positions: Plot specimen locations on a map
    plot_geographical_heatmap: Create density heatmaps
    plot_time_histogram: Show temporal distribution

For advanced usage (notebooks):
    Preprocessing: build_clean_dataframe, clean_for_clustering, merge_core_tables
    Pipeline components: SpatialDBSCAN, TemporalDBSCAN, Preprocessor, CombineClusters

Basic usage (database):
    >>> from expedition_clustering import create_pipeline, DatabaseConfig, load_core_tables
    >>> config = DatabaseConfig()
    >>> tables = load_core_tables(config)
    >>> clean_df = build_clean_dataframe(tables)
    >>> pipeline = create_pipeline(e_dist=10, e_days=7)
    >>> clustered_df = pipeline.fit_transform(clean_df)

Basic usage (CSV):
    >>> from expedition_clustering import create_pipeline, load_collection_csv
    >>> df = load_collection_csv("/path/to/incoming_data/botany")
    >>> pipeline = create_pipeline(e_dist=10, e_days=7)
    >>> clustered_df = pipeline.fit_transform(df)
"""

# Core clustering functionality (most users only need this)
# CSV data source utilities
from .csv_source import list_available_collections, load_collection_csv

# Database utilities (for advanced users and notebooks)
from .data import DatabaseConfig, fetch_table, load_core_tables

# Pipeline components (for customization)
# Evaluation utilities (experimental, not recommended for production)
from .pipeline import (
    CombineClusters,
    IterativeSpatiotemporalClustering,
    Preprocessor,
    SpatialDBSCAN,
    SpatialReconnectWithinSpatiotemporal,
    SpatialReconnectWithinTemporal,
    TemporalDBSCAN,
    TemporalDBSCANRecompute,
    TemporalReconnectWithinSpatiotemporal,
    ValidateSpatiotemporalConnectivity,
    cluster_pipeline_scorer,
    create_pipeline,
    custom_cv_search,
    kfold_analysis,
    partial_ari_with_penalty,
    penalized_ari_scorer,
)

# Visualization functions
from .plotting import (
    plot_geographical_heatmap,
    plot_geographical_heatmap_by_day,
    plot_geographical_positions,
    plot_time_histogram,
)

# Preprocessing utilities (for advanced users and notebooks)
from .preprocessing import (
    build_clean_dataframe,
    clean_for_clustering,
    merge_core_tables,
)
from .redaction import (
    RedactionDropVerificationResult,
    RedactionVerificationResult,
    fetch_redaction_flags,
    redact_clustered_csv,
    redact_clustered_dataframe,
    verify_redacted_csv,
    verify_redacted_csv_drop,
    verify_redaction,
    verify_redaction_drop,
)

__all__ = [
    "CombineClusters",
    "DatabaseConfig",
    "IterativeSpatiotemporalClustering",
    "Preprocessor",
    "RedactionDropVerificationResult",
    "RedactionVerificationResult",
    "SpatialDBSCAN",
    "SpatialReconnectWithinSpatiotemporal",
    "SpatialReconnectWithinTemporal",
    "TemporalDBSCAN",
    "TemporalDBSCANRecompute",
    "TemporalReconnectWithinSpatiotemporal",
    "ValidateSpatiotemporalConnectivity",
    "build_clean_dataframe",
    "clean_for_clustering",
    "cluster_pipeline_scorer",
    "create_pipeline",
    "custom_cv_search",
    "fetch_redaction_flags",
    "fetch_table",
    "kfold_analysis",
    "list_available_collections",
    "load_collection_csv",
    "load_core_tables",
    "merge_core_tables",
    "partial_ari_with_penalty",
    "penalized_ari_scorer",
    "plot_geographical_heatmap",
    "plot_geographical_heatmap_by_day",
    "plot_geographical_positions",
    "plot_time_histogram",
    "redact_clustered_csv",
    "redact_clustered_dataframe",
    "verify_redacted_csv",
    "verify_redacted_csv_drop",
    "verify_redaction",
    "verify_redaction_drop",
]

__version__ = "0.1.0"
