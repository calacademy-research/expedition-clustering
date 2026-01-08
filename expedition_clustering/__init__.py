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
