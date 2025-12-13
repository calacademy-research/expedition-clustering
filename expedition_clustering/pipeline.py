import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, make_scorer
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid
from sklearn.pipeline import Pipeline


# Step 1: Custom Transformer for Preprocessing
class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        required_columns = ["collectingeventid", "latitude1", "longitude1", "startdate"]
        missing = [col for col in required_columns if col not in X.columns]
        if missing:
            raise ValueError(f"Input dataframe is missing required columns: {missing}. Did the preprocessing step drop all rows?")

        # Make a copy to avoid modifying the original
        X = X.copy()

        # Remove duplicate collectingeventid, keeping the first occurrence
        X = X.drop_duplicates(subset="collectingeventid", keep="first")

        # Drop rows with null latitude1, longitude1, or startdate (should already be handled)
        X = X.dropna(subset=["latitude1", "longitude1", "startdate"])

        # Drop rows outside valid latitude and longitude ranges
        X = X[(X["latitude1"].between(-90, 90)) & (X["longitude1"].between(-180, 180))]

        # Drop rows outside valid startdate range
        # Use a timezone-naive timestamp to match parsed dates
        today = pd.Timestamp.utcnow().tz_localize(None)
        min_year = 1800
        X["startdate"] = pd.to_datetime(X["startdate"], errors="coerce")
        X = X[(X["startdate"].dt.year >= min_year) & (X["startdate"] <= today)]

        return X.reset_index(drop=True)

# Step 2: Custom Transformer for Spatial DBSCAN Clustering
class SpatialDBSCAN(BaseEstimator, TransformerMixin):
    def __init__(self, e_dist):
        self.e_dist = e_dist

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Coerce coordinates to numeric early to avoid string/object dtype surprises
        X["latitude1"] = pd.to_numeric(X["latitude1"], errors="coerce")
        X["longitude1"] = pd.to_numeric(X["longitude1"], errors="coerce")
        coords = np.radians(X[["latitude1", "longitude1"]].values)
        eps_rad = self.e_dist / 6371  # Convert e_dist to radians

        db = DBSCAN(
            eps=eps_rad,
            min_samples=1,
            metric="haversine",
            algorithm="ball_tree",
        )
        labels = db.fit_predict(coords)

        X["spatial_cluster_id"] = labels
        return X

# Step 3: Custom Transformer for Temporal DBSCAN Clustering
class TemporalDBSCAN(BaseEstimator, TransformerMixin):
    def __init__(self, e_days):
        self.e_days = e_days

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Initialize to noise; we will fill per spatial cluster
        X["temporal_cluster_id"] = -1
        eps_days = float(self.e_days)

        for spatial_id in X["spatial_cluster_id"].unique():
            mask = X["spatial_cluster_id"] == spatial_id
            times = X.loc[mask, "startdate"].values.astype("datetime64[D]").astype(float).reshape(-1, 1)

            db = DBSCAN(
                eps=eps_days,
                min_samples=1,
                metric="euclidean",
                algorithm="ball_tree",
            )
            temporal_labels = db.fit_predict(times)
            X.loc[mask, "temporal_cluster_id"] = temporal_labels

        return X

# Step 4: Custom Transformer to Combine Clusters
class CombineClusters(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Drop any existing spatiotemporal labels to avoid merge suffixes
        for col in ("spatiotemporal_cluster_id", "spatiotemporal_cluster_id_x", "spatiotemporal_cluster_id_y"):
            if col in X:
                X = X.drop(columns=[col])
        # Save original indices
        original_indices = X.index

        # Generate unique integer IDs for spatiotemporal clusters
        cluster_combinations = X[["spatial_cluster_id", "temporal_cluster_id"]].drop_duplicates().reset_index(drop=True)
        cluster_combinations["spatiotemporal_cluster_id"] = range(len(cluster_combinations))
        X = X.merge(cluster_combinations, on=["spatial_cluster_id", "temporal_cluster_id"], how="left")

        # Restore original indices
        X.index = original_indices
        return X


class ValidateSpatiotemporalConnectivity(BaseEstimator, TransformerMixin):
    """
    Validate that each spatiotemporal cluster is both spatially and temporally
    connected using the configured epsilons.

    Raises ValueError if any cluster fails the connectivity checks.
    """

    def __init__(self, e_dist, e_days):
        self.e_dist = e_dist
        self.e_days = e_days

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "spatiotemporal_cluster_id" not in X.columns:
            raise ValueError("Missing spatiotemporal_cluster_id after iteration; pipeline state is invalid.")

        eps_rad = self.e_dist / 6371
        time_eps = float(self.e_days)
        spatial_bad = []
        temporal_bad = []

        for stc_id, sub in X.groupby("spatiotemporal_cluster_id"):
            if len(sub) <= 1:
                continue
            coords = np.radians(sub[["latitude1", "longitude1"]].astype(float).to_numpy())
            labels = DBSCAN(
                eps=eps_rad,
                min_samples=1,
                metric="haversine",
                algorithm="ball_tree",
            ).fit_predict(coords)
            if labels.max() > 0:
                spatial_bad.append((stc_id, int(labels.max() + 1), len(sub)))

            # Temporal connectivity on 1D day offsets
            times = sub["startdate"].to_numpy(dtype="datetime64[D]").astype(float).reshape(-1, 1)
            if np.isnan(times).all():
                continue
            t_labels = DBSCAN(
                eps=time_eps,
                min_samples=1,
                metric="euclidean",
                algorithm="ball_tree",
            ).fit_predict(times)
            if t_labels.max() > 0:
                temporal_bad.append((stc_id, int(t_labels.max() + 1), len(sub)))

        if spatial_bad or temporal_bad:
            parts = []
            if spatial_bad:
                sample = ", ".join(
                    f"id={cid} comps={comps} size={n}" for cid, comps, n in spatial_bad[:10]
                )
                parts.append(
                    f"{len(spatial_bad)} clusters spatially disconnected at e_dist={self.e_dist}km (examples: {sample})"
                )
            if temporal_bad:
                sample = ", ".join(
                    f"id={cid} comps={comps} size={n}" for cid, comps, n in temporal_bad[:10]
                )
                parts.append(
                    f"{len(temporal_bad)} clusters temporally disconnected at e_days={self.e_days} (examples: {sample})"
                )
            raise ValueError("; ".join(parts))
        return X


class SpatialReconnectWithinTemporal(BaseEstimator, TransformerMixin):
    """
    After temporal clustering, re-run spatial DBSCAN inside each
    (spatial_cluster_id, temporal_cluster_id) to break apart any spatially
    disconnected blobs, then update spatial_cluster_id accordingly.
    """

    def __init__(self, e_dist):
        self.e_dist = e_dist

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        eps_rad = self.e_dist / 6371
        labels = X["spatial_cluster_id"].to_numpy()
        next_label = int(labels.max()) + 1 if labels.size else 0

        for (_sp_id, _t_id), sub in X.groupby(["spatial_cluster_id", "temporal_cluster_id"]):
            if len(sub) <= 1:
                continue
            coords = np.radians(sub[["latitude1", "longitude1"]].astype(float).to_numpy())
            comp_labels = DBSCAN(
                eps=eps_rad,
                min_samples=1,
                metric="haversine",
                algorithm="ball_tree",
            ).fit_predict(coords)
            unique = np.unique(comp_labels)
            if len(unique) == 1:
                continue

            base = unique[0]
            idxs = sub.index.to_numpy()
            for comp in unique:
                comp_rows = idxs[comp_labels == comp]
                if comp == base:
                    continue
                X.loc[comp_rows, "spatial_cluster_id"] = next_label
                next_label += 1

        return X


class TemporalDBSCANRecompute(BaseEstimator, TransformerMixin):
    """
    Recompute temporal DBSCAN per spatial cluster (after any spatial splitting)
    to ensure temporal labels reflect the final spatial partitions.
    """

    def __init__(self, e_days):
        self.e_days = e_days

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["temporal_cluster_id"] = -1
        eps_days = float(self.e_days)

        for spatial_id in X["spatial_cluster_id"].unique():
            mask = X["spatial_cluster_id"] == spatial_id
            times = X.loc[mask, "startdate"].values.astype("datetime64[D]").astype(float).reshape(-1, 1)
            db = DBSCAN(
                eps=eps_days,
                min_samples=1,
                metric="euclidean",
                algorithm="ball_tree",
            )
            labels = db.fit_predict(times)
            X.loc[mask, "temporal_cluster_id"] = labels
        return X


class SpatialReconnectWithinSpatiotemporal(BaseEstimator, TransformerMixin):
    """
    After combining, ensure each spatiotemporal cluster is spatially connected;
    if not, split into new spatiotemporal IDs per spatial component.
    """

    def __init__(self, e_dist):
        self.e_dist = e_dist

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "spatiotemporal_cluster_id" not in X.columns:
            return X
        eps_rad = self.e_dist / 6371
        next_id = int(X["spatiotemporal_cluster_id"].max()) + 1 if len(X) else 0

        for _stc_id, sub in X.groupby("spatiotemporal_cluster_id"):
            if len(sub) <= 1:
                continue
            coords = np.radians(sub[["latitude1", "longitude1"]].astype(float).to_numpy())
            labels = DBSCAN(
                eps=eps_rad,
                min_samples=1,
                metric="haversine",
                algorithm="ball_tree",
            ).fit_predict(coords)
            unique = np.unique(labels)
            if len(unique) == 1:
                continue
            base = unique[0]
            idxs = sub.index.to_numpy()
            for comp in unique:
                comp_rows = idxs[labels == comp]
                if comp == base:
                    continue
                X.loc[comp_rows, "spatiotemporal_cluster_id"] = next_id
                next_id += 1
        return X


class TemporalReconnectWithinSpatiotemporal(BaseEstimator, TransformerMixin):
    """
    After spatial reconnect at the spatiotemporal level, ensure temporal
    connectivity; split into new spatiotemporal IDs per temporal component.
    """

    def __init__(self, e_days):
        self.e_days = e_days

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "spatiotemporal_cluster_id" not in X.columns:
            return X
        eps_days = float(self.e_days)
        next_id = int(X["spatiotemporal_cluster_id"].max()) + 1 if len(X) else 0

        for _stc_id, sub in X.groupby("spatiotemporal_cluster_id"):
            if len(sub) <= 1:
                continue
            times = sub["startdate"].to_numpy(dtype="datetime64[D]").astype(float).reshape(-1, 1)
            labels = DBSCAN(
                eps=eps_days,
                min_samples=1,
                metric="euclidean",
                algorithm="ball_tree",
            ).fit_predict(times)
            unique = np.unique(labels)
            if len(unique) == 1:
                continue
            base = unique[0]
            idxs = sub.index.to_numpy()
            for comp in unique:
                comp_rows = idxs[labels == comp]
                if comp == base:
                    continue
                X.loc[comp_rows, "spatiotemporal_cluster_id"] = next_id
                next_id += 1
        return X


class IterativeSpatiotemporalClustering(BaseEstimator, TransformerMixin):
    """
    Run spatial→temporal clustering with reconnect/validation in a loop until
    no spatial or temporal disconnects remain (or max_iter is reached).
    """

    def __init__(self, e_dist, e_days, max_iter=5):
        self.e_dist = e_dist
        self.e_days = e_days
        self.max_iter = max_iter

        # Reuse the existing components
        self.temporal_dbscan = TemporalDBSCAN(e_days=e_days)
        self.spatial_reconnect = SpatialReconnectWithinTemporal(e_dist=e_dist)
        self.temporal_recompute = TemporalDBSCANRecompute(e_days=e_days)
        self.combiner = CombineClusters()
        self.spatial_reconnect_st = SpatialReconnectWithinSpatiotemporal(e_dist=e_dist)
        self.temporal_reconnect_st = TemporalReconnectWithinSpatiotemporal(e_days=e_days)
        self.validator = ValidateSpatiotemporalConnectivity(e_dist=e_dist, e_days=e_days)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        last_error = None
        for iteration in range(1, self.max_iter + 1):
            X = self.temporal_dbscan.transform(X)
            X = self.spatial_reconnect.transform(X)
            X = self.temporal_recompute.transform(X)
            X = self.combiner.transform(X)
            X = self.spatial_reconnect_st.transform(X)
            X = self.temporal_reconnect_st.transform(X)

            # Stats/logging
            logger = logging.getLogger(__name__)
            spatial_n = X["spatial_cluster_id"].nunique()
            temporal_n = X["temporal_cluster_id"].nunique()
            stc_n = X["spatiotemporal_cluster_id"].nunique() if "spatiotemporal_cluster_id" in X.columns else 0
            spatial_bad, temporal_bad = self._disconnect_stats(X)
            logger.info(
                "Iterative pass %s/%s -> spatial:%s temporal:%s spatiotemporal:%s | spatial_bad:%s temporal_bad:%s",
                iteration,
                self.max_iter,
                spatial_n,
                temporal_n,
                stc_n,
                len(spatial_bad),
                len(temporal_bad),
            )

            try:
                self.validator.transform(X)
                return X  # success
            except ValueError as e:
                last_error = e
                # loop again to further split problem clusters
                continue

        if last_error:
            raise last_error
        return X

    def _disconnect_stats(self, X):
        eps_rad = self.e_dist / 6371
        time_eps = float(self.e_days)
        spatial_bad = []
        temporal_bad = []
        if "spatiotemporal_cluster_id" not in X.columns:
            return spatial_bad, temporal_bad
        for stc_id, sub in X.groupby("spatiotemporal_cluster_id"):
            if len(sub) <= 1:
                continue
            coords = np.radians(sub[["latitude1", "longitude1"]].astype(float).to_numpy())
            labels = DBSCAN(
                eps=eps_rad,
                min_samples=1,
                metric="haversine",
                algorithm="ball_tree",
            ).fit_predict(coords)
            if labels.max() > 0:
                spatial_bad.append(stc_id)

            times = sub["startdate"].to_numpy(dtype="datetime64[D]").astype(float).reshape(-1, 1)
            t_labels = DBSCAN(
                eps=time_eps,
                min_samples=1,
                metric="euclidean",
                algorithm="ball_tree",
            ).fit_predict(times)
            if t_labels.max() > 0:
                temporal_bad.append(stc_id)
        return spatial_bad, temporal_bad

# Custom scorer for penalized ARI
# NOTE: This scoring metric isn't really working! Area to work on...
def partial_ari_with_penalty(true_labels, predicted_labels):
    """
    Compute a penalized Adjusted Rand Index for partially labeled data.
    Penalizes when unlabeled points are assigned the same cluster as labeled points.
    """
    # Convert inputs to numpy arrays for easier manipulation
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Mask for valid (non-NaN) labels
    valid_mask = ~np.isnan(true_labels) & ~np.isnan(predicted_labels)
    true_labels_filtered = true_labels[valid_mask]
    predicted_labels_filtered = predicted_labels[valid_mask]

    # Compute ARI for the filtered subset
    if len(true_labels_filtered) == 0 or len(predicted_labels_filtered) == 0:
        return 0.0  # Return 0 if no valid labels are available

    ari_score = adjusted_rand_score(true_labels_filtered, predicted_labels_filtered)

    # Penalize cases where unlabeled rows share cluster IDs with labeled rows
    labeled_mask = ~np.isnan(true_labels)
    unlabeled_mask = np.isnan(true_labels)
    labeled_cluster_ids = set(predicted_labels[labeled_mask])
    unlabeled_cluster_ids = predicted_labels[unlabeled_mask]
    penalty_count = sum(cid in labeled_cluster_ids for cid in unlabeled_cluster_ids)

    # Define a penalty factor (adjustable based on sensitivity)
    penalty_factor = 100
    penalty = penalty_factor * penalty_count / len(predicted_labels)

    # Return the penalized ARI score
    penalized_score = ari_score - penalty
    return max(0.0, penalized_score)  # Ensure the score is not negative

# Create a scorer object for GridSearchCV or cross-validation
penalized_ari_scorer = make_scorer(partial_ari_with_penalty, greater_is_better=True)

# Create the pipeline
def create_pipeline(e_dist, e_days):
    return Pipeline(
        [
            ("preprocessor", Preprocessor()),
            ("spatial_dbscan", SpatialDBSCAN(e_dist=e_dist)),
            ("iterative_spatiotemporal", IterativeSpatiotemporalClustering(e_dist=e_dist, e_days=e_days)),
            ("validate_connectivity", ValidateSpatiotemporalConnectivity(e_dist=e_dist, e_days=e_days)),
        ]
    )

# Custom scorer for GridSearchCV
# NOTE: Shouldn't be used until scorer is improved
def cluster_pipeline_scorer(estimator, X, y):
    """
    Custom scorer for clustering pipelines that evaluates using `transform` output.
    """
    # Transform the data to get the predicted labels
    transformed = estimator.transform(X)

    # Align transformed data with X_test's indices
    transformed = transformed.reindex(X.index)
    predicted_labels = transformed["spatiotemporal_cluster_id"].values

    # Ensure `y` is also aligned with X_test
    y_aligned = y.reindex(X.index).values

    # Compute the penalized Adjusted Rand Index
    return partial_ari_with_penalty(y_aligned, predicted_labels)

# Perform K-Fold Analysis
# NOTE: Shouldn't be used until scorer is improved
def kfold_analysis(df, e_dist_values, e_days_values):
    labeled_df = df.dropna(subset=["cluster"])
    X = labeled_df.drop(columns=["cluster"])
    y = labeled_df["cluster"]

    param_grid = {
        "spatial_dbscan__e_dist": e_dist_values,
        "temporal_dbscan__e_days": e_days_values
    }

    pipeline = create_pipeline(e_dist=0.01, e_days=30)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=cluster_pipeline_scorer,  # Use the updated scorer
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        refit=True
    )

    grid_search.fit(X, y)

    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)

    return grid_search.best_estimator_


def custom_cv_search(processed_df, pipeline, param_grid, n_clusters=10):
    """
    Perform cross-validation search to minimize the average percentage difference
    between manual and algorithm-determined cluster sizes.

    Parameters
    ----------
    - processed_df: pd.DataFrame
        DataFrame containing the clustering results and the `cluster` column.
    - pipeline: sklearn.pipeline.Pipeline
        Pipeline to evaluate.
    - param_grid: dict
        Dictionary of pipeline parameters to test.
    - n_clusters: int
        Number of manual clusters to evaluate.

    Returns
    -------
    - best_params: dict
        Best parameters that minimize the metric.
    - best_score: float
        Best average clust_len_diff_perc score.
    - scores: list
        List of average scores for each parameter combination.

    """
    # Initialize the parameter grid
    grid = ParameterGrid(param_grid)
    best_score = float("inf")
    best_params = None
    scores = []

    for params in grid:
        # Update pipeline parameters
        pipeline.set_params(**params)

        # Fit the pipeline and transform the data
        processed_data = pipeline.fit_transform(processed_df)

        total_diff_perc = 0

        for i in range(n_clusters):
            # Filter manual cluster
            df1 = processed_data[processed_data["cluster"] == i]

            if len(df1) == 0:  # Skip if the cluster is empty
                continue

            # Get the spatiotemporal_cluster_id for the manual cluster
            stc_id = df1.iloc[0]["spatiotemporal_cluster_id"]

            # Compute cluster sizes
            manual_clust_len = len(df1)
            algo_clust_len = len(processed_data[processed_data["spatiotemporal_cluster_id"] == stc_id])

            # Compute percentage difference
            clust_len_diff_perc = abs(manual_clust_len - algo_clust_len) / manual_clust_len
            total_diff_perc += clust_len_diff_perc

        # Average percentage difference for this parameter combination
        avg_diff_perc = total_diff_perc / n_clusters
        scores.append(avg_diff_perc)

        # Update best parameters if current score is better
        if avg_diff_perc < best_score:
            best_score = avg_diff_perc
            best_params = params

    return best_params, best_score, scores

# Example Usage
param_grid = {
    "spatial_dbscan__e_dist": [.1, 1, 5, 10, 15, 20],
    "temporal_dbscan__e_days": [3, 5, 7, 9, 10]
}
