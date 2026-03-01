import logging
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, make_scorer
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid
from sklearn.pipeline import Pipeline


def normalize_collector_name(name: str) -> str:
    """
    Normalize a single collector name for matching.

    This matches the normalization used by the CAS collector disambiguation system.
    Handles variations like:
    - "Beck, R. H." and "Beck, R.H." -> "beck r h"
    - "Möller, J." and "Moller, J." -> "moller j"

    Args:
        name: Raw collector name string (single person, not multi-collector)

    Returns:
        Lowercase, ASCII-only, punctuation-removed name
    """
    if pd.isna(name) or str(name).strip() == '' or str(name).lower() == 'unknown':
        return 'unknown'

    name = str(name).strip()

    # Normalize unicode (NFKD decomposes characters, e.g., é -> e + combining accent)
    name = unicodedata.normalize("NFKD", name)
    # Remove non-ASCII characters (removes combining accents)
    name = name.encode("ascii", "ignore").decode("ascii")

    # Lowercase
    name = name.lower()

    # Remove punctuation except spaces
    name = re.sub(r"[^\w\s]", "", name)

    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()

    return name if name else 'unknown'


# Patterns to filter out (not collector names)
COLLECTOR_IGNORE_PATTERNS = [
    re.compile(r'^et\s*al\.?$', re.IGNORECASE),
    re.compile(r'^al\.?$', re.IGNORECASE),
    re.compile(r'^others?$', re.IGNORECASE),
    re.compile(r'^party$', re.IGNORECASE),
    re.compile(r'^crew$', re.IGNORECASE),
    re.compile(r'^staff$', re.IGNORECASE),
    re.compile(r'^\d+$'),  # Just numbers
    re.compile(r'^[A-Z]$'),  # Single letters
    re.compile(r'^[IVX]+$'),  # Roman numerals
]


def _is_initials_or_short_name(s: str) -> bool:
    """Check if string looks like initials (e.g., 'R. H.') or a short first name."""
    s = s.strip()
    # Initials pattern: one or more "X." or "X. X." etc.
    if re.match(r'^([A-Z]\.?\s*)+$', s):
        return True
    # Short first name (1-3 letters, possibly with period)
    if re.match(r'^[A-Z][a-z]?\.?$', s):
        return True
    return False


def _looks_like_last_first_format(s: str) -> bool:
    """
    Check if string looks like "Last, First" or "Last, Initials" format.

    Examples that should return True:
    - "Beck, R. H."
    - "Smith, John"
    - "Mailliard, J."
    - "Heim, W.?" (question mark is common for uncertain attribution)
    - "Van Huizen, P. J." (multi-word last name)
    - "De Silva, A."

    Examples that should return False:
    - "Smith, Jones, Brown" (multiple people)
    - "CAS Invert. Zool. Dept." (institution)
    """
    if ',' not in s:
        return False

    parts = s.split(',', 1)  # Split on first comma only
    if len(parts) != 2:
        return False

    last, first = parts[0].strip(), parts[1].strip()

    # Remove trailing question mark (uncertain attribution)
    first = first.rstrip('?').strip()
    last = last.rstrip('?').strip()

    # Last name: single word or multi-word (Van Huizen, De Silva, etc.)
    # Must start with capital, can have spaces/hyphens, no periods
    if not re.match(r'^[A-Z][A-Za-z\-\'\s]+$', last):
        return False

    # But reject if last name has too many words (likely institution)
    if len(last.split()) > 3:
        return False

    # First should be initials, short name, or a first name
    # Check if it's initials or a reasonable first name
    if _is_initials_or_short_name(first):
        return True

    # Check if it's a first name (capitalized word, possibly with middle initial)
    if re.match(r'^[A-Z][a-z]+(\s+[A-Z]\.?)?$', first):
        return True

    return False


def parse_collector_string(collector_str: str) -> list[str]:
    """
    Parse a collector string into individual collector names.

    Handles formats like:
    - "R. Mooi & S. Lockhart" -> ["R. Mooi", "S. Lockhart"]
    - "Smith, John; Jones, Mary" -> ["Smith, John", "Jones, Mary"]
    - "Beck, R. H." -> ["Beck, R. H."]
    - "D.E. Breedlove and A. Clewell" -> ["D.E. Breedlove", "A. Clewell"]

    The key insight is that commas can either:
    1. Separate multiple collectors: "Smith, Jones, Brown"
    2. Be part of "Last, First" format: "Smith, John"

    We detect "Last, First" format and preserve those names intact.

    Args:
        collector_str: Raw collector field from specimen record

    Returns:
        List of individual collector names
    """
    if pd.isna(collector_str) or str(collector_str).strip() == '':
        return []

    collector_str = str(collector_str).strip()

    # Remove vessel information (e.g., "aboard R/V Searcher")
    collector_str = re.sub(
        r'\s+aboard\s+(?:the\s+)?(?:R/V|PFS|RV|M/V|S/V|USCGC|HMS|RRS)?\s*["\']?[^"\']+["\']?\s*$',
        '',
        collector_str,
        flags=re.IGNORECASE
    )

    # First, split on semicolon (always a person separator)
    semicolon_parts = collector_str.split(';')

    collectors = []
    for part in semicolon_parts:
        part = part.strip()
        if not part:
            continue

        # Check if this part is in "Last, First" format
        if _looks_like_last_first_format(part):
            collectors.append(part)
            continue

        # Otherwise, split on other separators: &, "and", "with", comma
        # But be careful with commas - only split if not "Last, First"
        subparts = re.split(r'\s*&\s*|\s+and\s+|\s+with\s+|\s+et\s+|\s+y\s+', part, flags=re.IGNORECASE)

        for subpart in subparts:
            subpart = subpart.strip()

            # Handle comma-separated within this subpart
            # Only split on comma if it doesn't look like "Last, First"
            if ',' in subpart and not _looks_like_last_first_format(subpart):
                comma_parts = subpart.split(',')
                for cp in comma_parts:
                    cp = cp.strip()
                    if cp and len(cp) >= 2:
                        if not any(p.match(cp) for p in COLLECTOR_IGNORE_PATTERNS):
                            collectors.append(cp)
            elif subpart and len(subpart) >= 2:
                if not any(p.match(subpart) for p in COLLECTOR_IGNORE_PATTERNS):
                    collectors.append(subpart)

    # Clean up whitespace in all collectors
    collectors = [re.sub(r'\s+', ' ', c).strip() for c in collectors]

    return collectors


def create_collector_group_key(
    collector_str: str,
    aliases: Optional[dict[str, str]] = None
) -> str:
    """
    Create a canonical key for a collector group.

    For multi-collector strings like "R. Mooi & S. Lockhart":
    1. Parse into individual collectors
    2. Normalize each (or look up in alias table)
    3. Sort alphabetically
    4. Join with "+"

    This ensures "R. Mooi & S. Lockhart" and "S. Lockhart & R. Mooi"
    produce the same key, and expeditions are only grouped when
    ALL collectors match.

    Args:
        collector_str: Raw collector field
        aliases: Optional dict mapping normalized names to canonical IDs

    Returns:
        Canonical collector group key (e.g., "lockhart s+mooi r")
    """
    collectors = parse_collector_string(collector_str)

    if not collectors:
        return 'unknown'

    # Normalize each collector
    normalized = []
    for coll in collectors:
        norm = normalize_collector_name(coll)
        # Try alias lookup
        if aliases and norm in aliases:
            norm = aliases[norm]
        normalized.append(norm)

    # Sort and join to create canonical key
    normalized.sort()
    return '+'.join(normalized)


def extract_primary_collector(s):
    """
    Extract the primary collector name from a collector string.

    Handles formats like:
    - "Smith, John"
    - "Smith, John; Jones, Mary" -> "Smith, John"
    - "Smith, John & Jones, Mary" -> "Smith, John & Jones, Mary"

    Returns 'unknown' for null/empty values.

    Note: For clustering, prefer create_collector_group_key() which handles
    multi-collector strings properly.
    """
    if pd.isna(s) or str(s).strip() == '' or str(s).lower() == 'unknown':
        return 'unknown'
    s = str(s).strip()
    # Split on semicolon first (separates collector groups in some formats)
    parts = s.split(';')
    return parts[0].strip()


def load_collector_aliases(alias_file: Path) -> dict[str, str]:
    """
    Load collector alias mappings from a CSV file.

    The file should have columns: normalized_name, canonical_id
    Where canonical_id is a stable identifier for the disambiguated collector.

    Args:
        alias_file: Path to CSV file with alias mappings

    Returns:
        Dict mapping normalized names to canonical IDs
    """
    if not alias_file.exists():
        return {}

    df = pd.read_csv(alias_file)
    if 'normalized_name' not in df.columns or 'canonical_id' not in df.columns:
        logging.warning(f"Alias file {alias_file} missing required columns")
        return {}

    return dict(zip(df['normalized_name'], df['canonical_id'].astype(str)))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two points in kilometers."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


# Vessel patterns in collector strings (e.g., "R/V Searcher", "aboard M/V Explorer")
_VESSEL_PATTERN = re.compile(
    r'(?:aboard\s+(?:the\s+)?)?'
    r'(?:R/V|M/V|S/V|RV|MV|SV|PFS|USCGC|HMS|RRS|F/V|FV)\s+'
    r'["\']?([A-Za-z][A-Za-z0-9\s\-\.]+)',
    re.IGNORECASE,
)


def extract_vessel(collector_str: str) -> Optional[str]:
    """Extract vessel name from a collector string, if present."""
    if pd.isna(collector_str) or not str(collector_str).strip():
        return None
    match = _VESSEL_PATTERN.search(str(collector_str))
    if match:
        return match.group(1).strip().rstrip('.')
    return None


def measure_collector_coherence(
    df: pd.DataFrame,
    cluster_col: str = 'spatiotemporal_cluster_id',
    collector_col: str = 'collectors',
) -> float:
    """
    Measure what fraction of expeditions have a single primary collector.

    Returns a float 0-1. High values (>0.7) mean collectors are reliable
    for merge scoring; low values mean multi-collector work is common.
    """
    if collector_col not in df.columns or cluster_col not in df.columns:
        return 0.5  # neutral default

    single_collector = 0
    total = 0

    for _, group in df.groupby(cluster_col):
        total += 1
        collectors = set()
        for raw in group[collector_col].dropna():
            parsed = parse_collector_string(str(raw))
            for c in parsed:
                collectors.add(normalize_collector_name(c))
        collectors.discard('unknown')
        if len(collectors) <= 1:
            single_collector += 1

    return single_collector / total if total > 0 else 0.5


@dataclass
class _ExpeditionSummary:
    """Summary statistics for an expedition used in merge scoring."""
    expedition_id: int
    specimen_count: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    centroid_lat: float
    centroid_lon: float
    collectors: set
    vessel: Optional[str]
    year: int


@dataclass
class _MergeCandidate:
    """A candidate pair of expeditions for merging."""
    exp_a: _ExpeditionSummary
    exp_b: _ExpeditionSummary
    gap_days: int
    distance_km: float
    merge_score: float


def _build_expedition_summary(
    expedition_id: int,
    group: pd.DataFrame,
    collector_col: str = 'collectors',
) -> _ExpeditionSummary:
    """Build summary statistics for a single expedition."""
    dates = pd.to_datetime(group['startdate']).dropna()
    start = dates.min()
    end = dates.max()

    centroid_lat = group['latitude1'].astype(float).mean()
    centroid_lon = group['longitude1'].astype(float).mean()

    collectors = set()
    vessel = None
    if collector_col in group.columns:
        for raw in group[collector_col].dropna():
            raw = str(raw)
            parsed = parse_collector_string(raw)
            for c in parsed:
                norm = normalize_collector_name(c)
                if norm != 'unknown':
                    collectors.add(norm)
            if vessel is None:
                vessel = extract_vessel(raw)

    return _ExpeditionSummary(
        expedition_id=expedition_id,
        specimen_count=len(group),
        start_date=start,
        end_date=end,
        centroid_lat=centroid_lat,
        centroid_lon=centroid_lon,
        collectors=collectors,
        vessel=vessel,
        year=start.year if pd.notna(start) else 0,
    )


def _compute_merge_score(
    exp_a: _ExpeditionSummary,
    exp_b: _ExpeditionSummary,
    max_merge_gap_days: int,
    max_merge_distance_km: float,
    collector_coherence: float,
    w_temporal: float = 0.25,
    w_spatial: float = 0.25,
    w_collector: float = 0.30,
    w_vessel: float = 0.15,
) -> float:
    """
    Compute merge score for two expeditions.

    Scoring components (each 0-1, linearly decaying from threshold):
    - temporal: proximity in days
    - spatial: proximity in km
    - collector overlap: Jaccard index
    - vessel match: binary
    """
    # Temporal: gap between end of earlier and start of later
    if exp_a.end_date <= exp_b.start_date:
        gap = (exp_b.start_date - exp_a.end_date).days
    else:
        gap = (exp_a.start_date - exp_b.end_date).days
    gap = max(0, gap)
    temporal_score = max(0.0, 1.0 - gap / max_merge_gap_days) if max_merge_gap_days > 0 else 0.0

    # Spatial: centroid distance
    dist = haversine_km(exp_a.centroid_lat, exp_a.centroid_lon,
                        exp_b.centroid_lat, exp_b.centroid_lon)
    spatial_score = max(0.0, 1.0 - dist / max_merge_distance_km) if max_merge_distance_km > 0 else 0.0

    # Collector overlap: Jaccard index
    if exp_a.collectors and exp_b.collectors:
        intersection = len(exp_a.collectors & exp_b.collectors)
        union = len(exp_a.collectors | exp_b.collectors)
        collector_overlap = intersection / union if union > 0 else 0.0
    else:
        collector_overlap = 0.0

    # Vessel match
    vessel_match = 0.0
    if (exp_a.vessel and exp_b.vessel and
            exp_a.vessel.lower() == exp_b.vessel.lower()):
        vessel_match = 1.0

    # Adaptive collector weighting based on collection-wide coherence
    effective_collector_weight = w_collector * collector_coherence
    fallback_weight = w_collector * (1 - collector_coherence) * 0.5

    score = (
        w_temporal * temporal_score +
        w_spatial * spatial_score +
        effective_collector_weight * collector_overlap +
        w_vessel * vessel_match +
        fallback_weight * (temporal_score + spatial_score) / 2
    )

    max_possible = w_temporal + w_spatial + w_collector + w_vessel
    return score / max_possible if max_possible > 0 else 0.0


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
            raise ValueError(
                f"Input dataframe is missing required columns: {missing}. Did the preprocessing step drop all rows?"
            )

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


# Step 1b: Collector Partitioning
class CollectorPartitioning(BaseEstimator, TransformerMixin):
    """
    Partition specimens by primary collector before spatial clustering.

    This ensures that different collectors are NEVER merged into the same
    expedition, even if they collected in the same place at the same time.

    For collections where collector-based partitioning is appropriate (like orn),
    this creates a collector_partition column that will be used to scope all
    subsequent clustering.

    Collector names are normalized to handle variations like:
    - "Beck, R. H." and "Beck, R.H." -> same partition
    - "Möller, J." and "Moller, J." -> same partition

    Optionally, an alias file can be provided for full disambiguation
    (mapping multiple name variants to canonical collector IDs).

    Parameters
    ----------
    enabled : bool
        Whether to enable collector partitioning. When False, all specimens
        get the same partition ID (effectively disabling partitioning).
    collector_col : str
        Column name containing collector information. Default: 'collectors'
    include_year : bool
        Whether to include year in the partition key. This prevents chaining
        across years for prolific collectors. Default: True for enabled mode.
    alias_file : Optional[Path]
        Path to a CSV file with collector alias mappings (normalized_name -> canonical_id).
        If provided, disambiguated collector IDs are used instead of normalized names.
    """

    def __init__(
        self,
        enabled=False,
        collector_col='collectors',
        include_year=True,
        alias_file: Optional[Path] = None
    ):
        self.enabled = enabled
        self.collector_col = collector_col
        self.include_year = include_year
        self.alias_file = alias_file
        self._aliases: dict[str, str] = {}

    def fit(self, X, y=None):
        # Load aliases if file provided
        if self.alias_file:
            self._aliases = load_collector_aliases(Path(self.alias_file))
            if self._aliases:
                logging.info(f"Loaded {len(self._aliases)} collector aliases")
        return self

    def _resolve_collector_group(self, raw_name: str) -> str:
        """
        Resolve a collector string to a canonical collector group key.

        For multi-collector strings like "R. Mooi & S. Lockhart":
        - Parses into individual collectors
        - Normalizes/disambiguates each
        - Sorts and joins to create canonical key

        This ensures "R. Mooi & S. Lockhart" and "S. Lockhart & R. Mooi"
        are treated as the same collector group.
        """
        return create_collector_group_key(raw_name, self._aliases)

    def transform(self, X):
        X = X.copy()

        if not self.enabled:
            # All specimens in same partition - standard behavior
            X['collector_partition'] = 0
            return X

        # Extract primary collector (for display) and collector group key (for clustering)
        if self.collector_col in X.columns:
            X['primary_collector'] = X[self.collector_col].apply(extract_primary_collector)
            # Use collector GROUP key for partitioning (handles multi-collector properly)
            X['collector_group'] = X[self.collector_col].apply(self._resolve_collector_group)
        else:
            X['primary_collector'] = 'unknown'
            X['collector_group'] = 'unknown'

        # Build partition key using collector group
        if self.include_year and 'startdate' in X.columns:
            X['_year'] = pd.to_datetime(X['startdate']).dt.year.fillna(0).astype(int)
            X['_partition_key'] = X['collector_group'] + '_' + X['_year'].astype(str)
            X = X.drop(columns=['_year'])
        else:
            X['_partition_key'] = X['collector_group']

        # Convert to integer partition IDs
        partition_map = {k: i for i, k in enumerate(X['_partition_key'].unique())}
        X['collector_partition'] = X['_partition_key'].map(partition_map)
        X = X.drop(columns=['_partition_key'])

        return X


# Step 2: Custom Transformer for Spatial DBSCAN Clustering
class SpatialDBSCAN(BaseEstimator, TransformerMixin):
    """
    Perform spatial DBSCAN clustering within each collector partition.

    If collector_partition column exists, clustering is done separately
    within each partition to prevent merging different collectors.
    """

    def __init__(self, e_dist):
        self.e_dist = e_dist

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Coerce coordinates to numeric early to avoid string/object dtype surprises
        X["latitude1"] = pd.to_numeric(X["latitude1"], errors="coerce")
        X["longitude1"] = pd.to_numeric(X["longitude1"], errors="coerce")

        eps_rad = self.e_dist / 6371  # Convert e_dist to radians

        # Check if we have collector partitions
        if 'collector_partition' in X.columns and X['collector_partition'].nunique() > 1:
            # Cluster within each partition, ensuring unique IDs across partitions
            X['spatial_cluster_id'] = -1
            next_cluster_id = 0

            for partition_id in X['collector_partition'].unique():
                mask = X['collector_partition'] == partition_id
                partition_data = X.loc[mask]

                if len(partition_data) == 0:
                    continue

                coords = np.radians(partition_data[["latitude1", "longitude1"]].values)

                db = DBSCAN(
                    eps=eps_rad,
                    min_samples=1,
                    metric="haversine",
                    algorithm="ball_tree",
                )
                labels = db.fit_predict(coords)

                # Offset labels to ensure uniqueness across partitions
                X.loc[mask, 'spatial_cluster_id'] = labels + next_cluster_id
                next_cluster_id += labels.max() + 1
        else:
            # Standard behavior - cluster all together
            coords = np.radians(X[["latitude1", "longitude1"]].values)
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
                sample = ", ".join(f"id={cid} comps={comps} size={n}" for cid, comps, n in spatial_bad[:10])
                parts.append(
                    f"{len(spatial_bad)} clusters spatially disconnected at e_dist={self.e_dist}km (examples: {sample})"
                )
            if temporal_bad:
                sample = ", ".join(f"id={cid} comps={comps} size={n}" for cid, comps, n in temporal_bad[:10])
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


class MergeExpeditions(BaseEstimator, TransformerMixin):
    """
    Merge nearby expeditions that likely belong together.

    After initial DBSCAN produces tight clusters, this stage merges
    clusters that share collectors, are close in space/time, and score
    above a threshold. This implements the "two-pass" approach from
    the collections_explorer tuning system.

    Process:
    1. Build per-expedition summaries (centroid, dates, collectors, vessel)
    2. Find candidate pairs (same year, within gap/distance thresholds)
    3. Score using weighted formula (temporal + spatial + collector + vessel)
    4. Greedily merge highest-scoring pairs above threshold
    5. Repeat until convergence or max_iterations
    """

    def __init__(
        self,
        max_merge_gap_days=3,
        max_merge_distance_km=30.0,
        merge_threshold=0.6,
        max_gap_days=3,
        max_iterations=10,
    ):
        self.max_merge_gap_days = max_merge_gap_days
        self.max_merge_distance_km = max_merge_distance_km
        self.merge_threshold = merge_threshold
        self.max_gap_days = max_gap_days
        self.max_iterations = max_iterations

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        logger = logging.getLogger(__name__)

        if 'spatiotemporal_cluster_id' not in X.columns:
            return X

        # Measure collection-wide collector coherence once
        collector_coherence = measure_collector_coherence(X)
        logger.info("Merge stage: collector_coherence=%.3f", collector_coherence)

        total_merges = 0

        for iteration in range(1, self.max_iterations + 1):
            # Build summaries for all current expeditions
            summaries = {}
            for exp_id, group in X.groupby('spatiotemporal_cluster_id'):
                summaries[exp_id] = _build_expedition_summary(exp_id, group)

            # Find and score merge candidates
            candidates = self._find_candidates(summaries, collector_coherence)

            if not candidates:
                logger.info("Merge iteration %d: no candidates, converged", iteration)
                break

            # Sort by score descending (greedy: best merges first)
            candidates.sort(key=lambda c: c.merge_score, reverse=True)

            # Execute merges greedily
            merged_this_round = set()
            round_merges = 0

            for candidate in candidates:
                if candidate.merge_score < self.merge_threshold:
                    break  # sorted desc, so no more will pass

                a_id = candidate.exp_a.expedition_id
                b_id = candidate.exp_b.expedition_id

                if a_id in merged_this_round or b_id in merged_this_round:
                    continue

                # Validate: merged expedition must not have internal gaps > max_gap_days
                if not self._validate_merge(X, candidate):
                    continue

                # Execute: reassign B's specimens to A's cluster
                X.loc[
                    X['spatiotemporal_cluster_id'] == b_id,
                    'spatiotemporal_cluster_id'
                ] = a_id

                merged_this_round.add(a_id)
                merged_this_round.add(b_id)
                round_merges += 1

            total_merges += round_merges
            logger.info(
                "Merge iteration %d: %d merges (from %d candidates)",
                iteration, round_merges, len(candidates),
            )

            if round_merges == 0:
                break

        if total_merges > 0:
            logger.info("Merge stage complete: %d total merges", total_merges)

        return X

    def _find_candidates(
        self,
        summaries: dict[int, _ExpeditionSummary],
        collector_coherence: float,
    ) -> list[_MergeCandidate]:
        """Find and score all valid merge candidate pairs."""
        candidates = []

        # Group by year for efficiency
        by_year: dict[int, list[_ExpeditionSummary]] = {}
        for s in summaries.values():
            by_year.setdefault(s.year, []).append(s)

        for year_exps in by_year.values():
            # Sort by start_date for efficient gap checking
            year_exps.sort(key=lambda s: s.start_date)

            for i, exp_a in enumerate(year_exps):
                for j in range(i + 1, len(year_exps)):
                    exp_b = year_exps[j]

                    # Temporal gap check (hard constraint)
                    if exp_a.end_date <= exp_b.start_date:
                        gap = (exp_b.start_date - exp_a.end_date).days
                    else:
                        gap = (exp_a.start_date - exp_b.end_date).days
                    gap = max(0, gap)

                    if gap > self.max_merge_gap_days:
                        # Since sorted by start_date, if gap is too large
                        # all subsequent will also be too large
                        break

                    # Spatial distance check (hard constraint)
                    dist = haversine_km(
                        exp_a.centroid_lat, exp_a.centroid_lon,
                        exp_b.centroid_lat, exp_b.centroid_lon,
                    )
                    if dist > self.max_merge_distance_km:
                        continue

                    # Score
                    score = _compute_merge_score(
                        exp_a, exp_b,
                        self.max_merge_gap_days,
                        self.max_merge_distance_km,
                        collector_coherence,
                    )

                    if score >= self.merge_threshold:
                        candidates.append(_MergeCandidate(
                            exp_a=exp_a,
                            exp_b=exp_b,
                            gap_days=gap,
                            distance_km=dist,
                            merge_score=score,
                        ))

        return candidates

    def _validate_merge(self, df: pd.DataFrame, candidate: _MergeCandidate) -> bool:
        """Validate that a merge won't create internal gaps exceeding threshold."""
        a_id = candidate.exp_a.expedition_id
        b_id = candidate.exp_b.expedition_id

        mask = df['spatiotemporal_cluster_id'].isin([a_id, b_id])
        combined = df.loc[mask]

        dates = pd.to_datetime(combined['startdate']).dropna().sort_values()
        if len(dates) < 2:
            return True

        # Use vessel-aware gap threshold
        has_vessel = candidate.exp_a.vessel is not None or candidate.exp_b.vessel is not None
        gap_threshold = self.max_gap_days * 2 if has_vessel else self.max_gap_days

        # Check all consecutive date gaps
        day_values = dates.values.astype('datetime64[D]').astype(float)
        gaps = np.diff(day_values)
        max_internal_gap = gaps.max() if len(gaps) > 0 else 0

        return max_internal_gap <= gap_threshold


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
def create_pipeline(
    e_dist,
    e_days,
    collector_aware=False,
    include_year_in_partition=True,
    collector_alias_file: Optional[Path] = None,
    merge_gap_days: int = 3,
    merge_distance_km: float = 30.0,
    merge_threshold: float = 0.6,
    enable_merge: bool = True,
):
    """
    Create the expedition clustering pipeline.

    Parameters
    ----------
    e_dist : float
        Spatial epsilon in kilometers for initial DBSCAN clustering.
    e_days : float
        Temporal epsilon in days for initial DBSCAN clustering.
    collector_aware : bool
        If True, partition by collector before clustering.
    include_year_in_partition : bool
        If True (and collector_aware=True), include year in partition key.
    collector_alias_file : Optional[Path]
        Path to CSV file with collector disambiguation mappings.
    merge_gap_days : int
        Maximum temporal gap (days) between expeditions to consider merging.
    merge_distance_km : float
        Maximum spatial distance (km) between expedition centroids to merge.
    merge_threshold : float
        Minimum merge score (0-1) required to execute a merge.
    enable_merge : bool
        If True, run the merge stage after initial clustering.

    Returns
    -------
    Pipeline
        Configured sklearn Pipeline for expedition clustering.
    """
    steps = [
        ("preprocessor", Preprocessor()),
        ("collector_partition", CollectorPartitioning(
            enabled=collector_aware,
            include_year=include_year_in_partition,
            alias_file=collector_alias_file
        )),
        ("spatial_dbscan", SpatialDBSCAN(e_dist=e_dist)),
        ("iterative_spatiotemporal", IterativeSpatiotemporalClustering(e_dist=e_dist, e_days=e_days)),
        ("validate_connectivity", ValidateSpatiotemporalConnectivity(e_dist=e_dist, e_days=e_days)),
    ]

    if enable_merge:
        steps.append(("merge_expeditions", MergeExpeditions(
            max_merge_gap_days=merge_gap_days,
            max_merge_distance_km=merge_distance_km,
            merge_threshold=merge_threshold,
            max_gap_days=int(e_days),
        )))

    return Pipeline(steps)


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

    param_grid = {"spatial_dbscan__e_dist": e_dist_values, "temporal_dbscan__e_days": e_days_values}

    pipeline = create_pipeline(e_dist=0.01, e_days=30)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=cluster_pipeline_scorer,  # Use the updated scorer
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        refit=True,
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
param_grid = {"spatial_dbscan__e_dist": [0.1, 1, 5, 10, 15, 20], "temporal_dbscan__e_days": [3, 5, 7, 9, 10]}
