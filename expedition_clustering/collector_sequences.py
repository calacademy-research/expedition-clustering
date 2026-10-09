"""
Collector-number sequences (CAS Lens feature #922).

Beside the spatial clustering, botany specimens are grouped by collector +
collector number (``stationFieldNumber``) + collecting day, so specimens with
no coordinates can join the expedition their numbered neighbours belong to.

The rules are the botany collector-number spot check's (cas-lens
``docs/analysis/botany-collector-number-spotcheck.md``), written out in
cas-lens ``docs/requirements/bugs/922/PLAN.md``:

* parse  ``parse_collector_number``: integers, decorated integers, thousands
  commas, letter suffixes, letter prefixes (own series), year-prefixed and
  year-suffixed numbers read as (year, n), ranges and number-parts; s.n.,
  slash/dotted pairs, plot codes and other mixed values give no number.
* date   ``day_precise``: day-precise dates only; a day of 1 that is a month
  or year fill is not a day.
* key    ``collector_key``: the first-listed collector, surname + first
  initial, folded the way the clustering folds names.
* link   ``build_sequences``: rows of one key, series and calendar year are
  linked when their numbers differ by <= 20 and their days by <= 2 (the same
  number: <= 1 day); a sequence is a connected set of >= 2 linked rows.
* exclude, per row, the first rule that applies (``RULES`` order).

The merge with the spatial clusters (``merge_with_clusters``) never moves a
clustered row; see that function.

Coordinates: every coordinate used here is the row's merged coordinate,
Specify's or else the georeference engine's, as the Lens computes it
(cas-lens ``portal.etl.merged_coordinates``) and passes it in a coordinate
file with a ``coordinate_source`` column (specify | engine | empty).
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass, field
from datetime import date as dt_date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# The name folding the spatial clustering uses. Loaded from the
# sibling file, as scripts/cluster_csv.py does, so importing this module does
# not import the package __init__ (cartopy).
_spec = importlib.util.spec_from_file_location("_ec_pipeline", Path(__file__).parent / "pipeline.py")
_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pipeline)
normalize_collector_name = _pipeline.normalize_collector_name

# ---------------------------------------------------------------- rules

RULE_SN = "s.n."
RULE_UNPARSED = "unparsed code"
RULE_NO_DAY = "no day-precise date"
RULE_BEFORE_1890 = "before 1890"
RULE_NONPERSONAL = "non-personal collector"
RULE_NONSEQUENTIAL = "numbers do not follow dates"
RULE_SPREAD = "spread over 500 km"
# The order a row is tested in; a row counts under the first rule it meets.
RULES = (RULE_SN, RULE_UNPARSED, RULE_NO_DAY, RULE_BEFORE_1890, RULE_NONPERSONAL,
         RULE_NONSEQUENTIAL, RULE_SPREAD)

NUMBER_GAP = 20          # numbers within 20 link (geonomia's eps; spot check §4)
DAY_GAP = 2              # days within 2 link (spot check: 5.7% > 500 km vs 8.0% at 7)
SAME_NUMBER_DAY_GAP = 1  # one number on dates > 1 day apart is two specimens' numbers
FIRST_YEAR = 1890        # before 1890: distribution and exsiccata numbers
SPREAD_KM = 500          # robust spread cut
SPREAD_QUANTILE = 0.9    # 90th percentile of distances from the median point
# Collectors whose numbers do not follow dates (Tharp, Knoche): a key with at
# least this many n / n+1 pairs ...
NONSEQ_MIN_PAIRS = 10
# ... of which fewer than this share have an n and an n+1 row within ...
NONSEQ_MIN_CLOSE = 0.5
NONSEQ_CLOSE_DAYS = 3    # ... 3 days of each other.
# Reused series (Bartholomew, Eastwood), reported, not excluded: a key with at
# least this many numbers, this share of them on dates > 1 day apart.
REUSED_MIN_NUMBERS = 10
REUSED_MIN_SHARE = 0.1
YEAR_SCALE = 100000      # (year, n) is stored as year * 100000 + n
EARTH_KM = 6371.0

# ---------------------------------------------------------------- parsing

_SN = re.compile(
    r"^\s*(s\.?\s*n\.?|n\.?\s*s\.?|sin\s+n[uú]m(ero)?\.?|no\s+number|none|without\s+number|s/n)\s*\.*$",
    re.IGNORECASE,
)
_SN_EXSICCATA = re.compile(r"^s\.?\s*n\.?\s*\[", re.IGNORECASE)
_LEAD = r"[#!\(\[\s]*"           # leading decoration
_TRAIL = r"[\s\.,;\*!\?\)\]]*"   # trailing decoration
_INTEGER = re.compile(_LEAD + r"(\d{1,3}(?:,\d{3})+|\d+)" + _TRAIL)
_PAIR = re.compile(r"(\d+)\s*-\s*(\d+)" + _TRAIL)
_SUFFIX = re.compile(_LEAD + r"(\d+)\s*(?:[a-zA-Z]{1,3}|bis|ter|½|-[a-zA-Z])\.?" + _TRAIL)
_PREFIX = re.compile(r"([A-Za-z][A-Za-z\.]*)[\s\.\-#]*(\d+)" + _TRAIL)
_RANGE_MAX_SPAN = 50
_PART_MAX = 100
_YY = 100


def _near_year(value: int, year: int) -> bool:
    """Whether value is year's last two digits, or the year before's or after's."""
    yy = year % _YY
    return value in (yy, (yy - 1) % _YY, (yy + 1) % _YY)


def parse_collector_number(raw: str, year: int | None) -> tuple[str, str, int | None]:  # noqa: PLR0911 - one return per format
    """
    Read a collector number (``stationFieldNumber``).

    Inputs:  raw  -- the cell as text; year -- the collecting year, or None.
    Output:  (rule, series, number). rule is "" when a number was read, else
             RULE_SN or RULE_UNPARSED (series "" and number None). series is
             "" for a plain number, the upper-cased prefix letters for a
             prefixed one ("LUTEYN 14018" -> "LUTEYN"), "Y" for yy-n and "YS"
             for n-yy, whose number is year * 100000 + n.
    """
    s = raw.strip()
    if s == "" or _SN.match(s) or _SN_EXSICCATA.match(s):
        return (RULE_SN, "", None)
    m = _INTEGER.fullmatch(s)
    if m:
        return ("", "", int(m.group(1).replace(",", "")))
    m = _PAIR.fullmatch(s)
    if m:
        first, second = m.group(1), m.group(2)
        a, b = int(first), int(second)
        if year is not None:
            if (len(first) <= 2 and _near_year(a, year)) or (len(first) == 4 and abs(a - year) <= 1):  # noqa: PLR2004
                return ("", "Y", year * YEAR_SCALE + b)
            if len(second) == 2 and _near_year(b, year):  # noqa: PLR2004
                return ("", "YS", year * YEAR_SCALE + a)
        if a < b <= a + _RANGE_MAX_SPAN and len(second) >= len(first):
            return ("", "", a)
        if len(second) < len(first) and b < _PART_MAX:
            return ("", "", a)
        return (RULE_UNPARSED, "", None)
    m = _SUFFIX.fullmatch(s)
    if m:
        return ("", "", int(m.group(1)))
    m = _PREFIX.fullmatch(s)
    if m:
        return ("", re.sub(r"[^A-Z]", "", m.group(1).upper()), int(m.group(2)))
    return (RULE_UNPARSED, "", None)


# ---------------------------------------------------------------- dating

_HAS_ONE = re.compile(r"(?<!\d)0?1(?!\d)|1st")
_YEAR_ONLY = re.compile(r"\s*\d{4}\s*")
_MONTH_ONLY = re.compile(r"\s*[A-Za-z]+\.?,?\s+\d{4}\s*|\s*\d{4}\s*[-/]\s*\d{1,2}\s*")
_FIRST_DATE_YEAR = 1700
_LAST_DATE_YEAR = 2026


def day_precise(year: int | None, month: int | None, day: int | None, verbatim: str) -> dt_date | None:
    """
    The collecting day, when the record knows it.

    Inputs:  year, month, day -- Specify's date components (None when empty);
             verbatim -- the verbatim date text ("" when empty).
    Output:  the date, or None when the day is not known: a component is
             missing, the year is outside 1700-2026, it is not a calendar
             date, the verbatim date is a year or a month alone, or the day is
             1 without a 1 in the verbatim date (or with an empty verbatim
             date in January): the 1st that pads a month or a year.
    """
    if year is None or month is None or day is None:
        return None
    if not (_FIRST_DATE_YEAR <= year <= _LAST_DATE_YEAR):
        return None
    if _YEAR_ONLY.fullmatch(verbatim) or _MONTH_ONLY.fullmatch(verbatim):
        return None
    if day == 1 and not _HAS_ONE.search(verbatim) and (verbatim != "" or month == 1):
        return None
    try:
        return dt_date(year, month, day)
    except ValueError:
        # Not a calendar date (30 February): the record does not know the day.
        return None


# ---------------------------------------------------------------- collector key

_NONPERSON = re.compile(
    r"unspecified|unknown|anonym|ex herb|herb\.|survey|expedition|club|society|universit|school|class|"
    r"collector|botanical|garden|museum|grupo|team|staff|^s\.?\s*n\.?$|^n/?a$",
    re.IGNORECASE,
)
_TITLE = re.compile(r"^(mrs|mr|dr|miss|rev|prof|sr|fr|brother|sister)\.?\s+")


def collector_key(collectors: str) -> tuple[str, bool]:
    """
    The collector-number key of a collectors string.

    Inputs:  the ``collectors`` cell.
    Output:  (key, personal). key is the first-listed collector's surname and
             first initial, folded by normalize_collector_name ("Thomas, John
             H." and "Thomas, J. H." -> "thomas j"); personal is False when
             the string is empty or its first collector is not a person
             (survey, herbarium, "unknown", ...), and then the key is "".
    """
    # The first person: the text before the first ";", "&" or " and ". Not
    # parse_collector_string, which reads "Thomas, John Hunter" and
    # "Breedlove6, D E" as two people (its "Last, First" test wants a capital
    # word and initials); the spot check's split keeps them one.
    first = re.split(r";|\s&\s|\s+and\s+", collectors or "")[0].strip()
    if not first:
        return ("", False)
    if _NONPERSON.search(first):
        return ("", False)
    if "," in first:
        last, given = first.split(",", 1)
    else:
        tokens = first.replace(".", ". ").split()
        last, given = (tokens[-1], " ".join(tokens[:-1])) if len(tokens) > 1 else (first, "")
    surname = re.sub(r"[^a-z ]", "", normalize_collector_name(last)).strip()
    if surname in ("", "unknown"):
        return ("", False)
    given = _TITLE.sub("", given.strip().lower())
    folded = re.sub(r"[^a-z]", "", normalize_collector_name(given)) if given.strip() else ""
    initial = "" if folded == "unknown" else folded[:1]
    return ((surname + " " + initial).strip(), True)


# ---------------------------------------------------------------- rows

RAW_COLUMNS = ("spid", "collectors", "stationFieldNumber", "startDate", "ce_startDate", "ce_startDate1",
               "verbatimDate", "latitude1", "longitude1", "coordinate_source")
COORDINATE_SOURCES = frozenset({"specify", "engine", ""})


def _int_or_none(text: str) -> int | None:
    t = text.strip()
    return int(float(t)) if re.fullmatch(r"-?\d+(\.0+)?", t) else None


def prepare_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """
    One row per specimen with what the sequence pass reads.

    Inputs:  raw -- the merged clustering input's columns RAW_COLUMNS as text:
             PortalData.csv with latitude1 / longitude1 set to the row's
             effective coordinate and coordinate_source (specify | engine |
             empty) naming it (cas-lens portal.etl.synthetic_localities
             write_clustering_input, bug #918).
    Output:  frame with spid, collectors, key, series, number (Int64), day
             (days since 1970-01-01, Int64), year (Int64), lat, lng (float,
             NaN without), coordinate_source, rule ("" or the first RULES
             entry met up to RULE_NONPERSONAL; the later rules are applied by
             build_sequences).
    Raises:  ValueError naming a missing column (a plain PortalData.csv has no
             coordinate_source: its engine-located rows would count as
             unlocated), a duplicated spid, an unknown coordinate_source, or a
             located row whose source is empty / an unlocated row with one.
    """
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        msg = (f"the clustering input lacks column(s) {', '.join(missing)} that the collector-number "
               "sequences read (coordinate_source comes from the merged input, bug #918)")
        raise ValueError(msg)
    raw = raw.loc[:, list(RAW_COLUMNS)].fillna("").astype(str)
    if raw["spid"].duplicated().any():
        msg = f"{int(raw['spid'].duplicated().sum())} duplicated spid(s) in the clustering input"
        raise ValueError(msg)
    bad = set(raw["coordinate_source"]) - COORDINATE_SOURCES
    if bad:
        msg = f"unknown coordinate_source value(s) {sorted(bad)}; expected specify, engine or empty"
        raise ValueError(msg)
    located = (raw["latitude1"].str.strip() != "") & (raw["longitude1"].str.strip() != "")
    mismatch = located != (raw["coordinate_source"] != "")
    if mismatch.any():
        msg = (f"{int(mismatch.sum())} clustering input row(s) disagree with their coordinate_source "
               f"(first spid {raw.loc[mismatch, 'spid'].iloc[0]}): a located row names its source, an unlocated one none")
        raise ValueError(msg)

    out = pd.DataFrame({"spid": raw["spid"].to_numpy(), "collectors": raw["collectors"].to_numpy()})
    years = [_int_or_none(v) for v in raw["startDate"]]
    months = [_int_or_none(v) for v in raw["ce_startDate"]]
    days = [_int_or_none(v) for v in raw["ce_startDate1"]]
    epoch = dt_date(1970, 1, 1)
    day_cache: dict[tuple, int | None] = {}
    ordinals = []
    for key in zip(years, months, days, raw["verbatimDate"], strict=True):
        if key not in day_cache:
            d = day_precise(key[0], key[1], key[2], key[3].strip())
            day_cache[key] = None if d is None else (d - epoch).days
        ordinals.append(day_cache[key])
    out["day"] = pd.array(ordinals, dtype="Int64")
    out["year"] = pd.array(years, dtype="Int64")

    parsed = [parse_collector_number(s, y) for s, y in zip(raw["stationFieldNumber"], years, strict=True)]
    out["series"] = [p[1] for p in parsed]
    out["number"] = pd.array([p[2] for p in parsed], dtype="Int64")
    keys = {c: collector_key(c) for c in pd.unique(out["collectors"])}
    out["key"] = out["collectors"].map(lambda c: keys[c][0])
    personal = out["collectors"].map(lambda c: keys[c][1]).to_numpy(dtype=bool)

    out["lat"] = pd.to_numeric(raw["latitude1"].str.strip().mask(~located), errors="raise").to_numpy(dtype=float)
    out["lng"] = pd.to_numeric(raw["longitude1"].str.strip().mask(~located), errors="raise").to_numpy(dtype=float)
    out["coordinate_source"] = raw["coordinate_source"].to_numpy()

    rule = np.array([p[0] for p in parsed], dtype=object)
    free = rule == ""
    rule[free & out["day"].isna().to_numpy()] = RULE_NO_DAY
    free = rule == ""
    rule[free & (out["year"].fillna(0).to_numpy() < FIRST_YEAR)] = RULE_BEFORE_1890
    free = rule == ""
    rule[free & ~personal] = RULE_NONPERSONAL
    out["rule"] = rule
    return out


# ---------------------------------------------------------------- sequences


@dataclass
class SequenceResult:
    """build_sequences' answer: the rows and the keys it flagged."""

    rows: pd.DataFrame
    reused_keys: set = field(default_factory=set)
    nonsequential_keys: set = field(default_factory=set)
    spread_km: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def excluded_counts(self) -> dict[str, int]:
        """Rows excluded per rule, every rule present (0 when none)."""
        counts = self.rows["rule"].value_counts()
        return {r: int(counts.get(r, 0)) for r in RULES}

    def alone_rows(self) -> int:
        """Rows that passed every rule but link to no other row."""
        return int(((self.rows["rule"] == "") & self.rows["sequence_id"].isna()).sum())


def _pairs_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Distinct (key, series, number, day) of the rows."""
    return rows[["key", "series", "number", "day"]].drop_duplicates()


def _nonsequential_keys(rows: pd.DataFrame) -> set:
    """
    (key, series) whose numbers do not follow their dates.

    For each number n with an n + 1 under the same key and series, the pair is
    close when some n row and some n + 1 row are within NONSEQ_CLOSE_DAYS
    days. A key with >= NONSEQ_MIN_PAIRS pairs and a close share under
    NONSEQ_MIN_CLOSE is returned (Tharp 28%, Knoche 41%; Bartholomew 88%).
    """
    t = _pairs_table(rows)
    nxt = t.assign(number=t["number"] - 1)
    j = t.merge(nxt, on=["key", "series", "number"], suffixes=("", "_next"))
    if j.empty:
        return set()
    j["close"] = (j["day"] - j["day_next"]).abs() <= NONSEQ_CLOSE_DAYS
    per_number = j.groupby(["key", "series", "number"])["close"].any()
    per_key = per_number.groupby(level=["key", "series"]).agg(["size", "mean"])
    bad = per_key[(per_key["size"] >= NONSEQ_MIN_PAIRS) & (per_key["mean"] < NONSEQ_MIN_CLOSE)]
    return set(bad.index)


def _reused_keys(rows: pd.DataFrame) -> set:
    """Keys whose numbers recur on dates > 1 day apart (restarted series)."""
    t = _pairs_table(rows)
    span = t.groupby(["key", "series", "number"])["day"].agg(lambda d: int(d.max() - d.min()))
    per_key = (span > SAME_NUMBER_DAY_GAP).groupby(level="key").agg(["size", "mean"])
    hit = per_key[(per_key["size"] >= REUSED_MIN_NUMBERS) & (per_key["mean"] >= REUSED_MIN_SHARE)]
    return set(hit.index)


def _components(rows: pd.DataFrame) -> np.ndarray:
    """
    Component label per row of the link graph (rows given in any order).

    Rows are sorted by (key, series, year, day, number); for growing offsets k
    every row i is compared with row i + k while both are in one group and
    within DAY_GAP days (the sort makes the day difference grow with k, so a
    row leaves the window for good). An edge needs the number gap <= NUMBER_GAP
    and, for the same number, the day gap <= SAME_NUMBER_DAY_GAP.
    """
    n = len(rows)
    order = np.lexsort((rows["number"].to_numpy(dtype=np.int64), rows["day"].to_numpy(dtype=np.int64),
                        rows["year"].to_numpy(dtype=np.int64),
                        pd.factorize(rows["key"].astype(str) + "\x1f" + rows["series"].astype(str))[0]))
    group = pd.factorize((rows["key"].astype(str) + "\x1f" + rows["series"].astype(str) + "\x1f"
                          + rows["year"].astype(str)).to_numpy()[order])[0]
    day = rows["day"].to_numpy(dtype=np.int64)[order]
    num = rows["number"].to_numpy(dtype=np.int64)[order]
    src, dst = [], []
    active = np.arange(n - 1)
    k = 1
    while active.size:
        j = active + k
        keep = j < n
        active, j = active[keep], j[keep]
        same = (group[active] == group[j]) & (day[j] - day[active] <= DAY_GAP)
        active, j = active[same], j[same]
        dnum = np.abs(num[j] - num[active])
        edge = (dnum <= NUMBER_GAP) & ((dnum > 0) | (day[j] - day[active] <= SAME_NUMBER_DAY_GAP))
        src.append(active[edge])
        dst.append(j[edge])
        k += 1
    s = np.concatenate(src) if src else np.array([], dtype=np.int64)
    d = np.concatenate(dst) if dst else np.array([], dtype=np.int64)
    graph = coo_matrix((np.ones(len(s), dtype=np.int8), (s, d)), shape=(n, n))
    _, labels_sorted = connected_components(graph, directed=False)
    labels = np.empty(n, dtype=np.int64)
    labels[order] = labels_sorted
    return labels


def _haversine_km(lat1, lng1, lat2, lng2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lng2 - lng1) * p / 2) ** 2)
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def robust_spread_km(located: pd.DataFrame, by: str) -> pd.DataFrame:
    """
    Per group: the median point and the 90th-percentile distance from it.

    Inputs:  located rows (lat, lng not NaN) and the grouping column.
    Output:  frame indexed by group: median_lat, median_lng, spread_km. One
             bad point (a sign flip, lat = lon) cannot set the figure.
    """
    med = located.groupby(by)[["lat", "lng"]].median().rename(columns={"lat": "median_lat", "lng": "median_lng"})
    j = located[[by, "lat", "lng"]].join(med, on=by)
    j["km"] = _haversine_km(j["lat"].to_numpy(), j["lng"].to_numpy(),
                            j["median_lat"].to_numpy(), j["median_lng"].to_numpy())
    med["spread_km"] = j.groupby(by)["km"].quantile(SPREAD_QUANTILE)
    return med


def build_sequences(rows: pd.DataFrame) -> SequenceResult:
    """
    Link the rows into collector-number sequences.

    Inputs:  prepare_rows' frame.
    Output:  SequenceResult whose rows gain sequence_id (Int64; <NA> for rows
             excluded or linked to nothing, ids 0.. in key/day order) and whose
             rule also carries RULE_NONSEQUENTIAL and RULE_SPREAD;
             candidate_sequence_id also keeps the spread-cut sequences' ids.
             reused_keys / nonsequential_keys name the flagged keys;
             spread_km is the robust spread per sequence id (sequences with
             >= 2 located rows, the spread-cut ones included).
    """
    rows = rows.copy()
    eligible = rows["rule"] == ""
    candidates = rows[eligible]
    nonseq = _nonsequential_keys(candidates)
    if nonseq:
        pairs = pd.MultiIndex.from_frame(rows[["key", "series"]])
        hit = eligible & pairs.isin(list(nonseq))
        rows.loc[hit, "rule"] = RULE_NONSEQUENTIAL
    reused = _reused_keys(rows[rows["rule"] == ""])

    linkable = rows[rows["rule"] == ""]
    seq = pd.Series(pd.NA, index=rows.index, dtype="Int64")
    spread = pd.Series(dtype=float)
    if not linkable.empty:
        labels = pd.Series(_components(linkable), index=linkable.index)
        sizes = labels.map(labels.value_counts())
        labels = labels[sizes >= 2]  # noqa: PLR2004 - a sequence is two rows or more
        # Stable ids: number the sequences in (key, first day, first number) order.
        first = (linkable.loc[labels.index].assign(label=labels)
                 .groupby("label").agg(key=("key", "first"), day=("day", "min"), number=("number", "min"))
                 .sort_values(["key", "day", "number"]))
        renumber = pd.Series(np.arange(len(first)), index=first.index)
        seq.loc[labels.index] = labels.map(renumber).to_numpy()

        rows["sequence_id"] = seq
        located = rows[rows["sequence_id"].notna() & rows["lat"].notna() & rows["lng"].notna()]
        counts = located.groupby("sequence_id").size()
        located = located[located["sequence_id"].map(counts) >= 2]  # noqa: PLR2004
        if not located.empty:
            spread = robust_spread_km(located, "sequence_id")["spread_km"]
            wide = set(spread[spread > SPREAD_KM].index)
            hit = rows["sequence_id"].isin(wide).fillna(value=False).to_numpy(dtype=bool)
            rows.loc[hit, "rule"] = RULE_SPREAD
    # candidate_sequence_id keeps the spread-cut sequences' ids for the sequence
    # table; sequence_id is the sequences that passed every rule.
    rows["candidate_sequence_id"] = seq
    rows["sequence_id"] = seq
    rows.loc[rows["rule"] != "", "sequence_id"] = pd.NA
    return SequenceResult(rows=rows, reused_keys=set(reused),
                          nonsequential_keys={k for k, _ in nonseq}, spread_km=spread)



# ---------------------------------------------------------------- merge with the spatial clusters

JOIN_SPATIAL = "spatial"
JOIN_NUMBER = "collector_number"
DISPOSITION_IN_CLUSTER = "in cluster"
DISPOSITION_ATTACHED = "attached"
DISPOSITION_SEQUENCE_ONLY = "sequence-only"
DISPOSITION_BELOW_FLOOR = "below floor"
DISPOSITION_SPREAD = RULE_SPREAD
SOURCES = ("specify", "engine", "none")
# Score weights for the nearest clustered row: number gap first, then day gap,
# then the cluster's rank. Days in one sequence are < 366 and a sequence holds
# fewer than 10**6 clusters, so the weights never overlap.
_DAY_WEIGHT = 10**6
_NUMBER_WEIGHT = 10**9
SEQUENCE_TABLE_COLUMNS = ["collector_sequence_id", "collector_key", "series", "number_min", "number_max",
                          "day_min", "day_max", "rows", "specify_rows", "engine_rows", "unlocated_rows",
                          "spread_km", "median_lat", "median_lng", "cluster_ids", "disposition"]
_EPOCH = dt_date(1970, 1, 1)


def _source_label(source: str) -> str:
    return source if source else "none"


def _by_source(sources: pd.Series) -> dict[str, int]:
    counts = sources.map(_source_label).value_counts()
    return {s: int(counts.get(s, 0)) for s in SOURCES}


@dataclass
class MergeResult:
    """merge_with_clusters' answer."""

    # spid, spatiotemporal_cluster_id, cluster_join (always JOIN_NUMBER): the
    # rows the sequences add to the output, nothing else.
    assignments: pd.DataFrame
    # One row per sequence (PLAN §4.2), the spread-cut ones included.
    sequences: pd.DataFrame
    sequence_only_expeditions: int
    sequence_only_rows: int
    # The rows in a sequence that passed every rule: spid, sequence_id,
    # coordinate_source, and the cluster they end in (-1: none).
    sequence_rows: pd.DataFrame

    def attached_rows(self) -> int:
        """Rows added to an existing spatial cluster."""
        return int(self.assignments["attached"].sum())

    def attached_by_source(self) -> dict[str, int]:
        """Rows added to an existing spatial cluster, by coordinate source."""
        spids = set(self.assignments.loc[self.assignments["attached"], "spid"])
        return _by_source(self.sequence_rows.loc[self.sequence_rows["spid"].isin(spids), "coordinate_source"])

    def in_sequences_by_source(self) -> dict[str, int]:
        """Rows in a sequence that passed every rule, by coordinate source."""
        return _by_source(self.sequence_rows["coordinate_source"])


def _nearest_clusters(seq: pd.DataFrame, cluster: np.ndarray) -> np.ndarray:
    """
    For one sequence's rows, the cluster each un-clustered row joins.

    Inputs:  the sequence's rows (number, day) and their cluster ids (-1 for
             none; at least one row clustered).
    Output:  cluster id per row: its own for a clustered row; for the others
             the cluster of the clustered row nearest in number, then in day,
             then the lower cluster id.
    """
    num = seq["number"].to_numpy(dtype=np.int64)
    day = seq["day"].to_numpy(dtype=np.int64)
    done = cluster >= 0
    out = cluster.copy()
    c_num, c_day, c_id = num[done], day[done], cluster[done]
    rank = np.unique(c_id, return_inverse=True)[1]
    todo = np.flatnonzero(~done)
    score = (np.abs(num[todo, None] - c_num[None, :]) * _NUMBER_WEIGHT
             + np.abs(day[todo, None] - c_day[None, :]) * _DAY_WEIGHT + rank[None, :])
    out[todo] = c_id[np.argmin(score, axis=1)]
    return out


def _iso(day: int) -> str:
    return (_EPOCH + pd.Timedelta(days=int(day))).isoformat()


def _sequence_row(sid: int, seq: pd.DataFrame, final: np.ndarray, disposition: str) -> list:
    """One row of the sequence table (PLAN §4.2); spread_km is filled by the caller."""
    src = seq["coordinate_source"]
    located = seq[seq["lat"].notna()]
    clusters = sorted({int(c) for c in final if c >= 0})
    return [
        int(sid), seq["key"].iloc[0], seq["series"].iloc[0], int(seq["number"].min()), int(seq["number"].max()),
        _iso(seq["day"].min()), _iso(seq["day"].max()), len(seq), int((src == "specify").sum()),
        int((src == "engine").sum()), int((src == "").sum()), np.nan,
        float(located["lat"].median()) if len(located) else np.nan,
        float(located["lng"].median()) if len(located) else np.nan,
        " ".join(str(c) for c in clusters), disposition,
    ]


def merge_with_clusters(result: SequenceResult, cluster_of: pd.Series, min_specimens: int) -> MergeResult:
    """
    Add the sequences to the spatial clusters (PLAN §3, EXP-SEQ-4).

    Inputs:  result -- build_sequences' answer;
             cluster_of -- spid -> spatiotemporal_cluster_id of every row the
             spatial clustering kept (after its size floor);
             min_specimens -- the size floor for a sequence-only expedition.
    Output:  MergeResult. A clustered row never moves and no two clusters
             merge. A sequence row in no kept cluster joins the cluster of the
             sequence's clustered row nearest in number (then day, then the
             lower cluster id). A sequence with no clustered row and at least
             min_specimens rows becomes a new cluster, numbered above the
             largest spatial id in sequence-id order; a smaller one joins
             nothing. assignments.attached is True for a row added to an
             existing spatial cluster, False for a sequence-only row.
    """
    rows = result.rows
    in_seq = rows[rows["sequence_id"].notna()].copy()
    in_seq["sequence_id"] = in_seq["sequence_id"].astype("int64")
    in_seq["cluster"] = in_seq["spid"].map(cluster_of).fillna(-1).astype("int64")
    in_seq = in_seq.sort_values(["sequence_id", "number", "day"], kind="stable")

    next_id = int(cluster_of.max()) + 1 if len(cluster_of) else 0
    spids, clusters, attached, finals, table = [], [], [], [], []
    only_count = only_rows = 0
    for sid, seq in in_seq.groupby("sequence_id", sort=True):
        cluster = seq["cluster"].to_numpy()
        clustered = cluster >= 0
        if clustered.all():
            disposition, final = DISPOSITION_IN_CLUSTER, cluster
        elif clustered.any():
            disposition, final = DISPOSITION_ATTACHED, _nearest_clusters(seq, cluster)
        elif len(seq) >= min_specimens:
            disposition, final = DISPOSITION_SEQUENCE_ONLY, np.full(len(seq), next_id)
            next_id += 1
            only_count += 1
            only_rows += len(seq)
        else:
            disposition, final = DISPOSITION_BELOW_FLOOR, cluster
        new = (~clustered) & (final >= 0)
        spids.extend(seq["spid"].to_numpy()[new])
        clusters.extend(final[new])
        attached.extend([disposition == DISPOSITION_ATTACHED] * int(new.sum()))
        finals.append(pd.Series(final, index=seq.index))
        table.append(_sequence_row(int(sid), seq, final, disposition))

    spread_rows = rows[(rows["rule"] == RULE_SPREAD) & rows["candidate_sequence_id"].notna()]
    for sid, seq in spread_rows.groupby("candidate_sequence_id", sort=True):
        table.append(_sequence_row(int(sid), seq, np.full(len(seq), -1), DISPOSITION_SPREAD))

    assignments = pd.DataFrame({
        "spid": pd.Series(spids, dtype=object),
        "spatiotemporal_cluster_id": pd.Series(clusters, dtype="int64"),
        "attached": pd.Series(attached, dtype=bool),
    })
    assignments["cluster_join"] = JOIN_NUMBER
    sequences = pd.DataFrame(table, columns=SEQUENCE_TABLE_COLUMNS)
    if len(sequences):
        spread = result.spread_km.reindex(sequences["collector_sequence_id"].to_numpy()).to_numpy()
        sequences["spread_km"] = np.round(spread.astype(float), 1)
        sequences = sequences.sort_values("collector_sequence_id", kind="stable").reset_index(drop=True)
    in_seq["final_cluster"] = pd.concat(finals) if finals else pd.Series(dtype="int64")
    return MergeResult(assignments=assignments, sequences=sequences, sequence_only_expeditions=only_count,
                       sequence_only_rows=only_rows,
                       sequence_rows=in_seq[["spid", "sequence_id", "coordinate_source", "final_cluster"]])
