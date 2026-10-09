"""
Feature #922 (CAS Lens): botany collector-number sequences.

The parse, dating, collector-key, link and exclusion rules of the botany
collector-number spot check (cas-lens docs/analysis/botany-collector-number-spotcheck.md,
plan docs/requirements/bugs/922/PLAN.md §1-2). Cases are the spot check's
hand-read rows and named collectors.

Requirements: EXP-SEQ-1, EXP-SEQ-2 (cas-lens docs/requirements/expedition-requirements.md).
"""

import datetime as dt

import pandas as pd
import pytest

from expedition_clustering.collector_sequences import (
    RULE_BEFORE_1890,
    RULE_NO_DAY,
    RULE_NONPERSONAL,
    RULE_NONSEQUENTIAL,
    RULE_SN,
    RULE_SPREAD,
    RULE_UNPARSED,
    build_sequences,
    collector_key,
    day_precise,
    parse_collector_number,
    prepare_rows,
)

# ---------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    ("raw", "year", "series", "number"),
    [
        ("3820", 1992, "", 3820),  # hand-read #1 Killeen
        ("LUTEYN 14018", 1990, "LUTEYN", 14018),  # #21
        ("cd02", 2024, "CD", 2),  # #50 Dooley
        ("4294 bis", 1987, "", 4294),  # #76 Geerinck-Coutrel
        ("12029,", 1981, "", 12029),  # #89 Sousa
        ("R16496", 1961, "R", 16496),  # #102 Raven & Cannon
        ("DS 4328", 1979, "DS", 4328),  # #118 Showers
        ("1566.", 1905, "", 1566),  # #73 Stewart
        ("2*", 1901, "", 2),  # #88 Rosendahl
        ("# 6871;", 1950, "", 6871),
        ("11,886", 1950, "", 11886),
        ("1475a.", 1950, "", 1475),
        ("48204A", 1950, "", 48204),
        ("12½", 1950, "", 12),
        ("L.L. 77", 1950, "LL", 77),
        ("M-64", 1950, "M", 64),
        ("92-157", 1992, "Y", 1992 * 100000 + 157),  # Maxwell
        ("91-20", 1992, "Y", 1992 * 100000 + 20),  # year - 1
        ("1992-157", 1992, "Y", 1992 * 100000 + 157),
        ("858-49", 1949, "YS", 1949 * 100000 + 858),
        ("70-100", 1950, "", 70),  # range
        ("341-342?", 1950, "", 341),
        ("98278-5", 1998, "", 98278),  # number-part
        ("1510-2", 1950, "", 1510),
    ],
)
def test_parses(raw, year, series, number):
    rule, got_series, got_number = parse_collector_number(raw, year)
    assert (rule, got_series, got_number) == ("", series, number)


@pytest.mark.parametrize(
    ("raw", "rule"),
    [
        ("s.n.", RULE_SN),
        ("S.n.", RULE_SN),
        ("s n", RULE_SN),
        ("N. S.", RULE_SN),
        ("s/n", RULE_SN),
        ("sin número", RULE_SN),
        ("", RULE_SN),
        ("s.n. [Exsiccatae #818.]", RULE_SN),
        ("10/32b", RULE_UNPARSED),  # #83 Mozingo plot code
        ("802/350", RULE_UNPARSED),
        ("164a-6845", RULE_UNPARSED),  # #66 Grant: two numbers
        ("260(79820)", RULE_UNPARSED),  # #77 Herter: + herbarium number
        ("74-W-13", RULE_UNPARSED),  # Williams plot code
        ("84-121-4", RULE_UNPARSED),
        ("6.556", RULE_UNPARSED),
        ("77-12", RULE_UNPARSED),  # hyphenated pair, not the year
        ("MEX-2./907.", RULE_UNPARSED),
        ("Unspecified", RULE_UNPARSED),
    ],
)
def test_does_not_parse(raw, rule):
    assert parse_collector_number(raw, 1950) == (rule, "", None)


def test_year_prefix_needs_the_collecting_year():
    # "92-157" collected in 1960 is not year-prefixed: a hyphenated pair.
    assert parse_collector_number("92-157", 1960) == (RULE_UNPARSED, "", None)
    # Without a year the yy cannot be checked.
    assert parse_collector_number("92-157", None) == (RULE_UNPARSED, "", None)


# ---------------------------------------------------------------- dating


@pytest.mark.parametrize(
    ("year", "month", "day", "verbatim", "expected"),
    [
        (1950, 6, 15, "", dt.date(1950, 6, 15)),
        (1950, 6, 1, "", dt.date(1950, 6, 1)),  # month not January, verbatim empty
        (1950, 6, 1, "1 June 1950", dt.date(1950, 6, 1)),
        (1950, 6, 1, "June 1st 1950", dt.date(1950, 6, 1)),
        (1950, 6, 1, "1950-06-01", dt.date(1950, 6, 1)),
        (1911, 1, 1, "", None),  # the year-only fill
        (1911, 1, 1, "1911", None),
        (1911, 6, 1, "June 1911", None),  # month-only verbatim
        (1911, 6, 1, "1911-06", None),
        (1911, 6, 1, "June 1911 coll.", None),  # a 1-day with no 1 in the verbatim date
        (1911, 2, 30, "", None),  # not a calendar date
        (1911, None, 4, "", None),
        (None, 6, 4, "", None),
        (1650, 6, 4, "", None),  # outside 1700-2026
    ],
)
def test_day_precise(year, month, day, verbatim, expected):
    assert day_precise(year, month, day, verbatim) == expected


# ---------------------------------------------------------------- collector key


@pytest.mark.parametrize(
    ("collectors", "key", "personal"),
    [
        ("Thomas, John H.", "thomas j", True),
        ("Thomas, J. H.", "thomas j", True),
        ("Thomas, John Hunter", "thomas j", True),
        ("D.E. Breedlove and A. Clewell", "breedlove d", True),
        ("Breedlove6, D E", "breedlove d", True),  # hand-read #105
        ("Raven, Peter H.; Breedlove, D. E.", "raven p", True),  # first-listed
        ("Eastwood, Mrs. Alice", "eastwood a", True),
        ("HangBiaoBen", "hangbiaoben", True),
        ("Gaoligong Shan Biodiversity Survey", "gaoligong shan biodiversity survey", False),
        ("Ex herb. Geo. B. Hinton", "", False),
        ("unknown", "", False),
        ("", "", False),
    ],
)
def test_collector_key(collectors, key, personal):
    got_key, got_personal = collector_key(collectors)
    assert got_personal is personal
    if personal:
        assert got_key == key


# ---------------------------------------------------------------- sequences


def _row(spid, collectors, number, date, *, lat=None, lng=None, source="", verbatim=""):
    year, month, day = (date.year, date.month, date.day) if isinstance(date, dt.date) else date
    return {
        "spid": spid,
        "collectors": collectors,
        "stationFieldNumber": str(number),
        "startDate": "" if year is None else str(year),
        "ce_startDate": "" if month is None else str(month),
        "ce_startDate1": "" if day is None else str(day),
        "verbatimDate": verbatim,
        "latitude1": "" if lat is None else str(lat),
        "longitude1": "" if lng is None else str(lng),
        "coordinate_source": source,
    }


def _trip(prefix, collectors, first_number, start, days, per_day, **kw):
    rows = []
    n = first_number
    for d in range(days):
        for _ in range(per_day):
            rows.append(_row(f"{prefix}-{n}", collectors, n, start + dt.timedelta(days=d), **kw))
            n += 1
    return rows


def _run(rows):
    return build_sequences(prepare_rows(pd.DataFrame(rows)))


def _sequences(result, spid_prefix):
    sel = result.rows[result.rows["spid"].str.startswith(spid_prefix)]
    return sel.groupby("sequence_id")["spid"].apply(sorted).tolist()


def test_bartholomew_reused_series_is_one_sequence_per_trip():
    # Bartholomew used the same numbers on the 1984 Yunnan, 1985 Arizona and
    # 1986 Fanjing Shan trips (spot check §2): three sequences, not 90 singles.
    rows = (
        _trip("y84", "Bartholomew, B.", 100, dt.date(1984, 6, 1), 3, 10)
        + _trip("a85", "Bartholomew, B.", 100, dt.date(1985, 7, 1), 3, 10)
        + _trip("f86", "Bartholomew, B.", 100, dt.date(1986, 8, 1), 3, 10)
    )
    result = _run(rows)
    assert result.rows["sequence_id"].notna().all()
    for prefix in ("y84", "a85", "f86"):
        seqs = _sequences(result, prefix)
        assert len(seqs) == 1
        assert len(seqs[0]) == 30
    assert result.rows["sequence_id"].nunique() == 3
    assert "bartholomew b" in result.reused_keys
    assert "bartholomew b" not in result.nonsequential_keys


def test_eastwood_restarts_split_by_trip():
    # Eastwood 820-823 used in 1906, 1912, 1914 and 1933.
    rows = []
    for year, (m, d) in ((1906, (7, 27)), (1912, (6, 29)), (1914, (7, 21)), (1933, (6, 20))):
        rows += [_row(f"e{year}-{n}", "Eastwood, Alice", n, dt.date(year, m, d)) for n in range(820, 824)]
    result = _run(rows)
    assert result.rows["sequence_id"].nunique() == 4
    for year in (1906, 1912, 1914, 1933):
        assert len(_sequences(result, f"e{year}")) == 1


def test_tharp_numbers_that_do_not_follow_dates_are_excluded():
    # Tharp: 147, 228, 270 each on 4-7 dates; consecutive numbers years apart.
    rows = [
        _row(f"t{n}", "Tharp, B. C.", n, dt.date(1925 + (n * 7) % 20, 1 + n % 12, 10))
        for n in range(1, 40)
    ]
    result = _run(rows)
    assert (result.rows["rule"] == RULE_NONSEQUENTIAL).all()
    assert result.rows["sequence_id"].isna().all()
    assert "tharp b" in result.nonsequential_keys


def test_maxwell_year_prefixed_numbers_split_by_year():
    rows = [_row(f"m92-{n}", "Maxwell, J. F.", f"92-{n}", dt.date(1992, 3, 1)) for n in range(1, 11)]
    rows += [_row(f"m93-{n}", "Maxwell, J. F.", f"93-{n}", dt.date(1993, 3, 1)) for n in range(1, 11)]
    result = _run(rows)
    assert len(_sequences(result, "m92")) == 1
    assert len(_sequences(result, "m93")) == 1
    assert result.rows["sequence_id"].nunique() == 2
    assert set(result.rows["series"]) == {"Y"}


def test_sn_rows_are_excluded_by_rule():
    rows = [_row(f"s{i}", "Pollard, Henry M.", "s.n.", dt.date(1950, 5, 1)) for i in range(5)]
    result = _run(rows)
    assert (result.rows["rule"] == RULE_SN).all()
    assert result.excluded_counts()[RULE_SN] == 5


def test_link_limits():
    d = dt.date(1950, 5, 1)
    rows = [
        _row("a1", "Smith, A.", 100, d),
        _row("a2", "Smith, A.", 120, d),  # gap 20: linked
        _row("b1", "Jones, B.", 100, d),
        _row("b2", "Jones, B.", 121, d),  # gap 21: not linked
        _row("c1", "Brown, C.", 100, d),
        _row("c2", "Brown, C.", 101, d + dt.timedelta(days=2)),  # 2 days: linked
        _row("d1", "Green, D.", 100, d),
        _row("d2", "Green, D.", 101, d + dt.timedelta(days=3)),  # 3 days: not linked
        _row("e1", "Gray, E.", 50, d),
        _row("e2", "Gray, E.", 50, d + dt.timedelta(days=2)),  # same number, 2 days: not linked
        _row("f1", "White, F.", 10, dt.date(1950, 12, 31)),
        _row("f2", "White, F.", 11, dt.date(1951, 1, 1), verbatim="1 Jan 1951"),  # year boundary: not linked
    ]
    result = _run(rows)
    seq = dict(zip(result.rows["spid"], result.rows["sequence_id"], strict=True))
    assert seq["a1"] == seq["a2"]
    assert pd.notna(seq["a1"])
    assert seq["c1"] == seq["c2"]
    for alone in ("b1", "b2", "d1", "d2", "e1", "e2", "f1", "f2"):
        assert pd.isna(seq[alone]), alone
    assert result.alone_rows() == 8


def test_spread_over_500_km_is_excluded():
    d = dt.date(1960, 4, 2)
    far = [
        _row("w1", "Far, A.", 1, d, lat=0, lng=0, source="specify"),
        _row("w2", "Far, A.", 2, d, lat=0, lng=0, source="engine"),
        _row("w3", "Far, A.", 3, d, lat=0, lng=10, source="specify"),
        _row("w4", "Far, A.", 4, d, lat=0, lng=10, source="specify"),
        _row("w5", "Far, A.", 5, d),
    ]
    near = [
        _row("n1", "Near, B.", 1, d, lat=0, lng=0, source="specify"),
        _row("n2", "Near, B.", 2, d, lat=0, lng=1, source="engine"),
        _row("n3", "Near, B.", 3, d),
    ]
    result = _run(far + near)
    rows = result.rows.set_index("spid")
    assert (rows.loc[["w1", "w2", "w3", "w4", "w5"], "rule"] == RULE_SPREAD).all()
    assert len(set(rows.loc[["n1", "n2", "n3"], "sequence_id"])) == 1
    assert (rows.loc[["n1", "n2", "n3"], "rule"] == "").all()
    assert result.excluded_counts()[RULE_SPREAD] == 5


def test_other_exclusions():
    rows = [
        _row("old1", "Drummond, J.", 152, dt.date(1845, 6, 3)),
        _row("old2", "Drummond, J.", 153, dt.date(1845, 6, 3)),
        _row("nd1", "Smith, A.", 10, (1950, 6, None)),
        _row("nd2", "Smith, A.", 11, (1911, 1, 1)),
        _row("np1", "Gaoligong Shan Biodiversity Survey", 4062, dt.date(2002, 11, 15)),
        _row("np2", "Gaoligong Shan Biodiversity Survey", 4063, dt.date(2002, 11, 15)),
        _row("pc1", "Mozingo, H. N.", "10/32b", dt.date(1946, 3, 17)),
    ]
    result = _run(rows)
    rule = dict(zip(result.rows["spid"], result.rows["rule"], strict=True))
    assert rule["old1"] == rule["old2"] == RULE_BEFORE_1890
    assert rule["nd1"] == rule["nd2"] == RULE_NO_DAY
    assert rule["np1"] == rule["np2"] == RULE_NONPERSONAL
    assert rule["pc1"] == RULE_UNPARSED
    assert result.rows["sequence_id"].isna().all()


def test_team_rows_use_the_first_collector():
    d = dt.date(1965, 1, 20)
    rows = [
        _row("t1", "Breedlove, D. E.", 8165, d),
        _row("t2", "Breedlove, D. E.; Raven, Peter H.", 8166, d),
        _row("t3", "Breedlove, Dennis E.", 8167, d),
    ]
    result = _run(rows)
    assert len(set(result.rows["sequence_id"])) == 1
    assert result.rows["sequence_id"].notna().all()


def test_coordinates_carry_their_source():
    d = dt.date(1970, 5, 5)
    rows = [
        _row("c1", "Smith, A.", 1, d, lat=37.5, lng=-122.1, source="specify"),
        _row("c2", "Smith, A.", 2, d, lat=37.6, lng=-122.2, source="engine"),
        _row("c3", "Smith, A.", 3, d),
    ]
    result = _run(rows)
    src = dict(zip(result.rows["spid"], result.rows["coordinate_source"], strict=True))
    assert src == {"c1": "specify", "c2": "engine", "c3": ""}


def test_prepare_rows_requires_the_merged_input():
    # A plain PortalData.csv has no coordinate_source: refused, not read as all-Specify.
    plain = pd.DataFrame([_row("a", "Smith, A.", 1, dt.date(1950, 1, 2))]).drop(columns=["coordinate_source"])
    with pytest.raises(ValueError, match="coordinate_source"):
        prepare_rows(plain)
    with pytest.raises(ValueError, match="stationFieldNumber"):
        prepare_rows(pd.DataFrame({"spid": ["a"], "collectors": ["Smith, A."]}))


def test_prepare_rows_checks_the_source_against_the_coordinates():
    rows = [_row("a", "Smith, A.", 1, dt.date(1950, 1, 2), lat=1.0, lng=2.0, source="")]
    with pytest.raises(ValueError, match="coordinate_source"):
        prepare_rows(pd.DataFrame(rows))
    rows = [_row("a", "Smith, A.", 1, dt.date(1950, 1, 2), source="specify")]
    with pytest.raises(ValueError, match="coordinate_source"):
        prepare_rows(pd.DataFrame(rows))
    rows = [_row("a", "Smith, A.", 1, dt.date(1950, 1, 2), lat=1.0, lng=2.0, source="gazetteer")]
    with pytest.raises(ValueError, match="gazetteer"):
        prepare_rows(pd.DataFrame(rows))
