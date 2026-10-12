#!/usr/bin/env python3
"""
Write the collector alias file cluster_csv.py reads from a CAS Lens spelling map
(CAS Lens feature #378, EXP-ALIAS-5).

Input:  collector_spelling_map.csv, written each week by CAS Lens's spelling-map
        stage (columns alias, person_id, slug, canonical_name: every spelling
        Lens links to exactly one person).
Output: the alias file (normalized_name, canonical_id): each spelling keyed by
        this repo's normalize_collector_name, the id "@<slug>"; a key two
        people's spellings give is left out. One line on stdout:

    collector aliases: <n> spellings, <k> keys for <p> people; <s> keys held by two or more people left out, <x> spellings skipped

Exit: 0 when written; 1, with the reason on stderr and no output file, when the
spelling map is missing, has no rows, lacks the alias or slug column, has a
blank slug, or gives no key at all.

Usage:
    python scripts/collector_alias_file.py incoming_data/people/collector_spelling_map.csv -o /tmp/collector_aliases.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from expedition_clustering.pipeline import alias_table_from_spellings

REQUIRED = ("alias", "slug")


def build(spelling_map: Path, output: Path) -> str:
    """
    Read the spelling map, write the alias file, return the summary line.

    Raises FileNotFoundError / ValueError with the reason; nothing is written then.
    """
    if not spelling_map.exists():
        msg = f"spelling map not found: {spelling_map}"
        raise FileNotFoundError(msg)
    if spelling_map.stat().st_size == 0:
        msg = f"spelling map {spelling_map} is empty"
        raise ValueError(msg)
    df = pd.read_csv(spelling_map, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        msg = f"spelling map {spelling_map} lacks column(s) {', '.join(missing)}; has {list(df.columns)}"
        raise ValueError(msg)
    if df.empty:
        msg = f"spelling map {spelling_map} has no rows"
        raise ValueError(msg)
    table = alias_table_from_spellings(zip(df["alias"], df["slug"], strict=True))
    if not table.aliases:
        msg = f"spelling map {spelling_map} gives no collector key ({table.spellings} spellings)"
        raise ValueError(msg)
    rows = sorted(table.aliases.items())
    tmp = output.with_name(output.name + ".building")
    pd.DataFrame(rows, columns=["normalized_name", "canonical_id"]).to_csv(tmp, index=False)
    tmp.replace(output)
    return (f"collector aliases: {table.spellings} spellings, {len(table.aliases)} keys for "
            f"{table.people} people; {len(table.shared_keys)} keys held by two or more people "
            f"left out, {table.skipped} spellings skipped")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("spelling_map", type=Path, help="CAS Lens collector_spelling_map.csv")
    parser.add_argument("-o", "--output", type=Path, required=True, help="alias file to write")
    args = parser.parse_args()
    try:
        line = build(args.spelling_map, args.output)
    except (FileNotFoundError, ValueError) as e:
        print(f"collector_alias_file: {e}", file=sys.stderr)
        return 1
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
