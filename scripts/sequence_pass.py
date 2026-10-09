#!/usr/bin/env python3
"""
Apply the collector-number sequence pass (CAS Lens feature #922) to an
existing clustering output, without re-running the spatial clustering.

Usage:
    python scripts/sequence_pass.py MERGED_INPUT CLUSTERED_CSV -o OUTPUT_CSV [--min-specimens 20]

Inputs:  MERGED_INPUT -- a merged clustering input (a collection directory
         holding PortalData.csv, or the file) with coordinate_source (CAS Lens
         #918 write_clustering_input); CLUSTERED_CSV -- a clustered_expeditions.csv
         of the same collection, taken as the spatial clusters as they are.
Outputs: OUTPUT_CSV (the clusters plus the rows the sequences add, with the
         sequence columns, geographic classification recomputed),
         collector_sequences.csv and collector_sequences_summary.txt beside
         it; the three log lines on stdout.

Used to measure the pass on a week's data (cas-lens docs/requirements/bugs/922/README.md);
the weekly run uses cluster_csv.py --collector-sequences.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

_pkg_dir = Path(__file__).parent.parent / "expedition_clustering"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _pkg_dir / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_csv = _load("csv_source")
_seq = _load("collector_sequences")
_geo = _load("geo_classify")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("merged_input", type=Path)
    parser.add_argument("clustered_csv", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--min-specimens", type=int, default=20)
    args = parser.parse_args()

    source = args.merged_input / "PortalData.csv" if args.merged_input.is_dir() else args.merged_input
    all_rows = _csv.load_collection_csv(source)
    clustered = pd.read_csv(args.clustered_csv, low_memory=False)
    clustered = clustered.drop(columns=[c for c in clustered.columns if c.startswith("geo_")])
    sequence_pass = _seq.run_sequence_pass(clustered, all_rows, _seq.read_sequence_columns(source),
                                           args.min_specimens)
    for line in sequence_pass.log_lines():
        print(line, flush=True)
    output = _geo.classify_expeditions(sequence_pass.output, cluster_col="spatiotemporal_cluster_id",
                                       lat_col="latitude1", lng_col="longitude1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    sequence_pass.write_files(args.output.parent)
    print(f"Saved {args.output}, {_seq.SEQUENCES_FILE} and {_seq.SUMMARY_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
