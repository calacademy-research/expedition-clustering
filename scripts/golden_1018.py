#!/usr/bin/env python3
"""
CAS Lens #1018: write tests/data/pipeline_1018_expected.csv, the cluster ids
an engine checkout gives on the synthetic input (tests/synthetic_1018.py).

Run once with the engine BEFORE the speed-up (628db2f) to record the
reference the speed-up must reproduce:
    python scripts/golden_1018.py --engine <checkout at 628db2f> --out tests/data/pipeline_1018_expected.csv

Inputs:  --engine (a checkout whose expedition_clustering/pipeline.py is
         used), --out.
Outputs: CSV: config, spid, and the three id columns.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from synthetic_1018 import CONFIGS, ID_COLUMNS, synthetic_input


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("pipeline", args.engine / "expedition_clustering" / "pipeline.py")
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules["pipeline"] = pipeline
    spec.loader.exec_module(pipeline)
    frames = []
    for name, cfg in CONFIGS.items():
        out = pipeline.create_pipeline(**cfg).fit_transform(synthetic_input())
        frames.append(out[["spid", *ID_COLUMNS]].assign(config=name))
    result = pd.concat(frames)[["config", "spid", *ID_COLUMNS]].sort_values(["config", "spid"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"{len(result)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
