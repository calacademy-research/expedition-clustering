#!/usr/bin/env bash
#
# Thin wrapper to run the expedition clustering pipeline via Python.
# Usage:
#   scripts/run_pipeline.sh --e-dist 15 --e-days 5 --sample 50000
#

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}

$PYTHON_BIN scripts/run_pipeline.py "$@"
