#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/outputs/logs/main
/usr/bin/python3 code/prepare_dsm_for_sar.py
/usr/bin/python3 code/run_tongji_gamma_geocode.py "$@" 2>&1 | tee results/outputs/logs/main/run.log
