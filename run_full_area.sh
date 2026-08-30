#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/outputs/tables/full_area results/outputs/logs/full_area
/usr/bin/python3 code/prepare_dsm_for_sar.py
/usr/bin/python3 code/run_full_area_geocode.py "$@" 2>&1 | tee results/outputs/logs/full_area/run_full_area.log
/usr/bin/python3 code/pic_all.py sync
