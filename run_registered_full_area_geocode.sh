#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p results/outputs/logs/registered_full_area results/picall/注册复现
/usr/bin/python3 code/prepare_dsm_for_sar.py
/usr/bin/python3 code/run_registered_full_area_geocode.py "$@" \
  2>&1 | tee results/outputs/logs/registered_full_area/run.log
/usr/bin/python3 code/reproduce_pic_all2_one_to_one.py
