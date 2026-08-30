#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
/usr/bin/python3 code/register_strict_triangle_projection.py "$@"
/usr/bin/python3 code/optimize_strict_local_registration.py
