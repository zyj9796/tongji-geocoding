from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR / "code"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from geocode_gamma_rslc_with_buildings import parse_gamma_par, read_rslc_amplitude  # noqa: E402
from io_paths import RSLC_DIR  # noqa: E402
from register_strict_triangle_projection import edge_map, load_triangles, rasterize_triangles  # noqa: E402


def shifted_values(image: np.ndarray, rr: np.ndarray, cc: np.ndarray, dr: int, dc: int) -> np.ndarray:
    rows = rr + dr
    cols = cc + dc
    keep = (rows >= 0) & (cols >= 0) & (rows < image.shape[0]) & (cols < image.shape[1])
    return image[rows[keep], cols[keep]] if np.any(keep) else np.zeros(0, dtype=np.float32)


def score(
    amplitude: np.ndarray,
    edges: np.ndarray,
    rr: np.ndarray,
    cc: np.ndarray,
    ring_rr: np.ndarray,
    ring_cc: np.ndarray,
    dr: int,
    dc: int,
    max_shift: int,
) -> dict:
    inside_amp = shifted_values(amplitude, rr, cc, dr, dc)
    ring_amp = shifted_values(amplitude, ring_rr, ring_cc, dr, dc)
    inside_edge = shifted_values(edges, rr, cc, dr, dc)
    ring_edge = shifted_values(edges, ring_rr, ring_cc, dr, dc)
    if inside_amp.size < 4:
        return {"score": -1e9, "inside_amp": 0.0, "ring_amp": 0.0, "inside_edge": 0.0, "ring_edge": 0.0}
    ia = float(np.mean(inside_amp))
    ra = float(np.mean(ring_amp)) if ring_amp.size else float(np.mean(amplitude))
    ie = float(np.mean(inside_edge)) if inside_edge.size else 0.0
    re = float(np.mean(ring_edge)) if ring_edge.size else float(np.mean(edges))
    penalty = 0.0015 * (dr * dr + dc * dc) / max(max_shift, 1)
    return {
        "score": 100.0 * (ia - ra) + 35.0 * (ie - re) - penalty,
        "inside_amp": ia,
        "ring_amp": ra,
        "inside_edge": ie,
        "ring_edge": re,
    }


def optimize_mask(
    mask: np.ndarray,
    amplitude: np.ndarray,
    edges: np.ndarray,
    max_shift: int,
    coarse_step: int,
    min_score_gain: float,
) -> dict:
    rr, cc = np.nonzero(mask)
    ring = binary_dilation(mask, iterations=5) & ~binary_dilation(mask, iterations=1)
    ring_rr, ring_cc = np.nonzero(ring)
    if rr.size < 4:
        return {
            "roof_pixels": int(rr.size),
            "base_score": -1e9,
            "best_score": -1e9,
            "score_gain": 0.0,
            "candidate_row_shift": 0,
            "candidate_col_shift": 0,
            "applied_row_shift": 0,
            "applied_col_shift": 0,
            "accepted": 0,
        }
    base = score(amplitude, edges, rr, cc, ring_rr, ring_cc, 0, 0, max_shift)
    best = {**base, "row_shift": 0, "col_shift": 0}
    for dr in range(-max_shift, max_shift + 1, coarse_step):
        for dc in range(-max_shift, max_shift + 1, coarse_step):
            candidate = score(amplitude, edges, rr, cc, ring_rr, ring_cc, dr, dc, max_shift)
            if candidate["score"] > best["score"]:
                best = {**candidate, "row_shift": dr, "col_shift": dc}
    coarse_row = int(best["row_shift"])
    coarse_col = int(best["col_shift"])
    for dr in range(max(-max_shift, coarse_row - coarse_step), min(max_shift, coarse_row + coarse_step) + 1):
        for dc in range(max(-max_shift, coarse_col - coarse_step), min(max_shift, coarse_col + coarse_step) + 1):
            candidate = score(amplitude, edges, rr, cc, ring_rr, ring_cc, dr, dc, max_shift)
            if candidate["score"] > best["score"]:
                best = {**candidate, "row_shift": dr, "col_shift": dc}
    gain = float(best["score"] - base["score"])
    accepted = int(gain >= min_score_gain and (int(best["row_shift"]) != 0 or int(best["col_shift"]) != 0))
    return {
        "roof_pixels": int(rr.size),
        "base_score": float(base["score"]),
        "best_score": float(best["score"]),
        "score_gain": gain,
        "candidate_row_shift": int(best["row_shift"]),
        "candidate_col_shift": int(best["col_shift"]),
        "applied_row_shift": int(best["row_shift"]) if accepted else 0,
        "applied_col_shift": int(best["col_shift"]) if accepted else 0,
        "accepted": accepted,
        "base_inside_amp": float(base["inside_amp"]),
        "best_inside_amp": float(best["inside_amp"]),
        "base_inside_edge": float(base["inside_edge"]),
        "best_inside_edge": float(best["inside_edge"]),
    }


def shift_payload(payload: dict, shifts: dict[int, tuple[int, int]]) -> dict:
    for feature in payload.get("features", []):
        props = feature.setdefault("properties", {})
        fid = int(props.get("fid", -1))
        dr, dc = shifts.get(fid, (0, 0))
        geometry = feature.get("geometry", {})
        if geometry.get("type") == "Polygon":
            geometry["coordinates"] = [
                [[float(x) + dc, float(y) + dr] for x, y, *rest in ring]
                for ring in geometry.get("coordinates", [])
            ]
        props["local_registration_row_shift"] = dr
        props["local_registration_col_shift"] = dc
    payload["local_registration"] = {
        "method": "roof-only real-SAR amplitude/edge optimization with conservative acceptance threshold",
        "coordinate_system": "x=range column, y=azimuth row",
    }
    return payload


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    par = parse_gamma_par(RSLC_DIR / f"{args.date}.rslc.par")
    raw = read_rslc_amplitude(RSLC_DIR / f"{args.date}.rslc", int(par["azimuth_lines"]), int(par["range_samples"]))
    amplitude = raw.astype(np.float32) / 255.0
    edges = edge_map(raw)
    payload, triangles = load_triangles(Path(args.registered_triangles))
    roof_by_fid: dict[int, list[np.ndarray]] = defaultdict(list)
    for item in triangles:
        props = item["feature"].get("properties", {})
        if str(props.get("surface", "")) == "roof":
            roof_by_fid[int(props.get("fid", -1))].append(item["xy"])

    rows: list[dict] = []
    shifts: dict[int, tuple[int, int]] = {}
    for index, (fid, roof_triangles) in enumerate(sorted(roof_by_fid.items()), start=1):
        mask = rasterize_triangles(roof_triangles, raw.shape)
        result = optimize_mask(mask, amplitude, edges, args.max_shift, args.coarse_step, args.min_score_gain)
        rows.append({"fid": fid, **result})
        shifts[fid] = (int(result["applied_row_shift"]), int(result["applied_col_shift"]))
        if index % 100 == 0:
            print(f"local registration {index}/{len(roof_by_fid)}", flush=True)

    metrics_path = out_dir / f"{args.date}_strict_local_registration_metrics.csv"
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    shifted_path = out_dir / f"{args.date}_locally_registered_strict_sar_surface_triangles.geojson"
    shifted_path.write_text(json.dumps(shift_payload(payload, shifts), ensure_ascii=False), encoding="utf-8")
    accepted = [row for row in rows if int(row["accepted"]) == 1]
    summary = {
        "date": args.date,
        "input_registered_triangles": str(Path(args.registered_triangles)),
        "buildings": len(rows),
        "accepted_local_shifts": len(accepted),
        "max_shift": args.max_shift,
        "coarse_step": args.coarse_step,
        "min_score_gain": args.min_score_gain,
        "median_applied_row_shift": float(np.median([row["applied_row_shift"] for row in accepted])) if accepted else 0.0,
        "median_applied_col_shift": float(np.median([row["applied_col_shift"] for row in accepted])) if accepted else 0.0,
        "median_accepted_score_gain": float(np.median([row["score_gain"] for row in accepted])) if accepted else 0.0,
        "metrics_csv": str(metrics_path),
        "locally_registered_triangles": str(shifted_path),
    }
    (out_dir / f"{args.date}_strict_local_registration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Conservative per-building roof registration after global strict-triangle alignment.")
    parser.add_argument("--date", default="20200708")
    parser.add_argument(
        "--registered-triangles",
        default=str(PROJECT_DIR / "results" / "outputs" / "strict_triangle_registration" / "20200708_registered_strict_sar_surface_triangles.geojson"),
    )
    parser.add_argument("--out-dir", default=str(PROJECT_DIR / "results" / "outputs" / "strict_local_registration"))
    parser.add_argument("--max-shift", type=int, default=10)
    parser.add_argument("--coarse-step", type=int, default=2)
    parser.add_argument("--min-score-gain", type=float, default=5.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
