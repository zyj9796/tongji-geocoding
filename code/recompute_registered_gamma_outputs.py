from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR / "code"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import geocode_tongji_all_buildings_compare_gamma as full_core  # noqa: E402
from geocode_gamma_rslc_with_buildings import make_orbit, parse_gamma_par, read_rslc_amplitude  # noqa: E402
from io_paths import DSM_TIF, RSLC_DIR, TIF_DIR  # noqa: E402
from raster_height import RasterHeightSampler  # noqa: E402
from run_full_area_geocode import make_full_area_figures, plot_fig54_with_gamma_dsm_background, read_points, read_tif, write_points_geojson  # noqa: E402
from run_registered_full_area_geocode import summarize, write_pic_readme  # noqa: E402


def read_buildings(path: Path) -> tuple[dict[int, np.ndarray], list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_fid: dict[int, np.ndarray] = {}
    plotting: list[dict] = []
    for feature in payload.get("features", []):
        fid = int(feature.get("properties", {}).get("fid", -1))
        ring = np.asarray(feature["geometry"]["coordinates"][0], dtype=np.float64)
        if ring.shape[0] > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        by_fid[fid] = ring[:, :2]
        plotting.append({"fid": fid, "ring_lonlat": ring[:, :2]})
    return by_fid, plotting


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    pic_dir = Path(args.pic_dir)
    points_path = output_dir / f"{args.date}_all_buildings_method_vs_gamma_points.csv"
    stats_path = output_dir / f"{args.date}_all_buildings_fig5_4_like_stats.csv"
    skipped_path = output_dir / f"{args.date}_all_buildings_skipped.csv"
    buildings_path = output_dir / f"{args.date}_all_valid_geocoded_buildings.geojson"
    rows = list(csv.DictReader(points_path.open(encoding="utf-8")))
    stats = list(csv.DictReader(stats_path.open(encoding="utf-8")))
    rings, plotting_buildings = read_buildings(buildings_path)
    local_shifts: dict[int, tuple[int, int]] = {}
    if args.local_shift_metrics:
        with Path(args.local_shift_metrics).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                local_shifts[int(row["fid"])] = (
                    int(float(row["applied_row_shift"])),
                    int(float(row["applied_col_shift"])),
                )

    par = parse_gamma_par(RSLC_DIR / f"{args.date}.rslc.par")
    orbit = make_orbit(par)
    dsm = RasterHeightSampler(DSM_TIF)
    grouped: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(int(row["fid"]), []).append(index)

    gamma_distance: dict[int, list[float]] = {}
    for number, (fid, indices) in enumerate(grouped.items(), start=1):
        local_row, local_col = local_shifts.get(fid, (0, 0))
        points = np.asarray(
            [
                [
                    float(rows[i]["row"]) - args.row_shift - local_row,
                    float(rows[i]["col"]) - args.col_shift - local_col,
                    float(rows[i]["method_lon"]),
                    float(rows[i]["method_lat"]),
                    float(rows[i]["method_height_m"]),
                    float(rows[i].get("triangle_index", 0)),
                ]
                for i in indices
            ],
            dtype=np.float64,
        )
        gamma = full_core.gamma_dsm_height_points(points, par, orbit, dsm)
        distance = full_core.boundary_distances(gamma[:, 2:4], rings[fid])
        gamma_distance[fid] = distance.tolist()
        for i, result in zip(indices, gamma):
            rows[i]["gamma_dsm_lon"] = f"{result[2]:.12f}"
            rows[i]["gamma_dsm_lat"] = f"{result[3]:.12f}"
            rows[i]["gamma_dsm_height_m"] = f"{result[4]:.8f}"
            rows[i]["gamma_dsm_ok"] = str(int(result[5]))
            rows[i]["gamma_dsm_residual"] = f"{result[6]:.8g}"
        if number % 100 == 0:
            print(f"corrected GAMMA inversion {number}/{len(grouped)} buildings", flush=True)

    with points_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in stats:
        values = np.asarray(gamma_distance[int(row["fid"])], dtype=np.float64)
        row["gamma_dsm_mean_boundary_distance_m"] = float(np.mean(values))
        row["gamma_dsm_median_boundary_distance_m"] = float(np.median(values))
        row["gamma_dsm_p90_boundary_distance_m"] = float(np.percentile(values, 90))
    with stats_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0].keys()))
        writer.writeheader()
        writer.writerows(stats)

    points_geojson = output_dir / f"{args.date}_registered_building_constrained_points.geojson"
    write_points_geojson(points_path, points_geojson)
    gamma_tif = TIF_DIR / f"{args.date}_gamma_dem_geocoded_wgs84.tif"
    amp = read_rslc_amplitude(RSLC_DIR / f"{args.date}.rslc", int(par["azimuth_lines"]), int(par["range_samples"]))
    method, gamma, intensity = read_points(points_path, amp)
    full_core.read_geotiff = read_tif
    if not hasattr(full_core, "_original_plot_fig54_like"):
        full_core._original_plot_fig54_like = full_core.plot_fig54_like
    full_core.plot_fig54_like = plot_fig54_with_gamma_dsm_background
    full_core.plot_fig54_like(
        pic_dir / f"{args.date}_fig5_4_like_all_buildings_map.png",
        gamma_tif,
        plotting_buildings,
        method,
        gamma,
        intensity,
    )
    make_full_area_figures(
        args.date,
        gamma_tif,
        stats_path,
        points_path,
        buildings_path,
        image_dir=pic_dir,
        include_statistics=False,
    )
    summary = summarize(args.date, stats_path, points_path, skipped_path, args.row_shift, args.col_shift)
    summary.update(
        {
            "buildings_shp": str(PROJECT_DIR / "data" / "shp" / "tongji_clip_rslc_extent_equal_height_clean.shp"),
            "gamma_dsm_raster": str(gamma_tif),
            "geographic_backdrop_warp_applied": False,
            "offset_application": "subtract registered row/column shift before DSM-height range-Doppler inversion",
            "local_shift_metrics": str(Path(args.local_shift_metrics)) if args.local_shift_metrics else None,
            "locally_shifted_buildings": sum(1 for value in local_shifts.values() if value != (0, 0)),
            "output_dir": str(output_dir),
            "pic_dir": str(pic_dir),
        }
    )
    (output_dir / f"{args.date}_registered_full_area_geocode_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_pic_readme(pic_dir, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute the comparison geocoding after applying the registered radar offset.")
    parser.add_argument("--date", default="20200708")
    parser.add_argument("--row-shift", type=int, default=34)
    parser.add_argument("--col-shift", type=int, default=-1)
    parser.add_argument(
        "--local-shift-metrics",
        default=str(
            PROJECT_DIR
            / "results"
            / "outputs"
            / "strict_local_registration"
            / "20200708_strict_local_registration_metrics.csv"
        ),
    )
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "results" / "outputs" / "registered_full_area_geocode"))
    parser.add_argument("--pic-dir", default=str(PROJECT_DIR / "results" / "picall" / "注册复现"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
