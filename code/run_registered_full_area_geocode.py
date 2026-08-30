from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR / "code"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import geocode_tongji_all_buildings_compare_gamma as full_core  # noqa: E402
from io_paths import BUILDINGS_SHP, DSM_TIF, RSLC_DIR, TIF_DIR  # noqa: E402
from run_full_area_geocode import (  # noqa: E402
    make_full_area_figures,
    plot_fig54_with_gamma_dsm_background,
    read_tif,
    safe_apply_dsm_heights,
    use_gamma_dsm_tif_without_control_point_warp,
    write_points_geojson,
)
from run_strict_triangle_projection import build_strict_model, rasterize_model, refine_triangle_mask  # noqa: E402


def registered_rasterizer(
    row_shift: int,
    col_shift: int,
    local_shifts: dict[int, tuple[int, int]] | None = None,
    state: dict | None = None,
):
    def rasterize(building: dict, par: dict, orbit, image_shape: tuple[int, int]) -> dict:
        model = build_strict_model(building, par, orbit)
        fid = int(building.get("fid", -1))
        local_row, local_col = (local_shifts or {}).get(fid, (0, 0))
        total_row = row_shift + local_row
        total_col = col_shift + local_col
        projected = np.asarray(model["projected_rc"], dtype=np.float64).copy()
        projected[:, 0] += total_row
        projected[:, 1] += total_col
        model["projected_rc"] = projected
        mask0, tri_idx = rasterize_model(model, image_shape)
        model["mask0"] = mask0
        model["tri_idx"] = tri_idx
        model["registration_row_shift"] = row_shift
        model["registration_col_shift"] = col_shift
        model["local_registration_row_shift"] = local_row
        model["local_registration_col_shift"] = local_col
        model["total_registration_row_shift"] = total_row
        model["total_registration_col_shift"] = total_col
        if state is not None:
            state["fid"] = fid
            state["total_row_shift"] = total_row
            state["total_col_shift"] = total_col
        return model

    return rasterize


def registered_refiner(kappa: float, min_component: int):
    def refine(mask0: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
        mask, _ = refine_triangle_mask(mask0, amplitude, kappa, min_component)
        return mask

    return refine


def geometry_corrected_gamma_solver(row_shift: int, col_shift: int, state: dict | None = None):
    original = full_core.gamma_dsm_height_points

    def solve(points: np.ndarray, par: dict, orbit, dsm) -> np.ndarray:
        corrected = np.asarray(points, dtype=np.float64).copy()
        corrected[:, 0] -= int((state or {}).get("total_row_shift", row_shift))
        corrected[:, 1] -= int((state or {}).get("total_col_shift", col_shift))
        result = original(corrected, par, orbit, dsm)
        # Keep the observed SAR pixel identifiers in the exported paired table;
        # only the range-Doppler inversion uses timing/range-corrected indices.
        result[:, :2] = np.asarray(points, dtype=np.float64)[:, :2]
        return result

    return solve


def summarize(
    date: str,
    stats_csv: Path,
    points_csv: Path,
    skipped_csv: Path,
    row_shift: int,
    col_shift: int,
) -> dict:
    stats = list(csv.DictReader(stats_csv.open(encoding="utf-8")))
    skipped = list(csv.DictReader(skipped_csv.open(encoding="utf-8"))) if skipped_csv.exists() else []
    point_count = max(0, sum(1 for _ in points_csv.open(encoding="utf-8")) - 1)
    method_mean = np.asarray([float(row["method_mean_boundary_distance_m"]) for row in stats], dtype=np.float64)
    gamma_mean = np.asarray([float(row["gamma_dsm_mean_boundary_distance_m"]) for row in stats], dtype=np.float64)
    return {
        "date": date,
        "processing_basis": "roof-registered strict bottom/wall/roof triangles, conservative per-building local SAR shifts, triangle-constrained amplitude refinement, and barycentric 3D inversion",
        "registration_row_shift": row_shift,
        "registration_col_shift": col_shift,
        "valid_buildings": len(stats),
        "skipped_buildings": len(skipped),
        "scatter_points": point_count,
        "method_mean_boundary_distance_m": float(np.mean(method_mean)),
        "method_median_boundary_distance_m": float(statistics.median(method_mean)),
        "method_p90_boundary_distance_m": float(np.percentile(method_mean, 90)),
        "gamma_dsm_mean_boundary_distance_m": float(np.mean(gamma_mean)),
        "gamma_dsm_median_boundary_distance_m": float(statistics.median(gamma_mean)),
        "gamma_dsm_p90_boundary_distance_m": float(np.percentile(gamma_mean, 90)),
    }


def write_pic_readme(pic_dir: Path, summary: dict) -> None:
    images = sorted(path.name for path in pic_dir.glob("*.png"))
    lines = [
        "# Registered strict-triangle full-area geocoding figures",
        "",
        f"- Registration shift: row `{summary['registration_row_shift']:+d}`, col `{summary['registration_col_shift']:+d}` pixels",
        f"- Valid buildings: `{summary['valid_buildings']}`",
        f"- Scatter points: `{summary['scatter_points']}`",
        f"- PNG count: `{len(images)}`",
        "",
        "## Files",
        "",
        *[f"- `{name}`" for name in images],
        "",
    ]
    (pic_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    pic_dir = Path(args.pic_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pic_dir.mkdir(parents=True, exist_ok=True)
    registration = json.loads(Path(args.registration_summary).read_text(encoding="utf-8"))
    row_shift = int(registration["row_shift"])
    col_shift = int(registration["col_shift"])
    local_shifts: dict[int, tuple[int, int]] = {}
    if args.local_shift_metrics:
        with Path(args.local_shift_metrics).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                local_shifts[int(row["fid"])] = (
                    int(float(row["applied_row_shift"])),
                    int(float(row["applied_col_shift"])),
                )
    gamma_tif = TIF_DIR / f"{args.date}_gamma_dem_geocoded_wgs84.tif"
    if not gamma_tif.exists():
        raise FileNotFoundError(f"Missing GAMMA/DSM raster: {gamma_tif}")

    full_core.DEFAULT_RSLC_DIR = RSLC_DIR
    full_core.apply_dsm_heights = safe_apply_dsm_heights
    registration_state: dict = {}
    full_core.rasterize_building = registered_rasterizer(row_shift, col_shift, local_shifts, registration_state)
    full_core.refine_mask = registered_refiner(args.kappa, args.min_component)
    full_core.gamma_dsm_height_points = geometry_corrected_gamma_solver(row_shift, col_shift, registration_state)
    # The radar timing/range correction belongs to the pixel inversion above;
    # it is not a WGS84 image deformation.  Keep the formal GAMMA/DSM terrain-
    # geocoded raster unchanged as the geographic backdrop.
    full_core.write_building_aligned_gamma_tif = use_gamma_dsm_tif_without_control_point_warp
    full_core.read_geotiff = read_tif
    if not hasattr(full_core, "_original_plot_fig54_like"):
        full_core._original_plot_fig54_like = full_core.plot_fig54_like
    full_core.plot_fig54_like = plot_fig54_with_gamma_dsm_background

    namespace = argparse.Namespace(
        date=args.date,
        buildings_shp=str(BUILDINGS_SHP),
        gamma_tif=str(gamma_tif),
        dsm=str(DSM_TIF),
        out_dir=str(output_dir),
        max_buildings=args.max_buildings,
        max_points_per_building=args.max_points_per_building,
        min_mask0_pixels=args.min_mask0_pixels,
        min_mask_pixels=args.min_mask_pixels,
    )
    full_core.run(namespace)

    map_image = output_dir / f"{args.date}_fig5_4_like_all_buildings_map.png"
    if map_image.exists():
        shutil.move(str(map_image), pic_dir / map_image.name)
    error_image = output_dir / f"{args.date}_fig5_4_like_all_buildings_error_scatter.png"
    error_image.unlink(missing_ok=True)

    stats_csv = output_dir / f"{args.date}_all_buildings_fig5_4_like_stats.csv"
    points_csv = output_dir / f"{args.date}_all_buildings_method_vs_gamma_points.csv"
    skipped_csv = output_dir / f"{args.date}_all_buildings_skipped.csv"
    buildings_geojson = output_dir / f"{args.date}_all_valid_geocoded_buildings.geojson"
    points_geojson = output_dir / f"{args.date}_registered_building_constrained_points.geojson"
    write_points_geojson(points_csv, points_geojson)
    make_full_area_figures(
        args.date,
        gamma_tif,
        stats_csv,
        points_csv,
        buildings_geojson,
        image_dir=pic_dir,
        include_statistics=False,
    )

    for source in [
        PROJECT_DIR / "results" / "pic_all" / f"{args.date}_registered_strict_triangle_projection.png",
        PROJECT_DIR / "results" / "pic_all" / f"{args.date}_registered_strict_triangle_refined_mask.png",
    ]:
        if source.exists():
            shutil.copy2(source, pic_dir / source.name)

    summary = summarize(args.date, stats_csv, points_csv, skipped_csv, row_shift, col_shift)
    summary.update(
        {
            "buildings_shp": str(BUILDINGS_SHP),
            "gamma_dsm_raster": str(gamma_tif),
            "geographic_backdrop_warp_applied": False,
            "local_shift_metrics": str(Path(args.local_shift_metrics)) if args.local_shift_metrics else None,
            "locally_shifted_buildings": sum(1 for shift in local_shifts.values() if shift != (0, 0)),
            "registration_summary": str(Path(args.registration_summary)),
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
    parser = argparse.ArgumentParser(description="Run full-area geocoding from the registered strict triangle projection.")
    parser.add_argument("--date", default="20200708")
    parser.add_argument(
        "--registration-summary",
        default=str(
            PROJECT_DIR
            / "results"
            / "outputs"
            / "strict_triangle_registration"
            / "20200708_strict_triangle_registration_summary.json"
        ),
    )
    parser.add_argument(
        "--output-dir", default=str(PROJECT_DIR / "results" / "outputs" / "registered_full_area_geocode")
    )
    parser.add_argument("--pic-dir", default=str(PROJECT_DIR / "results" / "picall" / "注册复现"))
    parser.add_argument("--max-points-per-building", type=int, default=60)
    parser.add_argument("--max-buildings", type=int, default=0)
    parser.add_argument("--min-mask0-pixels", type=int, default=4)
    parser.add_argument("--min-mask-pixels", type=int, default=2)
    parser.add_argument("--kappa", type=float, default=0.25)
    parser.add_argument("--min-component", type=int, default=2)
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
    run(parser.parse_args())


if __name__ == "__main__":
    main()
