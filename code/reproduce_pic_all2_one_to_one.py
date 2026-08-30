from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR / "code"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from chinese_matplotlib import install_chinese_labels  # noqa: E402

install_chinese_labels()

import compare_psinsar_3d_with_shizhan as legacy_ps  # noqa: E402
import compare_same_ps_pixel_3d as same_ps  # noqa: E402
import geocode_tongji_all_buildings_compare_gamma as full_core  # noqa: E402
import make_additional_full_area_ppt_figures as additional_figures  # noqa: E402
import make_full_area_planar_comparison as planar_figure  # noqa: E402
import make_same_pixel_ppt_figures as same_ps_figures  # noqa: E402
import optimize_strict_local_registration as strict_local  # noqa: E402
import register_strict_triangle_projection as registration_plot  # noqa: E402
import run_result_tongji_geocoding as paper_core  # noqa: E402
from geocode_gamma_rslc_with_buildings import make_orbit, parse_gamma_par, read_rslc_amplitude  # noqa: E402
from io_paths import DSM_TIF, PS_POINTS_CSV, RSLC_DIR, TIF_DIR  # noqa: E402
from run_full_area_geocode import make_full_area_figures, plot_fig54_with_gamma_dsm_background, read_points, read_tif  # noqa: E402
from run_registered_full_area_geocode import registered_rasterizer, registered_refiner  # noqa: E402


REFERENCE_NAMES = {
    "20200708_图件_317829351926.png",
    "20200708_图件_1009510162568.png",
    "20200708_图件_1045917436903_版本2.png",
    "20200708_配准后严格三角面投影.png",
    "20200708_配准后严格三角面精化掩膜.png",
    "图_01_雷达总览与选定建筑像素.png",
    "图_01_同济校区雷达强度与建筑轮廓.png",
    "图_02_初始投影掩膜.png",
    "图_02_逐建筑雷达裁剪图.png",
    "图_03_精化掩膜.png",
    "图_04_同像素地理定位对比.png",
    "图_04_初始与精化掩膜对比.png",
    "图_05_建筑约束地理编码点.png",
    "图_06_伽马软件与建筑约束方法对比图.png",
    "图_08_三维散射点.png",
    "图_09_地理编码点局部放大.png",
    "图件_982942697266.png",
    "图件_292635425203.png",
    "图_同像素永久散射体三维建筑.png",
    "图_同像素永久散射体雷达像素.png",
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_buildings(path: Path) -> tuple[list[dict], dict[int, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    buildings: list[dict] = []
    by_fid: dict[int, dict] = {}
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        ring = np.asarray(feature["geometry"]["coordinates"][0], dtype=np.float64)
        if ring.shape[0] > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        item = {
            "fid": int(props["fid"]),
            "floor": int(props.get("floor", 0)),
            "height_m": float(props.get("height_m", 0.0)),
            "base_height_m": float(props.get("base_height_m", 0.0)),
            "top_height_m": float(props.get("top_height_m", 0.0)),
            "ring_lonlat": ring[:, :2],
        }
        buildings.append(item)
        by_fid[item["fid"]] = item
    return buildings, by_fid


def point_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    method = np.asarray(
        [[float(row["method_lon"]), float(row["method_lat"]), float(row["method_height_m"])] for row in rows],
        dtype=np.float64,
    )
    gamma = np.asarray(
        [[float(row["gamma_dsm_lon"]), float(row["gamma_dsm_lat"])] for row in rows], dtype=np.float64
    )
    return method, gamma


def plot_zoom(
    path: Path,
    gamma_tif: Path,
    building: dict,
    method: np.ndarray,
    gamma: np.ndarray,
) -> None:
    bg, extent = read_tif(gamma_tif)
    ring = building["ring_lonlat"]
    padx = max(float(np.ptp(ring[:, 0])) * 1.6, 0.00035)
    pady = max(float(np.ptp(ring[:, 1])) * 1.6, 0.00035)
    fig, ax = plt.subplots(figsize=(5.0, 4.4), dpi=500)
    ax.imshow(bg, cmap="gray", extent=extent, origin="upper", interpolation="nearest")
    ax.add_patch(MplPolygon(ring, closed=True, fill=False, edgecolor="white", linewidth=1.6, zorder=3))
    ax.add_patch(MplPolygon(ring, closed=True, fill=False, edgecolor="#111111", linewidth=0.8, zorder=4))
    ax.scatter(gamma[:, 0], gamma[:, 1], s=5, c="#f28e2b", alpha=0.45, linewidths=0, label="GAMMA/DEM")
    ax.scatter(method[:, 0], method[:, 1], s=6, c="#2c7fb8", alpha=0.88, linewidths=0, label="Building constrained")
    ax.set_xlim(float(np.mean(ring[:, 0]) - padx), float(np.mean(ring[:, 0]) + padx))
    ax.set_ylim(float(np.mean(ring[:, 1]) - pady), float(np.mean(ring[:, 1]) + pady))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude / deg")
    ax.set_ylabel("Latitude / deg")
    ax.set_title(f"Zoomed geocoding points, FID {building['fid']}")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(color="white", alpha=0.24, linewidth=0.25)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    fig.tight_layout(pad=0.25)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def reproduce_main_paper_figures(
    pic_dir: Path,
    points_rows: list[dict],
    stats_rows: list[dict],
    buildings_by_fid: dict[int, dict],
    gamma_tif: Path,
    row_shift: int,
    col_shift: int,
    local_shifts: dict[int, tuple[int, int]],
) -> None:
    ranked = sorted(stats_rows, key=lambda row: int(float(row["sample_points"])), reverse=True)
    selected_fids = [int(row["fid"]) for row in ranked[:24] if int(row["fid"]) in buildings_by_fid]
    selected = [buildings_by_fid[fid] for fid in selected_fids]
    selected_set = set(selected_fids)
    selected_rows = [row for row in points_rows if int(row["fid"]) in selected_set]
    par = parse_gamma_par(RSLC_DIR / "20200708.rslc.par")
    orbit = make_orbit(par)
    amp = read_rslc_amplitude(RSLC_DIR / "20200708.rslc", int(par["azimuth_lines"]), int(par["range_samples"]))
    rasterizer = registered_rasterizer(row_shift, col_shift, local_shifts)
    refiner = registered_refiner(0.25, 2)
    models = []
    for building in selected:
        model = rasterizer(building, par, orbit, amp.shape)
        model["mask"] = refiner(model["mask0"], amp)
        models.append(model)

    paper_core.read_geotiff = read_tif
    paper_core.plot_intensity_with_buildings(pic_dir / "图_01_同济校区雷达强度与建筑轮廓.png", gamma_tif, selected)
    paper_core.plot_masks(pic_dir / "图_02_初始投影掩膜.png", amp, selected, models, "initial")
    paper_core.plot_masks(pic_dir / "图_03_精化掩膜.png", amp, selected, models, "refined")
    paper_core.plot_initial_vs_refined(pic_dir / "图_04_初始与精化掩膜对比.png", amp, models)

    method, gamma = point_arrays(selected_rows)
    paper_core.plot_points_map(pic_dir / "图_05_建筑约束地理编码点.png", gamma_tif, selected, method)
    paper_core.plot_points_map(pic_dir / "图_06_伽马软件与建筑约束方法对比图.png", gamma_tif, selected, method, gamma)
    points_by_building = []
    for fid in selected_fids:
        fid_rows = [row for row in selected_rows if int(row["fid"]) == fid]
        points_by_building.append(
            np.asarray(
                [
                    [
                        float(row["row"]),
                        float(row["col"]),
                        float(row["method_lon"]),
                        float(row["method_lat"]),
                        float(row["method_height_m"]),
                        float(row.get("triangle_index", 0)),
                    ]
                    for row in fid_rows
                ],
                dtype=np.float64,
            )
        )
    paper_core.plot_3d(pic_dir / "图_08_三维散射点.png", selected, points_by_building)
    densest_fid = max(selected_fids, key=lambda fid: sum(int(row["fid"]) == fid for row in selected_rows))
    dense_rows = [row for row in selected_rows if int(row["fid"]) == densest_fid]
    dense_method, dense_gamma = point_arrays(dense_rows)
    plot_zoom(pic_dir / "图_09_地理编码点局部放大.png", gamma_tif, buildings_by_fid[densest_fid], dense_method, dense_gamma)


def reproduce_full_area_figures(
    pic_dir: Path,
    data_dir: Path,
    points_path: Path,
    stats_path: Path,
    buildings_path: Path,
    gamma_tif: Path,
    buildings: list[dict],
    points_rows: list[dict],
) -> None:
    par = parse_gamma_par(RSLC_DIR / "20200708.rslc.par")
    amp = read_rslc_amplitude(RSLC_DIR / "20200708.rslc", int(par["azimuth_lines"]), int(par["range_samples"]))
    method, gamma, intensity = read_points(points_path, amp)
    full_core.read_geotiff = read_tif
    if not hasattr(full_core, "_original_plot_fig54_like"):
        full_core._original_plot_fig54_like = full_core.plot_fig54_like
    full_core.plot_fig54_like = plot_fig54_with_gamma_dsm_background
    full_core.plot_fig54_like(
        pic_dir / "20200708_图件_317829351926.png",
        gamma_tif,
        buildings,
        method,
        gamma,
        intensity,
    )
    make_full_area_figures(
        "20200708",
        gamma_tif,
        stats_path,
        points_path,
        buildings_path,
        image_dir=pic_dir,
        include_statistics=False,
    )
    historical_name = pic_dir / "20200708_图件_1009510162568.png"
    chinese_name = pic_dir / "20200708_图件_1009510162568.png"
    if historical_name.exists():
        historical_name.replace(chinese_name)

    planar_figure.POINTS_CSV = points_path
    planar_figure.BUILDINGS_GEOJSON = buildings_path
    planar_figure.STATS_CSV = stats_path
    planar_figure.OUT_PNG = pic_dir / "20200708_图件_1045917436903_版本2.png"
    planar_figure.PPT_FIG_DIR = data_dir / "unused_ppt"
    planar_figure.PPT_COPY = planar_figure.PPT_FIG_DIR / planar_figure.OUT_PNG.name
    planar_figure.make_figure()

    additional_figures.POINTS_CSV = points_path
    additional_figures.STATS_CSV = stats_path
    additional_figures.BUILDINGS_GEOJSON = buildings_path
    additional_figures.FIG_DIR = pic_dir
    additional_figures.FULL_AREA_IMAGE_DIR = pic_dir
    method2, gamma2, fids = additional_figures.arrays(points_rows)
    building_map = additional_figures.load_buildings(buildings_path)
    stats = additional_figures.read_csv(stats_path)
    additional_figures.fig_zoom_cases(method2, gamma2, fids, building_map, stats)


def reproduce_registered_projection_figures(
    pic_dir: Path,
    registered_triangles: Path,
    row_shift: int,
    col_shift: int,
) -> None:
    par = parse_gamma_par(RSLC_DIR / "20200708.rslc.par")
    amp = read_rslc_amplitude(RSLC_DIR / "20200708.rslc", int(par["azimuth_lines"]), int(par["range_samples"]))
    _payload, triangles = registration_plot.load_triangles(registered_triangles)
    segments = registration_plot.triangle_segments(triangles, 0, 0)
    by_fid: dict[int, list[np.ndarray]] = {}
    for item in triangles:
        fid = int(item["feature"].get("properties", {}).get("fid", -1))
        by_fid.setdefault(fid, []).append(item["xy"])
    refined = np.zeros(amp.shape, dtype=bool)
    for items in by_fid.values():
        mask0 = registration_plot.rasterize_triangles(items, amp.shape)
        mask, _threshold = registration_plot.refine_triangle_mask(mask0, amp, 0.25, 2)
        refined |= mask
    registration_plot.plot_projection(
        pic_dir / "20200708_配准后严格三角面投影.png",
        amp,
        segments,
        "20200708",
        row_shift,
        col_shift,
    )
    registration_plot.plot_refined(
        pic_dir / "20200708_配准后严格三角面精化掩膜.png",
        amp,
        refined,
        segments["roof"],
        "20200708",
        row_shift,
        col_shift,
    )


def reproduce_ps_figures(
    pic_dir: Path,
    data_dir: Path,
    points_path: Path,
    buildings_path: Path,
    row_shift: int,
    col_shift: int,
) -> None:
    legacy_dir = data_dir / "legacy_ps"
    ps = legacy_ps.load_ps(PS_POINTS_CSV)
    method = legacy_ps.load_shizhan(points_path)
    nearest_rows, summary = legacy_ps.compare(ps, method, 15.0)
    by_building = legacy_ps.summarize_by_fid(nearest_rows, 15.0)
    write_csv(legacy_dir / "psinsar_3dlut_vs_bc_nearest_points.csv", nearest_rows)
    write_csv(legacy_dir / "psinsar_3dlut_vs_bc_by_building.csv", by_building)
    (legacy_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    legacy_ps.plot_map(pic_dir / "图件_292635425203.png", nearest_rows)

    same_dir = data_dir / "same_pixel"
    ps_buildings = same_ps.load_buildings(buildings_path)
    candidate_fids = same_ps.candidate_fids(legacy_dir / "psinsar_3dlut_vs_bc_by_building.csv", ps_buildings, 120)
    ps_par = parse_gamma_par(RSLC_DIR / "20220601.rslc.par")
    ps_orbit = make_orbit(ps_par)
    ps_amp_raw = read_rslc_amplitude(
        RSLC_DIR / "20220601.rslc", int(ps_par["azimuth_lines"]), int(ps_par["range_samples"])
    )
    ps_amp = ps_amp_raw.astype(np.float32) / 255.0
    ps_edges = strict_local.edge_map(ps_amp_raw)
    global_rasterizer = registered_rasterizer(row_shift, col_shift)
    ps_local_shifts: dict[int, tuple[int, int]] = {}
    ps_local_rows: list[dict] = []
    for fid in candidate_fids:
        model = global_rasterizer(ps_buildings[fid], ps_par, ps_orbit, ps_amp_raw.shape)
        roof_xy = [
            np.column_stack([model["projected_rc"][tri, 1], model["projected_rc"][tri, 0]])
            for tri, surface in zip(model["triangles"], model["surfaces"])
            if str(surface) == "roof"
        ]
        roof_mask = registration_plot.rasterize_triangles(roof_xy, ps_amp_raw.shape)
        result = strict_local.optimize_mask(roof_mask, ps_amp, ps_edges, 10, 2, 5.0)
        ps_local_shifts[fid] = (int(result["applied_row_shift"]), int(result["applied_col_shift"]))
        ps_local_rows.append({"fid": fid, **result})
    write_csv(same_dir / "20220601_strict_local_registration_metrics.csv", ps_local_rows)
    same_ps.SAME_PIXEL_IMAGE_DIR = pic_dir
    same_ps.rasterize_building = registered_rasterizer(row_shift, col_shift, ps_local_shifts)
    same_args = argparse.Namespace(
        ps_csv=str(PS_POINTS_CSV),
        buildings_geojson=str(buildings_path),
        by_building_csv=str(legacy_dir / "psinsar_3dlut_vs_bc_by_building.csv"),
        rslc_par=str(RSLC_DIR / "20220601.rslc.par"),
        out_dir=str(same_dir),
        candidate_buildings=120,
        count=10,
        height_tolerance_m=3.0,
    )
    same_ps.run(same_args)
    matches = read_csv(same_dir / "same_ps_pixel_ps_gamma_vs_bc_points.csv")
    stats = read_csv(same_dir / "same_ps_pixel_selected_buildings_stats.csv")
    if not matches or not stats:
        raise RuntimeError("Strict registered same-PS-pixel workflow produced no valid selected buildings")

    same_ps_figures.FIG_DIR = pic_dir
    same_ps_figures.MATCH_CSV = same_dir / "same_ps_pixel_ps_gamma_vs_bc_points.csv"
    same_ps_figures.STATS_CSV = same_dir / "same_ps_pixel_selected_buildings_stats.csv"
    same_ps_figures.BUILDINGS_GEOJSON = buildings_path
    same_ps_figures.RSLC_PAR = RSLC_DIR / "20220601.rslc.par"
    same_ps_figures.rasterize_building = registered_rasterizer(row_shift, col_shift, ps_local_shifts)
    fids = same_ps_figures.selected_fids(stats)
    valid = same_ps_figures.height_valid_matches(matches, fids)
    sar = same_ps_figures.load_sar_image()
    buildings, models = same_ps_figures.build_models(fids)
    same_ps_figures.fig_sar_overview(sar, valid, fids, models)
    same_ps_figures.fig_per_building_crops(sar, valid, fids, models)
    same_ps_figures.fig_geographic_map(valid, buildings, fids)


def write_readme(pic_dir: Path, row_shift: int, col_shift: int, locally_shifted_buildings: int) -> None:
    names = sorted(path.name for path in pic_dir.glob("*.png"))
    lines = [
        "# pic_all2 one-to-one reproduction",
        "",
        "All figures are regenerated from the updated SHP and strict registered triangle projection; no image is copied from `pic_all`.",
        f"Radar registration: row `{row_shift:+d}`, column `{col_shift:+d}` pixels.",
        f"Conservative per-building local registration: `{locally_shifted_buildings}` buildings accepted within `±10 px` at score gain `>=5`.",
        "In comparison figures, orange PS/GAMMA points are the traditional baseline and may remain displaced; blue/height-colored points are the corrected building-constrained result.",
        f"PNG count: `{len(names)}`.",
        "",
        "## Files",
        "",
        *[f"- `{name}`" for name in names],
        "",
    ]
    (pic_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    pic_dir = Path(args.pic_dir)
    data_dir = Path(args.data_dir)
    pic_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    for path in pic_dir.glob("*.png"):
        path.unlink()

    registered_dir = PROJECT_DIR / "results" / "outputs" / "registered_full_area_geocode"
    points_path = registered_dir / "20200708_all_buildings_method_vs_gamma_points.csv"
    stats_path = registered_dir / "20200708_all_buildings_fig5_4_like_stats.csv"
    buildings_path = registered_dir / "20200708_all_valid_geocoded_buildings.geojson"
    summary_path = registered_dir / "20200708_registered_full_area_geocode_summary.json"
    registration_summary = json.loads(
        (PROJECT_DIR / "results" / "outputs" / "strict_triangle_registration" / "20200708_strict_triangle_registration_summary.json").read_text(encoding="utf-8")
    )
    row_shift = int(registration_summary["row_shift"])
    col_shift = int(registration_summary["col_shift"])
    local_metrics = (
        PROJECT_DIR
        / "results"
        / "outputs"
        / "strict_local_registration"
        / "20200708_strict_local_registration_metrics.csv"
    )
    local_shifts: dict[int, tuple[int, int]] = {}
    with local_metrics.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            local_shifts[int(row["fid"])] = (
                int(float(row["applied_row_shift"])),
                int(float(row["applied_col_shift"])),
            )
    gamma_tif = TIF_DIR / "20200708_gamma_dem_geocoded_wgs84.tif"
    points_rows = read_csv(points_path)
    stats_rows = read_csv(stats_path)
    buildings, buildings_by_fid = load_buildings(buildings_path)

    print("[1/5] full-area and planar figures", flush=True)
    reproduce_full_area_figures(pic_dir, data_dir, points_path, stats_path, buildings_path, gamma_tif, buildings, points_rows)
    print("[2/5] selected-building paper figures", flush=True)
    reproduce_main_paper_figures(
        pic_dir,
        points_rows,
        stats_rows,
        buildings_by_fid,
        gamma_tif,
        row_shift,
        col_shift,
        local_shifts,
    )
    print("[3/5] registered projection figures", flush=True)
    reproduce_registered_projection_figures(
        pic_dir,
        PROJECT_DIR
        / "results"
        / "outputs"
        / "strict_local_registration"
        / "20200708_locally_registered_strict_sar_surface_triangles.geojson",
        row_shift,
        col_shift,
    )
    print("[4/5] PS-InSAR and strict same-pixel figures", flush=True)
    reproduce_ps_figures(pic_dir, data_dir, points_path, buildings_path, row_shift, col_shift)
    print("[5/5] one-to-one validation", flush=True)
    actual = {path.name for path in pic_dir.glob("*.png")}
    if actual != REFERENCE_NAMES:
        raise RuntimeError(
            f"Figure manifest mismatch: missing={sorted(REFERENCE_NAMES - actual)}, extra={sorted(actual - REFERENCE_NAMES)}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = {
        "reference_png_count": len(REFERENCE_NAMES),
        "generated_png_count": len(actual),
        "one_to_one_filename_match": True,
        "updated_shp": summary.get("buildings_shp"),
        "registration_row_shift": row_shift,
        "registration_col_shift": col_shift,
        "locally_shifted_buildings": sum(1 for value in local_shifts.values() if value != (0, 0)),
        "valid_buildings": summary.get("valid_buildings"),
        "scatter_points": summary.get("scatter_points"),
        "files": sorted(actual),
    }
    (data_dir / "one_to_one_reproduction_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_readme(
        pic_dir,
        row_shift,
        col_shift,
        sum(1 for value in local_shifts.values() if value != (0, 0)),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="按当前中文图件清单逐图重建注册复现结果。")
    parser.add_argument("--pic-dir", default=str(PROJECT_DIR / "results" / "picall" / "注册复现"))
    parser.add_argument("--data-dir", default=str(PROJECT_DIR / "results" / "outputs" / "pic_all2_reproduction"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
