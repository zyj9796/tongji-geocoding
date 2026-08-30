from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("FONTCONFIG_PATH", "/etc/fonts")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from osgeo import gdal

from io_paths import (
    BUILDINGS_SHP,
    DATA_DIR,
    DSM_TIF,
    FULL_AREA_DIR as FULL_DIR,
    FULL_AREA_GEOJSON_DIR,
    FULL_AREA_IMAGE_DIR,
    FULL_AREA_RASTER_DIR,
    FULL_AREA_LOG_DIR,
    PROJECT_DIR,
    REPO_ROOT,
    RESULTS_DIR,
    RSLC_DIR,
    SUMMARY_DIR,
    TIF_DIR,
    ensure_core_output_dirs,
)
from sar_display import stretch_sar_grayscale

sys.path.insert(0, str(REPO_ROOT / "src"))

import geocode_tongji_all_buildings_compare_gamma as full_core


def read_tif(path: Path) -> tuple[np.ndarray, list[float]]:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(path)
    arr = ds.ReadAsArray().astype(np.float32)
    gt = ds.GetGeoTransform()
    extent = [gt[0], gt[0] + ds.RasterXSize * gt[1], gt[3] + ds.RasterYSize * gt[5], gt[3]]
    ds = None
    return stretch_sar_grayscale(arr), extent


def read_buildings(path: Path) -> list[np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rings = []
    for feat in data.get("features", []):
        ring = np.asarray(feat["geometry"]["coordinates"][0], dtype=np.float64)
        if ring.shape[0] > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        rings.append(ring[:, :2])
    return rings


def read_points(path: Path, amp: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    method = []
    gamma = []
    intensity = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method.append((float(row["method_lon"]), float(row["method_lat"]), float(row["method_height_m"])))
            gamma.append((float(row["gamma_dsm_lon"]), float(row["gamma_dsm_lat"])))
            if amp is not None:
                rr = int(round(float(row["row"])))
                cc = int(round(float(row["col"])))
                if 0 <= rr < amp.shape[0] and 0 <= cc < amp.shape[1]:
                    intensity.append(float(amp[rr, cc]) / 255.0)
                else:
                    intensity.append(0.0)
    return np.asarray(method, dtype=np.float64), np.asarray(gamma, dtype=np.float64), np.asarray(intensity, dtype=np.float64)


def write_points_geojson(points_csv: Path, out_geojson: Path) -> None:
    features = []
    with points_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "fid": int(row["fid"]),
                        "row": float(row["row"]),
                        "col": float(row["col"]),
                        "height_m": float(row["method_height_m"]),
                        "gamma_lon": float(row["gamma_dsm_lon"]),
                        "gamma_lat": float(row["gamma_dsm_lat"]),
                        "gamma_height_m": float(row["gamma_dsm_height_m"]),
                        "gamma_ok": int(row["gamma_dsm_ok"]),
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(row["method_lon"]), float(row["method_lat"]), float(row["method_height_m"])],
                    },
                }
            )
    out_geojson.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False), encoding="utf-8")


def safe_apply_dsm_heights(buildings: list[dict], dsm) -> list[dict]:
    out = []
    for b in buildings:
        try:
            top_h = dsm.building_surface_height(b["ring_lonlat"])
        except Exception as exc:
            print(f"skip building {b.get('fid')}: DSM sample failed: {exc}", flush=True)
            continue
        item = dict(b)
        item["top_height_m"] = top_h
        item["base_height_m"] = max(0.0, top_h - float(b["height_m"]))
        out.append(item)
    return out


def use_gamma_dsm_tif_without_control_point_warp(
    _out_tif: Path,
    _date: str,
    _out_dir: Path,
    _amp: np.ndarray,
    source_gamma_tif: Path,
    _point_rows: list[dict],
    **_kwargs,
) -> Path:
    source_gamma_tif = Path(source_gamma_tif)
    out_tif = Path(_out_tif)
    out_tif.unlink(missing_ok=True)
    try:
        os.link(source_gamma_tif, out_tif)
    except OSError:
        shutil.copy2(source_gamma_tif, out_tif)
    return out_tif


def plot_fig54_with_gamma_dsm_background(
    out_png: Path,
    gamma_tif: Path,
    valid_buildings: list[dict],
    method_points: np.ndarray,
    gamma_points: np.ndarray,
    _sar_intensity: np.ndarray | None = None,
) -> None:
    original_line = full_core.Line2D

    def corrected_line(*args, **kwargs):
        if kwargs.get("label") == "Building-aligned GAMMA/DSM GeoTIFF":
            kwargs["label"] = "GAMMA/DSM terrain-geocoded RSLC"
        return original_line(*args, **kwargs)

    full_core.Line2D = corrected_line
    old_facecolor = plt.rcParams["axes.facecolor"]
    old_legend_facecolor = plt.rcParams["legend.facecolor"]
    plt.rcParams["axes.facecolor"] = "#050505"
    plt.rcParams["legend.facecolor"] = "white"
    try:
        full_core._original_plot_fig54_like(
            out_png, gamma_tif, valid_buildings, method_points, gamma_points, None
        )
    finally:
        full_core.Line2D = original_line
        plt.rcParams["axes.facecolor"] = old_facecolor
        plt.rcParams["legend.facecolor"] = old_legend_facecolor


def make_full_area_figures(
    date: str,
    gamma_tif: Path,
    stats_csv: Path,
    points_csv: Path,
    buildings_geojson: Path,
    image_dir: Path = FULL_AREA_IMAGE_DIR,
    include_statistics: bool = True,
) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    bg, extent = read_tif(gamma_tif)
    rings = read_buildings(buildings_geojson)
    par = full_core.parse_gamma_par(RSLC_DIR / f"{date}.rslc.par")
    amp = full_core.read_rslc_amplitude(RSLC_DIR / f"{date}.rslc", int(par["azimuth_lines"]), int(par["range_samples"]))
    method, gamma, intensity = read_points(points_csv, amp)
    all_rings = np.vstack(rings)
    xpad = max(float(np.ptp(all_rings[:, 0])) * 0.035, 0.00015)
    ypad = max(float(np.ptp(all_rings[:, 1])) * 0.035, 0.00015)
    xlim = (float(np.min(all_rings[:, 0]) - xpad), float(np.max(all_rings[:, 0]) + xpad))
    ylim = (float(np.min(all_rings[:, 1]) - ypad), float(np.max(all_rings[:, 1]) + ypad))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 6.3), dpi=450)
    ax.set_facecolor("#050505")
    ax.imshow(bg, cmap="gray", extent=extent, origin="upper", interpolation="nearest", alpha=0.78, zorder=1)
    step_ring = max(1, len(rings) // 1000)
    for ring in rings[::step_ring]:
        ax.add_patch(MplPolygon(ring, closed=True, fill=False, edgecolor="white", linewidth=0.32, alpha=0.62, zorder=3))
        ax.add_patch(MplPolygon(ring, closed=True, fill=False, edgecolor="#222222", linewidth=0.16, alpha=0.7, zorder=4))
    step_gamma = max(1, gamma.shape[0] // 60000)
    step_method = max(1, method.shape[0] // 60000)
    ax.scatter(gamma[::step_gamma, 0], gamma[::step_gamma, 1], s=0.55, c="#f28e2b", alpha=0.24, linewidths=0, label="GAMMA/DEM", zorder=5)
    sc = ax.scatter(
        method[::step_method, 0],
        method[::step_method, 1],
        s=0.8,
        c=method[::step_method, 2],
        cmap="viridis",
        alpha=0.72,
        linewidths=0,
        label="Building constrained",
        zorder=6,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude / deg")
    ax.set_ylabel("Latitude / deg")
    ax.set_title(f"Tongji full-area building-constrained geocoding ({date})")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(color="white", linewidth=0.2, alpha=0.2)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.018)
    cbar.set_label("Height / m")
    fig.tight_layout(pad=0.25)
    fig.savefig(image_dir / f"{date}_fig_full_area_gamma_vs_proposed.png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    if not include_statistics:
        return

    rows = list(csv.DictReader(stats_csv.open(encoding="utf-8")))
    method_mean = np.asarray([float(r["method_mean_boundary_distance_m"]) for r in rows], dtype=np.float64)
    gamma_mean = np.asarray([float(r["gamma_dsm_mean_boundary_distance_m"]) for r in rows], dtype=np.float64)
    order = np.argsort(gamma_mean)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), dpi=450)
    x = np.arange(len(rows))
    axes[0].scatter(x, gamma_mean[order], s=3.0, c="#f28e2b", alpha=0.62, linewidths=0, label="GAMMA/DEM")
    axes[0].scatter(x, np.maximum(method_mean[order], 1e-6), s=3.0, c="#2c7fb8", alpha=0.68, linewidths=0, label="Building constrained")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Buildings sorted by GAMMA/DEM error")
    axes[0].set_ylabel("Mean boundary distance / m")
    axes[0].grid(axis="y", which="both", color="#dddddd", linewidth=0.28)
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].boxplot([gamma_mean, np.maximum(method_mean, 1e-6)], labels=["GAMMA/DEM", "Building\nconstrained"], showfliers=True)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Mean boundary distance / m")
    axes[1].grid(axis="y", which="both", color="#dddddd", linewidth=0.28)
    fig.suptitle("Full-area horizontal error statistics")
    fig.tight_layout(pad=0.35)
    fig.savefig(image_dir / f"{date}_fig_full_area_error_statistics.png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def collect_full_area_images() -> None:
    FULL_AREA_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for path in FULL_DIR.glob("*.png"):
        path.replace(FULL_AREA_IMAGE_DIR / path.name)


def collect_full_area_data() -> None:
    FULL_AREA_GEOJSON_DIR.mkdir(parents=True, exist_ok=True)
    FULL_AREA_RASTER_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    for path in FULL_DIR.glob("*.geojson"):
        path.replace(FULL_AREA_GEOJSON_DIR / path.name)
    for path in FULL_DIR.glob("*.json"):
        path.replace(SUMMARY_DIR / path.name)
    for pattern in ["*.tif", "*.vrt"]:
        for path in FULL_DIR.glob(pattern):
            path.replace(FULL_AREA_RASTER_DIR / path.name)
    meta_path = SUMMARY_DIR / "20200708_building_aligned_gamma_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tif_path = FULL_AREA_RASTER_DIR / "20200708_building_aligned_gamma_dsm_geocoded_wgs84.tif"
        if tif_path.exists():
            meta["output_tif"] = str(tif_path)
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(date: str, stats_csv: Path, points_csv: Path, skipped_csv: Path) -> dict:
    rows = list(csv.DictReader(stats_csv.open(encoding="utf-8")))
    skipped = list(csv.DictReader(skipped_csv.open(encoding="utf-8"))) if skipped_csv.exists() else []
    point_count = sum(1 for _ in points_csv.open(encoding="utf-8")) - 1
    method_mean = [float(r["method_mean_boundary_distance_m"]) for r in rows]
    gamma_mean = [float(r["gamma_dsm_mean_boundary_distance_m"]) for r in rows]
    out = {
        "date": date,
        "valid_buildings": len(rows),
        "skipped_buildings": len(skipped),
        "scatter_points": point_count,
        "method_mean_boundary_distance_m": float(np.mean(method_mean)),
        "method_median_boundary_distance_m": float(statistics.median(method_mean)),
        "method_p90_boundary_distance_m": float(np.percentile(method_mean, 90)),
        "gamma_dem_mean_boundary_distance_m": float(np.mean(gamma_mean)),
        "gamma_dem_median_boundary_distance_m": float(statistics.median(gamma_mean)),
        "gamma_dem_p90_boundary_distance_m": float(np.percentile(gamma_mean, 90)),
    }
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    (SUMMARY_DIR / f"{date}_full_area_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_full_area_readme(summary: dict) -> None:
    text = f"""# Tongji All-Building Geocoding vs GAMMA

- Valid geocoded buildings: {summary['valid_buildings']}
- Skipped buildings: {summary['skipped_buildings']}
- Scatter points: {summary['scatter_points']}
- Main Figure 5.4-like map: `20200708_图件_317829351926.png`
- Error scatter: `20200708_fig5_4_like_all_buildings_error_scatter.png`
- Point coordinate pairs: `20200708_all_buildings_method_vs_gamma_points.csv`
- Per-building statistics: `20200708_all_buildings_fig5_4_like_stats.csv`
- Skipped list: `20200708_all_buildings_skipped.csv`

Each valid building is processed with footprint-height model projection, SAR-amplitude mask refinement, and model-surface coordinate inversion. Building top heights are sampled from `{DSM_TIF}`; base heights are DSM top minus vector height and are clamped to 0 m when necessary.

The GAMMA comparison solves the same sampled SAR pixels on the DSM-height surface. The visible SAR backdrop is independently terrain-geocoded from RSLC and DSM with `dem_import -> multi_look -> gc_map2 -> geocode_back -> data2geotiff`; GDAL only reprojects it to EPSG:4326. No zero-height GCP/TPS or building-control-point polynomial warp is used.

Display grayscale matches `图_02_初始投影掩膜.png`: square-root intensity to amplitude, valid-pixel 2%-98% clipping, and linear black-white mapping. This does not alter raster values or coordinates.

Current method mean/median/P90 boundary distance: {summary['method_mean_boundary_distance_m']:.6f}/{summary['method_median_boundary_distance_m']:.6f}/{summary['method_p90_boundary_distance_m']:.6f} m. Current GAMMA/DSM mean/median/P90: {summary['gamma_dem_mean_boundary_distance_m']:.6f}/{summary['gamma_dem_median_boundary_distance_m']:.6f}/{summary['gamma_dem_p90_boundary_distance_m']:.6f} m.
"""
    (FULL_DIR / "README.md").write_text(text, encoding="utf-8")


def update_markdown(summary: dict) -> None:
    block = f"""

## Tongji 全区域地理编码补充实验

已新增全区域批处理脚本：

```bash
cd /home/u/geocoding/geo_bc/a_geo_tongji
bash run_full_area.sh
```

全区域结果目录：

```text
results/outputs/tables/full_area
```

本次全区域实验使用 `data/shp/tongji_clip.shp` 中位于 `{summary['date']}` SAR 地理编码覆盖范围内的所有可处理建筑物，而不是只选择高层样例建筑。处理流程仍采用建筑物轮廓、高度属性和 DSM 构建三维挤出模型，再用 GAMMA `.rslc.par` 参数完成零多普勒/斜距几何投影，最后将精炼后的强散射像素反算到建筑物三角网表面，并与传统 GAMMA/DEM 高程面反算结果对比。

全区域统计结果：

- 有效建筑物：{summary['valid_buildings']} 栋。
- 跳过建筑物：{summary['skipped_buildings']} 栋，主要原因包括投影掩膜过小、精炼掩膜过小或无有效散射点。
- 建筑物约束散射点：{summary['scatter_points']} 个。
- 建筑物约束方法平均边界距离：{summary['method_mean_boundary_distance_m']:.6g} m。
- 建筑物约束方法 90 分位边界距离：{summary['method_p90_boundary_distance_m']:.6g} m。
- 传统 GAMMA/DEM 平均边界距离：{summary['gamma_dem_mean_boundary_distance_m']:.3f} m。
- 传统 GAMMA/DEM 90 分位边界距离：{summary['gamma_dem_p90_boundary_distance_m']:.3f} m。

全区域关键输出：

- `results/outputs/tables/full_area/{summary['date']}_all_buildings_fig5_4_like_stats.csv`
- `results/outputs/tables/full_area/{summary['date']}_all_buildings_method_vs_gamma_points.csv`
- `results/outputs/tables/full_area/{summary['date']}_all_buildings_skipped.csv`
- `results/outputs/geodata/full_area/{summary['date']}_all_valid_geocoded_buildings.geojson`
- `results/outputs/geodata/full_area/{summary['date']}_all_buildings_proposed_points.geojson`
- `results/picall/主流程/{summary['date']}_fig_full_area_gamma_vs_proposed.png`
- `results/picall/主流程/{summary['date']}_fig_full_area_error_statistics.png`

全区域图中，传统 GAMMA/DEM 点云在多个建筑群附近呈现沿雷达几何方向的侧向偏移；建筑物约束方法则将散射点压回建筑物轮廓、屋顶和立面三角网附近，更符合“附加轮廓矢量的 SAR 建筑物精细地理编码”的文献思路。
"""
    for md in [RESULTS_DIR / "README.md"]:
        text = md.read_text(encoding="utf-8") if md.exists() else ""
        marker = "## Tongji 全区域地理编码补充实验"
        if marker in text:
            text = text[: text.index(marker)].rstrip() + "\n"
        md.write_text(text.rstrip() + block + "\n", encoding="utf-8")


def write_runner() -> None:
    runner = PROJECT_DIR / "run_full_area.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd \"$(dirname \"$0\")\"\n"
        "mkdir -p results/outputs/tables/full_area results/outputs/logs/full_area\n"
        "/usr/bin/python3 code/prepare_dsm_for_sar.py\n"
        "/usr/bin/python3 code/run_full_area_geocode.py \"$@\" 2>&1 | tee results/outputs/logs/full_area/run_full_area.log\n"
        "/usr/bin/python3 code/pic_all.py sync\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20200708")
    parser.add_argument("--max-points-per-building", type=int, default=60)
    parser.add_argument("--max-buildings", type=int, default=0, help="0 means all buildings in SAR coverage")
    args = parser.parse_args()

    ensure_core_output_dirs()
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    FULL_AREA_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    write_runner()
    gamma_tif = TIF_DIR / f"{args.date}_gamma_dem_geocoded_wgs84.tif"
    if not gamma_tif.exists():
        raise FileNotFoundError(f"Missing GAMMA geocoded TIF: {gamma_tif}. Run bash run.sh first.")

    full_core.DEFAULT_RSLC_DIR = RSLC_DIR
    full_core.apply_dsm_heights = safe_apply_dsm_heights
    # The input is now a real GAMMA/DSM terrain-geocoded raster. Do not apply
    # the former building-control-point polynomial warp to the SAR background.
    full_core.write_building_aligned_gamma_tif = use_gamma_dsm_tif_without_control_point_warp
    full_core.read_geotiff = read_tif
    if not hasattr(full_core, "_original_plot_fig54_like"):
        full_core._original_plot_fig54_like = full_core.plot_fig54_like
    full_core.plot_fig54_like = plot_fig54_with_gamma_dsm_background
    ns = argparse.Namespace(
        date=args.date,
        buildings_shp=str(BUILDINGS_SHP),
        gamma_tif=str(gamma_tif),
        dsm=str(DSM_TIF),
        out_dir=str(FULL_DIR),
        max_buildings=args.max_buildings,
        max_points_per_building=args.max_points_per_building,
        min_mask0_pixels=4,
        min_mask_pixels=2,
    )
    full_core.run(ns)
    collect_full_area_images()
    collect_full_area_data()

    stats_csv = FULL_DIR / f"{args.date}_all_buildings_fig5_4_like_stats.csv"
    points_csv = FULL_DIR / f"{args.date}_all_buildings_method_vs_gamma_points.csv"
    skipped_csv = FULL_DIR / f"{args.date}_all_buildings_skipped.csv"
    buildings_geojson = FULL_AREA_GEOJSON_DIR / f"{args.date}_all_valid_geocoded_buildings.geojson"
    write_points_geojson(points_csv, FULL_AREA_GEOJSON_DIR / f"{args.date}_all_buildings_proposed_points.geojson")
    make_full_area_figures(args.date, gamma_tif, stats_csv, points_csv, buildings_geojson)
    collect_full_area_images()
    collect_full_area_data()
    summary = summarize(args.date, stats_csv, points_csv, skipped_csv)
    write_full_area_readme(summary)
    update_markdown(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
