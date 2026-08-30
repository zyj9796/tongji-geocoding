from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("FONTCONFIG_PATH", "/etc/fonts")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 registers the "3d" projection
from osgeo import gdal
from osgeo import ogr

from io_paths import (
    BUILDINGS_SHP,
    DATA_DIR,
    DSM_TIF,
    GEOJSON_DIR,
    LOG_DIR,
    MAIN_IMAGE_DIR,
    PROJECT_DIR,
    REPO_ROOT,
    RESULTS_DIR,
    RSLC_DIR,
    TABLE_DIR,
    TIF_DIR,
    WORK_DIR,
    SUMMARY_DIR,
    ensure_core_output_dirs,
)
from gamma_dsm_geocode import geocode_rslc_with_dsm
from sar_display import stretch_sar_grayscale

sys.path.insert(0, str(REPO_ROOT / "src"))

import run_result_tongji_geocoding as core


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_a_geo_dirs(_out_root: Path) -> dict[str, Path]:
    ensure_core_output_dirs()
    dirs = {
        "logs": LOG_DIR,
        "tif": TIF_DIR,
        "png": WORK_DIR / "main_png_duplicates",
        "csv": TABLE_DIR,
        "geojson": GEOJSON_DIR,
        "figures": MAIN_IMAGE_DIR,
        "work": WORK_DIR,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


KEEP_MAIN_IMAGES = {
    "图_01_同济校区雷达强度与建筑轮廓.png",
    "图_02_初始投影掩膜.png",
    "图_03_精化掩膜.png",
    "图_04_初始与精化掩膜对比.png",
    "图_05_建筑约束地理编码点.png",
    "图_06_伽马软件与建筑约束方法对比图.png",
    "fig_07_error_statistics.png",
    "图_08_三维散射点.png",
    "图_09_地理编码点局部放大.png",
}


def prune_main_images() -> None:
    for path in MAIN_IMAGE_DIR.glob("*.png"):
        if re.match(r"^\d{8}_fig_0[1-9]_", path.name) and path.name not in KEEP_MAIN_IMAGES:
            path.unlink()
    shutil.rmtree(WORK_DIR / "main_png_duplicates", ignore_errors=True)
    try:
        WORK_DIR.rmdir()
    except OSError:
        pass


def archive_previous_error_logs() -> None:
    archive = LOG_DIR / "failed_attempts"
    for path in LOG_DIR.glob("*_error.txt"):
        archive.mkdir(parents=True, exist_ok=True)
        path.replace(archive / path.name)


def choose_local_buildings(bounds, dsm, max_buildings: int):
    buildings = core.load_area_buildings(BUILDINGS_SHP, bounds)
    enriched = []
    for item in buildings:
        item = dict(item)
        item["height_source"] = "height"
        if float(item.get("height_m", 0.0)) <= 0 and float(item.get("floor", 0.0)) > 0:
            item["height_source"] = "Floor*3m"
        try:
            top_h = dsm.building_surface_height(item["ring_lonlat"])
        except Exception as exc:
            log(f"skip building {item.get('fid')}: DSM sample failed: {exc}")
            continue
        item["top_height_m"] = top_h
        item["base_height_m"] = max(0.0, top_h - float(item["height_m"]))
        enriched.append(item)
        if max_buildings > 0 and len(enriched) >= max_buildings:
            break
    return enriched


def write_local_runner() -> None:
    runner = PROJECT_DIR / "run.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd \"$(dirname \"$0\")\"\n"
        "mkdir -p results/outputs/logs/main\n"
        "/usr/bin/python3 code/prepare_dsm_for_sar.py\n"
        "/usr/bin/python3 code/run_tongji_gamma_geocode.py \"$@\" 2>&1 | tee results/outputs/logs/main/run.log\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)


def shp_fields_note() -> str:
    ds = ogr.Open(str(BUILDINGS_SHP))
    if ds is None:
        return "建筑物 shp 未能读取。"
    lyr = ds.GetLayer(0)
    defn = lyr.GetLayerDefn()
    fields = [defn.GetFieldDefn(i).GetNameRef() for i in range(defn.GetFieldCount())]
    return f"建筑物字段：{fields}。本实验优先使用 `height`，并保留 `Floor` 层数字段；若高度缺失，可按 `Floor*3 m` 估算。"


def write_local_readme(dirs, processed_dates, all_stats, missing) -> None:
    method_mean = float(np.mean([s["method_mean_m"] for s in all_stats]))
    gamma_mean = float(np.mean([s["gamma_mean_m"] for s in all_stats]))
    total_points = int(sum(int(s.get("valid_points", 0)) for s in all_stats))
    n_buildings = len({(s.get("scene", ""), s.get("building", "")) for s in all_stats if "scene" in s and "building" in s})
    lines = [
        "# a_geo_tongji: Tongji GAMMA 建筑物约束 SAR 精细地理编码",
        "",
        "## 目录约定",
        "- `data/`: 原始输入数据，只读使用。",
        "- `code/`: 本次整理后的实验脚本。",
        "- `results/`: 本次全部输出结果。",
        "",
        "## 输入数据",
        f"- RSLC/GAMMA 参数: `{RSLC_DIR}`",
        f"- 处理日期: {', '.join(processed_dates)}",
        f"- Tongji 建筑物矢量: `{BUILDINGS_SHP}`",
        f"- Tongji DSM: `{DSM_TIF}`",
        f"- {shp_fields_note()}",
        "",
        "## 方法说明",
        "- 文献流程中的 ISCE 地理编码环节在本实验中用 GAMMA `.rslc.par` 参数替代。",
        "- 读取轨道状态向量、斜距采样、方位时间、多普勒多项式，按零多普勒与斜距方程将三维建筑物模型投影到 SAR 雷达坐标。",
        "- 由建筑物轮廓、DSM 和 `height/Floor` 属性构建底面、屋顶面、立面三角网挤出模型。",
        "- 在三维模型投影掩膜附近用 SAR 幅度统计、连通/形态学约束提取强散射像素。",
        "- 将强散射像素按投影三角面重心坐标反算到建筑物三维表面，并与 GAMMA/DEM 高程面反算结果对比。",
        "",
        "## 运行",
        "```bash",
        "cd /home/u/geocoding/geo_bc/a_geo_tongji",
        "bash run.sh",
        "```",
        "",
        "## 输出",
        f"- 图片: `results/picall/主流程/`",
        f"- GeoTIFF: `results/outputs/rasters/main/`",
        f"- 散射点与误差 CSV: `results/outputs/tables/main/`",
        f"- 建筑物和散射点 GeoJSON: `results/outputs/geodata/main/`",
        f"- 日志: `results/outputs/logs/main/run.log`",
        "",
        "## 关键论文图",
        "- `results/picall/主流程/图_01_同济校区雷达强度与建筑轮廓.png`",
        "- `results/picall/主流程/图_02_初始投影掩膜.png`",
        "- `results/picall/主流程/图_03_精化掩膜.png`",
        "- `results/picall/主流程/图_04_初始与精化掩膜对比.png`",
        "- `results/picall/主流程/图_05_建筑约束地理编码点.png`",
        "- `results/picall/主流程/图_06_伽马软件与建筑约束方法对比图.png`",
        "- `results/picall/主流程/fig_07_error_statistics.png`",
        "- `results/picall/主流程/图_08_三维散射点.png`",
        "- `results/picall/主流程/图_09_地理编码点局部放大.png`",
        "",
        "## 本次结果摘要",
        f"- 有效场景: {len(processed_dates)} 景。",
        f"- 有效建筑-时相组合: {n_buildings}。",
        f"- 输出建筑物约束散射点: {total_points} 个。",
        f"- 本文/GAMMA 替代方法平均轮廓边界距离: {method_mean:.4g} m。",
        f"- 传统 GAMMA/DEM 对比平均轮廓边界距离: {gamma_mean:.2f} m。",
        "- 本文方法点被约束回建筑物屋顶/立面三角网，传统 GAMMA/DEM 结果更容易落在地面高程面或建筑物外侧。",
    ]
    if missing:
        lines += ["", "## 缺失输入", *[f"- `{m}`" for m in missing]]
    (RESULTS_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_no_old_reference(_dirs) -> None:
    return None


def make_gamma_dsm_geocoded(date: str, dirs: dict[str, Path]):
    radar_tif, gamma_tif, _metadata = geocode_rslc_with_dsm(
        date, RSLC_DIR, DSM_TIF, dirs["tif"], WORK_DIR / "gamma_dsm_geocode"
    )
    return radar_tif, gamma_tif, core.parse_gamma_par(RSLC_DIR / f"{date}.rslc.par")


def _read_tif(path: Path):
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    arr = ds.ReadAsArray().astype(np.float32)
    gt = ds.GetGeoTransform()
    extent = [gt[0], gt[0] + ds.RasterXSize * gt[1], gt[3] + ds.RasterYSize * gt[5], gt[3]]
    ds = None
    return stretch_sar_grayscale(arr), extent


def _read_buildings(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for feat in data.get("features", []):
        ring = np.asarray(feat["geometry"]["coordinates"][0], dtype=float)
        if ring.shape[0] > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        out.append((feat.get("properties", {}), ring[:, :2]))
    return out


def make_polished_geocode_figures(date: str = "20200708") -> None:
    figures = MAIN_IMAGE_DIR
    gamma_tif = TIF_DIR / f"{date}_gamma_dem_geocoded_wgs84.tif"
    point_csv = TABLE_DIR / f"{date}_scatter_points_method_vs_gamma.csv"
    buildings_geojson = GEOJSON_DIR / f"{date}_valid_buildings.geojson"
    if not gamma_tif.exists() or not point_csv.exists() or not buildings_geojson.exists():
        return

    bg, extent = _read_tif(gamma_tif)
    buildings = _read_buildings(buildings_geojson)
    rows = list(csv.DictReader(point_csv.open(encoding="utf-8")))
    method = np.asarray([[float(r["method_lon"]), float(r["method_lat"]), float(r["method_height_m"])] for r in rows], dtype=float)
    gamma = np.asarray([[float(r["gamma_lon"]), float(r["gamma_lat"])] for r in rows], dtype=float)
    rings = np.vstack([ring for _, ring in buildings])
    xpad = max(float(np.ptp(rings[:, 0])) * 0.18, 0.00028)
    ypad = max(float(np.ptp(rings[:, 1])) * 0.18, 0.00028)
    xlim = (float(np.min(rings[:, 0]) - xpad), float(np.max(rings[:, 0]) + xpad))
    ylim = (float(np.min(rings[:, 1]) - ypad), float(np.max(rings[:, 1]) + ypad))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )
    fig, ax = plt.subplots(figsize=(6.4, 5.8), dpi=450)
    ax.imshow(bg, cmap="gray", extent=extent, origin="upper", interpolation="nearest")
    for props, ring in buildings:
        ax.add_patch(MplPolygon(ring, closed=True, facecolor="none", edgecolor="white", linewidth=1.0, zorder=3))
        ax.add_patch(MplPolygon(ring, closed=True, facecolor="none", edgecolor="#1f1f1f", linewidth=0.45, zorder=4))
    ax.scatter(gamma[:, 0], gamma[:, 1], s=1.4, c="#f28e2b", alpha=0.32, linewidths=0, label="GAMMA/DEM", zorder=5)
    sc = ax.scatter(method[:, 0], method[:, 1], s=1.8, c=method[:, 2], cmap="viridis", alpha=0.82, linewidths=0, label="Building constrained", zorder=6)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude / deg")
    ax.set_ylabel("Latitude / deg")
    ax.set_title("Tongji building-constrained SAR geocoding")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(color="white", alpha=0.22, linewidth=0.25)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, borderpad=0.4, handletextpad=0.4)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.76, pad=0.018)
    cbar.set_label("Height / m")
    fig.tight_layout(pad=0.25)
    out = figures / "图_06_伽马软件与建筑约束方法对比图.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    # Zoomed paper panel around the densest building to show point-level behavior.
    counts = {}
    for r in rows:
        counts[int(r["building"])] = counts.get(int(r["building"]), 0) + 1
    target = max(counts, key=counts.get)
    target_rows = [r for r in rows if int(r["building"]) == target]
    target_method = np.asarray([[float(r["method_lon"]), float(r["method_lat"]), float(r["method_height_m"])] for r in target_rows], dtype=float)
    target_gamma = np.asarray([[float(r["gamma_lon"]), float(r["gamma_lat"])] for r in target_rows], dtype=float)
    target_ring = buildings[target - 1][1]
    zpadx = max(float(np.ptp(target_ring[:, 0])) * 1.6, 0.00035)
    zpady = max(float(np.ptp(target_ring[:, 1])) * 1.6, 0.00035)
    zx = (float(np.mean(target_ring[:, 0]) - zpadx), float(np.mean(target_ring[:, 0]) + zpadx))
    zy = (float(np.mean(target_ring[:, 1]) - zpady), float(np.mean(target_ring[:, 1]) + zpady))
    fig, ax = plt.subplots(figsize=(5.0, 4.4), dpi=500)
    ax.imshow(bg, cmap="gray", extent=extent, origin="upper", interpolation="nearest")
    ax.add_patch(MplPolygon(target_ring, closed=True, facecolor="none", edgecolor="white", linewidth=1.6, zorder=3))
    ax.add_patch(MplPolygon(target_ring, closed=True, facecolor="none", edgecolor="#111111", linewidth=0.8, zorder=4))
    ax.scatter(target_gamma[:, 0], target_gamma[:, 1], s=5, c="#f28e2b", alpha=0.45, linewidths=0, label="GAMMA/DEM", zorder=5)
    ax.scatter(target_method[:, 0], target_method[:, 1], s=6, c="#2c7fb8", alpha=0.88, linewidths=0, label="Building constrained", zorder=6)
    ax.set_xlim(*zx)
    ax.set_ylim(*zy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude / deg")
    ax.set_ylabel("Latitude / deg")
    ax.set_title(f"Zoomed geocoding points, building {target}")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(color="white", alpha=0.24, linewidth=0.25)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    fig.tight_layout(pad=0.25)
    zoom = figures / "图_09_地理编码点局部放大.png"
    fig.savefig(zoom, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    core.RSLC_DIR = RSLC_DIR
    core.BUILDINGS_SHP = BUILDINGS_SHP
    core.DSM_TIF = DSM_TIF
    core.OUT_ROOT = RESULTS_DIR
    core.OLD_ALL_DIR = PROJECT_DIR / "_unused_old_results"
    core.ensure_dirs = ensure_a_geo_dirs
    core.choose_buildings = choose_local_buildings
    core.write_runner = write_local_runner
    core.write_readme = write_local_readme
    core.copy_all_building_reference = copy_no_old_reference
    core.make_gamma_geocoded = make_gamma_dsm_geocoded
    core.read_geotiff = _read_tif

    archive_previous_error_logs()
    if "--max-buildings" not in sys.argv:
        sys.argv.extend(["--max-buildings", "24"])
    if "--max-points-per-building" not in sys.argv:
        sys.argv.extend(["--max-points-per-building", "350"])
    core.main()
    make_polished_geocode_figures("20200708")
    prune_main_images()

    # Compact method comparison for quick paper-table use.
    stats_path = TABLE_DIR / "multi_scene_error_statistics.csv"
    rows = list(csv.DictReader(stats_path.open(encoding="utf-8")))
    summary = {
        "scenes": sorted({r["scene"] for r in rows}),
        "building_scene_records": len(rows),
        "scatter_points": sum(int(r["valid_points"]) for r in rows),
        "proposed_mean_boundary_distance_m": float(np.mean([float(r["method_mean_m"]) for r in rows])),
        "gamma_dem_mean_boundary_distance_m": float(np.mean([float(r["gamma_mean_m"]) for r in rows])),
        "proposed_p90_boundary_distance_m": float(np.mean([float(r["method_p90_m"]) for r in rows])),
        "gamma_dem_p90_boundary_distance_m": float(np.mean([float(r["gamma_p90_m"]) for r in rows])),
    }
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    (SUMMARY_DIR / "main_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"a_geo_tongji summary: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
