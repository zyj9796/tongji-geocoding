from __future__ import annotations

import csv
import json
import math
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from io_paths import (
    FULL_AREA_DIR,
    FULL_AREA_GEOJSON_DIR,
    FULL_AREA_IMAGE_DIR,
    PPT_DIR,
    PPT_DOC_DIR as DOC_DIR,
    PPT_IMAGE_DIR as FIG_DIR,
    PPT_ZIP as ZIP_PATH,
    RESULTS_DIR,
    TRASH_DIR,
)

POINTS_CSV = FULL_AREA_DIR / "20200708_all_buildings_method_vs_gamma_points.csv"
STATS_CSV = FULL_AREA_DIR / "20200708_all_buildings_fig5_4_like_stats.csv"
BUILDINGS_GEOJSON = FULL_AREA_GEOJSON_DIR / "20200708_all_valid_geocoded_buildings.geojson"

EARTH_R = 6378137.0


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_buildings(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    buildings = {}
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        fid = int(props["fid"])
        ring = np.asarray(feat["geometry"]["coordinates"][0], dtype=np.float64)
        if ring.shape[0] > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        buildings[fid] = {"fid": fid, "props": props, "ring": ring[:, :2]}
    return buildings


def arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = [r for r in rows if int(float(r.get("gamma_dsm_ok", 0))) == 1]
    method = np.asarray([[float(r["method_lon"]), float(r["method_lat"])] for r in valid], dtype=np.float64)
    gamma = np.asarray([[float(r["gamma_dsm_lon"]), float(r["gamma_dsm_lat"])] for r in valid], dtype=np.float64)
    fids = np.asarray([int(r["fid"]) for r in valid], dtype=np.int64)
    return method, gamma, fids


def lonlat_delta_m(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat0 = np.deg2rad((src[:, 1] + dst[:, 1]) / 2.0)
    dx = (dst[:, 0] - src[:, 0]) * math.pi / 180.0 * EARTH_R * np.cos(lat0)
    dy = (dst[:, 1] - src[:, 1]) * math.pi / 180.0 * EARTH_R
    return dx, dy, np.hypot(dx, dy)


def extent_from_points(*pts: np.ndarray) -> tuple[float, float, float, float]:
    all_xy = np.vstack(pts)
    padx = np.ptp(all_xy[:, 0]) * 0.035
    pady = np.ptp(all_xy[:, 1]) * 0.035
    return (
        float(np.min(all_xy[:, 0]) - padx),
        float(np.max(all_xy[:, 0]) + padx),
        float(np.min(all_xy[:, 1]) - pady),
        float(np.max(all_xy[:, 1]) + pady),
    )


def set_map_axis(ax, extent: tuple[float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect(1.0 / max(math.cos(math.radians((ymin + ymax) / 2.0)), 0.1), adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(color="#9ca3af", linewidth=0.25, alpha=0.25)
    ax.tick_params(labelsize=8)


def building_patches(buildings: dict[int, dict], fids: set[int] | None = None) -> list[Polygon]:
    patches = []
    for fid, b in buildings.items():
        if fids is not None and fid not in fids:
            continue
        patches.append(Polygon(b["ring"], closed=True))
    return patches


def add_buildings(ax, buildings: dict[int, dict], fids: set[int] | None = None, alpha: float = 0.38) -> None:
    patches = building_patches(buildings, fids)
    coll = PatchCollection(patches, facecolor="#e5e7eb", edgecolor="#6b7280", linewidth=0.12, alpha=alpha)
    ax.add_collection(coll)


def sample_idx(n: int, target: int) -> np.ndarray:
    if n <= target:
        return np.arange(n)
    return np.linspace(0, n - 1, target, dtype=np.int64)


def local_xy(lonlat: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    x = (lonlat[:, 0] - lon0) * math.pi / 180.0 * EARTH_R * math.cos(math.radians(lat0))
    y = (lonlat[:, 1] - lat0) * math.pi / 180.0 * EARTH_R
    return np.column_stack([x, y])


def fig_vector_field(method: np.ndarray, gamma: np.ndarray, buildings: dict[int, dict]) -> Path:
    dx, dy, dist = lonlat_delta_m(gamma, method)
    lon = gamma[:, 0]
    lat = gamma[:, 1]
    bins_x = np.linspace(np.min(lon), np.max(lon), 26)
    bins_y = np.linspace(np.min(lat), np.max(lat), 24)
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    ix = np.clip(np.digitize(lon, bins_x) - 1, 0, len(bins_x) - 2)
    iy = np.clip(np.digitize(lat, bins_y) - 1, 0, len(bins_y) - 2)
    for i, key in enumerate(zip(ix, iy)):
        cells[key].append(i)

    centers = []
    u = []
    v = []
    c = []
    for ids in cells.values():
        if len(ids) < 20:
            continue
        arr = np.asarray(ids, dtype=np.int64)
        centers.append([float(np.mean(lon[arr])), float(np.mean(lat[arr]))])
        u.append(float(np.median(dx[arr])))
        v.append(float(np.median(dy[arr])))
        c.append(float(np.median(dist[arr])))
    centers = np.asarray(centers, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.8, 8.0), dpi=300)
    add_buildings(ax, buildings)
    q = ax.quiver(centers[:, 0], centers[:, 1], u, v, c, angles="xy", scale_units="xy", scale=115000,
                  cmap="viridis", width=0.0038, headwidth=3.7, alpha=0.92)
    set_map_axis(ax, extent_from_points(method, gamma))
    cbar = fig.colorbar(q, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("Median pair distance / m")
    ax.set_title("Full-Area Displacement Vector Field: GAMMA/DSM to Building-Constrained", fontsize=13, weight="bold")
    ax.text(0.02, 0.02, "Arrow direction: traditional GAMMA/DSM point -> building-constrained point",
            transform=ax.transAxes, fontsize=8, bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#9ca3af", "alpha": 0.9})
    fig.tight_layout()
    out = FIG_DIR / "fig_10_full_area_displacement_vector_field.png"
    fig.savefig(out)
    plt.close(fig)
    dst = FULL_AREA_IMAGE_DIR / out.name
    if out.resolve() != dst.resolve():
        shutil.copy2(out, dst)
    return out


def fig_error_heatmap(method: np.ndarray, gamma: np.ndarray, buildings: dict[int, dict]) -> Path:
    _dx, _dy, dist = lonlat_delta_m(method, gamma)
    fig, ax = plt.subplots(figsize=(9.8, 8.0), dpi=300)
    add_buildings(ax, buildings, alpha=0.25)
    hb = ax.hexbin(method[:, 0], method[:, 1], C=dist, reduce_C_function=np.median, gridsize=80, mincnt=3, cmap="magma")
    set_map_axis(ax, extent_from_points(method, gamma))
    cbar = fig.colorbar(hb, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("Median pair distance / m")
    ax.set_title("Full-Area Planar Difference Heatmap", fontsize=13, weight="bold")
    ax.text(0.02, 0.02, "Hexbin color shows median distance between two results for the same SAR sampled pixels.",
            transform=ax.transAxes, fontsize=8, bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#9ca3af", "alpha": 0.9})
    fig.tight_layout()
    out = FIG_DIR / "fig_11_full_area_planar_difference_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    dst = FULL_AREA_IMAGE_DIR / out.name
    if out.resolve() != dst.resolve():
        shutil.copy2(out, dst)
    return out


def top_fids(stats: list[dict], n: int = 6) -> list[int]:
    rows = sorted(stats, key=lambda r: float(r["gamma_dsm_mean_boundary_distance_m"]), reverse=True)
    return [int(r["fid"]) for r in rows[:n]]


def fig_zoom_cases(method: np.ndarray, gamma: np.ndarray, fids: np.ndarray, buildings: dict[int, dict], stats: list[dict]) -> Path:
    chosen = top_fids(stats, 6)
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2), dpi=300, squeeze=False)
    for ax, fid in zip(axes.ravel(), chosen):
        ids = np.where(fids == fid)[0]
        if ids.size == 0:
            ax.axis("off")
            continue
        ring = buildings[fid]["ring"]
        lon0 = float(np.mean(ring[:, 0]))
        lat0 = float(np.mean(ring[:, 1]))
        ring_xy = local_xy(ring, lon0, lat0)
        method_xy = local_xy(method[ids], lon0, lat0)
        gamma_xy = local_xy(gamma[ids], lon0, lat0)
        all_xy = np.vstack([ring_xy, method_xy, gamma_xy])
        padx = max(np.ptp(all_xy[:, 0]) * 0.28, 10.0)
        pady = max(np.ptp(all_xy[:, 1]) * 0.28, 10.0)
        extent = (float(np.min(all_xy[:, 0]) - padx), float(np.max(all_xy[:, 0]) + padx),
                  float(np.min(all_xy[:, 1]) - pady), float(np.max(all_xy[:, 1]) + pady))
        ax.add_patch(Polygon(ring_xy, closed=True, facecolor="#e5e7eb", edgecolor="#111827", linewidth=1.0, alpha=0.50))
        sids = ids[sample_idx(ids.size, 120)]
        local_index = {idx: pos for pos, idx in enumerate(ids)}
        for j in sids[:: max(1, len(sids) // 42)]:
            k = local_index[int(j)]
            ax.plot([gamma_xy[k, 0], method_xy[k, 0]], [gamma_xy[k, 1], method_xy[k, 1]],
                    color="#6b7280", linewidth=0.55, alpha=0.45)
        sk = np.asarray([local_index[int(j)] for j in sids], dtype=np.int64)
        ax.scatter(gamma_xy[sk, 0], gamma_xy[sk, 1], s=13, color="#f97316", alpha=0.62, linewidths=0, label="GAMMA/DSM")
        ax.scatter(method_xy[sk, 0], method_xy[sk, 1], s=13, color="#2563eb", alpha=0.80, linewidths=0, label="Method")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("East offset / m", fontsize=8)
        ax.set_ylabel("North offset / m", fontsize=8)
        ax.grid(color="#9ca3af", linewidth=0.3, alpha=0.28)
        ax.tick_params(labelsize=7)
        stat = next((r for r in stats if int(r["fid"]) == fid), None)
        title = f"FID {fid}"
        if stat:
            title += f" | GAMMA mean {float(stat['gamma_dsm_mean_boundary_distance_m']):.1f} m"
        ax.set_title(title, fontsize=9, weight="bold")
    axes.ravel()[0].legend(loc="best", fontsize=7, markerscale=1.5)
    fig.suptitle("Typical Buildings With Large GAMMA/DSM Boundary Error", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIG_DIR / "图件_982942697266.png"
    fig.savefig(out)
    plt.close(fig)
    dst = FULL_AREA_IMAGE_DIR / out.name
    if out.resolve() != dst.resolve():
        shutil.copy2(out, dst)
    return out


def fig_ranked_improvement(stats: list[dict]) -> Path:
    rows = sorted(stats, key=lambda r: float(r["gamma_dsm_mean_boundary_distance_m"]), reverse=True)[:30]
    fids = [r["fid"] for r in rows]
    gamma = np.asarray([float(r["gamma_dsm_mean_boundary_distance_m"]) for r in rows])
    method = np.asarray([float(r["method_mean_boundary_distance_m"]) for r in rows])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10.6, 9.2), dpi=300)
    for yi, before, after in zip(y, gamma, method):
        ax.plot([max(after, 0.001), before], [yi, yi], color="#9ca3af", linewidth=0.85, alpha=0.62, zorder=1)
        ax.annotate(
            "",
            xy=(max(after, 0.001), yi),
            xytext=(before, yi),
            arrowprops={"arrowstyle": "->", "color": "#6b7280", "lw": 0.65, "alpha": 0.6},
            zorder=1,
        )
    ax.scatter(
        gamma,
        y,
        s=38,
        color="#f97316",
        edgecolor="white",
        linewidth=0.35,
        label="Traditional GAMMA/DSM: distance before building constraint",
        zorder=3,
    )
    ax.scatter(
        np.maximum(method, 0.001),
        y,
        s=42,
        color="#2563eb",
        marker="D",
        edgecolor="white",
        linewidth=0.35,
        label="Building-constrained method: distance after constraint",
        zorder=4,
    )
    ax.axvline(float(np.mean(gamma)), color="#f97316", linestyle="--", linewidth=1.0, alpha=0.72)
    ax.axvline(float(np.mean(method)), color="#2563eb", linestyle="--", linewidth=1.0, alpha=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(fids, fontsize=7)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Mean distance to building footprint boundary / m (log scale; lower is better)")
    ax.set_ylabel("Building FID")
    ax.set_title("Boundary-Distance Improvement for the 30 Hardest GAMMA/DSM Buildings", fontsize=13, weight="bold")
    ax.text(
        0.02,
        0.985,
        f"Mean over these 30 buildings: GAMMA/DSM {np.mean(gamma):.2f} m -> building-constrained {np.mean(method):.3f} m",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        bbox={"boxstyle": "round,pad=0.32", "fc": "white", "ec": "#9ca3af", "alpha": 0.94},
    )
    ax.legend(loc="lower right", fontsize=8, frameon=True, title="Point meaning")
    ax.grid(axis="x", which="both", alpha=0.25, linewidth=0.35)
    fig.tight_layout()
    out = FIG_DIR / "fig_13_ranked_boundary_error_improvement.png"
    fig.savefig(out)
    plt.close(fig)
    dst = FULL_AREA_IMAGE_DIR / out.name
    if out.resolve() != dst.resolve():
        shutil.copy2(out, dst)
    return out


def fig_height_error_relation(stats: list[dict]) -> Path:
    height = np.asarray([float(r["height_m"]) for r in stats], dtype=np.float64)
    gamma = np.asarray([float(r["gamma_dsm_mean_boundary_distance_m"]) for r in stats], dtype=np.float64)
    method = np.asarray([float(r["method_mean_boundary_distance_m"]) for r in stats], dtype=np.float64)
    points = np.asarray([float(r["sample_points"]) for r in stats], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2), dpi=300, sharex=True)
    size = 10 + 40 * np.sqrt(points / max(np.max(points), 1.0))
    size_levels = [25, 50, 100]

    axes[0].scatter(
        height,
        gamma,
        s=size,
        color="#f97316",
        alpha=0.55,
        edgecolor="white",
        linewidth=0.25,
        label="Traditional GAMMA/DSM boundary distance",
    )
    axes[0].axhline(float(np.mean(gamma)), color="#9a3412", linestyle="--", linewidth=1.0, label=f"GAMMA/DSM mean = {np.mean(gamma):.2f} m")
    axes[0].set_title("Traditional GAMMA/DSM\n(boundary distance before building constraint)", fontsize=11, weight="bold", pad=8)
    axes[0].set_ylabel("Mean distance to building footprint boundary / m")

    axes[1].scatter(
        height,
        method,
        s=size,
        color="#2563eb",
        alpha=0.58,
        edgecolor="white",
        linewidth=0.25,
        label="Building-constrained boundary distance",
    )
    axes[1].axhline(float(np.mean(method)), color="#1d4ed8", linestyle="--", linewidth=1.0, label=f"Building-constrained mean = {np.mean(method):.2f} m")
    axes[1].set_title("Building-Constrained Method\n(boundary distance after constraint; zoomed y-axis)", fontsize=11, weight="bold", pad=8)
    axes[1].set_ylabel("Mean distance to building footprint boundary / m")
    axes[1].set_ylim(-0.05, max(2.2, float(np.percentile(method, 98)) * 1.25))

    for ax in axes:
        ax.set_xlabel("Building height / m")
        ax.grid(alpha=0.25, linewidth=0.35)
        ax.legend(loc="upper left", fontsize=8, frameon=True)

    handles = []
    labels = []
    for level in size_levels:
        handles.append(axes[1].scatter([], [], s=10 + 40 * np.sqrt(level / max(np.max(points), 1.0)), color="#6b7280", alpha=0.45))
        labels.append(f"{level} SAR samples")
    method_legend = axes[1].legend(loc="upper right", fontsize=8, frameon=True, title="Method line/points")
    axes[1].add_artist(method_legend)
    axes[1].legend(handles=handles, labels=labels, loc="lower right", fontsize=8, frameon=True, title="Point size")

    fig.suptitle("Building Height Versus Boundary Distance", fontsize=13.5, weight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = FIG_DIR / "fig_14_height_vs_boundary_error.png"
    fig.savefig(out)
    plt.close(fig)
    dst = FULL_AREA_IMAGE_DIR / out.name
    if out.resolve() != dst.resolve():
        shutil.copy2(out, dst)
    return out


def fig_cdf(stats: list[dict]) -> Path:
    gamma = np.sort(np.asarray([float(r["gamma_dsm_mean_boundary_distance_m"]) for r in stats], dtype=np.float64))
    method = np.sort(np.asarray([float(r["method_mean_boundary_distance_m"]) for r in stats], dtype=np.float64))
    y_g = np.arange(1, gamma.size + 1) / gamma.size
    y_m = np.arange(1, method.size + 1) / method.size
    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=300)
    ax.plot(gamma, y_g, color="#f97316", linewidth=2.0, label="GAMMA/DSM")
    ax.plot(method, y_m, color="#2563eb", linewidth=2.0, label="Building-constrained")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("Mean boundary distance / m")
    ax.set_ylabel("Cumulative proportion of buildings")
    ax.set_title("Cumulative Distribution of Building-Level Boundary Error", fontsize=13, weight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25, linewidth=0.35)
    fig.tight_layout()
    out = FIG_DIR / "fig_15_boundary_error_cdf.png"
    fig.savefig(out)
    plt.close(fig)
    dst = FULL_AREA_IMAGE_DIR / out.name
    if out.resolve() != dst.resolve():
        shutil.copy2(out, dst)
    return out


def write_doc(extra_figs: list[Path], stats: list[dict], method: np.ndarray, gamma: np.ndarray) -> Path:
    _dx, _dy, dist = lonlat_delta_m(method, gamma)
    method_mean = np.mean([float(r["method_mean_boundary_distance_m"]) for r in stats])
    gamma_mean = np.mean([float(r["gamma_dsm_mean_boundary_distance_m"]) for r in stats])
    doc = DOC_DIR / "图件说明.md"
    lines = [
        "# PPT 图件详细说明",
        "",
        "## 总体口径",
        "",
        "本图件包分为两组：第一组是 PS-InSAR 同像素典型建筑对比，第二组是 Tongji 全区域建筑物约束地理编码与传统 GAMMA/DSM 地理编码的平面对比。全区域图使用同一批 SAR 采样像素，分别给出本文建筑物约束方法坐标和传统 GAMMA/DSM 坐标。",
        "",
        "当前地理坐标 SAR 底图由 `dem_import -> multi_look -> gc_map2 -> geocode_back -> data2geotiff` 使用裁剪 DSM 正式生成，GDAL 仅执行 EPSG:4326 重投影；不使用零高程 GCP/TPS 或建筑控制点多项式变形。显示采用强度开平方后的幅度域 2%-98% 线性拉伸，NoData 保持黑色。",
        "",
        "## 全区域关键统计",
        "",
        f"- 有效建筑物：{len(stats)} 栋。",
        f"- 全区域点对：{method.shape[0]:,} 对。",
        f"- 同像素点对两方法平面距离中位数：{np.median(dist):.2f} m。",
        f"- 同像素点对两方法平面距离 P90：{np.percentile(dist, 90):.2f} m。",
        f"- 建筑物约束方法平均轮廓边界距离：{method_mean:.2f} m。",
        f"- 传统 GAMMA/DSM 平均轮廓边界距离：{gamma_mean:.2f} m。",
        "",
        "## 推荐 PPT 使用顺序",
        "",
        "1. `20200708_图件_1045917436903_版本2.png`：全区域三联平面对比，先建立整体视觉结论。",
        "2. `fig_10_full_area_displacement_vector_field.png`：解释传统结果到建筑约束结果的空间偏移方向。",
        "3. `fig_11_full_area_planar_difference_heatmap.png`：展示差异较大的空间热点。",
        "4. `图件_982942697266.png`：用局部建筑案例解释点云如何从 GAMMA/DSM 结果回到建筑轮廓附近。",
        "5. `fig_13_ranked_boundary_error_improvement.png`、`fig_15_boundary_error_cdf.png`：用统计图支撑定量结论。",
        "6. `fig_14_height_vs_boundary_error.png`：讨论建筑高度和传统地理编码误差之间的关系。",
        "",
        "## 新增图件说明",
        "",
        "- `fig_10_full_area_displacement_vector_field.png`：全区域位移矢量场。箭头从传统 GAMMA/DSM 点指向建筑物约束点，颜色表示该网格内两方法点对距离的中位数。适合说明偏移方向具有明显空间一致性。",
        "- `fig_11_full_area_planar_difference_heatmap.png`：全区域平面差异热力图。以建筑物约束点所在位置为基准，颜色表示同一 SAR 像素下两方法平面距离的局部中位数。适合寻找误差热点。",
        "- `图件_982942697266.png`：选取传统 GAMMA/DSM 轮廓边界误差较大的 6 栋建筑做局部放大。灰色为建筑轮廓，橙色为 GAMMA/DSM 点，蓝色为建筑物约束点，连线表示同一 SAR 像素的两种反算结果。",
        "- `fig_13_ranked_boundary_error_improvement.png`：按传统 GAMMA/DSM 平均轮廓边界误差排序的前 30 栋建筑。蓝色柱表示建筑物约束方法，橙色柱表示传统方法。适合直接展示误差改善幅度。",
        "- `fig_14_height_vs_boundary_error.png`：建筑高度与平均轮廓边界误差关系图。橙色表示传统 GAMMA/DSM，蓝色表示建筑物约束方法，点大小/颜色反映每栋建筑的采样点数。",
        "- `fig_15_boundary_error_cdf.png`：建筑级平均轮廓边界误差累计分布。曲线越靠左说明整体误差越小，适合概括全区域统计优势。",
        "",
        "## 已有图件说明",
        "",
        "- `图_01_雷达总览与选定建筑像素.png`：SAR 幅度底图上的典型建筑投影三角网和同像素 PS 点。",
        "- `图_02_逐建筑雷达裁剪图.png`：典型建筑 SAR 局部放大图。",
        "- `fig_03_sar_pixel_density.png`：典型建筑同像素 PS 点密度图。",
        "- `图_04_同像素地理定位对比.png`：典型建筑地理坐标下 PS/GAMMA 与建筑约束点对比。",
        "- `fig_05_difference_histograms.png`：典型建筑同像素水平差异和高程差异直方图。",
        "- `fig_06_per_building_horizontal_boxplot.png`：典型建筑水平差异箱线图。",
        "- `fig_07_same_pixel_diagnostics.png`：典型建筑同像素诊断散点图。",
        "- `fig_08_building_constrained_height_profiles.png`：典型建筑建筑约束相对高度分布。",
        "- `fig_09_selected_building_statistics_table.png`：典型建筑统计表。",
        "- `图_同像素永久散射体三维建筑.png`：典型建筑同像素三维对比。",
        "- `图_同像素永久散射体雷达像素.png`：典型建筑 SAR 行列坐标投影图。",
        "",
        "## 结论表述建议",
        "",
        "全区域平面对比表明，传统 GAMMA/DSM 方法在建筑群附近存在较明显的侧向偏移，而建筑物约束方法利用建筑轮廓、高度和三维投影关系，将同一批 SAR 散射像素约束回建筑物轮廓及其三维模型附近。统计结果中，建筑物约束方法的平均轮廓边界距离显著小于传统 GAMMA/DSM 方法；矢量场和局部放大图进一步说明这种改善不是单个样例现象，而是在全区域多个建筑群中具有一致性。",
        "",
    ]
    doc.write_text("\n".join(lines), encoding="utf-8")
    return doc


def update_zip() -> None:
    if not ZIP_PATH.exists():
        return
    tmp = ZIP_PATH.with_suffix(".tmp.zip")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root in [PPT_DIR, FIG_DIR, TRASH_DIR]:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(RESULTS_DIR))
    tmp.replace(ZIP_PATH)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FULL_AREA_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_csv(POINTS_CSV)
    stats = read_csv(STATS_CSV)
    buildings = load_buildings(BUILDINGS_GEOJSON)
    method, gamma, fids = arrays(rows)

    outs = [
        fig_vector_field(method, gamma, buildings),
        fig_error_heatmap(method, gamma, buildings),
        fig_zoom_cases(method, gamma, fids, buildings, stats),
        fig_ranked_improvement(stats),
        fig_height_error_relation(stats),
        fig_cdf(stats),
    ]
    doc = write_doc(outs, stats, method, gamma)
    update_zip()
    print("generated:")
    for out in outs:
        print(out)
    print(f"doc={doc}")
    print(f"zip={ZIP_PATH}")


if __name__ == "__main__":
    main()
