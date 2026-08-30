#!/usr/bin/env python3
"""Make a Fig. 009-style SAR scatter-point elevation map from adaptive ruler heights."""

from __future__ import annotations

import csv
import os
import pickle
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-siwei-adaptive-scatter")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.ndimage import binary_closing, binary_dilation, label
from skimage.draw import polygon as raster_polygon

from gamma_projection_core import BASE_WUSONG_M, INPUT, PICALL, WORK
from gamma_simulated_sar_registration import simulated_sar_registration
from project_volume_mesh_and_refine import BuildingMesh, compact_svg, display_image, interpolate_surface_elevation


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "svg.fonttype": "none",
    "axes.spines.right": False,
    "axes.spines.top": False,
})


def load_estimated_meshes() -> list[BuildingMesh]:
    # Mesh pickles were produced by a command-line script where the dataclass was
    # recorded as __main__.BuildingMesh; expose the compatible class for loading.
    setattr(sys.modules["__main__"], "BuildingMesh", BuildingMesh)
    cache = WORK / "gamma_adaptive_extended_ruler_meshes.pkl"
    with cache.open("rb") as handle:
        ruler_meshes: list[BuildingMesh] = pickle.load(handle)["meshes"]
    with (WORK / "adaptive_ruler_sar_feature_height_estimates.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    estimates = {
        int(row["fid"]): float(row["height_estimate_m"])
        for row in rows
        if row["height_estimate_m"] and np.isfinite(float(row["height_estimate_m"]))
    }
    result = []
    for ruler in ruler_meshes:
        if ruler.fid not in estimates:
            continue
        height = estimates[ruler.fid]
        n = len(ruler.top_indices)
        bottom = ruler.xy_initial[:n]
        top = bottom + (height / ruler.height) * (ruler.xy_initial[n:] - bottom)
        result.append(replace(ruler, height=height, xy_initial=np.vstack([bottom, top])))
    return result


def assign_los_owner(meshes: list[BuildingMesh], shift: np.ndarray, shape: tuple[int, int]):
    owner = np.full(shape, -1, dtype=np.int16)
    bounds: dict[int, tuple[int, int, int, int]] = {}
    for mesh_index in sorted(range(len(meshes)), key=lambda index: meshes[index].near_col + shift[0]):
        xy = meshes[mesh_index].xy(shift)
        r0 = max(0, int(np.floor(np.min(xy[:, 1]))) - 2)
        r1 = min(shape[0], int(np.ceil(np.max(xy[:, 1]))) + 3)
        c0 = max(0, int(np.floor(np.min(xy[:, 0]))) - 2)
        c1 = min(shape[1], int(np.ceil(np.max(xy[:, 0]))) + 3)
        bounds[mesh_index] = (r0, r1, c0, c1)
        for vertices in meshes[mesh_index].triangles:
            projected = xy[vertices]
            p0, p1, p2 = projected
            denominator = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
            if abs(denominator) < 1e-10:
                continue
            rr, cc = raster_polygon(projected[:, 1], projected[:, 0], shape=shape)
            if len(rr):
                free = owner[rr, cc] < 0
                owner[rr[free], cc[free]] = mesh_index
    return owner, bounds


def refine_owned_scatter(amplitude: np.ndarray, owner: np.ndarray, bounds: dict, count: int) -> np.ndarray:
    refined = np.zeros(owner.shape, dtype=bool)
    for mesh_index in range(count):
        r0, r1, c0, c1 = bounds[mesh_index]
        geometry = owner[r0:r1, c0:c1] == mesh_index
        if not np.any(geometry):
            continue
        window = binary_dilation(geometry, iterations=3)
        values = amplitude[r0:r1, c0:c1][window]
        threshold = float(values.mean() + 0.20 * values.std())
        candidate = geometry & (amplitude[r0:r1, c0:c1] > threshold)
        candidate = binary_closing(candidate, iterations=1)
        components, component_count = label(candidate)
        if component_count:
            sizes = np.bincount(components.ravel())
            keep = np.flatnonzero(sizes >= 4)
            keep = keep[keep != 0]
            candidate = np.isin(components, keep)
        refined[r0:r1, c0:c1] |= candidate
    return refined


def save_figure(path: Path, amplitude: np.ndarray, elevation: np.ndarray, counts: dict[str, int]) -> None:
    rows, cols = amplitude.shape
    fig, ax = plt.subplots(figsize=(12.0, 8.5))
    ax.imshow(
        display_image(amplitude[::2, ::2]), cmap="gray", vmin=0, vmax=1,
        interpolation="none", resample=False, extent=(0, cols, rows, 0),
    )
    rr, cc = np.where(np.isfinite(elevation))
    values = elevation[rr, cc]
    points = ax.scatter(
        cc, rr, c=values, s=0.10, marker="s", linewidths=0, cmap="turbo",
        vmin=BASE_WUSONG_M, vmax=float(np.nanpercentile(values, 99.5)),
        alpha=0.90, rasterized=True,
    )
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_title("SAR建筑散射点自适应量尺估计高程标记", loc="left", fontsize=17, fontweight="bold", pad=13)
    ax.text(
        0.01, 0.985,
        "图015估计高度截短GAMMA投影体；LOS唯一建筑归属 + SAR强度精炼 + 三角面重心高程插值",
        transform=ax.transAxes, va="top", color="white", fontsize=10,
        bbox={"facecolor": "black", "alpha": 0.72, "edgecolor": "none", "pad": 5},
    )
    ax.text(
        0.99, 0.985,
        f"有效散射点 {len(values):,}｜墙面 {counts['wall_pixels']:,}｜屋顶 {counts['roof_pixels']:,}｜底面 {counts['bottom_pixels']:,}",
        transform=ax.transAxes, ha="right", va="top", color="white", fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.72, "edgecolor": "none", "pad": 5},
    )
    colorbar = fig.colorbar(points, ax=ax, fraction=0.028, pad=0.018)
    colorbar.set_label("建筑散射点绝对高程 / m（统一4 m底面）")
    ax.set_xlabel("距离向列号 / pixel")
    ax.set_ylabel("方位向行号 / pixel")
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    compact_svg(path)


def main() -> None:
    with rasterio.open(INPUT / "amplitude_crop.tif") as source:
        amplitude = source.read(1).astype(np.float32)
    registration, _ = simulated_sar_registration(amplitude)
    shift = np.asarray([registration["col_shift_px"], registration["row_shift_px"]], dtype=np.float64)
    meshes = load_estimated_meshes()
    owner, bounds = assign_los_owner(meshes, shift, amplitude.shape)
    refined = refine_owned_scatter(amplitude, owner, bounds, len(meshes))
    elevation, counts = interpolate_surface_elevation(
        meshes, shift, owner, refined,
        overlap_rule="visible_surface_priority",
        elevation_mode="linear_vertex_height",
        allow_boundary_missing=True,
    )
    output = PICALL / "016_图件_510790231558.svg"
    save_figure(output, amplitude, elevation, counts)
    values = elevation[np.isfinite(elevation)]
    print({
        "output": str(output), "buildings": len(meshes), "scatter_pixels": int(len(values)),
        "elevation_min_m": float(values.min()), "elevation_median_m": float(np.median(values)),
        "elevation_max_m": float(values.max()), "surface_counts": counts,
    })


if __name__ == "__main__":
    main()
