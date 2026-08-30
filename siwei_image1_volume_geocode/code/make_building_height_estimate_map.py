#!/usr/bin/env python3
"""Aggregate Fig. 009/010 pixel elevations into building heights and make Fig. 012."""

from __future__ import annotations

import base64
import csv
import io
import os
import pickle
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-siwei-building-height-map")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import Normalize
from PIL import Image, ImageFilter

from gamma_projection_core import BASE_WUSONG_M, BUILDINGS, INPUT, PICALL, WORK
from gamma_simulated_sar_registration import simulated_sar_registration
from project_volume_mesh_and_refine import (
    BuildingMesh,
    assigned_initial_mask,
    build_ecef_vertices,
    coordinated_building_elevation,
    interpolate_surface_elevation,
    refine_mask,
)


GEOCODE_WORK = WORK / "gamma_geocoded_sar"
SIM_WORK = WORK / "gamma_simulated_sar_ellipsoid"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "svg.fonttype": "none",
})


def par_value(path: Path, key: str) -> float:
    match = re.search(rf"^{re.escape(key)}:\s+([^\s]+)", path.read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise ValueError(f"GAMMA参数缺失: {key}")
    return float(match.group(1))


def compact_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'data:image/png;base64,\s*([^\"]+)', text)
    if not match:
        return
    image = Image.open(io.BytesIO(base64.b64decode(match.group(1)))).convert("L")
    if image.width > 3200:
        image = image.resize((3200, round(image.height * 3200 / image.width)), Image.Resampling.LANCZOS)
        image = image.filter(ImageFilter.UnsharpMask(radius=0.8, percent=125, threshold=2))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90, subsampling=0, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    path.write_text(text[:match.start()] + "data:image/jpeg;base64,\n" + encoded + text[match.end():], encoding="utf-8")


def upper_height(values: np.ndarray) -> float:
    """Robust top envelope relative to the common 4 m base."""
    valid = values[np.isfinite(values)]
    if not len(valid):
        return float("nan")
    # The upper 0.2% suppresses isolated triangle-edge samples while retaining roof/top support.
    return max(0.0, float(np.percentile(valid, 99.8) - BASE_WUSONG_M))


def main() -> None:
    import __main__
    __main__.BuildingMesh = BuildingMesh
    with (WORK / "gamma_building_volume_meshes.pkl").open("rb") as handle:
        meshes = pickle.load(handle)["meshes"]
    with rasterio.open(INPUT / "amplitude_crop.tif") as source:
        amplitude = source.read(1).astype(np.float32)
    registration, _ = simulated_sar_registration(amplitude)
    shift = np.asarray([registration["col_shift_px"], registration["row_shift_px"]], dtype=np.float64)
    initial, owner = assigned_initial_mask(meshes, shift, amplitude.shape)
    refined = refine_mask(amplitude, owner, len(meshes), kappa=0.20)
    vertices_ecef = build_ecef_vertices(meshes)
    figure09_elevation, _ = interpolate_surface_elevation(
        meshes, shift, owner, refined, overlap_rule="visible_surface_priority",
        vertices_ecef_by_fid=vertices_ecef, elevation_mode="ecef_barycentric_to_wgs84",
    )
    figure10_elevation, _ = coordinated_building_elevation(meshes, shift, owner, initial)

    records = []
    for mesh_index, mesh in enumerate(meshes):
        mask09 = (owner == mesh_index) & np.isfinite(figure09_elevation)
        mask10 = (owner == mesh_index) & np.isfinite(figure10_elevation)
        height09 = upper_height(figure09_elevation[mask09])
        height10 = upper_height(figure10_elevation[mask10])
        available = np.asarray([height09, height10], dtype=np.float64)
        fused = float(np.nanmedian(available)) if np.any(np.isfinite(available)) else float("nan")
        difference = abs(height09 - height10) if np.isfinite(height09) and np.isfinite(height10) else float("nan")
        records.append({
            "fid": mesh.fid,
            "clean_id": mesh.clean_id,
            "figure09_height_m": height09,
            "figure10_height_m": height10,
            "fused_height_m": fused,
            "figure09_pixels": int(mask09.sum()),
            "figure10_pixels": int(mask10.sum()),
            "method_difference_m": difference,
        })

    table_path = WORK / "building_height_estimates_from_fig09_10.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    estimates = {int(item["fid"]): item for item in records}

    dem_par = SIM_WORK / "dem_seg.par"
    width = int(par_value(dem_par, "width"))
    lines = int(par_value(dem_par, "nlines"))
    east = par_value(dem_par, "corner_east")
    north = par_value(dem_par, "corner_north")
    post_east = par_value(dem_par, "post_east")
    post_north = par_value(dem_par, "post_north")
    left = east - 0.5 * post_east
    right = left + width * post_east
    top = north - 0.5 * post_north
    bottom = top + lines * post_north
    geocoded = np.fromfile(GEOCODE_WORK / "amplitude_map.gamma", dtype=">f4").reshape(lines, width).astype(np.float32)
    geocoded[geocoded <= 0] = np.nan
    finite = geocoded[np.isfinite(geocoded)]
    low, high = np.percentile(finite, [2.0, 99.7])
    display = np.clip((geocoded - low) / max(float(high - low), 1e-6), 0, 1) ** 0.55

    buildings = gpd.read_file(BUILDINGS, engine="pyogrio").to_crs(32651)
    buildings["height_estimate_m"] = [estimates.get(int(fid), {}).get("fused_height_m", np.nan) for fid in buildings.index]
    buildings["method_difference_m"] = [estimates.get(int(fid), {}).get("method_difference_m", np.nan) for fid in buildings.index]
    buildings = buildings.cx[min(left, right):max(left, right), min(bottom, top):max(bottom, top)]
    valid_buildings = buildings[np.isfinite(buildings.height_estimate_m)].copy()
    missing_buildings = buildings[~np.isfinite(buildings.height_estimate_m)].copy()
    values = valid_buildings.height_estimate_m.to_numpy()
    vmax = float(np.percentile(values, 99.0))
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("turbo")

    fig, ax = plt.subplots(figsize=(11.2, 10.0))
    ax.imshow(display, cmap="gray", vmin=0, vmax=1, interpolation="none", resample=False,
              extent=(left, right, bottom, top))
    if len(missing_buildings):
        missing_buildings.plot(ax=ax, facecolor="#6B7280", edgecolor="#D1D5DB", linewidth=0.25, alpha=0.42, zorder=2)
    valid_buildings.plot(
        ax=ax, column="height_estimate_m", cmap=cmap, norm=norm, edgecolor="#E5FFFF",
        linewidth=0.24, alpha=0.68, zorder=3,
    )
    # SVG retains vector text, so all estimates remain readable when zoomed even if dense at page scale.
    for fid, feature in valid_buildings.iterrows():
        point = feature.geometry.representative_point()
        ax.text(
            point.x, point.y, f"{feature.height_estimate_m:.0f}", ha="center", va="center",
            fontsize=2.15, color="white", fontweight="bold", zorder=4,
            path_effects=[],
        )
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=ax, fraction=0.030, pad=0.018)
    colorbar.set_label("图009/010融合建筑高度估计 / m")
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.set_title("GAMMA地理编码SAR上的建筑高度估计", loc="left", fontsize=17, fontweight="bold", pad=13)
    ax.text(
        0.01, 0.985,
        "图009精炼散射点与图010完整投影体高程上包络融合；数字为建筑高度/m，灰色为无有效投影结果",
        transform=ax.transAxes, va="top", color="white", fontsize=9.5,
        bbox={"facecolor": "black", "alpha": 0.74, "edgecolor": "none", "pad": 5},
    )
    differences = np.asarray([item["method_difference_m"] for item in records], dtype=np.float64)
    ax.text(
        0.99, 0.985,
        f"有效 {len(valid_buildings):,}/{len(buildings):,}栋｜两方法差异中位数 {np.nanmedian(differences):.2f} m",
        transform=ax.transAxes, ha="right", va="top", color="white", fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.74, "edgecolor": "none", "pad": 5},
    )
    ax.set_xlabel("UTM东向坐标 / m")
    ax.set_ylabel("UTM北向坐标 / m")
    fig.tight_layout()
    output = PICALL / "012_图009与图010融合的建筑高度估计.svg"
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)
    compact_svg(output)
    print({
        "output": str(output), "table": str(table_path), "map_buildings": len(buildings),
        "estimated_buildings": len(valid_buildings), "missing_buildings": len(missing_buildings),
        "height_min_m": float(np.min(values)), "height_median_m": float(np.median(values)),
        "height_max_m": float(np.max(values)), "method_difference_median_m": float(np.nanmedian(differences)),
        "method_difference_p90_m": float(np.nanpercentile(differences, 90)),
    })


if __name__ == "__main__":
    main()
