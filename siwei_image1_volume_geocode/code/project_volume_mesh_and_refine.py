#!/usr/bin/env python3
"""Paper-style building volume projection, registration and mask refinement."""

from __future__ import annotations

import base64
import io
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-siwei-volume-mesh")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.collections import LineCollection
from PIL import Image, ImageFilter
from pyproj import Transformer
from scipy.interpolate import CubicSpline
from scipy.ndimage import binary_closing, binary_dilation, label
from shapely.geometry import Polygon
from shapely.ops import triangulate
from skimage.draw import polygon as raster_polygon

from gamma_projection_core import (
    BASE_WUSONG_M,
    BUILDINGS,
    INPUT,
    PICALL,
    PROJECTION_DATUM_VERSION,
    WORK,
    ellipsoid_to_wusong,
    gamma_project,
    wusong_to_ellipsoid,
    write_gamma_par,
)
from gamma_simulated_sar_registration import simulated_sar_registration

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "svg.fonttype": "none",
    "axes.spines.right": False,
    "axes.spines.top": False,
})


@dataclass
class BuildingMesh:
    fid: int
    clean_id: int
    height: float
    xy_initial: np.ndarray
    triangles: np.ndarray
    surfaces: np.ndarray
    top_indices: np.ndarray
    near_col: float

    def xy(self, shift: np.ndarray) -> np.ndarray:
        return self.xy_initial + shift


def clean_ring(geometry) -> np.ndarray:
    if geometry.geom_type == "MultiPolygon":
        geometry = max(geometry.geoms, key=lambda item: item.area)
    ring = np.asarray(geometry.exterior.coords, dtype=np.float64)[:, :2]
    if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    keep = [0]
    for index in range(1, len(ring)):
        if not np.allclose(ring[index], ring[keep[-1]], atol=1e-12, rtol=0):
            keep.append(index)
    ring = ring[keep]
    if len(ring) < 3:
        raise ValueError("建筑轮廓少于3个独立顶点")
    return ring


def constrained_roof_triangles(ring: np.ndarray) -> np.ndarray:
    polygon = Polygon(ring).buffer(0)
    result = []
    for triangle in triangulate(polygon):
        if not polygon.covers(triangle.representative_point()):
            continue
        coords = np.asarray(triangle.exterior.coords, dtype=np.float64)[:3, :2]
        indices = []
        for point in coords:
            distance = np.linalg.norm(ring - point, axis=1)
            index = int(np.argmin(distance))
            if distance[index] > 1e-9:
                raise ValueError("三角剖分生成非轮廓顶点")
            indices.append(index)
        if len(set(indices)) == 3:
            result.append(tuple(indices))
    if not result:
        raise ValueError("屋顶三角剖分失败")
    return np.asarray(result, dtype=np.int32)


def make_mesh(fid: int, feature, par: Path, cache: dict) -> BuildingMesh:
    ring = clean_ring(feature.geometry)
    height = float(feature.height)
    n = len(ring)
    xy = []
    for z_wusong in (BASE_WUSONG_M, BASE_WUSONG_M + height):
        for lon, lat in ring:
            z = float(wusong_to_ellipsoid(z_wusong, lat, lon))
            key = (round(float(lon), 12), round(float(lat), 12), round(float(z), 6))
            if key not in cache:
                cache[key] = gamma_project(par, float(lat), float(lon), float(z))
            xy.append(cache[key])
    xy = np.asarray(xy, dtype=np.float64)
    roof_triangles = constrained_roof_triangles(ring)
    triangles: list[tuple[int, int, int]] = []
    surfaces: list[str] = []
    for index in range(n):
        following = (index + 1) % n
        triangles.extend([(index, following, n + following), (index, n + following, n + index)])
        surfaces.extend(["wall", "wall"])
    for a, b, c in roof_triangles:
        triangles.append((n + int(a), n + int(b), n + int(c)))
        surfaces.append("roof")
        triangles.append((int(c), int(b), int(a)))
        surfaces.append("bottom")
    return BuildingMesh(
        fid=fid,
        clean_id=int(feature.clean_id),
        height=height,
        xy_initial=xy,
        triangles=np.asarray(triangles, dtype=np.int32),
        surfaces=np.asarray(surfaces),
        top_indices=np.arange(n, 2 * n, dtype=np.int32),
        near_col=float(np.min(xy[:n, 0])),
    )


def display_image(amplitude: np.ndarray) -> np.ndarray:
    valid = amplitude[amplitude > 0]
    lo, hi = np.percentile(valid, [2.0, 99.7])
    return np.clip((amplitude - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.55


def compact_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"data:image/png;base64,\s*([^\"]+)", text)
    if not match:
        return
    image = Image.open(io.BytesIO(base64.b64decode(match.group(1)))).convert("L")
    if image.width > 3200:
        image = image.resize((3200, round(image.height * 3200 / image.width)), Image.Resampling.LANCZOS)
        image = image.filter(ImageFilter.UnsharpMask(radius=0.8, percent=125, threshold=2))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=88, subsampling=0, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    path.write_text(text[: match.start()] + "data:image/jpeg;base64,\n" + encoded + text[match.end() :], encoding="utf-8")


def base_axes(amplitude: np.ndarray, title: str, subtitle: str):
    rows, cols = amplitude.shape
    preview = amplitude[::2, ::2]
    fig, ax = plt.subplots(figsize=(12.0, 8.5))
    ax.imshow(display_image(preview), cmap="gray", vmin=0, vmax=1, interpolation="none", resample=False,
              extent=(0, cols, rows, 0))
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_title(title, loc="left", fontsize=17, fontweight="bold", pad=13)
    ax.text(0.01, 0.985, subtitle, transform=ax.transAxes, va="top", color="white", fontsize=10,
            bbox={"facecolor": "black", "alpha": 0.72, "edgecolor": "none", "pad": 5})
    ax.set_xlabel("距离向列号 / pixel")
    ax.set_ylabel("方位向行号 / pixel")
    return fig, ax


def save_mesh_svg(path: Path, amplitude: np.ndarray, meshes: list[BuildingMesh], shift: np.ndarray, title: str, subtitle: str) -> None:
    segments = {"bottom": [], "wall": [], "roof": []}
    for mesh in meshes:
        xy = mesh.xy(shift)
        for triangle, surface in zip(mesh.triangles, mesh.surfaces, strict=True):
            vertices = xy[triangle]
            segments[str(surface)].append(np.vstack([vertices, vertices[0]]))
    fig, ax = base_axes(amplitude, title, subtitle)
    styles = {"bottom": ("#8B5CF6", 0.20, 0.48), "wall": ("#FF4FD8", 0.20, 0.48), "roof": ("#00E5FF", 0.28, 0.76)}
    for surface in ("bottom", "wall", "roof"):
        color, width, alpha = styles[surface]
        collection = LineCollection(segments[surface], colors=color, linewidths=width, alpha=alpha, label=surface)
        collection.set_rasterized(True)
        ax.add_collection(collection)
    ax.legend(["底面", "墙面", "屋顶"], loc="lower left", framealpha=0.78)
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    compact_svg(path)


def rasterize_triangle(mask: np.ndarray, xy: np.ndarray) -> None:
    rr, cc = raster_polygon(xy[:, 1], xy[:, 0], shape=mask.shape)
    mask[rr, cc] = True


def assigned_initial_mask(meshes: list[BuildingMesh], shift: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    union = np.zeros(shape, dtype=bool)
    owner = np.full(shape, -1, dtype=np.int16)
    # Paper LOS rule: smaller near-range column owns overlap pixels.
    for mesh_index in sorted(range(len(meshes)), key=lambda index: meshes[index].near_col + shift[0]):
        mesh = meshes[mesh_index]
        local = np.zeros(shape, dtype=bool)
        xy = mesh.xy(shift)
        for triangle in mesh.triangles:
            rasterize_triangle(local, xy[triangle])
        union |= local
        owner[local & (owner < 0)] = mesh_index
    return union, owner


def refine_mask(amplitude: np.ndarray, owner: np.ndarray, building_count: int, kappa: float = 0.20) -> np.ndarray:
    refined = np.zeros(owner.shape, dtype=bool)
    sar_amplitude = np.maximum(amplitude, 0).astype(np.float32, copy=False)
    for index in range(building_count):
        rr, cc = np.where(owner == index)
        if len(rr) == 0:
            continue
        r0, r1 = max(0, int(rr.min()) - 3), min(owner.shape[0], int(rr.max()) + 4)
        c0, c1 = max(0, int(cc.min()) - 3), min(owner.shape[1], int(cc.max()) + 4)
        geometry = owner[r0:r1, c0:c1] == index
        window = binary_dilation(geometry, iterations=3)
        values = sar_amplitude[r0:r1, c0:c1][window]
        threshold = float(values.mean() + kappa * values.std())
        candidate = geometry & (sar_amplitude[r0:r1, c0:c1] > threshold)
        candidate = binary_closing(candidate, iterations=1)
        components, count = label(candidate)
        if count:
            sizes = np.bincount(components.ravel())
            keep = np.where(sizes >= 4)[0]
            keep = keep[keep != 0]
            candidate = np.isin(components, keep)
        refined[r0:r1, c0:c1] |= candidate & geometry
    return refined


def interpolate_surface_elevation(
    meshes: list[BuildingMesh],
    shift: np.ndarray,
    owner: np.ndarray,
    interpolation_mask: np.ndarray,
    overlap_rule: str = "visible_surface_priority",
    vertices_ecef_by_fid: dict[int, np.ndarray] | None = None,
    sensor_ecef_by_row: np.ndarray | None = None,
    elevation_mode: str = "linear_vertex_height",
    allow_boundary_missing: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    """Interpolate absolute elevation at every SAR pixel centre covered by the 3-D triangles."""
    elevation = np.full(owner.shape, np.nan, dtype=np.float32)
    surface_code = np.zeros(owner.shape, dtype=np.uint8)
    if overlap_rule not in {"visible_surface_priority", "upper_envelope", "radar_los_nearest"}:
        raise ValueError(f"未知三角面重叠规则: {overlap_rule}")
    if overlap_rule == "radar_los_nearest" and (vertices_ecef_by_fid is None or sensor_ecef_by_row is None):
        raise ValueError("雷达LOS深度判定缺少建筑ECEF顶点或逐行卫星位置")
    if elevation_mode not in {"linear_vertex_height", "ecef_barycentric_to_wgs84"}:
        raise ValueError(f"未知高程反算模式: {elevation_mode}")
    if elevation_mode == "ecef_barycentric_to_wgs84" and vertices_ecef_by_fid is None:
        raise ValueError("ECEF重心反算缺少建筑三维顶点")
    ecef_to_llh = Transformer.from_crs(4978, 4979, always_xy=True) if elevation_mode == "ecef_barycentric_to_wgs84" else None
    slant_depth = np.full(owner.shape, np.inf, dtype=np.float32) if overlap_rule == "radar_los_nearest" else None
    exterior_coverage = np.zeros(owner.shape, dtype=bool) if overlap_rule == "radar_los_nearest" else None
    surface_rank = {"wall": 0, "roof": 1, "bottom": 2}
    surface_value = {"wall": 1, "roof": 2, "bottom": 3}
    for mesh_index, mesh in enumerate(meshes):
        xy = mesh.xy(shift)
        vertex_elevation = np.full(len(xy), BASE_WUSONG_M, dtype=np.float64)
        vertex_elevation[mesh.top_indices] += mesh.height
        triangle_order = sorted(
            range(len(mesh.triangles)),
            key=lambda index: (
                surface_rank[str(mesh.surfaces[index])],
                float(np.min(xy[mesh.triangles[index], 0])),
                index,
            ),
        )
        for triangle_index in triangle_order:
            if overlap_rule == "radar_los_nearest" and str(mesh.surfaces[triangle_index]) == "bottom":
                # The 4 m bottom is a construction cap on the ground plane, not a radar-visible exterior face.
                continue
            vertices = mesh.triangles[triangle_index]
            projected = xy[vertices]
            rr, cc = raster_polygon(projected[:, 1], projected[:, 0], shape=owner.shape)
            if not len(rr):
                continue
            target = interpolation_mask[rr, cc]
            if overlap_rule != "radar_los_nearest":
                target &= owner[rr, cc] == mesh_index
            else:
                exterior_coverage[rr[target], cc[target]] = True
            if overlap_rule == "visible_surface_priority":
                keep = target & ~np.isfinite(elevation[rr, cc])
            else:
                keep = target
            if not np.any(keep):
                continue
            rr, cc = rr[keep], cc[keep]
            p0, p1, p2 = projected
            denominator = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
            if abs(denominator) < 1e-10:
                continue
            col = cc.astype(np.float64)
            row = rr.astype(np.float64)
            weight0 = ((p1[1] - p2[1]) * (col - p2[0]) + (p2[0] - p1[0]) * (row - p2[1])) / denominator
            weight1 = ((p2[1] - p0[1]) * (col - p2[0]) + (p0[0] - p2[0]) * (row - p2[1])) / denominator
            weight2 = 1.0 - weight0 - weight1
            weights = np.column_stack([weight0, weight1, weight2])
            if elevation_mode == "ecef_barycentric_to_wgs84":
                points_ecef_for_height = weights @ vertices_ecef_by_fid[mesh.fid][vertices]
                point_lon, point_lat, values = ecef_to_llh.transform(
                    points_ecef_for_height[:, 0],
                    points_ecef_for_height[:, 1],
                    points_ecef_for_height[:, 2],
                )
                values = ellipsoid_to_wusong(np.asarray(values), point_lat, point_lon)
                values = np.asarray(values, dtype=np.float64)
            else:
                values = weights @ vertex_elevation[vertices]
            # Numerical clipping enforces the known extruded-building vertical interval.
            values = np.clip(values, BASE_WUSONG_M, BASE_WUSONG_M + mesh.height)
            if overlap_rule == "upper_envelope":
                current = elevation[rr, cc]
                update = ~np.isfinite(current) | (values > current + 1e-5)
                rr, cc, values = rr[update], cc[update], values[update]
                if not len(rr):
                    continue
            elif overlap_rule == "radar_los_nearest":
                ecef_vertices = vertices_ecef_by_fid[mesh.fid][vertices]
                points_ecef = weights @ ecef_vertices
                sensor_ecef = sensor_ecef_by_row[rr]
                candidate_depth = np.linalg.norm(sensor_ecef - points_ecef, axis=1)
                update = candidate_depth < slant_depth[rr, cc].astype(np.float64)
                rr, cc, values = rr[update], cc[update], values[update]
                candidate_depth = candidate_depth[update]
                if not len(rr):
                    continue
                slant_depth[rr, cc] = candidate_depth.astype(np.float32)
            elevation[rr, cc] = values.astype(np.float32)
            surface_code[rr, cc] = surface_value[str(mesh.surfaces[triangle_index])]
    required = exterior_coverage if overlap_rule == "radar_los_nearest" else interpolation_mask
    missing = required & ~np.isfinite(elevation)
    if np.any(missing) and not allow_boundary_missing:
        raise RuntimeError(f"完整建筑体三角面内有{int(missing.sum())}个像素未获得插值高程")
    stats = {
        "wall_pixels": int(np.sum(surface_code == 1)),
        "roof_pixels": int(np.sum(surface_code == 2)),
        "bottom_pixels": int(np.sum(surface_code == 3)),
        "boundary_missing_pixels": int(missing.sum()),
    }
    if overlap_rule == "radar_los_nearest":
        stats["bottom_only_excluded_pixels"] = int(np.sum(interpolation_mask & ~exterior_coverage))
    return elevation, stats


def build_ecef_vertices(meshes: list[BuildingMesh]) -> dict[int, np.ndarray]:
    """Rebuild the physical 3-D vertices corresponding to the cached GAMMA image vertices."""
    buildings = gpd.read_file(BUILDINGS, engine="pyogrio").to_crs(4326)
    transformer = Transformer.from_crs(4979, 4978, always_xy=True)
    result: dict[int, np.ndarray] = {}
    for mesh in meshes:
        ring = clean_ring(buildings.loc[mesh.fid].geometry)
        n = len(ring)
        if 2 * n != len(mesh.xy_initial):
            raise RuntimeError(f"建筑{mesh.fid}的矢量顶点与GAMMA投影缓存不一致")
        lon = np.r_[ring[:, 0], ring[:, 0]]
        lat = np.r_[ring[:, 1], ring[:, 1]]
        height = np.r_[
            wusong_to_ellipsoid(BASE_WUSONG_M, ring[:, 1], ring[:, 0]),
            wusong_to_ellipsoid(BASE_WUSONG_M + mesh.height, ring[:, 1], ring[:, 0]),
        ]
        x, y, z = transformer.transform(lon, lat, height)
        result[mesh.fid] = np.column_stack([x, y, z]).astype(np.float64)
    return result


def coordinated_building_elevation(
    meshes: list[BuildingMesh],
    shift: np.ndarray,
    owner: np.ndarray,
    interpolation_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit one bottom-to-top linear height field per building in radar coordinates."""
    elevation = np.full(owner.shape, np.nan, dtype=np.float32)
    clipped_low = 0
    clipped_high = 0
    assigned = 0
    for mesh_index, mesh in enumerate(meshes):
        rr, cc = np.where(interpolation_mask & (owner == mesh_index))
        if not len(rr):
            continue
        xy = mesh.xy(shift)
        n = len(mesh.top_indices)
        design = np.column_stack([xy[:, 0], xy[:, 1], np.ones(2 * n)])
        fraction_observation = np.r_[np.zeros(n), np.ones(n)]
        coefficients, *_ = np.linalg.lstsq(design, fraction_observation, rcond=None)
        raw_fraction = coefficients[0] * cc + coefficients[1] * rr + coefficients[2]
        clipped_low += int(np.sum(raw_fraction < 0))
        clipped_high += int(np.sum(raw_fraction > 1))
        fraction = np.clip(raw_fraction, 0.0, 1.0)
        elevation[rr, cc] = (BASE_WUSONG_M + mesh.height * fraction).astype(np.float32)
        assigned += len(rr)
    missing = interpolation_mask & ~np.isfinite(elevation)
    if np.any(missing):
        raise RuntimeError(f"建筑级协调高程场存在{int(missing.sum())}个漏插值像素")
    return elevation, {
        "assigned_pixels": float(assigned),
        "clipped_to_base_pixels": float(clipped_low),
        "clipped_to_roof_pixels": float(clipped_high),
    }


def sensor_ecef_by_image_row(par: Path, rows: int) -> np.ndarray:
    """Interpolate orbit state vectors at every SAR azimuth row."""
    text = par.read_text(encoding="utf-8")
    def scalar(key: str) -> float:
        match = re.search(rf"^{re.escape(key)}:\s+([-+0-9.eE]+)", text, re.MULTILINE)
        if match is None:
            raise ValueError(f"SLC参数缺失: {key}")
        return float(match.group(1))
    count = int(scalar("number_of_state_vectors"))
    first_time = scalar("time_of_first_state_vector")
    interval = scalar("state_vector_interval")
    start_time = scalar("start_time")
    line_time = scalar("azimuth_line_time")
    positions = []
    for index in range(1, count + 1):
        match = re.search(rf"^state_vector_position_{index}:\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", text, re.MULTILINE)
        if match is None:
            raise ValueError(f"SLC轨道状态向量缺失: {index}")
        positions.append([float(match.group(i)) for i in range(1, 4)])
    state_times = first_time + np.arange(count) * interval
    image_times = start_time + np.arange(rows) * line_time
    return CubicSpline(state_times, np.asarray(positions), axis=0)(image_times).astype(np.float64)


def save_mask_svg(path: Path, amplitude: np.ndarray, mask: np.ndarray, title: str, subtitle: str, color: tuple[float, float, float]) -> None:
    fig, ax = base_axes(amplitude, title, subtitle)
    preview = mask[::2, ::2]
    overlay = np.zeros((*preview.shape, 4), dtype=np.uint8)
    overlay[preview] = tuple(round(channel * 255) for channel in color) + (148,)
    ax.imshow(overlay, interpolation="none", resample=False, extent=(0, mask.shape[1], mask.shape[0], 0))
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    compact_svg(path)


def save_simulated_svg(path: Path, simulated: np.ndarray, full_shape: tuple[int, int]) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 8.5))
    ax.imshow(simulated, cmap="gray", vmin=0, vmax=1, interpolation="none", resample=False,
              extent=(0, full_shape[1], full_shape[0], 0))
    ax.set_xlim(0, full_shape[1])
    ax.set_ylim(full_shape[0], 0)
    ax.set_title("GAMMA建筑DSM模拟SAR影像", loc="left", fontsize=17, fontweight="bold", pad=13)
    ax.text(0.01, 0.985, "GAMMA gc_map2查找表 + sim_sar，已转换到当前裁剪影像雷达坐标",
            transform=ax.transAxes, va="top", color="white", fontsize=10,
            bbox={"facecolor": "black", "alpha": 0.72, "edgecolor": "none", "pad": 5})
    ax.set_xlabel("距离向列号 / pixel")
    ax.set_ylabel("方位向行号 / pixel")
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    compact_svg(path)


def save_elevation_svg(
    path: Path,
    amplitude: np.ndarray,
    elevation: np.ndarray,
    surface_counts: dict[str, int],
    subtitle: str,
) -> None:
    fig, ax = base_axes(
        amplitude,
        "建筑体三角面重心插值SAR像素高程",
        subtitle,
    )
    preview = elevation[::2, ::2]
    valid = np.isfinite(preview)
    displayed = np.ma.masked_where(~valid, preview)
    image = ax.imshow(
        displayed,
        cmap="turbo",
        vmin=BASE_WUSONG_M,
        vmax=float(np.nanpercentile(elevation, 99.5)),
        alpha=0.84,
        interpolation="none",
        resample=False,
        extent=(0, elevation.shape[1], elevation.shape[0], 0),
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.018)
    colorbar.set_label("绝对高程 / m（统一4 m底面基准）")
    ax.text(
        0.99,
        0.985,
        f"墙面 {surface_counts['wall_pixels']:,}｜屋顶 {surface_counts['roof_pixels']:,}｜底面 {surface_counts['bottom_pixels']:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="white",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.72, "edgecolor": "none", "pad": 5},
    )
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    compact_svg(path)


def save_paper_height_svg(path: Path, amplitude: np.ndarray, elevation: np.ndarray, surface_counts: dict[str, int]) -> None:
    rows, cols = amplitude.shape
    fig, ax = plt.subplots(figsize=(12.0, 8.5))
    preview_amplitude = amplitude[::2, ::2]
    ax.imshow(
        display_image(preview_amplitude), cmap="gray", vmin=0, vmax=1, interpolation="none", resample=False,
        extent=(0, cols, rows, 0),
    )
    rr, cc = np.where(np.isfinite(elevation))
    values = elevation[rr, cc]
    # Paper Fig. 5.3 represents discrete building scattering points colored by elevation.
    points = ax.scatter(
        cc,
        rr,
        c=values,
        s=0.10,
        marker="s",
        linewidths=0,
        cmap="turbo",
        vmin=BASE_WUSONG_M,
        vmax=float(np.nanpercentile(values, 99.5)),
        alpha=0.90,
        rasterized=True,
    )
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_title("SAR建筑散射点三维高程标记", loc="left", fontsize=17, fontweight="bold", pad=13)
    ax.text(
        0.01, 0.985,
        "参照初稿式(2.13)-(2.16)、(3.3)-(3.4)：LOS建筑归属 + 精炼散射像素 + ECEF重心反算",
        transform=ax.transAxes, va="top", color="white", fontsize=10,
        bbox={"facecolor": "black", "alpha": 0.72, "edgecolor": "none", "pad": 5},
    )
    ax.text(
        0.99, 0.985,
        f"墙面 {surface_counts['wall_pixels']:,}｜屋顶 {surface_counts['roof_pixels']:,}｜底面 {surface_counts['bottom_pixels']:,}",
        transform=ax.transAxes, ha="right", va="top", color="white", fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.72, "edgecolor": "none", "pad": 5},
    )
    colorbar = fig.colorbar(points, ax=ax, fraction=0.028, pad=0.018)
    colorbar.set_label("WGS84椭球高 / m")
    ax.set_xlabel("距离向列号 / pixel")
    ax.set_ylabel("方位向行号 / pixel")
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    compact_svg(path)


def save_owned_volume_height_svg(
    path: Path,
    amplitude: np.ndarray,
    elevation: np.ndarray,
    surface_counts: dict[str, int],
) -> None:
    fig, ax = base_axes(
        amplitude,
        "投影建筑体内建筑级协调线性高程",
        "先按LOS近距顺序归属建筑；每栋建筑以全部底/顶投影顶点拟合统一线性高度轴，消除三角面拼块跳变",
    )
    preview = elevation[::2, ::2]
    image = ax.imshow(
        np.ma.masked_invalid(preview),
        cmap="turbo",
        vmin=BASE_WUSONG_M,
        vmax=float(np.nanpercentile(elevation, 99.5)),
        alpha=0.86,
        interpolation="none",
        resample=False,
        extent=(0, elevation.shape[1], elevation.shape[0], 0),
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.018)
    colorbar.set_label("WGS84椭球高 / m")
    ax.text(
        0.99,
        0.985,
        f"有效像素 {int(surface_counts['assigned_pixels']):,}｜底端约束 {int(surface_counts['clipped_to_base_pixels']):,}｜顶端约束 {int(surface_counts['clipped_to_roof_pixels']):,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="white",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.72, "edgecolor": "none", "pad": 5},
    )
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    compact_svg(path)


def main() -> None:
    par = WORK / "image1_full_scene.slc.par"
    write_gamma_par(INPUT / "source.meta.xml", par)
    with rasterio.open(INPUT / "amplitude_crop.tif") as source:
        amplitude = source.read(1).astype(np.float32)
    mesh_cache = WORK / "gamma_building_volume_meshes.pkl"
    if mesh_cache.exists():
        with mesh_cache.open("rb") as handle:
            stored = pickle.load(handle)
        if stored.get("projection_datum_version") == PROJECTION_DATUM_VERSION:
            meshes, skipped, gamma_calls = stored["meshes"], stored["skipped"], stored["gamma_calls"]
            print(f"读取GAMMA建筑体缓存：{len(meshes)}栋", flush=True)
        else:
            meshes = None
    else:
        meshes = None
    if meshes is None:
        buildings = gpd.read_file(BUILDINGS, engine="pyogrio")
        if buildings.crs is None:
            raise ValueError("建筑矢量缺少CRS")
        buildings = buildings.to_crs(4326)
        cache = {}
        meshes = []
        skipped = 0
        for fid, feature in buildings.iterrows():
            try:
                mesh = make_mesh(int(fid), feature, par, cache)
                maximum = mesh.xy_initial.max(axis=0)
                minimum = mesh.xy_initial.min(axis=0)
                if maximum[0] < 0 or maximum[1] < 0 or minimum[0] >= amplitude.shape[1] or minimum[1] >= amplitude.shape[0]:
                    skipped += 1
                    continue
                meshes.append(mesh)
            except Exception:
                skipped += 1
            if (len(meshes) + skipped) % 100 == 0:
                print(f"GAMMA建筑体投影 {len(meshes) + skipped}/{len(buildings)}", flush=True)
        gamma_calls = len(cache)
        with mesh_cache.open("wb") as handle:
            pickle.dump({
                "meshes": meshes, "skipped": skipped, "gamma_calls": gamma_calls,
                "projection_datum_version": PROJECTION_DATUM_VERSION,
            }, handle, protocol=pickle.HIGHEST_PROTOCOL)
    registration, simulated = simulated_sar_registration(amplitude)
    shift = np.asarray([registration["col_shift_px"], registration["row_shift_px"]], dtype=np.float64)
    for stale in PICALL.glob("*.svg"):
        if not stale.name.startswith("001_"):
            stale.unlink()
    save_simulated_svg(PICALL / "002_图件_90843926537.svg", simulated, amplitude.shape)
    save_mesh_svg(PICALL / "003_图件_353273323986.svg", amplitude, meshes, np.zeros(2),
                  "GAMMA建筑体三角网初始投影", "按初稿式(2.9)-(2.11)：底面4 m，顶面4 m+height，墙面每边双三角形")
    save_mesh_svg(PICALL / "004_图件_697279743889.svg", amplitude, meshes, shift,
                  "GAMMA模拟SAR配准后的建筑体三角网投影",
                  f"模拟与实测SAR相位相关：距离向{shift[0]:+.2f} pixel，方位向{shift[1]:+.2f} pixel；整栋建筑统一校正")
    initial, owner = assigned_initial_mask(meshes, shift, amplitude.shape)
    refined = refine_mask(amplitude, owner, len(meshes), kappa=0.20)
    save_mask_svg(PICALL / "005_拉伸建筑体投影初始掩膜.svg", amplitude, initial,
                  "建筑体投影初始掩膜", f"三角面栅格化并按LOS近距离向归属；候选像素{int(initial.sum()):,}", (0.0, 0.85, 1.0))
    save_mask_svg(PICALL / "006_图件_148609788448.svg", amplitude, refined,
                  "SAR强度特征精炼掩膜",
                  f"初稿式(3.2)-(3.3)：局部阈值 μ+0.20σ，闭运算与小连通域剔除；保留{int(refined.sum()):,}像素", (0.20, 1.0, 0.28))
    vertices_ecef = build_ecef_vertices(meshes)
    refined_elevation, refined_surface_counts = interpolate_surface_elevation(
        meshes, shift, owner, refined, overlap_rule="visible_surface_priority"
    )
    save_elevation_svg(
        PICALL / "007_图件_926220070457.svg",
        amplitude,
        refined_elevation,
        refined_surface_counts,
        "三角形顶点三维高程按像素重心坐标插值；仅显示SAR强度精炼掩膜内的有效像素",
    )
    sensor_positions = sensor_ecef_by_image_row(par, amplitude.shape[0])
    full_elevation, full_surface_counts = interpolate_surface_elevation(
        meshes,
        shift,
        owner,
        initial,
        overlap_rule="radar_los_nearest",
        vertices_ecef_by_fid=vertices_ecef,
        sensor_ecef_by_row=sensor_positions,
    )
    save_elevation_svg(
        PICALL / "008_图件_1042156626526.svg",
        amplitude,
        full_elevation,
        full_surface_counts,
        "图004完整三角网逐像素重心插值；重叠处按成像时刻卫星斜距选择LOS最前表面",
    )
    paper_elevation, paper_surface_counts = interpolate_surface_elevation(
        meshes,
        shift,
        owner,
        refined,
        overlap_rule="visible_surface_priority",
        vertices_ecef_by_fid=vertices_ecef,
        elevation_mode="ecef_barycentric_to_wgs84",
    )
    save_paper_height_svg(
        PICALL / "009_图件_1006799566252.svg",
        amplitude,
        paper_elevation,
        paper_surface_counts,
    )
    owned_volume_elevation, owned_volume_surface_counts = coordinated_building_elevation(
        meshes, shift, owner, initial
    )
    save_owned_volume_height_svg(
        PICALL / "010_图件_904716073273.svg",
        amplitude,
        owned_volume_elevation,
        owned_volume_surface_counts,
    )
    refined_valid = refined_elevation[np.isfinite(refined_elevation)]
    full_valid = full_elevation[np.isfinite(full_elevation)]
    print({
        "buildings": len(meshes), "skipped": skipped, "gamma_calls": gamma_calls,
        "triangles": int(sum(len(mesh.triangles) for mesh in meshes)),
        "registration": registration, "initial_mask_pixels": int(initial.sum()),
        "refined_mask_pixels": int(refined.sum()), "refined_fraction": float(refined.sum() / max(initial.sum(), 1)),
        "refined_surface_pixels": refined_surface_counts,
        "refined_elevation_min_m": float(refined_valid.min()),
        "refined_elevation_max_m": float(refined_valid.max()),
        "full_surface_pixels": full_surface_counts,
        "full_elevation_min_m": float(full_valid.min()),
        "full_elevation_max_m": float(full_valid.max()),
        "full_elevation_mean_m": float(full_valid.mean()),
        "paper_surface_pixels": paper_surface_counts,
        "paper_elevation_min_m": float(np.nanmin(paper_elevation)),
        "paper_elevation_max_m": float(np.nanmax(paper_elevation)),
        "owned_volume_linear_field": owned_volume_surface_counts,
        "owned_volume_elevation_min_m": float(np.nanmin(owned_volume_elevation)),
        "owned_volume_elevation_max_m": float(np.nanmax(owned_volume_elevation)),
    })


if __name__ == "__main__":
    main()
