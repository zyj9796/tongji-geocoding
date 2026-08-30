#!/usr/bin/env python3
"""Estimate heights by SAR feature matching along height-extended GAMMA building rulers."""

from __future__ import annotations

import base64
import csv
import io
import os
import pickle
import re
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-siwei-fixed-ruler")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import Normalize
from PIL import Image, ImageFilter
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter1d, sobel
from skimage.draw import polygon as raster_polygon

from gamma_projection_core import (
    BASE_WUSONG_M, BUILDINGS, INPUT, PICALL, PROJECTION_DATUM_VERSION, WORK,
    gamma_project, wusong_to_ellipsoid, write_gamma_par,
)
from gamma_simulated_sar_registration import simulated_sar_registration
from project_volume_mesh_and_refine import BuildingMesh, clean_ring, constrained_roof_triangles


MINIMUM_HEIGHT_M = 3.0
GEOCODE_WORK = WORK / "gamma_geocoded_sar"
SIM_WORK = WORK / "gamma_simulated_sar_ellipsoid"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "svg.fonttype": "none",
})


def make_fixed_mesh(fid: int, clean_id: int, geometry, ruler_height_m: float, par: Path, cache: dict) -> BuildingMesh:
    ring = clean_ring(geometry)
    n = len(ring)
    xy = []
    for elevation_wusong in (BASE_WUSONG_M, BASE_WUSONG_M + ruler_height_m):
        for lon, lat in ring:
            elevation = float(wusong_to_ellipsoid(elevation_wusong, lat, lon))
            key = (round(float(lon), 12), round(float(lat), 12), round(elevation, 6))
            if key not in cache:
                cache[key] = gamma_project(par, float(lat), float(lon), elevation)
            xy.append(cache[key])
    xy = np.asarray(xy, dtype=np.float64)
    roof = constrained_roof_triangles(ring)
    triangles, surfaces = [], []
    for index in range(n):
        following = (index + 1) % n
        triangles.extend([(index, following, n + following), (index, n + following, n + index)])
        surfaces.extend(["wall", "wall"])
    for a, b, c in roof:
        triangles.extend([(n + int(a), n + int(b), n + int(c)), (int(c), int(b), int(a))])
        surfaces.extend(["roof", "bottom"])
    return BuildingMesh(
        fid=fid, clean_id=clean_id, height=ruler_height_m, xy_initial=xy,
        triangles=np.asarray(triangles, dtype=np.int32), surfaces=np.asarray(surfaces),
        top_indices=np.arange(n, 2 * n, dtype=np.int32), near_col=float(np.min(xy[:n, 0])),
    )


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, float(np.std(values)) * 0.25, 1e-6)
    return np.clip((values - median) / scale, -4.0, 4.0)


def estimate_from_moving_roof(
    amplitude_norm: np.ndarray,
    edge_norm: np.ndarray,
    mesh: BuildingMesh,
    shift: np.ndarray,
    owner: np.ndarray,
    mesh_index: int,
) -> dict:
    """Read the fixed ruler by matching candidate roof slices to SAR building features."""
    xy = mesh.xy(shift)
    n = len(mesh.top_indices)
    bottom_xy, ruler_top_xy = xy[:n], xy[n:]

    def evaluate(heights: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        metrics = []
        for height in heights:
            fraction = float(height / mesh.height)
            roof_xy = bottom_xy + fraction * (ruler_top_xy - bottom_xy)
            rr, cc = raster_polygon(roof_xy[:, 1], roof_xy[:, 0], shape=amplitude_norm.shape)
            if not len(rr):
                metrics.append({"height_m": float(height), "pixels": 0, "edge": 0.0, "contrast": -1.0, "bright": 0.0})
                continue
            local_owner = owner[rr, cc] == mesh_index
            rr, cc = rr[local_owner], cc[local_owner]
            if len(rr) < 6:
                metrics.append({"height_m": float(height), "pixels": len(rr), "edge": 0.0, "contrast": -1.0, "bright": 0.0})
                continue
            r0, r1 = max(0, int(rr.min()) - 5), min(amplitude_norm.shape[0], int(rr.max()) + 6)
            c0, c1 = max(0, int(cc.min()) - 5), min(amplitude_norm.shape[1], int(cc.max()) + 6)
            mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
            mask[rr - r0, cc - c0] = True
            boundary = binary_dilation(mask, iterations=1) & ~binary_erosion(mask, iterations=1)
            outside = binary_dilation(mask, iterations=4) & ~binary_dilation(mask, iterations=1)
            # The ownership mask prevents candidate support from jumping to another ruler.
            owner_crop = owner[r0:r1, c0:c1] == mesh_index
            outside &= owner_crop
            values = amplitude_norm[r0:r1, c0:c1][mask]
            ordered = np.sort(values)
            bright = float(np.mean(ordered[max(0, int(0.70 * len(ordered))):]))
            outside_mean = float(np.mean(amplitude_norm[r0:r1, c0:c1][outside])) if np.any(outside) else float(np.median(values))
            metrics.append({
                "height_m": float(height), "pixels": len(values),
                "edge": float(np.mean(edge_norm[r0:r1, c0:c1][boundary])) if np.any(boundary) else 0.0,
                "contrast": float(np.mean(values) - outside_mean), "bright": bright,
            })
        valid = np.asarray([item["pixels"] >= 6 for item in metrics], dtype=bool)
        if not np.any(valid):
            return np.full(len(heights), -1e9), metrics
        score = np.full(len(heights), -1e9, dtype=np.float64)
        combined = (
            0.50 * robust_z(np.asarray([item["edge"] for item in metrics])[valid])
            + 0.30 * robust_z(np.asarray([item["contrast"] for item in metrics])[valid])
            + 0.20 * robust_z(np.asarray([item["bright"] for item in metrics])[valid])
        )
        combined -= 0.0005 * heights[valid]
        score[valid] = gaussian_filter1d(combined, 0.8, mode="nearest")
        for index, item in enumerate(metrics): item["score"] = float(score[index])
        return score, metrics

    maximum_estimate = max(MINIMUM_HEIGHT_M, mesh.height - 1.0)
    coarse = np.arange(MINIMUM_HEIGHT_M, maximum_estimate + 0.1, 2.0)
    coarse_score, coarse_metrics = evaluate(coarse)
    if np.max(coarse_score) <= -1e8:
        return {"height_estimate_m": np.nan, "score": np.nan, "margin": np.nan, "roof_pixels": 0}
    coarse_best = float(coarse[int(np.argmax(coarse_score))])
    fine = np.arange(max(MINIMUM_HEIGHT_M, coarse_best - 3), min(maximum_estimate, coarse_best + 3) + 0.01, 0.25)
    fine_score, fine_metrics = evaluate(fine)
    best_index = int(np.argmax(fine_score)); best = float(fine[best_index]); best_score = float(fine_score[best_index])
    separated = (np.abs(fine - best) >= 1.0) & (fine_score > -1e8)
    margin = best_score - float(np.max(fine_score[separated])) if np.any(separated) else 0.0
    return {"height_estimate_m": best, "score": best_score, "margin": margin, "roof_pixels": int(fine_metrics[best_index]["pixels"])}


def par_value(path: Path, key: str) -> float:
    match = re.search(rf"^{re.escape(key)}:\s+([^\s]+)", path.read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise ValueError(key)
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
    buffer = io.BytesIO(); image.save(buffer, "JPEG", quality=90, subsampling=0, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    path.write_text(text[:match.start()] + "data:image/jpeg;base64,\n" + encoded + text[match.end():], encoding="utf-8")


def main() -> None:
    par = WORK / "image1_full_scene.slc.par"
    write_gamma_par(INPUT / "source.meta.xml", par)
    vector = gpd.read_file(BUILDINGS, engine="pyogrio")
    if vector.crs is None:
        raise ValueError("建筑矢量缺少CRS")
    if "height" not in vector.columns:
        raise ValueError("建筑矢量缺少height字段")
    # height only controls how far the ruler is extended. It is never used as a
    # candidate preference, regularizer, label, or output substitution.
    estimation_input = vector[["clean_id", "height", "geometry"]].to_crs(4326).copy()
    estimation_input["ruler_height_m"] = np.maximum(
        1.5 * estimation_input["height"].astype(float),
        estimation_input["height"].astype(float) + 20.0,
    )
    with rasterio.open(INPUT / "amplitude_crop.tif") as source:
        amplitude = source.read(1).astype(np.float32)
    # Registration must see the original amplitude. Afterwards reuse this full
    # array for features so the full-scene workflow stays below 4 GB RAM.
    registration, _ = simulated_sar_registration(amplitude)
    shift = np.asarray([registration["col_shift_px"], registration["row_shift_px"]], dtype=np.float64)
    np.maximum(amplitude, 0, out=amplitude)
    np.log1p(amplitude, out=amplitude)
    positive = amplitude[amplitude > 0]
    lo, hi = np.percentile(positive, [2.0, 99.7])
    amplitude -= np.float32(lo)
    amplitude /= np.float32(max(float(hi - lo), 1e-6))
    np.clip(amplitude, 0, 1, out=amplitude)
    amplitude_norm = amplitude
    edge_y = sobel(amplitude_norm, axis=0, output=np.float32)
    edge_norm = sobel(amplitude_norm, axis=1, output=np.float32)
    np.hypot(edge_y, edge_norm, out=edge_norm)
    del edge_y
    edge_norm /= max(float(np.percentile(edge_norm, 99.0)), 1e-6)
    np.clip(edge_norm, 0, 1, out=edge_norm)

    cache_path = WORK / "gamma_adaptive_extended_ruler_meshes.pkl"
    meshes = None
    if cache_path.exists():
        stored = pickle.load(open(cache_path, "rb"))
        if stored.get("projection_datum_version") == PROJECTION_DATUM_VERSION:
            meshes = stored["meshes"]
    if meshes is None:
        cache, meshes = {}, []
        for fid, feature in estimation_input.iterrows():
            mesh = make_fixed_mesh(int(fid), int(feature.clean_id), feature.geometry, float(feature.ruler_height_m), par, cache)
            maximum, minimum = mesh.xy_initial.max(axis=0), mesh.xy_initial.min(axis=0)
            if maximum[0] >= 0 and maximum[1] >= 0 and minimum[0] < amplitude.shape[1] and minimum[1] < amplitude.shape[0]:
                meshes.append(mesh)
            if (fid + 1) % 100 == 0:
                print(f"固定量尺GAMMA投影 {fid + 1}/{len(estimation_input)}", flush=True)
        pickle.dump({
            "meshes": meshes, "ruler_rule": "max(1.5*height,height+20m)",
            "height_used_for_search_extent_only": True,
            "projection_datum_version": PROJECTION_DATUM_VERSION,
        }, open(cache_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    # Build one LOS-visible owner image directly. Near-range buildings claim
    # pixels first, hence the fixed ruler masks do not overlap.
    owner = np.full(amplitude.shape, -1, dtype=np.int16)
    for mesh_index in sorted(range(len(meshes)), key=lambda index: meshes[index].near_col + shift[0]):
        mesh = meshes[mesh_index]
        xy = mesh.xy(shift)
        for vertices in mesh.triangles:
            projected = xy[vertices]
            rr, cc = raster_polygon(projected[:, 1], projected[:, 0], shape=amplitude.shape)
            if len(rr):
                free = owner[rr, cc] < 0
                owner[rr[free], cc[free]] = mesh_index
        if (mesh_index + 1) % 100 == 0:
            print(f"固定量尺LOS归属 {mesh_index + 1}/{len(meshes)}", flush=True)
    assigned = owner[owner >= 0]
    owner_counts = np.bincount(assigned, minlength=len(meshes)) if len(assigned) else np.zeros(len(meshes), dtype=int)
    records = []
    for mesh_index, mesh in enumerate(meshes):
        result = estimate_from_moving_roof(amplitude_norm, edge_norm, mesh, shift, owner, mesh_index)
        records.append({
            "fid": mesh.fid, "clean_id": mesh.clean_id,
            "height_estimate_m": result["height_estimate_m"], "score": result["score"],
            "score_margin": result["margin"], "candidate_roof_pixels": result["roof_pixels"],
            "ruler_height_m": mesh.height, "assigned_volume_pixels": int(owner_counts[mesh_index]),
            "height_used_for_search_extent_only": True, "height_used_as_estimate_or_score_prior": False,
            "base_elevation_m": BASE_WUSONG_M,
        })
        if (mesh_index + 1) % 100 == 0:
            print(f"移动屋顶SAR特征匹配 {mesh_index + 1}/{len(meshes)}", flush=True)
    table_path = WORK / "adaptive_ruler_sar_feature_height_estimates.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)

    estimate_by_fid = {int(item["fid"]): item["height_estimate_m"] for item in records}
    dem_par = SIM_WORK / "dem_seg.par"; width, lines = int(par_value(dem_par, "width")), int(par_value(dem_par, "nlines"))
    east, north = par_value(dem_par, "corner_east"), par_value(dem_par, "corner_north")
    pe, pn = par_value(dem_par, "post_east"), par_value(dem_par, "post_north")
    left, top = east - 0.5 * pe, north - 0.5 * pn; right, bottom = left + width * pe, top + lines * pn
    geocoded = np.fromfile(GEOCODE_WORK / "amplitude_map.gamma", dtype=">f4").reshape(lines, width).astype(np.float32)
    geocoded[geocoded <= 0] = np.nan; finite = geocoded[np.isfinite(geocoded)]; lo, hi = np.percentile(finite, [2, 99.7])
    display = np.clip((geocoded - lo) / max(float(hi - lo), 1e-6), 0, 1) ** 0.55
    mapped = vector.to_crs(32651).copy()
    mapped["height_estimate_m"] = [estimate_by_fid.get(int(fid), np.nan) for fid in mapped.index]
    mapped = mapped.cx[min(left, right):max(left, right), min(bottom, top):max(bottom, top)]
    valid = mapped[np.isfinite(mapped.height_estimate_m)].copy(); missing = mapped[~np.isfinite(mapped.height_estimate_m)].copy()
    values = valid.height_estimate_m.to_numpy(); norm = Normalize(0, max(20.0, float(np.nanpercentile(values, 99)))); cmap = plt.get_cmap("turbo")
    fig, ax = plt.subplots(figsize=(11.2, 10.0)); ax.imshow(display, cmap="gray", vmin=0, vmax=1, extent=(left, right, bottom, top), interpolation="none")
    if len(missing): missing.plot(ax=ax, facecolor="#6B7280", edgecolor="#D1D5DB", linewidth=.22, alpha=.42)
    valid.plot(ax=ax, column="height_estimate_m", cmap=cmap, norm=norm, edgecolor="#E5FFFF", linewidth=.22, alpha=.72)
    for _, feature in valid.iterrows():
        p=feature.geometry.representative_point(); ax.text(p.x,p.y,f"{feature.height_estimate_m:.0f}",ha="center",va="center",fontsize=2.1,color="white",fontweight="bold")
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,fraction=.030,pad=.018);cb.set_label("扩展量尺SAR特征匹配建筑高度估计 / m")
    ax.set_xlim(left,right);ax.set_ylim(bottom,top);ax.set_aspect("equal")
    ax.set_title("GAMMA扩展量尺与SAR特征匹配建筑高度估计",loc="left",fontsize=17,fontweight="bold",pad=13)
    ax.text(.01,.985,"统一4 m底面；量尺=max(1.5×height, height+20 m)，仅限定搜索范围；图004配准；LOS唯一归属；SAR特征确定高度",transform=ax.transAxes,va="top",color="white",fontsize=9.3,bbox={"facecolor":"black","alpha":.74,"edgecolor":"none","pad":5})
    ax.text(.99,.985,f"有效 {len(valid):,}/{len(mapped):,}栋",transform=ax.transAxes,ha="right",va="top",color="white",fontsize=9,bbox={"facecolor":"black","alpha":.74,"edgecolor":"none","pad":5})
    ax.set_xlabel("UTM东向坐标 / m");ax.set_ylabel("UTM北向坐标 / m");fig.tight_layout()
    output=PICALL/"015_图件_750393910634.svg";fig.savefig(output,format="svg",bbox_inches="tight");plt.close(fig);compact_svg(output)
    print({"output":str(output),"table":str(table_path),"ruler_rule":"max(1.5*height,height+20m)","height_used_as_score_prior":False,"projected_buildings":len(meshes),"estimated_buildings":len(valid),"height_min_m":float(np.min(values)),"height_median_m":float(np.median(values)),"height_max_m":float(np.max(values)),"assigned_pixels":int(len(assigned))})


if __name__ == "__main__":
    main()
