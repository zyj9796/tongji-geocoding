#!/usr/bin/env python3
"""Long-ruler building height estimation by SAR point-cluster contour matching.

The attribute height controls only the ruler extent.  SAR points are selected
inside the long projected building corridor. Candidate projected roof contours
are matched to the selected point-cluster contours. The selected roof is then
projected once more by GAMMA at its estimated elevation, rather than retained
as an endpoint interpolation.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-siwei-long-contour-ruler")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from osgeo import ogr
from scipy.ndimage import (
    binary_closing, binary_dilation, binary_erosion, distance_transform_edt,
    gaussian_filter1d, label, sobel,
)

from assign_sar_pixel_heights_from_refined_ruler import (
    BASE_VERTICES, COORD_TO_SARPIX_LIST, DIFF_PAR, SLC_PAR,
    build_los_owner, polygon_local_mask, project_ruler_tops, rasterize_surfaces,
    read_amplitude, read_building_attributes, read_refined_bases, robust_z,
    roof_polygons, write_tiff,
)
from gamma_projection_core import BASE_WUSONG_M, BUILDINGS, GAMMA_LIB, PICALL, WORK, X_OFFSET, Y_OFFSET


MIN_HEIGHT_M = 3.0
COARSE_STEP_M = 2.0
FINE_STEP_M = 0.25
OUTPUT = WORK / "long_contour_ruler"
INITIAL_ESTIMATES = WORK / "refined_ruler_sar_pixel_height_estimates.csv"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["AR PL UKai CN", "Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


def extend_rulers(buildings: dict[int, dict]) -> None:
    """Use a deliberately long ruler without using attribute height as a prior."""
    for building in buildings.values():
        height = building["attribute_height_m"]
        building["ruler_height_m"] = min(220.0, max(80.0, 2.2 * height, height + 50.0))


def read_initial_sar_estimates() -> dict[int, dict]:
    """Read the previous SAR-only ruler solution as the first-stage height interval."""
    result = {}
    with INITIAL_ESTIMATES.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = float(row["height_estimate_m"])
            result[int(row["fid"])] = {"height_m": value, "quality": row["quality"]}
    return result


def select_building_points(amplitude: np.ndarray, owner: np.ndarray, buildings: dict[int, dict], kappa: float = 0.20) -> np.ndarray:
    """Figure-7-style SAR point selection inside each long-ruler volume."""
    selected = np.zeros(owner.shape, dtype=bool)
    raw = np.maximum(amplitude, 0).astype(np.float32, copy=False)
    for count, fid in enumerate(sorted(buildings), 1):
        # Get the local extent from the projected volume, avoiding a full-array
        # owner comparison for every building.
        polygons = []
        for part in buildings[fid]["parts"]:
            bottom, top = part["bottom_xy"], part["ruler_top_xy"]
            polygons.append(top)
            for index in range(len(bottom)):
                following = (index + 1) % len(bottom)
                polygons.append(np.asarray([bottom[index], bottom[following], top[following], top[index]]))
        local = polygon_local_mask(polygons, owner.shape, pad=3)
        if local is None:
            continue
        r0, r1, c0, c1, _ = local
        geometry = owner[r0:r1, c0:c1] == fid
        if not np.any(geometry):
            continue
        window = binary_dilation(geometry, iterations=3)
        values = raw[r0:r1, c0:c1][window]
        threshold = float(values.mean() + kappa * values.std())
        candidate = geometry & (raw[r0:r1, c0:c1] > threshold)
        candidate = binary_closing(candidate, iterations=1)
        components, component_count = label(candidate)
        if component_count:
            sizes = np.bincount(components.ravel())
            keep = np.where(sizes >= 4)[0]
            keep = keep[keep != 0]
            candidate = np.isin(components, keep)
        selected[r0:r1, c0:c1] |= candidate & geometry
        if count % 100 == 0:
            print(f"长尺走廊SAR建筑点筛选 {count}/{len(buildings)}", flush=True)
    return selected


def candidate_metrics(amplitude: np.ndarray, edge: np.ndarray, points: np.ndarray, owner: np.ndarray,
                      building: dict, heights: np.ndarray):
    metrics = []
    for height in heights:
        local = polygon_local_mask(roof_polygons(building, float(height)), amplitude.shape, pad=7)
        if local is None:
            metrics.append({"pixels": 0, "forward": 0.0, "reverse": 0.0, "density": 0.0, "contrast": 0.0, "edge": 0.0, "amp_contrast": -1.0, "bright": 0.0})
            continue
        r0, r1, c0, c1, roof = local
        owned = owner[r0:r1, c0:c1] == building["fid"]
        roof &= owned
        if np.count_nonzero(roof) < 6:
            metrics.append({"pixels": 0, "forward": 0.0, "reverse": 0.0, "density": 0.0, "contrast": 0.0, "edge": 0.0, "amp_contrast": -1.0, "bright": 0.0})
            continue
        roof_boundary = binary_dilation(roof, iterations=1) & ~binary_erosion(roof, iterations=1)
        selected = points[r0:r1, c0:c1] & owned
        near_roof = binary_dilation(roof, iterations=3)
        cluster_points = selected & near_roof
        point_count = int(np.count_nonzero(cluster_points))
        if point_count < 4 or not np.any(roof_boundary):
            metrics.append({"pixels": point_count, "forward": 0.0, "reverse": 0.0, "density": 0.0, "contrast": 0.0, "edge": 0.0, "amp_contrast": -1.0, "bright": 0.0})
            continue
        # Forward contour support: projected roof boundary should be close to
        # selected SAR building points.
        distance_to_points = distance_transform_edt(~selected)
        forward = float(np.mean(np.exp(-0.5 * (distance_to_points[roof_boundary] / 2.0) ** 2)))
        # Reverse contour support: the envelope of the nearby SAR point group
        # should be close to the projected roof contour.
        cluster = binary_closing(binary_dilation(cluster_points, iterations=1), iterations=1)
        cluster_boundary = cluster & ~binary_erosion(cluster, iterations=1)
        distance_to_roof_boundary = distance_transform_edt(~roof_boundary)
        reverse = float(np.mean(np.exp(-0.5 * (distance_to_roof_boundary[cluster_boundary] / 2.0) ** 2))) if np.any(cluster_boundary) else 0.0
        density = float(np.mean(binary_dilation(selected, iterations=1)[roof]))
        outside = binary_dilation(roof, iterations=6) & ~binary_dilation(roof, iterations=1) & owned
        outside_density = float(np.mean(binary_dilation(selected, iterations=1)[outside])) if np.any(outside) else density
        contrast = density - outside_density
        amplitude_crop = amplitude[r0:r1, c0:c1]
        roof_values = amplitude_crop[roof]
        ordered = np.sort(roof_values)
        bright = float(np.mean(ordered[max(0, int(0.70 * len(ordered))):]))
        outside_amplitude = float(np.mean(amplitude_crop[outside])) if np.any(outside) else float(np.median(roof_values))
        amplitude_contrast = float(np.mean(roof_values) - outside_amplitude)
        edge_support = float(np.mean(edge[r0:r1, c0:c1][roof_boundary]))
        metrics.append({
            "pixels": point_count, "forward": forward, "reverse": reverse,
            "density": density, "contrast": contrast, "edge": edge_support,
            "amp_contrast": amplitude_contrast, "bright": bright,
        })
    valid = np.asarray([item["pixels"] >= 4 for item in metrics])
    score = np.full(len(heights), -1e9, dtype=np.float64)
    if np.any(valid):
        density_all = np.asarray([m["density"] for m in metrics])
        terminal_drop = np.zeros(len(heights), dtype=np.float64)
        for index, height in enumerate(heights):
            after = (heights >= height + 1.5) & (heights <= height + 8.0) & valid
            if np.any(after):
                terminal_drop[index] = density_all[index] - float(np.mean(density_all[after]))
        score[valid] = (
            0.25 * robust_z(np.asarray([m["edge"] for m in metrics])[valid])
            + 0.20 * robust_z(np.asarray([m["amp_contrast"] for m in metrics])[valid])
            + 0.10 * robust_z(np.asarray([m["bright"] for m in metrics])[valid])
            + 0.15 * robust_z(np.asarray([m["forward"] for m in metrics])[valid])
            + 0.07 * robust_z(np.asarray([m["reverse"] for m in metrics])[valid])
            + 0.08 * robust_z(np.asarray([m["contrast"] for m in metrics])[valid])
            + 0.15 * robust_z(terminal_drop[valid])
            - 0.0010 * heights[valid]
        )
        if np.count_nonzero(valid) > 2:
            score[valid] = gaussian_filter1d(score[valid], 0.8, mode="nearest")
    return score, metrics


def estimate_height(amplitude: np.ndarray, edge: np.ndarray, points: np.ndarray, owner: np.ndarray,
                    building: dict, initial: dict | None) -> dict:
    maximum = building["ruler_height_m"] - 1.0
    initial_height = initial["height_m"] if initial and np.isfinite(initial["height_m"]) else np.nan
    if np.isfinite(initial_height):
        # The long ruler supplies an untruncated point corridor. The previous
        # SAR-only solution identifies which building cluster along that long
        # corridor belongs to this footprint; contour matching then refines it.
        radius = 12.0 if initial["quality"] == "good" else 20.0
        search_low = max(MIN_HEIGHT_M, initial_height - radius)
        search_high = min(maximum, initial_height + radius)
        coarse = np.arange(search_low, search_high + 0.01, 1.0)
    else:
        search_low, search_high = MIN_HEIGHT_M, maximum
        coarse = np.arange(search_low, search_high + 0.01, COARSE_STEP_M)
    coarse_score, _ = candidate_metrics(amplitude, edge, points, owner, building, coarse)
    if float(np.max(coarse_score)) <= -1e8:
        return {"height_estimate_m": np.nan, "quality": "insufficient_points", "score": np.nan,
                "score_margin": np.nan, "endpoint_peak": False, "roof_point_count": 0,
                "initial_sar_height_m": initial_height, "contour_adjustment_m": np.nan}
    coarse_best = float(coarse[int(np.argmax(coarse_score))])
    low, high = max(search_low, coarse_best - 2.5), min(search_high, coarse_best + 2.5)
    fine = np.arange(low, high + 0.01, FINE_STEP_M)
    fine_score, metrics = candidate_metrics(amplitude, edge, points, owner, building, fine)
    index = int(np.argmax(fine_score)); best = float(fine[index]); best_score = float(fine_score[index])
    separated = (np.abs(fine - best) >= 1.0) & (fine_score > -1e8)
    margin = best_score - float(np.max(fine_score[separated])) if np.any(separated) else 0.0
    endpoint = best <= MIN_HEIGHT_M + 0.26 or best >= maximum - 0.26
    interval_boundary = best <= search_low + 0.26 or best >= search_high - 0.26
    metric = metrics[index]
    if endpoint:
        quality = "endpoint_peak"
    elif interval_boundary:
        quality = "interval_boundary_peak"
    elif metric["pixels"] < 8:
        quality = "few_roof_points"
    elif margin < 0.15:
        quality = "weak_peak"
    elif metric["forward"] < 0.10:
        quality = "weak_contour"
    else:
        quality = "good"
    return {
        "height_estimate_m": best, "quality": quality, "score": best_score,
        "score_margin": margin, "endpoint_peak": endpoint, "search_interval_boundary_peak": interval_boundary,
        "roof_point_count": int(metric["pixels"]), "contour_forward": metric["forward"],
        "contour_reverse": metric["reverse"], "point_density": metric["density"],
        "point_inside_outside_contrast": metric["contrast"], "amplitude_inside_outside_contrast": metric["amp_contrast"],
        "roof_bright_quantile_mean": metric["bright"], "edge_support": metric["edge"],
        "initial_sar_height_m": initial_height,
        "contour_adjustment_m": best - initial_height if np.isfinite(initial_height) else np.nan,
    }


def project_selected_roofs(buildings: dict[int, dict], estimates: dict[int, dict]) -> None:
    rows, references = [], []
    for fid in sorted(buildings):
        height = estimates[fid]["height_estimate_m"]
        if not np.isfinite(height):
            continue
        for part_index, part in enumerate(buildings[fid]["parts"]):
            for vertex in range(len(part["bottom_xy"])):
                rows.append([part["latitude"][vertex], part["longitude"][vertex], part["base_ellipsoid_m"][vertex] + height])
                references.append((fid, part_index, vertex))
    map_path = OUTPUT / "selected_roof_map_coordinates.txt"
    sar_path = OUTPUT / "selected_roof_sar_coordinates.txt"
    np.savetxt(map_path, np.asarray(rows), fmt="%.12f %.12f %.6f")
    environment = os.environ.copy(); environment["LD_LIBRARY_PATH"] = str(GAMMA_LIB)
    subprocess.run([
        str(COORD_TO_SARPIX_LIST), str(SLC_PAR), "-", "-", str(map_path), str(sar_path), str(DIFF_PAR),
    ], check=True, env=environment, capture_output=True, text=True)
    projected = np.loadtxt(sar_path)
    if projected.ndim == 1:
        projected = projected[None, :]
    selected = {}
    for fid, building in buildings.items():
        for part_index, part in enumerate(building["parts"]):
            selected[(fid, part_index)] = np.empty_like(part["bottom_xy"])
    for xy, (fid, part_index, vertex) in zip(projected, references):
        selected[(fid, part_index)][vertex] = [xy[0] - X_OFFSET, xy[1] - Y_OFFSET]
    for fid, building in buildings.items():
        height = estimates[fid]["height_estimate_m"]
        if not np.isfinite(height):
            continue
        displacements = []
        base_centroids, roof_centroids = [], []
        for part_index, part in enumerate(building["parts"]):
            roof = selected[(fid, part_index)]
            part["selected_roof_xy"] = roof
            displacements.append(roof - part["bottom_xy"])
            base_centroids.append(np.mean(part["bottom_xy"], axis=0)); roof_centroids.append(np.mean(roof, axis=0))
        displacement = np.vstack(displacements)
        offset = np.mean(roof_centroids, axis=0) - np.mean(base_centroids, axis=0)
        estimates[fid].update({
            "base_centroid_col_px": float(np.mean(base_centroids, axis=0)[0]),
            "base_centroid_row_px": float(np.mean(base_centroids, axis=0)[1]),
            "roof_offset_col_px": float(offset[0]), "roof_offset_row_px": float(offset[1]),
            "roof_offset_magnitude_px": float(np.linalg.norm(offset)),
            "offset_col_px_per_m": float(np.median(displacement[:, 0]) / height),
            "offset_row_px_per_m": float(np.median(displacement[:, 1]) / height),
            "offset_magnitude_px_per_m": float(np.linalg.norm(offset) / height),
        })


def set_selected_roofs_as_final_geometry(buildings: dict[int, dict], estimates: dict[int, dict]) -> None:
    for fid, building in buildings.items():
        height = estimates[fid]["height_estimate_m"]
        if not np.isfinite(height):
            continue
        building["ruler_height_m"] = float(height)
        for part in building["parts"]:
            part["ruler_top_xy"] = part["selected_roof_xy"]


def write_vector(path: Path, records: dict[int, dict]) -> None:
    source = ogr.Open(str(BUILDINGS), 0); source_layer = source.GetLayer(0)
    driver = ogr.GetDriverByName("GPKG")
    if path.exists():
        driver.DeleteDataSource(str(path))
    target = driver.CreateDataSource(str(path))
    layer = target.CreateLayer("building_height_estimates", source_layer.GetSpatialRef(), source_layer.GetGeomType())
    source_definition = source_layer.GetLayerDefn()
    for index in range(source_definition.GetFieldCount()):
        layer.CreateField(source_definition.GetFieldDefn(index))
    new_fields = [
        ("h_est_m", ogr.OFTReal), ("roof_z_m", ogr.OFTReal), ("h_quality", ogr.OFTString),
        ("score", ogr.OFTReal), ("margin", ogr.OFTReal), ("dcol_px", ogr.OFTReal),
        ("drow_px", ogr.OFTReal), ("dpx_per_m", ogr.OFTReal), ("roof_pts", ogr.OFTInteger),
    ]
    for name, field_type in new_fields:
        definition = ogr.FieldDefn(name, field_type)
        if field_type == ogr.OFTString: definition.SetWidth(24)
        layer.CreateField(definition)
    target_definition = layer.GetLayerDefn()
    for fid, source_feature in enumerate(source_layer):
        feature = ogr.Feature(target_definition)
        feature.SetGeometry(source_feature.GetGeometryRef().Clone())
        for index in range(source_definition.GetFieldCount()):
            feature.SetField(source_definition.GetFieldDefn(index).GetNameRef(), source_feature.GetField(index))
        record = records.get(fid)
        if record and np.isfinite(record["height_estimate_m"]):
            feature.SetField("h_est_m", record["height_estimate_m"])
            feature.SetField("roof_z_m", BASE_WUSONG_M + record["height_estimate_m"])
            feature.SetField("h_quality", record["quality"]); feature.SetField("score", record["score"])
            feature.SetField("margin", record["score_margin"]); feature.SetField("dcol_px", record["roof_offset_col_px"])
            feature.SetField("drow_px", record["roof_offset_row_px"]); feature.SetField("dpx_per_m", record["offset_magnitude_px_per_m"])
            feature.SetField("roof_pts", record["roof_point_count"])
        else:
            feature.SetField("h_quality", record["quality"] if record else "not_projected")
        layer.CreateFeature(feature)
    target.FlushCache(); target = None; source = None


def plot_vector(path: Path, records: dict[int, dict]) -> None:
    source = ogr.Open(str(BUILDINGS), 0); patches, values = [], []
    for fid, feature in enumerate(source.GetLayer(0)):
        record = records.get(fid)
        if not record or not np.isfinite(record["height_estimate_m"]):
            continue
        geometry = feature.GetGeometryRef()
        polygons = [geometry] if geometry.GetGeometryName() == "POLYGON" else [geometry.GetGeometryRef(i) for i in range(geometry.GetGeometryCount())]
        for polygon in polygons:
            ring = polygon.GetGeometryRef(0)
            patches.append(Polygon(np.asarray([ring.GetPoint(i)[:2] for i in range(ring.GetPointCount())]), closed=True))
            values.append(record["height_estimate_m"])
    fig, ax = plt.subplots(figsize=(9.2, 9.0))
    collection = PatchCollection(patches, cmap="turbo", edgecolor="#E5FFFF", linewidth=.20)
    collection.set_array(np.asarray(values)); collection.set_clim(0, float(np.percentile(values, 99)))
    ax.add_collection(collection); ax.autoscale(); ax.set_aspect("equal")
    cb = fig.colorbar(collection, ax=ax, fraction=.035, pad=.02); cb.set_label("长尺轮廓匹配建筑高度估计 / m")
    ax.set_title("建筑矢量中的长尺轮廓匹配高度估计", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("经度 / °"); ax.set_ylabel("纬度 / °"); fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    for required in (BASE_VERTICES, DIFF_PAR, SLC_PAR):
        if not required.exists(): raise FileNotFoundError(required)
    OUTPUT.mkdir(parents=True, exist_ok=True); PICALL.mkdir(parents=True, exist_ok=True)
    amplitude_norm, amplitude_source = read_amplitude()
    # Point selection in the previous method used raw amplitude, whereas the
    # contour edge term uses normalized log amplitude.
    raw_amplitude = amplitude_source.ReadAsArray().astype(np.float32)
    edge_y = sobel(amplitude_norm, axis=0, output=np.float32); edge = sobel(amplitude_norm, axis=1, output=np.float32)
    np.hypot(edge_y, edge, out=edge); del edge_y
    edge /= max(float(np.percentile(edge, 99)), 1e-6); np.clip(edge, 0, 1, out=edge)

    attributes = read_building_attributes(); buildings = read_refined_bases(attributes)
    initial_estimates = read_initial_sar_estimates()
    extend_rulers(buildings); project_ruler_tops(buildings)
    owner = build_los_owner(buildings, amplitude_norm.shape)
    points = select_building_points(raw_amplitude, owner, buildings)
    write_tiff(OUTPUT / "long_ruler_selected_building_points.tif", points.astype(np.uint8), amplitude_source, 0,
               "SAR building points selected within long projected building rulers")

    estimates = {}
    for count, fid in enumerate(sorted(buildings), 1):
        estimates[fid] = estimate_height(amplitude_norm, edge, points, owner, buildings[fid], initial_estimates.get(fid))
        if count % 100 == 0:
            print(f"长尺顶面点群轮廓匹配 {count}/{len(buildings)}", flush=True)
    project_selected_roofs(buildings, estimates)

    records = []
    for fid in sorted(buildings):
        result = estimates[fid]
        record = {
            "fid": fid, "clean_id": buildings[fid]["clean_id"], "base_wusong_m": BASE_WUSONG_M,
            "attribute_height_m_search_extent_only": buildings[fid]["attribute_height_m"],
            "long_ruler_height_m": buildings[fid]["ruler_height_m"], **result,
        }
        record["roof_wusong_elevation_m"] = BASE_WUSONG_M + result["height_estimate_m"] if np.isfinite(result["height_estimate_m"]) else np.nan
        records.append(record)
    table_path = OUTPUT / "long_contour_ruler_height_estimates.csv"
    fieldnames = sorted({key for record in records for key in record})
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(records)

    set_selected_roofs_as_final_geometry(buildings, estimates)
    elevation, surface, building_id = rasterize_surfaces(buildings, estimates, amplitude_norm.shape)
    point_elevation = np.where(points & np.isfinite(elevation), elevation, np.nan).astype(np.float32)
    point_surface = np.where(points & np.isfinite(elevation), surface, 0).astype(np.uint8)
    write_tiff(OUTPUT / "building_surface_wusong_elevation_m.tif", elevation, amplitude_source, np.nan,
               "Direct-GAMMA selected roof; roof constant; wall barycentric linear elevation, m Wusong")
    write_tiff(OUTPUT / "selected_sar_building_point_wusong_elevation_m.tif", point_elevation, amplitude_source, np.nan,
               "Height assigned only to selected SAR building points")
    write_tiff(OUTPUT / "selected_sar_building_point_surface_class.tif", point_surface, amplitude_source, 0,
               "Selected SAR point class: 1=wall, 2=roof")
    vector_path = OUTPUT / "building_height_estimates.gpkg"; write_vector(vector_path, estimates)
    vector_figure = PICALL / "022_长尺顶面轮廓匹配建筑矢量高度估计.png"; plot_vector(vector_figure, estimates)

    finite = np.asarray([r["height_estimate_m"] for r in records], float)
    valid_records = [r for r in records if np.isfinite(r["height_estimate_m"])]
    quality_counts = {key: sum(r["quality"] == key for r in records) for key in sorted({r["quality"] for r in records})}
    per_m = np.asarray([r["offset_magnitude_px_per_m"] for r in valid_records])
    dcol = np.asarray([r["offset_col_px_per_m"] for r in valid_records]); drow = np.asarray([r["offset_row_px_per_m"] for r in valid_records])
    summary = {
        "ruler_rule": "min(220, max(80, 2.2*attribute_height, attribute_height+50)) m; search extent only",
        "point_selection": "within long-ruler LOS owner; raw amplitude > local mean + 0.20 std; closing; components >=4 px",
        "roof_match": "bidirectional roof/point-cluster contours + inside/outside contrast + post-roof point-density termination",
        "two_stage_constraint": "previous refined SAR-only ruler estimate defines local height interval; vector height is not a score prior",
        "final_roof_projection": "direct GAMMA coord_to_sarpix_list at estimated roof elevation with refined DIFF_par",
        "finite_estimates": int(np.count_nonzero(np.isfinite(finite))), "quality_counts": quality_counts,
        "height_m": {"min": float(np.nanmin(finite)), "median": float(np.nanmedian(finite)), "max": float(np.nanmax(finite))},
        "pixel_offset_per_height": {
            "median_col_px_per_m": float(np.median(dcol)), "median_row_px_per_m": float(np.median(drow)),
            "median_magnitude_px_per_m": float(np.median(per_m)), "median_m_per_pixel": float(np.median(1.0 / per_m)),
        },
        "pixels": {"selected_long_ruler_points": int(points.sum()), "assigned_final_points": int(np.isfinite(point_elevation).sum()),
                   "roof": int(np.count_nonzero(point_surface == 2)), "wall": int(np.count_nonzero(point_surface == 1))},
        "outputs": {"table": str(table_path), "vector": str(vector_path), "vector_figure": str(vector_figure), "directory": str(OUTPUT)},
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
