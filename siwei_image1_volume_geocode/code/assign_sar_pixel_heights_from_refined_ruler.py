#!/usr/bin/env python3
"""Assign building-surface heights to SAR pixels with the refined GAMMA ruler.

The building base is fixed at 4 m Wusong elevation.  Roof height is selected
from SAR amplitude features along the GAMMA-projected height ruler.  Pixels in
the selected roof polygon receive one constant roof elevation; pixels on the
projected side faces receive barycentrically interpolated elevations between
the 4 m base and the selected roof elevation.

The Wusong-to-WGS84 ellipsoid conversion remains the explicitly provisional
GAMMA EGM96 proxy used by the refined base projection.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-siwei-ruler-height")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal, ogr
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter1d, sobel

from gamma_projection_core import (
    BASE_WUSONG_M, BUILDINGS, GAMMA_LIB, INPUT, PICALL, WORK, X_OFFSET, Y_OFFSET,
)


COORD_TO_SARPIX_LIST = Path("/usr/local/GAMMA/DIFF/bin/coord_to_sarpix_list")
BASE_VERTICES = WORK / "building_base_gamma_projection_vertices_refined.csv"
DIFF_PAR = WORK / "gamma_base_refinement/base_refinement.diff_par"
SLC_PAR = WORK / "image1_full_scene.slc.par"
MIN_HEIGHT_M = 3.0
COARSE_STEP_M = 2.0
FINE_STEP_M = 0.25

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["AR PL UKai CN", "Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    scale = max(1.4826 * mad, 0.25 * float(np.std(values)), 1e-6)
    return np.clip((values - med) / scale, -4.0, 4.0)


def read_amplitude() -> tuple[np.ndarray, object]:
    dataset = gdal.Open(str(INPUT / "amplitude_crop.tif"), gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(INPUT / "amplitude_crop.tif")
    amplitude = dataset.ReadAsArray().astype(np.float32)
    amplitude[amplitude < 0] = 0
    np.log1p(amplitude, out=amplitude)
    positive = amplitude[amplitude > 0]
    lo, hi = np.percentile(positive, [2.0, 99.7])
    amplitude -= np.float32(lo)
    amplitude /= np.float32(max(float(hi - lo), 1e-6))
    np.clip(amplitude, 0, 1, out=amplitude)
    return amplitude, dataset


def read_building_attributes() -> dict[int, dict]:
    source = ogr.Open(str(BUILDINGS), 0)
    if source is None:
        raise FileNotFoundError(BUILDINGS)
    result = {}
    for fid, feature in enumerate(source.GetLayer(0)):
        height = feature.GetField("height")
        height = float(height) if height is not None else 30.0
        if not np.isfinite(height) or height <= 0:
            height = 30.0
        result[fid] = {
            "clean_id": int(feature.GetField("clean_id")),
            # Attribute height controls only how far the ruler is extended.
            "attribute_height_m": height,
            "ruler_height_m": max(1.5 * height, height + 20.0),
        }
    return result


def read_refined_bases(attributes: dict[int, dict]) -> dict[int, dict]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    with BASE_VERTICES.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            fid = int(row["fid"])
            if fid not in attributes:
                continue
            grouped[(fid, int(row["part"]))].append(row)
    buildings: dict[int, dict] = {}
    for (fid, part), rows in grouped.items():
        rows.sort(key=lambda item: int(item["vertex"]))
        lon = np.asarray([float(item["longitude_deg"]) for item in rows])
        lat = np.asarray([float(item["latitude_deg"]) for item in rows])
        ellipsoid = np.asarray([float(item["wgs84_ellipsoid_height_m"]) for item in rows])
        bottom = np.asarray([[float(item["crop_col_px"]), float(item["crop_row_px"])] for item in rows])
        # Shapefile rings repeat their first vertex.  Keep only unique edge vertices.
        if len(bottom) > 2 and np.allclose(bottom[0], bottom[-1], atol=1e-5):
            lon, lat, ellipsoid, bottom = lon[:-1], lat[:-1], ellipsoid[:-1], bottom[:-1]
        item = buildings.setdefault(fid, {**attributes[fid], "fid": fid, "parts": []})
        item["parts"].append({
            "part": part, "longitude": lon, "latitude": lat,
            "base_ellipsoid_m": ellipsoid, "bottom_xy": bottom,
        })
    return buildings


def project_ruler_tops(buildings: dict[int, dict]) -> None:
    map_rows, references = [], []
    for fid in sorted(buildings):
        building = buildings[fid]
        for part_index, part in enumerate(building["parts"]):
            for vertex in range(len(part["bottom_xy"])):
                map_rows.append([
                    part["latitude"][vertex], part["longitude"][vertex],
                    part["base_ellipsoid_m"][vertex] + building["ruler_height_m"],
                ])
                references.append((fid, part_index, vertex))
    map_path = WORK / "refined_ruler_top_map_coordinates.txt"
    sar_path = WORK / "refined_ruler_top_sar_coordinates.txt"
    np.savetxt(map_path, np.asarray(map_rows), fmt="%.12f %.12f %.6f")
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(GAMMA_LIB)
    command = [
        str(COORD_TO_SARPIX_LIST), str(SLC_PAR), "-", "-",
        str(map_path), str(sar_path), str(DIFF_PAR),
    ]
    subprocess.run(command, check=True, env=environment, capture_output=True, text=True)
    projected = np.loadtxt(sar_path)
    if projected.ndim == 1:
        projected = projected[None, :]
    if len(projected) != len(references):
        raise RuntimeError("GAMMA ruler-top output count mismatch")
    top_arrays = {}
    for fid, building in buildings.items():
        for part_index, part in enumerate(building["parts"]):
            top_arrays[(fid, part_index)] = np.empty_like(part["bottom_xy"])
    for xy, (fid, part_index, vertex) in zip(projected, references):
        top_arrays[(fid, part_index)][vertex] = [xy[0] - X_OFFSET, xy[1] - Y_OFFSET]
    for fid, building in buildings.items():
        for part_index, part in enumerate(building["parts"]):
            part["ruler_top_xy"] = top_arrays[(fid, part_index)]
        building["near_col"] = min(float(np.min(part["bottom_xy"][:, 0])) for part in building["parts"])


def polygon_local_mask(polygons: list[np.ndarray], shape: tuple[int, int], pad: int = 0):
    valid = [p for p in polygons if len(p) >= 3 and np.all(np.isfinite(p))]
    if not valid:
        return None
    c0 = max(0, int(np.floor(min(np.min(p[:, 0]) for p in valid))) - pad)
    c1 = min(shape[1], int(np.ceil(max(np.max(p[:, 0]) for p in valid))) + pad + 1)
    r0 = max(0, int(np.floor(min(np.min(p[:, 1]) for p in valid))) - pad)
    r1 = min(shape[0], int(np.ceil(max(np.max(p[:, 1]) for p in valid))) + pad + 1)
    if c1 <= c0 or r1 <= r0:
        return None
    image = Image.new("1", (c1 - c0, r1 - r0), 0)
    draw = ImageDraw.Draw(image)
    for polygon in valid:
        draw.polygon([(float(x - c0), float(y - r0)) for x, y in polygon], fill=1)
    return r0, r1, c0, c1, np.asarray(image, dtype=bool)


def volume_polygons(building: dict) -> list[np.ndarray]:
    polygons = []
    for part in building["parts"]:
        bottom, top = part["bottom_xy"], part["ruler_top_xy"]
        polygons.append(top)
        for index in range(len(bottom)):
            following = (index + 1) % len(bottom)
            polygons.append(np.asarray([bottom[index], bottom[following], top[following], top[index]]))
    return polygons


def build_los_owner(buildings: dict[int, dict], shape: tuple[int, int]) -> np.ndarray:
    owner = np.full(shape, -1, dtype=np.int16)
    for count, fid in enumerate(sorted(buildings, key=lambda key: buildings[key]["near_col"]), 1):
        local = polygon_local_mask(volume_polygons(buildings[fid]), shape)
        if local is None:
            continue
        r0, r1, c0, c1, mask = local
        view = owner[r0:r1, c0:c1]
        free = mask & (view < 0)
        view[free] = fid
        if count % 100 == 0:
            print(f"最大量尺LOS归属 {count}/{len(buildings)}", flush=True)
    return owner


def roof_polygons(building: dict, height_m: float) -> list[np.ndarray]:
    fraction = height_m / building["ruler_height_m"]
    return [
        part["bottom_xy"] + fraction * (part["ruler_top_xy"] - part["bottom_xy"])
        for part in building["parts"]
    ]


def evaluate_heights(amplitude: np.ndarray, edge: np.ndarray, owner: np.ndarray,
                     building: dict, heights: np.ndarray):
    metrics = []
    for height in heights:
        local = polygon_local_mask(roof_polygons(building, float(height)), amplitude.shape, pad=5)
        if local is None:
            metrics.append({"pixels": 0, "edge": 0.0, "contrast": -1.0, "bright": 0.0})
            continue
        r0, r1, c0, c1, mask = local
        owned = owner[r0:r1, c0:c1] == building["fid"]
        mask &= owned
        pixel_count = int(np.count_nonzero(mask))
        if pixel_count < 6:
            metrics.append({"pixels": pixel_count, "edge": 0.0, "contrast": -1.0, "bright": 0.0})
            continue
        boundary = binary_dilation(mask, iterations=1) & ~binary_erosion(mask, iterations=1)
        outside = binary_dilation(mask, iterations=4) & ~binary_dilation(mask, iterations=1)
        outside &= owned
        crop = amplitude[r0:r1, c0:c1]
        values = crop[mask]
        upper = np.partition(values, max(0, int(0.7 * len(values))))[int(0.7 * len(values)):]
        outside_mean = float(np.mean(crop[outside])) if np.any(outside) else float(np.median(values))
        metrics.append({
            "pixels": pixel_count,
            "edge": float(np.mean(edge[r0:r1, c0:c1][boundary])) if np.any(boundary) else 0.0,
            "contrast": float(np.mean(values) - outside_mean),
            "bright": float(np.mean(upper)),
        })
    valid = np.asarray([item["pixels"] >= 6 for item in metrics])
    scores = np.full(len(heights), -1e9, dtype=np.float64)
    if np.any(valid):
        scores[valid] = (
            0.50 * robust_z(np.asarray([m["edge"] for m in metrics])[valid])
            + 0.30 * robust_z(np.asarray([m["contrast"] for m in metrics])[valid])
            + 0.20 * robust_z(np.asarray([m["bright"] for m in metrics])[valid])
            - 0.0005 * heights[valid]
        )
        if np.count_nonzero(valid) > 2:
            scores[valid] = gaussian_filter1d(scores[valid], 0.8, mode="nearest")
    return scores, metrics


def estimate_roof(amplitude: np.ndarray, edge: np.ndarray, owner: np.ndarray, building: dict) -> dict:
    maximum = max(MIN_HEIGHT_M, building["ruler_height_m"] - 1.0)
    coarse = np.arange(MIN_HEIGHT_M, maximum + 0.01, COARSE_STEP_M)
    coarse_score, _ = evaluate_heights(amplitude, edge, owner, building, coarse)
    if float(np.max(coarse_score)) <= -1e8:
        return {"height_estimate_m": np.nan, "quality": "insufficient_pixels", "roof_detected": False,
                "score": np.nan, "score_margin": np.nan, "roof_pixels": 0, "endpoint_peak": False}
    coarse_best = float(coarse[int(np.argmax(coarse_score))])
    fine_low = max(MIN_HEIGHT_M, coarse_best - 3.0)
    fine_high = min(maximum, coarse_best + 3.0)
    fine = np.arange(fine_low, fine_high + 0.01, FINE_STEP_M)
    fine_score, metrics = evaluate_heights(amplitude, edge, owner, building, fine)
    index = int(np.argmax(fine_score))
    best, best_score = float(fine[index]), float(fine_score[index])
    separated = (np.abs(fine - best) >= 1.0) & (fine_score > -1e8)
    margin = best_score - float(np.max(fine_score[separated])) if np.any(separated) else 0.0
    # A peak at either global ruler endpoint means the roof was not bracketed.
    endpoint = best <= MIN_HEIGHT_M + 0.26 or best >= maximum - 0.26
    pixels = int(metrics[index]["pixels"])
    if endpoint:
        quality = "endpoint_peak"
    elif pixels < 10:
        quality = "few_roof_pixels"
    elif margin < 0.15:
        quality = "weak_peak"
    else:
        quality = "good"
    return {
        "height_estimate_m": best, "quality": quality, "roof_detected": quality == "good",
        "score": best_score, "score_margin": margin, "roof_pixels": pixels,
        "endpoint_peak": endpoint, "boundary_edge": float(metrics[index]["edge"]),
        "inside_outside_contrast": float(metrics[index]["contrast"]),
    }


def triangle_pixels(xy: np.ndarray, z: np.ndarray, shape: tuple[int, int]):
    c0 = max(0, int(np.floor(np.min(xy[:, 0])))); c1 = min(shape[1] - 1, int(np.ceil(np.max(xy[:, 0]))))
    r0 = max(0, int(np.floor(np.min(xy[:, 1])))); r1 = min(shape[0] - 1, int(np.ceil(np.max(xy[:, 1]))))
    if c1 < c0 or r1 < r0:
        return np.empty(0, int), np.empty(0, int), np.empty(0, np.float32)
    cols, rows = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
    x, y = cols.astype(np.float64), rows.astype(np.float64)
    x0, y0 = xy[0]; x1, y1 = xy[1]; x2, y2 = xy[2]
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denominator) < 1e-9:
        return np.empty(0, int), np.empty(0, int), np.empty(0, np.float32)
    w0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denominator
    w1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denominator
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7)
    values = (w0 * z[0] + w1 * z[1] + w2 * z[2]).astype(np.float32)
    return rows[inside], cols[inside], values[inside]


def rasterize_surfaces(buildings: dict[int, dict], estimates: dict[int, dict], shape: tuple[int, int]):
    elevation = np.full(shape, np.nan, dtype=np.float32)
    surface = np.zeros(shape, dtype=np.uint8)  # 0 background, 1 wall, 2 roof
    building_id = np.zeros(shape, dtype=np.uint16)  # fid+1, preserving 0 as background
    ordered = sorted(buildings, key=lambda key: buildings[key]["near_col"])
    for count, fid in enumerate(ordered, 1):
        height = estimates[fid]["height_estimate_m"]
        if not np.isfinite(height):
            continue
        building = buildings[fid]
        fraction = float(height / building["ruler_height_m"])
        # Roof is written before walls, so same-building layover pixels are
        # explicitly classified as top surface. Near-range buildings still own
        # pixels before farther buildings.
        local = polygon_local_mask(roof_polygons(building, float(height)), shape)
        if local is not None:
            r0, r1, c0, c1, mask = local
            free = mask & (surface[r0:r1, c0:c1] == 0)
            elevation[r0:r1, c0:c1][free] = np.float32(BASE_WUSONG_M + height)
            surface[r0:r1, c0:c1][free] = 2
            building_id[r0:r1, c0:c1][free] = fid + 1
        for part in building["parts"]:
            bottom = part["bottom_xy"]
            top = bottom + fraction * (part["ruler_top_xy"] - bottom)
            for index in range(len(bottom)):
                following = (index + 1) % len(bottom)
                for xy, z in (
                    (np.asarray([bottom[index], bottom[following], top[following]]),
                     np.asarray([BASE_WUSONG_M, BASE_WUSONG_M, BASE_WUSONG_M + height])),
                    (np.asarray([bottom[index], top[following], top[index]]),
                     np.asarray([BASE_WUSONG_M, BASE_WUSONG_M + height, BASE_WUSONG_M + height])),
                ):
                    rr, cc, zz = triangle_pixels(xy, z, shape)
                    if len(rr):
                        free = surface[rr, cc] == 0
                        elevation[rr[free], cc[free]] = zz[free]
                        surface[rr[free], cc[free]] = 1
                        building_id[rr[free], cc[free]] = fid + 1
        if count % 100 == 0:
            print(f"顶面/侧面高度栅格化 {count}/{len(ordered)}", flush=True)
    return elevation, surface, building_id


def write_tiff(path: Path, array: np.ndarray, source, nodata, description: str) -> None:
    options = ["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=3", "BIGTIFF=IF_SAFER"] if array.dtype.kind == "f" else ["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"]
    data_type = gdal.GDT_Float32 if array.dtype.kind == "f" else (gdal.GDT_Byte if array.dtype == np.uint8 else gdal.GDT_UInt16)
    target = gdal.GetDriverByName("GTiff").Create(str(path), array.shape[1], array.shape[0], 1, data_type, options=options)
    target.SetGeoTransform(source.GetGeoTransform())
    target.SetProjection(source.GetProjection())
    band = target.GetRasterBand(1); band.SetNoDataValue(nodata); band.SetDescription(description)
    band.WriteArray(array); band.FlushCache(); target.FlushCache(); target = None


def main() -> None:
    for required in (BASE_VERTICES, DIFF_PAR, SLC_PAR):
        if not required.exists():
            raise FileNotFoundError(required)
    WORK.mkdir(parents=True, exist_ok=True); PICALL.mkdir(parents=True, exist_ok=True)
    amplitude, source = read_amplitude()
    edge_y = sobel(amplitude, axis=0, output=np.float32)
    edge = sobel(amplitude, axis=1, output=np.float32)
    np.hypot(edge_y, edge, out=edge); del edge_y
    edge /= max(float(np.percentile(edge, 99.0)), 1e-6); np.clip(edge, 0, 1, out=edge)

    attributes = read_building_attributes()
    buildings = read_refined_bases(attributes)
    project_ruler_tops(buildings)
    owner = build_los_owner(buildings, amplitude.shape)
    estimates, records = {}, []
    for count, fid in enumerate(sorted(buildings), 1):
        result = estimate_roof(amplitude, edge, owner, buildings[fid])
        estimates[fid] = result
        records.append({
            "fid": fid, "clean_id": buildings[fid]["clean_id"],
            "base_wusong_m": BASE_WUSONG_M,
            "height_estimate_m": result["height_estimate_m"],
            "roof_wusong_elevation_m": BASE_WUSONG_M + result["height_estimate_m"] if np.isfinite(result["height_estimate_m"]) else np.nan,
            "roof_detected": result["roof_detected"], "quality": result["quality"],
            "score": result["score"], "score_margin": result["score_margin"],
            "candidate_roof_pixels": result["roof_pixels"], "endpoint_peak": result["endpoint_peak"],
            "boundary_edge": result.get("boundary_edge", np.nan),
            "inside_outside_contrast": result.get("inside_outside_contrast", np.nan),
            "attribute_height_m_search_extent_only": buildings[fid]["attribute_height_m"],
            "ruler_height_m": buildings[fid]["ruler_height_m"],
        })
        if count % 100 == 0:
            print(f"精化量尺屋顶判定 {count}/{len(buildings)}", flush=True)
    table_path = WORK / "refined_ruler_sar_pixel_height_estimates.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)

    del edge, owner
    elevation, surface, building_id = rasterize_surfaces(buildings, estimates, amplitude.shape)
    relative = elevation - np.float32(BASE_WUSONG_M)
    out_dir = WORK / "refined_ruler_pixel_heights"; out_dir.mkdir(parents=True, exist_ok=True)
    write_tiff(out_dir / "building_surface_wusong_elevation_m.tif", elevation, source, np.nan,
               "Building surface elevation, m, Wusong datum; roof constant, wall linear")
    write_tiff(out_dir / "building_height_above_4m_base_m.tif", relative, source, np.nan,
               "Building surface height above fixed 4 m Wusong base")
    write_tiff(out_dir / "surface_class.tif", surface, source, 0, "0=background, 1=wall, 2=roof")
    write_tiff(out_dir / "building_fid_plus_one.tif", building_id, source, 0, "Source building fid plus one")
    quality_codes = {"good": 1, "weak_peak": 2, "endpoint_peak": 3, "few_roof_pixels": 4, "insufficient_pixels": 5}
    quality_lut = np.zeros(max(buildings) + 2, dtype=np.uint8)
    for fid, result in estimates.items():
        quality_lut[fid + 1] = quality_codes[result["quality"]]
    quality = quality_lut[building_id]
    write_tiff(out_dir / "roof_detection_quality.tif", quality, source, 0,
               "0=background, 1=good, 2=weak peak, 3=endpoint peak, 4=few roof pixels, 5=insufficient")

    display = amplitude[::2, ::2]
    height_display = relative[::2, ::2]
    class_display = surface[::2, ::2]
    vmax = max(20.0, float(np.nanpercentile(relative, 99)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True)
    axes[0].imshow(display, cmap="gray", vmin=0, vmax=1, interpolation="none")
    overlay = np.ma.masked_invalid(height_display)
    image = axes[0].imshow(overlay, cmap="turbo", vmin=0, vmax=vmax, alpha=0.72, interpolation="none")
    axes[0].set_title("SAR像素建筑高度（顶面恒高、侧面线性）", loc="left")
    fig.colorbar(image, ax=axes[0], fraction=.035, pad=.02, label="相对4 m底面的高度 / m")
    classes = np.ma.masked_equal(class_display, 0)
    axes[1].imshow(display, cmap="gray", vmin=0, vmax=1, interpolation="none")
    axes[1].imshow(classes, cmap=matplotlib.colors.ListedColormap(["#ff9f1c", "#00e5ff"]), vmin=1, vmax=2, alpha=.78, interpolation="none")
    axes[1].set_title("表面判断：橙色=侧面，青色=顶面", loc="left")
    for axis in axes:
        axis.set_xlabel("距离向列号 / 2 pixel"); axis.set_ylabel("方位向行号 / 2 pixel")
    figure_path = PICALL / "020_图件_326662715140.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight"); plt.close(fig)

    finite = np.asarray([r["height_estimate_m"] for r in records], dtype=float)
    quality_counts = {key: sum(r["quality"] == key for r in records) for key in sorted({r["quality"] for r in records})}
    summary = {
        "method": "refined GAMMA height ruler with SAR roof-feature matching",
        "base": "fixed 4.000 m Wusong elevation",
        "projection": "coord_to_sarpix_list with base_refinement.diff_par",
        "vertical_conversion": "PROVISIONAL Wusong + GAMMA EGM96 proxy",
        "surface_rule": {"roof": "constant 4 m + estimated building height", "wall": "linear barycentric interpolation from 4 m base to roof"},
        "overlap_rule": "roof overrides wall within a building; smaller range-column building owns inter-building overlap",
        "buildings": len(buildings), "finite_estimates": int(np.count_nonzero(np.isfinite(finite))),
        "quality_counts": quality_counts,
        "quality_raster_codes": quality_codes,
        "height_m": {"min": float(np.nanmin(finite)), "median": float(np.nanmedian(finite)), "max": float(np.nanmax(finite))},
        "pixels": {"wall": int(np.count_nonzero(surface == 1)), "roof": int(np.count_nonzero(surface == 2))},
        "outputs": {"directory": str(out_dir), "estimates_csv": str(table_path), "figure": str(figure_path)},
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
