#!/usr/bin/env python3
"""Project the fixed 4 m Wusong building base into the SAR image with GAMMA.

Until a surveyed Wusong-to-WGS84 vertical transformation is supplied, the
GAMMA EGM96 geoid is used explicitly as a provisional vertical-datum proxy.
No building height, roof edge, or SAR brightness is used to define the base.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-siwei-gamma-base")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from osgeo import gdal, ogr

from gamma_projection_core import BASE_WUSONG_M, BUILDINGS, GAMMA_LIB, INPUT, PICALL, WORK, X_OFFSET, Y_OFFSET


GAMMA_COORD_LIST = Path("/usr/local/GAMMA/DIFF/bin/coord_to_sarpix_list")
GAMMA_EGM96 = Path("/usr/local/GAMMA/DIFF/scripts/egm96_wgs84_diff.tif")

plt.rcParams.update({"font.family": "AR PL UKai CN", "axes.unicode_minus": False})


def exterior_rings(geometry: ogr.Geometry):
    if geometry.GetGeometryName() == "POLYGON":
        yield 0, geometry.GetGeometryRef(0)
    elif geometry.GetGeometryName() == "MULTIPOLYGON":
        for part in range(geometry.GetGeometryCount()):
            yield part, geometry.GetGeometryRef(part).GetGeometryRef(0)


def bilinear_geoid(dataset, longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    band = dataset.GetRasterBand(1)
    grid = band.ReadAsArray().astype(np.float64)
    inverse = gdal.InvGeoTransform(dataset.GetGeoTransform())
    if inverse is None:
        raise RuntimeError("EGM96 geotransform is not invertible")
    col = inverse[0] + inverse[1] * longitude + inverse[2] * latitude
    row = inverse[3] + inverse[4] * longitude + inverse[5] * latitude
    # AREA raster: interpolation is referenced to pixel centres.
    col -= 0.5; row -= 0.5
    c0 = np.floor(col).astype(int); r0 = np.floor(row).astype(int)
    c0 = np.clip(c0, 0, dataset.RasterXSize - 2)
    r0 = np.clip(r0, 0, dataset.RasterYSize - 2)
    dc = col - c0; dr = row - r0
    return (
        grid[r0, c0] * (1 - dc) * (1 - dr)
        + grid[r0, c0 + 1] * dc * (1 - dr)
        + grid[r0 + 1, c0] * (1 - dc) * dr
        + grid[r0 + 1, c0 + 1] * dc * dr
    )


def normalized_amplitude(path: Path) -> np.ndarray:
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    image = dataset.ReadAsArray().astype(np.float32)
    positive = image[image > 0]
    low, high = np.percentile(positive, [2.0, 99.7])
    return np.clip((image - low) / max(float(high - low), 1e-6), 0, 1) ** 0.55


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diff-par", type=Path, help="optional GAMMA DIFF_par refinement polynomial")
    args = parser.parse_args()
    refined = args.diff_par is not None
    WORK.mkdir(parents=True, exist_ok=True); PICALL.mkdir(parents=True, exist_ok=True)
    vector = ogr.Open(str(BUILDINGS), 0)
    geoid = gdal.Open(str(GAMMA_EGM96), gdal.GA_ReadOnly)
    if vector is None or geoid is None:
        raise FileNotFoundError("building vector or GAMMA EGM96 grid is missing")

    points = []
    layer = vector.GetLayer(0)
    for fid, feature in enumerate(layer):
        clean_id = feature.GetField("clean_id")
        for part, ring in exterior_rings(feature.GetGeometryRef()):
            for vertex in range(ring.GetPointCount()):
                longitude, latitude, _ = ring.GetPoint(vertex)
                points.append({
                    "fid": fid, "clean_id": clean_id, "part": part, "vertex": vertex,
                    "longitude_deg": longitude, "latitude_deg": latitude,
                })
    longitude = np.asarray([item["longitude_deg"] for item in points])
    latitude = np.asarray([item["latitude_deg"] for item in points])
    undulation = bilinear_geoid(geoid, longitude, latitude)
    ellipsoid_height = BASE_WUSONG_M + undulation

    map_path = WORK / "building_base_gamma_map_coordinates.txt"
    suffix = "_refined" if refined else ""
    sar_path = WORK / f"building_base_gamma_full_scene_sar_coordinates{suffix}.txt"
    np.savetxt(map_path, np.column_stack([latitude, longitude, ellipsoid_height]), fmt="%.12f %.12f %.6f")
    environment = os.environ.copy(); environment["LD_LIBRARY_PATH"] = str(GAMMA_LIB)
    command = [
        str(GAMMA_COORD_LIST), str(WORK / "image1_full_scene.slc.par"), "-", "-",
        str(map_path), str(sar_path),
    ]
    if refined:
        command.append(str(args.diff_par))
    subprocess.run(command, check=True, env=environment, capture_output=True, text=True)
    projected = np.loadtxt(sar_path)
    if projected.ndim == 1:
        projected = projected[None, :]
    if len(projected) != len(points):
        raise RuntimeError(f"GAMMA returned {len(projected)} points for {len(points)} inputs")
    # This installed GAMMA build writes range pixel first and azimuth line
    # second (the HTML table labels the columns in the opposite order).
    range_pixel = projected[:, 0]; azimuth = projected[:, 1]
    crop_col = range_pixel - X_OFFSET; crop_row = azimuth - Y_OFFSET

    csv_path = WORK / f"building_base_gamma_projection_vertices{suffix}.csv"
    fieldnames = [
        "fid", "clean_id", "part", "vertex", "longitude_deg", "latitude_deg",
        "base_wusong_m", "egm96_undulation_m", "wgs84_ellipsoid_height_m",
        "full_scene_range_pixel", "full_scene_azimuth_line", "crop_col_px", "crop_row_px",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader()
        for index, item in enumerate(points):
            writer.writerow({
                **item, "base_wusong_m": BASE_WUSONG_M,
                "egm96_undulation_m": float(undulation[index]),
                "wgs84_ellipsoid_height_m": float(ellipsoid_height[index]),
                "full_scene_range_pixel": float(range_pixel[index]),
                "full_scene_azimuth_line": float(azimuth[index]),
                "crop_col_px": float(crop_col[index]), "crop_row_px": float(crop_row[index]),
            })

    segments = []
    building_ids_in_crop = set()
    start = 0
    while start < len(points):
        key = (points[start]["fid"], points[start]["part"]); end = start + 1
        while end < len(points) and (points[end]["fid"], points[end]["part"]) == key:
            end += 1
        segment = np.column_stack([crop_col[start:end], crop_row[start:end]])
        segments.append(segment)
        if np.any((segment[:, 0] >= 0) & (segment[:, 0] < 6726) & (segment[:, 1] >= 0) & (segment[:, 1] < 4703)):
            building_ids_in_crop.add(key[0])
        start = end

    amplitude = normalized_amplitude(INPUT / "amplitude_crop.tif")
    rows, cols = amplitude.shape
    fig, ax = plt.subplots(figsize=(11.8, 8.5))
    ax.imshow(amplitude[::2, ::2], cmap="gray", vmin=0, vmax=1, extent=(0, cols, rows, 0), interpolation="none")
    ax.add_collection(LineCollection(segments, colors="#00E5FF", linewidths=0.38, alpha=0.90, rasterized=True))
    ax.set_xlim(0, cols); ax.set_ylim(rows, 0)
    ax.set_title("GAMMA精化投影的建筑底面" if refined else "GAMMA投影的建筑底面", loc="left", fontsize=17, fontweight="bold", pad=12)
    ax.text(
        0.01, 0.985,
        "底面固定为吴淞高程4.000 m；暂以GAMMA EGM96作为吴淞→WGS84椭球高代理；"
        + ("已应用DIFF_par常数残差改正" if refined else "coord_to_sarpix_list直接正投影"),
        transform=ax.transAxes, va="top", color="white", fontsize=9.5,
        bbox={"facecolor": "black", "alpha": 0.74, "edgecolor": "none", "pad": 5},
    )
    ax.set_xlabel("距离向列号 / pixel"); ax.set_ylabel("方位向行号 / pixel")
    fig.tight_layout()
    figure_path = PICALL / ("019_图件_715556404984.png" if refined else "018_图件_239081998462.png")
    fig.savefig(figure_path, dpi=220, bbox_inches="tight"); plt.close(fig)

    summary = {
        "definition": "building footprint at fixed 4.000 m Wusong elevation",
        "gamma_function": "coord_to_sarpix_list",
        "gamma_coordinates": "WGS84 latitude/longitude; WGS84 ellipsoid height",
        "vertical_conversion": "PROVISIONAL: h_WGS84 = 4.000 m Wusong + GAMMA EGM96 undulation",
        "vertical_conversion_limit": "EGM96 is not a surveyed Wusong datum transformation",
        "egm96_undulation_m": {
            "min": float(np.min(undulation)), "median": float(np.median(undulation)), "max": float(np.max(undulation)),
        },
        "ellipsoid_height_m": {
            "min": float(np.min(ellipsoid_height)), "median": float(np.median(ellipsoid_height)), "max": float(np.max(ellipsoid_height)),
        },
        "vertices": len(points), "buildings_in_crop": len(building_ids_in_crop),
        "crop_offset": {"column": X_OFFSET, "row": Y_OFFSET},
        "diff_par_applied": refined,
        "diff_par": str(args.diff_par) if refined else None,
        "sar_brightness_used_to_define_base": False,
        "outputs": {"vertices_csv": str(csv_path), "figure": str(figure_path), "gamma_raw_output": str(sar_path)},
        "command": command,
    }
    summary_path = WORK / f"building_base_gamma_projection_summary{suffix}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
