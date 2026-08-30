#!/usr/bin/env python3
"""Audit and visualize the physical building-ground projection in the SAR crop.

Run with /usr/bin/python3 on this workstation.  It intentionally uses GDAL/OGR
instead of the plotting workflow's optional GeoPandas stack.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-siwei-base-audit")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from osgeo import gdal, ogr

from gamma_projection_core import (
    BASE_WUSONG_M, BUILDINGS, INPUT, PICALL, PROJECTION_DATUM_VERSION,
    WORK, wusong_to_ellipsoid,
)
from rpc_projection_core import RPCModel

plt.rcParams.update({
    "font.family": "AR PL UKai CN",
    "axes.unicode_minus": False,
})


# Recomputed against gamma_simulated_sar_ellipsoid at factors 2 and 4.  This is
# a residual image-registration diagnostic, not part of the vertical datum
# conversion and not evidence that a bright layover edge is the wall foot.
RESIDUAL_COL_SHIFT_PX = 0.0
RESIDUAL_ROW_SHIFT_PX = 0.0
LEGACY_COL_SHIFT_PX = -42.5
LEGACY_ROW_SHIFT_PX = -18.5


def exterior_rings(geometry: ogr.Geometry):
    if geometry.GetGeometryName() == "POLYGON":
        yield geometry.GetGeometryRef(0)
    elif geometry.GetGeometryName() == "MULTIPOLYGON":
        for index in range(geometry.GetGeometryCount()):
            yield geometry.GetGeometryRef(index).GetGeometryRef(0)


def normalized_amplitude(path: Path) -> np.ndarray:
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(path)
    image = dataset.ReadAsArray().astype(np.float32)
    positive = image[image > 0]
    low, high = np.percentile(positive, [2.0, 99.7])
    return np.clip((image - low) / max(float(high - low), 1e-6), 0, 1) ** 0.55


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    PICALL.mkdir(parents=True, exist_ok=True)
    rpc = RPCModel.read()
    source = ogr.Open(str(BUILDINGS), 0)
    if source is None:
        raise FileNotFoundError(BUILDINGS)
    layer = source.GetLayer(0)
    wrong_segments, physical_segments, final_segments = [], [], []
    records = []
    datum_displacements = []
    final_vs_legacy = []
    for fid, feature in enumerate(layer):
        geometry = feature.GetGeometryRef()
        all_wrong, all_physical, all_final = [], [], []
        for ring in exterior_rings(geometry):
            points = np.asarray([ring.GetPoint(i)[:2] for i in range(ring.GetPointCount())], dtype=np.float64)
            if len(points) < 4:
                continue
            wrong_col, wrong_row = rpc.project(points[:, 1], points[:, 0], BASE_WUSONG_M)
            base_height = wusong_to_ellipsoid(BASE_WUSONG_M, points[:, 1], points[:, 0])
            base_col, base_row = rpc.project(points[:, 1], points[:, 0], base_height)
            final_col = base_col + RESIDUAL_COL_SHIFT_PX
            final_row = base_row + RESIDUAL_ROW_SHIFT_PX
            wrong = np.column_stack([wrong_col, wrong_row])
            physical = np.column_stack([base_col, base_row])
            final = np.column_stack([final_col, final_row])
            wrong_segments.append(wrong); physical_segments.append(physical); final_segments.append(final)
            all_wrong.append(wrong); all_physical.append(physical); all_final.append(final)
            datum_displacements.extend(np.linalg.norm(physical - wrong, axis=1).tolist())
            legacy = wrong + np.asarray([LEGACY_COL_SHIFT_PX, LEGACY_ROW_SHIFT_PX])
            final_vs_legacy.extend(np.linalg.norm(final - legacy, axis=1).tolist())
        if not all_final:
            continue
        wrong = np.vstack(all_wrong); physical = np.vstack(all_physical); final = np.vstack(all_final)
        records.append({
            "fid": fid,
            "clean_id": feature.GetField("clean_id"),
            "base_wusong_m": BASE_WUSONG_M,
            "base_ellipsoid_m": float(np.mean(base_height)),
            "vertical_datum_offset_m": float(np.mean(base_height - BASE_WUSONG_M)),
            "rpc_physical_base_col_px": float(np.mean(physical[:, 0])),
            "rpc_physical_base_row_px": float(np.mean(physical[:, 1])),
            "residual_col_shift_px": RESIDUAL_COL_SHIFT_PX,
            "residual_row_shift_px": RESIDUAL_ROW_SHIFT_PX,
            "final_base_col_px": float(np.mean(final[:, 0])),
            "final_base_row_px": float(np.mean(final[:, 1])),
            "wrong_datum_base_col_px": float(np.mean(wrong[:, 0])),
            "wrong_datum_base_row_px": float(np.mean(wrong[:, 1])),
        })

    csv_path = WORK / "building_base_projection_rpc_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)

    amplitude = normalized_amplitude(INPUT / "amplitude_crop.tif")
    rows, cols = amplitude.shape
    preview = amplitude[::2, ::2]
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.2), sharex=True, sharey=True)
    panels = [
        (axes[0], wrong_segments, "错误基准：4 m 被当成椭球高", "#FF4D4D"),
        (axes[1], final_segments, "暂行物理底面：4 m + GAMMA EGM96", "#00E5FF"),
    ]
    for ax, segments, title, color in panels:
        ax.imshow(preview, cmap="gray", vmin=0, vmax=1, extent=(0, cols, rows, 0), interpolation="none")
        ax.add_collection(LineCollection(segments, colors=color, linewidths=0.30, alpha=0.82, rasterized=True))
        ax.set_xlim(0, cols); ax.set_ylim(rows, 0); ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_xlabel("距离向列号 / pixel")
    axes[0].set_ylabel("方位向行号 / pixel")
    fig.suptitle("建筑底面在四维 SAR 影像上的 RPC 投影审计", fontsize=16, fontweight="bold")
    fig.text(
        0.5, 0.025,
        "底面由建筑地理轮廓和地面高程定义；SAR亮线仅作验收。\n"
        "红色未做垂直基准转换；青色按GAMMA EGM96暂行转换为WGS84椭球高。",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.105, 1, 0.95))
    png_path = PICALL / "017_图件_859104265266.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight"); plt.close(fig)

    summary = {
        "projection_datum_version": PROJECTION_DATUM_VERSION,
        "definition": "building footprint at local ground elevation, forward-projected by vendor RPC",
        "base_wusong_m": BASE_WUSONG_M,
        "base_wgs84_ellipsoid_m": "spatially varying: see CSV",
        "wusong_to_ellipsoid": "PROVISIONAL GAMMA EGM96 proxy",
        "residual_registration_px": {"column": RESIDUAL_COL_SHIFT_PX, "row": RESIDUAL_ROW_SHIFT_PX},
        "legacy_registration_px": {"column": LEGACY_COL_SHIFT_PX, "row": LEGACY_ROW_SHIFT_PX},
        "datum_error_displacement_px": {
            "median": float(np.median(datum_displacements)),
            "p90": float(np.percentile(datum_displacements, 90)),
        },
        "new_final_vs_legacy_final_px": {
            "median": float(np.median(final_vs_legacy)),
            "p90": float(np.percentile(final_vs_legacy, 90)),
        },
        "buildings": len(records),
        "csv": str(csv_path), "figure": str(png_path),
        "important_limit": "Absolute confirmation still requires surveyed ground targets/corner reflectors; a layover bright edge is not a ground control point.",
    }
    json_path = WORK / "building_base_projection_rpc_audit_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
