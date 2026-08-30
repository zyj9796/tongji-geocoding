#!/usr/bin/env python3
"""Plot SAR point elevations and roof/base pixel-offset height relations."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-siwei-long-contour-output")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

from gamma_projection_core import BASE_WUSONG_M, INPUT, PICALL, WORK
from plot_refined_ruler_height_figure7_style import compact_svg, display_image


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["AR PL UKai CN", "Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans"],
    "font.size": 9, "svg.fonttype": "none", "axes.unicode_minus": False,
})


def raster(path: Path) -> np.ndarray:
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None: raise FileNotFoundError(path)
    return dataset.ReadAsArray()


def linear_fit(height: np.ndarray, offset: np.ndarray) -> dict:
    design = np.column_stack([height, np.ones(len(height))])
    slope, intercept = np.linalg.lstsq(design, offset, rcond=None)[0]
    fitted = slope * height + intercept
    r2 = 1.0 - float(np.sum((offset - fitted) ** 2) / np.sum((offset - np.mean(offset)) ** 2))
    return {"slope_px_per_m": float(slope), "intercept_px": float(intercept), "r_squared": r2}


def main() -> None:
    output = WORK / "long_contour_ruler"
    amplitude = raster(INPUT / "amplitude_crop.tif")
    elevation = raster(output / "selected_sar_building_point_wusong_elevation_m.tif").astype(np.float32)
    surface = raster(output / "selected_sar_building_point_surface_class.tif")
    rows, cols = amplitude.shape

    fig, ax = plt.subplots(figsize=(12.0, 8.5))
    ax.imshow(display_image(amplitude[::2, ::2]), cmap="gray", vmin=0, vmax=1, interpolation="none",
              resample=False, extent=(0, cols, rows, 0))
    preview = elevation[::2, ::2]
    shown = np.ma.masked_where(~np.isfinite(preview), preview)
    vmax = float(np.nanpercentile(elevation, 99.5))
    image = ax.imshow(shown, cmap="turbo", vmin=BASE_WUSONG_M, vmax=vmax, alpha=.86,
                      interpolation="none", resample=False, extent=(0, cols, rows, 0))
    wall_count = int(np.count_nonzero(surface == 1)); roof_count = int(np.count_nonzero(surface == 2))
    ax.set_xlim(0, cols); ax.set_ylim(rows, 0)
    ax.set_title("长尺顶面轮廓匹配的SAR建筑点高程", loc="left", fontsize=17, fontweight="bold", pad=13)
    ax.text(.01, .985, "长尺走廊筛选建筑点；估计屋顶由GAMMA直接正投影；屋顶恒高，墙面按三角面重心坐标线性赋高",
            transform=ax.transAxes, va="top", color="white", fontsize=10,
            bbox={"facecolor":"black", "alpha":.72, "edgecolor":"none", "pad":5})
    ax.text(.99, .985, f"墙面 {wall_count:,}｜屋顶 {roof_count:,}", transform=ax.transAxes, ha="right", va="top",
            color="white", fontsize=9, bbox={"facecolor":"black", "alpha":.72, "edgecolor":"none", "pad":5})
    colorbar = fig.colorbar(image, ax=ax, fraction=.028, pad=.018)
    colorbar.set_label("绝对高程 / m（吴淞高程，统一4 m底面）")
    ax.set_xlabel("距离向列号 / pixel"); ax.set_ylabel("方位向行号 / pixel"); fig.tight_layout()
    sar_svg = PICALL / "023_图件_514775775242.svg"
    sar_png = PICALL / "023_图件_514775775242.png"
    fig.savefig(sar_svg, format="svg", bbox_inches="tight"); fig.savefig(sar_png, dpi=220, bbox_inches="tight")
    plt.close(fig); compact_svg(sar_svg)

    records = [row for row in csv.DictReader((output / "long_contour_ruler_height_estimates.csv").open())
               if row["height_estimate_m"].lower() != "nan"]
    height = np.asarray([float(row["height_estimate_m"]) for row in records])
    col = np.asarray([float(row["roof_offset_col_px"]) for row in records])
    row = np.asarray([float(item["roof_offset_row_px"]) for item in records])
    magnitude = np.hypot(col, row)
    fits = {"range_column": linear_fit(height, col), "azimuth_row": linear_fit(height, row),
            "offset_magnitude": linear_fit(height, magnitude)}
    relation_path = output / "pixel_offset_height_relation.json"
    relation_path.write_text(json.dumps({
        "definition": "roof centroid pixel offset relative to refined 4 m base centroid",
        "height_unit": "m", "pixel_unit": "pixel", "fits": fits,
        "median_m_per_pixel": float(np.median(height / magnitude)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)
    panels = [(col, "距离向偏移 / pixel", "range_column"), (row, "方位向偏移 / pixel", "azimuth_row"),
              (magnitude, "偏移模长 / pixel", "offset_magnitude")]
    order = np.argsort(height); xline = height[order]
    for ax, (values, ylabel, key) in zip(axes, panels):
        ax.scatter(height, values, s=7, alpha=.32, color="#2563EB", edgecolors="none")
        fit = fits[key]; ax.plot(xline, fit["slope_px_per_m"] * xline + fit["intercept_px"], color="#DC2626", lw=1.5)
        ax.text(.04, .96, f"斜率 {fit['slope_px_per_m']:.6f} pixel/m\n$R^2$ = {fit['r_squared']:.7f}",
                transform=ax.transAxes, va="top", bbox={"facecolor":"white", "alpha":.86, "edgecolor":"#CBD5E1"})
        ax.set_xlabel("建筑高度估计 / m"); ax.set_ylabel(ylabel); ax.grid(alpha=.18)
    fig.suptitle("屋顶相对4 m底面的像素偏移—高程变化关系", fontsize=16, fontweight="bold")
    relation_png = PICALL / "024_屋顶相对底面像素偏移与高度关系.png"
    relation_svg = PICALL / "024_屋顶相对底面像素偏移与高度关系.svg"
    fig.savefig(relation_png, dpi=220, bbox_inches="tight"); fig.savefig(relation_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    print({"sar_png": str(sar_png), "sar_svg": str(sar_svg), "relation_png": str(relation_png),
           "relation_svg": str(relation_svg), "relation_json": str(relation_path), "fits": fits})


if __name__ == "__main__":
    main()
