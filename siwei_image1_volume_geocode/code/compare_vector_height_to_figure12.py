#!/usr/bin/env python3
"""Map the source height field and compare it with Figure 012 building estimates."""

from __future__ import annotations

import base64
import csv
import io
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-siwei-height-comparison")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from PIL import Image, ImageFilter

from gamma_projection_core import BUILDINGS, PICALL, WORK


SIM_WORK = WORK / "gamma_simulated_sar_ellipsoid"
GEOCODE_WORK = WORK / "gamma_geocoded_sar"

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


def load_context():
    parameter = SIM_WORK / "dem_seg.par"
    width = int(par_value(parameter, "width"))
    lines = int(par_value(parameter, "nlines"))
    east = par_value(parameter, "corner_east")
    north = par_value(parameter, "corner_north")
    post_east = par_value(parameter, "post_east")
    post_north = par_value(parameter, "post_north")
    left = east - 0.5 * post_east
    right = left + width * post_east
    top = north - 0.5 * post_north
    bottom = top + lines * post_north
    geocoded = np.fromfile(GEOCODE_WORK / "amplitude_map.gamma", dtype=">f4").reshape(lines, width).astype(np.float32)
    geocoded[geocoded <= 0] = np.nan
    finite = geocoded[np.isfinite(geocoded)]
    low, high = np.percentile(finite, [2.0, 99.7])
    display = np.clip((geocoded - low) / max(float(high - low), 1e-6), 0, 1) ** 0.55
    return display, (left, right, bottom, top)


def map_axes(ax, display, extent, title: str):
    left, right, bottom, top = extent
    ax.imshow(display, cmap="gray", vmin=0, vmax=1, interpolation="none", resample=False,
              extent=(left, right, bottom, top))
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("UTM东向 / m")
    ax.set_ylabel("UTM北向 / m")


def annotate(ax, frame, column: str, fontsize: float = 2.1):
    for _, feature in frame[np.isfinite(frame[column])].iterrows():
        point = feature.geometry.representative_point()
        ax.text(point.x, point.y, f"{feature[column]:.0f}", ha="center", va="center",
                fontsize=fontsize, color="white", fontweight="bold", zorder=5)


def main() -> None:
    display, extent = load_context()
    left, right, bottom, top = extent
    buildings = gpd.read_file(BUILDINGS, engine="pyogrio").to_crs(32651)
    estimates = {int(row["fid"]): float(row["fused_height_m"])
                 for row in csv.DictReader(open(WORK / "building_height_estimates_from_fig09_10.csv", encoding="utf-8"))}
    buildings["height_vector_m"] = np.asarray(buildings["height"], dtype=float)
    buildings["height_estimate_m"] = [estimates.get(int(fid), np.nan) for fid in buildings.index]
    buildings["difference_m"] = buildings.height_estimate_m - buildings.height_vector_m
    buildings = buildings.cx[min(left, right):max(left, right), min(bottom, top):max(bottom, top)].copy()
    paired = buildings[np.isfinite(buildings.height_vector_m) & np.isfinite(buildings.height_estimate_m)].copy()
    source_valid = buildings[np.isfinite(buildings.height_vector_m)].copy()
    common_vmax = float(np.percentile(np.r_[source_valid.height_vector_m, paired.height_estimate_m], 99.0))
    height_norm = Normalize(vmin=0.0, vmax=common_vmax)
    height_cmap = plt.get_cmap("turbo")

    fig, ax = plt.subplots(figsize=(11.2, 10.0))
    map_axes(ax, display, extent, "建筑矢量height字段")
    source_valid.plot(ax=ax, column="height_vector_m", cmap=height_cmap, norm=height_norm,
                      edgecolor="#E5FFFF", linewidth=0.24, alpha=0.70, zorder=3)
    annotate(ax, source_valid, "height_vector_m")
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=height_norm, cmap=height_cmap), ax=ax, fraction=0.030, pad=0.018)
    colorbar.set_label("建筑矢量height字段 / m")
    ax.text(0.01, 0.985, "数字为原始矢量height字段；底图为图011的GAMMA模拟SAR校正后地理编码影像",
            transform=ax.transAxes, va="top", color="white", fontsize=9.5,
            bbox={"facecolor": "black", "alpha": 0.74, "edgecolor": "none", "pad": 5})
    fig.tight_layout()
    figure13 = PICALL / "013_图件_245150484619.svg"
    fig.savefig(figure13, format="svg", bbox_inches="tight")
    plt.close(fig)
    compact_svg(figure13)

    vector = paired.height_vector_m.to_numpy()
    estimate = paired.height_estimate_m.to_numpy()
    difference = estimate - vector
    correlation = float(np.corrcoef(vector, estimate)[0, 1])
    mae = float(np.mean(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    bias = float(np.mean(difference))
    diff_limit = float(max(5.0, np.percentile(np.abs(difference), 98.0)))
    diff_norm = TwoSlopeNorm(vmin=-diff_limit, vcenter=0.0, vmax=diff_limit)

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.1))
    for ax, column, title in [
        (axes[0], "height_vector_m", "a  建筑矢量height字段"),
        (axes[1], "height_estimate_m", "b  图12融合高度估计"),
    ]:
        map_axes(ax, display, extent, title)
        paired.plot(ax=ax, column=column, cmap=height_cmap, norm=height_norm,
                    edgecolor="#E5FFFF", linewidth=0.20, alpha=0.72, zorder=3)
    map_axes(axes[2], display, extent, "c  图12估计 − height字段")
    paired.plot(ax=axes[2], column="difference_m", cmap="coolwarm", norm=diff_norm,
                edgecolor="#F3F4F6", linewidth=0.20, alpha=0.76, zorder=3)
    height_bar = fig.colorbar(plt.cm.ScalarMappable(norm=height_norm, cmap=height_cmap), ax=axes[:2], fraction=0.020, pad=0.012)
    height_bar.set_label("建筑高度 / m（a、b统一色标）")
    diff_bar = fig.colorbar(plt.cm.ScalarMappable(norm=diff_norm, cmap="coolwarm"), ax=axes[2], fraction=0.045, pad=0.018)
    diff_bar.set_label("高度差 / m")
    fig.suptitle("建筑矢量height字段与图12高度估计对比", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.925,
             f"共同有效建筑 {len(paired):,}栋｜相关系数 r={correlation:.3f}｜偏差={bias:+.2f} m｜MAE={mae:.2f} m｜RMSE={rmse:.2f} m",
             ha="center", va="top", fontsize=10)
    fig.subplots_adjust(left=0.045, right=0.965, bottom=0.10, top=0.88, wspace=0.18)
    figure14 = PICALL / "014_图件_233072718418.svg"
    fig.savefig(figure14, format="svg", bbox_inches="tight")
    plt.close(fig)
    compact_svg(figure14)

    comparison_path = WORK / "building_height_vector_vs_figure12.csv"
    paired[["height_vector_m", "height_estimate_m", "difference_m", "geometry"]].drop(columns="geometry").to_csv(
        comparison_path, index_label="fid", encoding="utf-8"
    )
    print({
        "figure13": str(figure13), "figure14": str(figure14), "comparison": str(comparison_path),
        "source_height_buildings": len(source_valid), "paired_buildings": len(paired),
        "correlation": correlation, "bias_m": bias, "mae_m": mae, "rmse_m": rmse,
        "difference_p90_abs_m": float(np.percentile(np.abs(difference), 90)),
    })


if __name__ == "__main__":
    main()
