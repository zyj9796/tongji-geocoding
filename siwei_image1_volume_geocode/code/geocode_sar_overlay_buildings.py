#!/usr/bin/env python3
"""GAMMA-geocode the cropped SAR amplitude and overlay source building vectors."""

from __future__ import annotations

import base64
import io
import os
import re
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-siwei-geocoded")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image, ImageFilter

from gamma_projection_core import BUILDINGS, GAMMA_LIB, INPUT, PICALL, WORK, X_OFFSET, Y_OFFSET
from gamma_simulated_sar_registration import evaluate_spatial_shift, simulated_sar_spatial_registration


SIM_WORK = WORK / "gamma_simulated_sar_ellipsoid"
GEOCODE_WORK = WORK / "gamma_geocoded_sar"
GAMMA_GEOCODE_BACK = Path("/usr/local/GAMMA/DIFF/bin/geocode_back")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "svg.fonttype": "none",
})


def par_value(path: Path, key: str, cast=float):
    match = re.search(rf"^{re.escape(key)}:\s+([^\s]+)", path.read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise ValueError(f"GAMMA参数缺失: {key}")
    return cast(float(match.group(1))) if cast is int else cast(match.group(1))


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


def main() -> None:
    GEOCODE_WORK.mkdir(parents=True, exist_ok=True)
    dem_par = SIM_WORK / "dem_seg.par"
    map_width = par_value(dem_par, "width", int)
    map_lines = par_value(dem_par, "nlines", int)
    corner_east = par_value(dem_par, "corner_east")
    corner_north = par_value(dem_par, "corner_north")
    post_east = par_value(dem_par, "post_east")
    post_north = par_value(dem_par, "post_north")
    lut = np.fromfile(SIM_WORK / "image1.lt", dtype=">f4").reshape(map_lines, map_width, 2).astype(np.float32)
    with rasterio.open(INPUT / "amplitude_crop.tif") as source:
        amplitude = source.read(1).astype(np.float32)
    registration, _ = simulated_sar_spatial_registration(amplitude)
    # gc_map2 maps each map point to the uncorrected radar coordinate. The
    # simulated-to-observed registration shift must therefore be added to the
    # lookup coordinates before sampling the observed cropped amplitude.
    crop_col = lut[:, :, 0] - X_OFFSET
    crop_row = lut[:, :, 1] - Y_OFFSET
    col_shift, row_shift = evaluate_spatial_shift(registration, crop_col, crop_row, amplitude.shape)
    lut[:, :, 0] = crop_col + col_shift
    lut[:, :, 1] = crop_row + row_shift
    valid = np.isfinite(lut).all(axis=2)
    valid &= (lut[:, :, 0] >= 0) & (lut[:, :, 0] < amplitude.shape[1] - 1)
    valid &= (lut[:, :, 1] >= 0) & (lut[:, :, 1] < amplitude.shape[0] - 1)
    lut[~valid] = -9999.0
    amplitude_path = GEOCODE_WORK / "amplitude_crop.gamma"
    lut_path = GEOCODE_WORK / "image1_crop.lt"
    output_path = GEOCODE_WORK / "amplitude_map.gamma"
    amplitude.astype(">f4").tofile(amplitude_path)
    lut.astype(">f4").tofile(lut_path)
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(GAMMA_LIB)
    subprocess.run([
        str(GAMMA_GEOCODE_BACK), str(amplitude_path), str(amplitude.shape[1]), str(lut_path),
        str(output_path), str(map_width), str(map_lines), "3", "0", "1", "1", "5", "0",
    ], check=True, env=environment)
    geocoded = np.fromfile(output_path, dtype=">f4").reshape(map_lines, map_width).astype(np.float32)
    geocoded[~valid] = np.nan
    finite = geocoded[np.isfinite(geocoded) & (geocoded > 0)]
    low, high = np.percentile(finite, [2.0, 99.7])
    display = np.clip((geocoded - low) / max(float(high - low), 1e-6), 0, 1) ** 0.55
    left = corner_east - 0.5 * post_east
    right = left + map_width * post_east
    top = corner_north - 0.5 * post_north
    bottom = top + map_lines * post_north
    buildings = gpd.read_file(BUILDINGS, engine="pyogrio").to_crs(32651)
    buildings = buildings.cx[min(left, right):max(left, right), min(bottom, top):max(bottom, top)]

    fig, ax = plt.subplots(figsize=(10.0, 10.0))
    ax.imshow(display, cmap="gray", vmin=0, vmax=1, interpolation="none", resample=False,
              extent=(left, right, bottom, top))
    buildings.boundary.plot(ax=ax, color="#00E5FF", linewidth=0.42, alpha=0.92, zorder=3)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.set_title("偏差校正后的GAMMA地理编码SAR与建筑轮廓叠加", loc="left", fontsize=17, fontweight="bold", pad=13)
    ax.text(0.01, 0.985, f"GAMMA模拟/实测SAR分块配准：{registration['accepted_windows']}/{registration['total_windows']}个可靠窗口拟合缓变LUT校正场；UTM 51N，1 m",
            transform=ax.transAxes, va="top", color="white", fontsize=10,
            bbox={"facecolor": "black", "alpha": 0.72, "edgecolor": "none", "pad": 5})
    ax.text(0.99, 0.985, f"叠加建筑 {len(buildings):,}栋", transform=ax.transAxes, ha="right", va="top",
            color="white", fontsize=9, bbox={"facecolor": "black", "alpha": 0.72, "edgecolor": "none", "pad": 5})
    ax.set_xlabel("UTM东向坐标 / m")
    ax.set_ylabel("UTM北向坐标 / m")
    fig.tight_layout()
    output = PICALL / "011_图件_440218767479.svg"
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)
    compact_svg(output)
    print({
        "output": str(output), "map_shape": [map_lines, map_width], "valid_map_pixels": int(valid.sum()),
        "utm_extent": [left, bottom, right, top], "buildings": int(len(buildings)),
        "gamma_command": "geocode_back", "interpolation": "bicubic_sqrt",
        "lut_col_shift_range_px": [float(np.nanmin(col_shift)), float(np.nanmax(col_shift))],
        "lut_row_shift_range_px": [float(np.nanmin(row_shift)), float(np.nanmax(row_shift))],
        "accepted_registration_windows": registration["accepted_windows"],
        "fit_residual_p90_px": registration["fit_residual_p90_px"],
        "registration_cross_scale_disagreement_px": registration["cross_scale_disagreement_px"],
    })


if __name__ == "__main__":
    main()
