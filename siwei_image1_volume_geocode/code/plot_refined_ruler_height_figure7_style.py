#!/usr/bin/env python3
"""Plot refined GAMMA ruler pixel heights using the original Figure 7 style."""

from __future__ import annotations

import base64
import io
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-siwei-figure7-style")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from PIL import Image, ImageFilter
from scipy.ndimage import binary_closing, binary_dilation, label

from gamma_projection_core import BASE_WUSONG_M, INPUT, PICALL, WORK


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["AR PL UKai CN", "Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans"],
    "font.size": 9,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})


def read_array(path: Path) -> np.ndarray:
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(path)
    return dataset.ReadAsArray()


def display_image(amplitude: np.ndarray) -> np.ndarray:
    amplitude = np.maximum(amplitude.astype(np.float32), 0)
    valid = amplitude[amplitude > 0]
    lo, hi = np.percentile(valid, [2.0, 99.7])
    return np.clip((amplitude - lo) / max(float(hi - lo), 1e-6), 0, 1) ** 0.55


def compact_svg(path: Path) -> None:
    """Use the same embedded-background compression convention as Figure 7."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r'data:image/png;base64,\s*([^\"]+)', text)
    if not match:
        return
    image = Image.open(io.BytesIO(base64.b64decode(match.group(1)))).convert("L")
    if image.width > 3200:
        lanczos = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        image = image.resize((3200, round(image.height * 3200 / image.width)), lanczos)
        image = image.filter(ImageFilter.UnsharpMask(radius=0.8, percent=125, threshold=2))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=88, subsampling=0, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    path.write_text(text[:match.start()] + "data:image/jpeg;base64,\n" + encoded + text[match.end():], encoding="utf-8")


def refine_mask(amplitude: np.ndarray, building_id: np.ndarray, kappa: float = 0.20) -> np.ndarray:
    """Reproduce Figure 7's per-building SAR-amplitude mask refinement."""
    refined = np.zeros(building_id.shape, dtype=bool)
    sar_amplitude = np.maximum(amplitude, 0).astype(np.float32, copy=False)
    ids = np.unique(building_id)
    ids = ids[ids > 0]
    for count, value in enumerate(ids, 1):
        rr, cc = np.where(building_id == value)
        r0, r1 = max(0, int(rr.min()) - 3), min(building_id.shape[0], int(rr.max()) + 4)
        c0, c1 = max(0, int(cc.min()) - 3), min(building_id.shape[1], int(cc.max()) + 4)
        geometry = building_id[r0:r1, c0:c1] == value
        window = binary_dilation(geometry, iterations=3)
        values = sar_amplitude[r0:r1, c0:c1][window]
        threshold = float(values.mean() + kappa * values.std())
        candidate = geometry & (sar_amplitude[r0:r1, c0:c1] > threshold)
        candidate = binary_closing(candidate, iterations=1)
        components, component_count = label(candidate)
        if component_count:
            sizes = np.bincount(components.ravel())
            keep = np.where(sizes >= 4)[0]
            keep = keep[keep != 0]
            candidate = np.isin(components, keep)
        refined[r0:r1, c0:c1] |= candidate & geometry
        if count % 100 == 0:
            print(f"图7式SAR强度精炼 {count}/{len(ids)}", flush=True)
    return refined


def write_tiff(path: Path, array: np.ndarray, source, nodata, description: str) -> None:
    is_float = array.dtype.kind == "f"
    data_type = gdal.GDT_Float32 if is_float else gdal.GDT_Byte
    options = ["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"]
    if is_float:
        options.append("PREDICTOR=3")
    target = gdal.GetDriverByName("GTiff").Create(
        str(path), array.shape[1], array.shape[0], 1, data_type, options=options,
    )
    target.SetGeoTransform(source.GetGeoTransform()); target.SetProjection(source.GetProjection())
    band = target.GetRasterBand(1); band.SetNoDataValue(nodata); band.SetDescription(description)
    band.WriteArray(array); band.FlushCache(); target.FlushCache(); target = None


def main() -> None:
    result = WORK / "refined_ruler_pixel_heights"
    amplitude_source = gdal.Open(str(INPUT / "amplitude_crop.tif"), gdal.GA_ReadOnly)
    if amplitude_source is None:
        raise FileNotFoundError(INPUT / "amplitude_crop.tif")
    amplitude = amplitude_source.ReadAsArray()
    elevation = read_array(result / "building_surface_wusong_elevation_m.tif").astype(np.float32)
    surface = read_array(result / "surface_class.tif")
    building_id = read_array(result / "building_fid_plus_one.tif")
    rows, cols = amplitude.shape

    # Figure 7 displays only SAR-intensity-refined pixels, rather than the
    # complete projected building volume (the latter corresponds to Figure 8).
    refined = refine_mask(amplitude, building_id, kappa=0.20)
    refined &= np.isfinite(elevation)
    elevation = np.where(refined, elevation, np.nan).astype(np.float32)
    surface = np.where(refined, surface, 0).astype(np.uint8)
    write_tiff(result / "sar_intensity_refined_mask_figure7.tif", refined.astype(np.uint8), amplitude_source, 0,
               "Figure 7 mask: per-building SAR amplitude > mean + 0.20 std; closing; components >= 4 pixels")
    write_tiff(result / "building_surface_wusong_elevation_figure7_m.tif", elevation, amplitude_source, np.nan,
               "Figure 7 refined-mask building surface elevation, m Wusong")

    fig, ax = plt.subplots(figsize=(12.0, 8.5))
    ax.imshow(
        display_image(amplitude[::2, ::2]), cmap="gray", vmin=0, vmax=1,
        interpolation="none", resample=False, extent=(0, cols, rows, 0),
    )
    preview = elevation[::2, ::2]
    displayed = np.ma.masked_where(~np.isfinite(preview), preview)
    vmax = float(np.nanpercentile(elevation, 99.5))
    image = ax.imshow(
        displayed, cmap="turbo", vmin=BASE_WUSONG_M, vmax=vmax, alpha=0.84,
        interpolation="none", resample=False, extent=(0, cols, rows, 0),
    )
    ax.set_xlim(0, cols); ax.set_ylim(rows, 0)
    ax.set_title("建筑体三角面重心插值SAR像素高程", loc="left", fontsize=17, fontweight="bold", pad=13)
    ax.text(
        0.01, 0.985,
        "GAMMA精化4 m吴淞高程底面；尺子法判定顶面；顶面恒高，墙面按底—顶三角面重心坐标线性插值",
        transform=ax.transAxes, va="top", color="white", fontsize=10,
        bbox={"facecolor": "black", "alpha": 0.72, "edgecolor": "none", "pad": 5},
    )
    wall_pixels = int(np.count_nonzero(surface == 1))
    roof_pixels = int(np.count_nonzero(surface == 2))
    ax.text(
        0.99, 0.985,
        f"墙面 {wall_pixels:,}｜屋顶 {roof_pixels:,}",
        transform=ax.transAxes, ha="right", va="top", color="white", fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.72, "edgecolor": "none", "pad": 5},
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.018)
    colorbar.set_label("绝对高程 / m（统一4 m底面基准）")
    ax.set_xlabel("距离向列号 / pixel")
    ax.set_ylabel("方位向行号 / pixel")
    fig.tight_layout()

    svg_path = PICALL / "021_图件_785283998164.svg"
    png_path = PICALL / "021_图件_785283998164.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    compact_svg(svg_path)
    print({
        "svg": str(svg_path), "png": str(png_path),
        "wall_pixels": wall_pixels, "roof_pixels": roof_pixels,
        "refined_pixels": int(refined.sum()), "colorbar_min_m": BASE_WUSONG_M,
        "colorbar_max_m": vmax,
    })


if __name__ == "__main__":
    main()
