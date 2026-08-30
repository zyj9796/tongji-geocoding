from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fig-5-9-5-10")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import zoom
from shapely.geometry import Polygon
from shapely.ops import unary_union

from chinese_matplotlib import install_chinese_labels


PROJECT_DIR = Path(__file__).resolve().parents[1]
RSLC_DIR = PROJECT_DIR / "data" / "RE_SLAVES"


DATE = "20200708"
TRIANGLES = (
    PROJECT_DIR
    / "results"
    / "outputs"
    / "strict_local_registration"
    / f"{DATE}_locally_registered_strict_sar_surface_triangles.geojson"
)
POINTS = (
    PROJECT_DIR
    / "results"
    / "outputs"
    / "registered_full_area_geocode"
    / f"{DATE}_all_buildings_method_vs_gamma_points.csv"
)
OUT_DIR = PROJECT_DIR / "results" / "picall" / "注册复现" / "雷达重采样与论文图件"

# The labels follow the vertical/left-to-right arrangement of the thesis figures.
NON_OVERLAP = [(267, "建筑一"), (268, "建筑二"), (285, "建筑三"), (287, "建筑四")]
LAYOVER = [(556, "建筑十一"), (559, "建筑八"), (557, "建筑二")]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def parse_radar_grid(path: Path) -> dict[str, float | int]:
    values: dict[str, float | int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        if key.strip() in {"range_samples", "azimuth_lines"}:
            values[key.strip()] = int(float(rest.split()[0]))
        elif key.strip() in {"range_pixel_spacing", "azimuth_pixel_spacing"}:
            values[key.strip()] = float(rest.split()[0])
    required = {"range_samples", "azimuth_lines", "range_pixel_spacing", "azimuth_pixel_spacing"}
    missing = required.difference(values)
    if missing:
        raise ValueError(f"Missing radar-grid fields: {sorted(missing)}")
    return values


def read_amplitude(path: Path, rows: int, cols: int) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=">i2")
    expected = rows * cols * 2
    if raw.size != expected:
        raise ValueError(f"Unexpected RSLC size: got {raw.size}, expected {expected} int16 values")
    complex_parts = raw.reshape(rows, cols, 2).astype(np.float32)
    return np.hypot(complex_parts[:, :, 0], complex_parts[:, :, 1]).astype(np.float32)


def resample_amplitude_to_one_meter(
    amplitude: np.ndarray, *, azimuth_spacing_m: float, range_spacing_m: float
) -> np.ndarray:
    # Original axes are (azimuth row, range column).  Multiplying image indices
    # by the native spacing maps every overlay and the raster to the same 1 m grid.
    return zoom(
        amplitude.astype(np.float32),
        zoom=(azimuth_spacing_m, range_spacing_m),
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32)


def percentile_preview(amplitude: np.ndarray) -> np.ndarray:
    positive = amplitude[np.isfinite(amplitude) & (amplitude > 0)]
    lo, hi = np.percentile(positive, (2.0, 98.0)) if positive.size else (0.0, 1.0)
    scaled = np.clip((amplitude - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    return np.round(255.0 * scaled**0.65).astype(np.uint8)


def save_resampled_scene(amplitude: np.ndarray, metadata: dict) -> dict[str, str]:
    tif_path = OUT_DIR / f"{DATE}_图件_621563374170.tif"
    png_path = OUT_DIR / f"{DATE}_图件_444307131627.png"
    meta_path = OUT_DIR / f"{DATE}_一米重采样元数据.json"
    tifffile.imwrite(
        tif_path,
        amplitude.astype(np.float32),
        compression="deflate",
        metadata={
            "axes": "YX",
            "coordinate_system": "radar range/azimuth grid",
            "range_pixel_spacing_m": 1.0,
            "azimuth_pixel_spacing_m": 1.0,
            "value": "linear SAR amplitude",
        },
    )
    Image.fromarray(percentile_preview(amplitude), mode="L").save(png_path, optimize=True)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"float32_tiff": str(tif_path), "preview_png": str(png_path), "metadata_json": str(meta_path)}


def stretch_with_limits(amplitude: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scaled = np.clip((amplitude - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    return scaled**0.65


def plot_native_vs_resampled(
    native: np.ndarray,
    resampled: np.ndarray,
    *,
    range_spacing_m: float,
    azimuth_spacing_m: float,
) -> Path:
    positive = native[np.isfinite(native) & (native > 0)]
    lo, hi = np.percentile(positive, (2.0, 98.0)) if positive.size else (0.0, 1.0)
    native_display = stretch_with_limits(native, float(lo), float(hi))
    resampled_display = stretch_with_limits(resampled, float(lo), float(hi))
    physical_width_m = float(native.shape[1]) * range_spacing_m
    physical_height_m = float(native.shape[0]) * azimuth_spacing_m
    extent = [0.0, physical_width_m, physical_height_m, 0.0]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 7.0), constrained_layout=True, sharex=True, sharey=True)
    axes[0].imshow(native_display, cmap="gray", extent=extent, interpolation="nearest", aspect="equal")
    axes[1].imshow(resampled_display, cmap="gray", extent=extent, interpolation="nearest", aspect="equal")
    axes[0].set_title(
        f"(a) Native SAR\n{range_spacing_m:.3f} m × {azimuth_spacing_m:.3f} m; "
        f"{native.shape[1]} × {native.shape[0]}",
        fontsize=10,
    )
    axes[1].set_title(
        f"(b) Resampled SAR\n1.000 m × 1.000 m; {resampled.shape[1]} × {resampled.shape[0]}",
        fontsize=10,
    )
    for ax in axes:
        ax.set_xlabel("Range distance / m")
        ax.set_ylabel("Azimuth distance / m")
        ax.tick_params(width=0.7, length=3)
    out_path = OUT_DIR / f"{DATE}_图件_43812585258.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def load_triangles(*, range_spacing_m: float, azimuth_spacing_m: float) -> dict[int, dict[str, list[Polygon]]]:
    payload = json.loads(TRIANGLES.read_text(encoding="utf-8"))
    grouped: dict[int, dict[str, list[Polygon]]] = defaultdict(lambda: defaultdict(list))
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        fid = int(props["fid"])
        surface = str(props.get("surface", "wall"))
        xy = np.asarray(feature["geometry"]["coordinates"][0], dtype=np.float64)
        xy[:, 0] *= range_spacing_m
        xy[:, 1] *= azimuth_spacing_m
        polygon = Polygon(xy[:, :2]).buffer(0)
        if not polygon.is_empty:
            grouped[fid][surface].append(polygon)
    return grouped


def load_points(*, range_spacing_m: float, azimuth_spacing_m: float) -> dict[int, np.ndarray]:
    grouped: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    with POINTS.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[int(row["fid"])].append(
                (
                    float(row["col"]) * range_spacing_m,
                    float(row["row"]) * azimuth_spacing_m,
                    float(row["method_height_m"]),
                )
            )
    return {fid: np.asarray(values, dtype=np.float64) for fid, values in grouped.items()}


def union_surface(surface_polygons: dict[str, list[Polygon]], surface: str):
    polygons = surface_polygons.get(surface, [])
    return unary_union(polygons) if polygons else None


def all_union(surface_polygons: dict[str, list[Polygon]]):
    return unary_union([polygon for values in surface_polygons.values() for polygon in values])


def polygon_rings(geometry):
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield np.asarray(geometry.exterior.coords, dtype=np.float64)
    elif geometry.geom_type == "MultiPolygon":
        for polygon in geometry.geoms:
            yield np.asarray(polygon.exterior.coords, dtype=np.float64)


def draw_outline(ax: plt.Axes, geometry, *, roof: bool, label: str | None) -> None:
    first = True
    for ring in polygon_rings(geometry):
        ax.plot(
            ring[:, 0],
            ring[:, 1],
            color="#FFE66D" if roof else "#00D5FF",
            linewidth=1.6 if roof else 1.25,
            linestyle="-" if roof else "--",
            alpha=0.96 if roof else 0.88,
            label=label if first else None,
            zorder=6,
        )
        first = False


def crop_extent(
    selected: list[tuple[int, str]],
    triangles: dict[int, dict[str, list[Polygon]]],
    points: dict[int, np.ndarray],
    image_shape: tuple[int, int],
    pad_x: int,
    pad_y: int,
) -> tuple[int, int, int, int]:
    xs: list[float] = []
    ys: list[float] = []
    for fid, _label in selected:
        geom = all_union(triangles[fid])
        xs.extend([geom.bounds[0], geom.bounds[2]])
        ys.extend([geom.bounds[1], geom.bounds[3]])
        if fid in points and points[fid].size:
            xs.extend(points[fid][:, 0].tolist())
            ys.extend(points[fid][:, 1].tolist())
    rows, cols = image_shape
    min_c = max(0, int(math.floor(min(xs))) - pad_x)
    max_c = min(cols - 1, int(math.ceil(max(xs))) + pad_x)
    min_r = max(0, int(math.floor(min(ys))) - pad_y)
    max_r = min(rows - 1, int(math.ceil(max(ys))) + pad_y)
    return min_c, max_c, min_r, max_r


def display_amplitude(amplitude: np.ndarray, extent: tuple[int, int, int, int]) -> np.ndarray:
    min_c, max_c, min_r, max_r = extent
    crop = amplitude[min_r : max_r + 1, min_c : max_c + 1].astype(np.float32)
    valid = crop[np.isfinite(crop)]
    lo, hi = np.percentile(valid, (1.0, 99.5))
    shown = np.clip((crop - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    return shown**0.65


def overlap_report(selected: list[tuple[int, str]], triangles: dict[int, dict[str, list[Polygon]]]) -> list[dict]:
    report: list[dict] = []
    unions = {fid: all_union(triangles[fid]) for fid, _ in selected}
    for index, (fid_a, label_a) in enumerate(selected):
        for fid_b, label_b in selected[index + 1 :]:
            intersection = float(unions[fid_a].intersection(unions[fid_b]).area)
            denominator = min(float(unions[fid_a].area), float(unions[fid_b].area))
            report.append(
                {
                    "a": label_a,
                    "fid_a": fid_a,
                    "b": label_b,
                    "fid_b": fid_b,
                    "intersection_px2": intersection,
                    "overlap_fraction_of_smaller": intersection / denominator if denominator > 0 else 0.0,
                }
            )
    return report


def reproduce(
    *,
    filename: str,
    selected: list[tuple[int, str]],
    triangles: dict[int, dict[str, list[Polygon]]],
    points: dict[int, np.ndarray],
    amplitude: np.ndarray,
    pad_x: int,
    pad_y: int,
) -> dict:
    extent = crop_extent(selected, triangles, points, amplitude.shape, pad_x, pad_y)
    min_c, max_c, min_r, max_r = extent
    amp = display_amplitude(amplitude, extent)
    heights = np.concatenate([points[fid][:, 2] for fid, _ in selected])
    vmin = max(0.0, 10.0 * math.floor(float(np.nanmin(heights)) / 10.0))
    vmax = 10.0 * math.ceil(float(np.nanmax(heights)) / 10.0)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=True)
    image_extent = [min_c, max_c, max_r, min_r]
    for ax in axes:
        ax.imshow(amp, cmap="gray", extent=image_extent, interpolation="nearest")
        ax.set_xlim(min_c, max_c)
        ax.set_ylim(max_r, min_r)
        ax.set_xlabel("Range Pixel (1 m grid)")
        ax.set_ylabel("Azimuth Pixel (1 m grid)")
        ax.tick_params(width=0.7, length=3)

    axes[0].text(
        0.02,
        0.96,
        "(a) Original SAR",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#333333", "linewidth": 0.4, "alpha": 0.90, "pad": 2.0},
        zorder=10,
    )

    scatter = None
    for index, (fid, label) in enumerate(selected):
        roof_geometry = union_surface(triangles[fid], "roof")
        draw_outline(
            axes[1],
            union_surface(triangles[fid], "bottom"),
            roof=False,
            label="Bottom footprint" if index == 0 else None,
        )
        draw_outline(
            axes[1],
            roof_geometry,
            roof=True,
            label="Roof outline" if index == 0 else None,
        )
        values = points[fid]
        scatter = axes[1].scatter(
            values[:, 0],
            values[:, 1],
            c=values[:, 2],
            cmap="jet",
            vmin=vmin,
            vmax=vmax,
            s=8.0,
            alpha=0.94,
            edgecolors="none",
            label=label,
            zorder=7,
        )
        label_x = float(roof_geometry.centroid.x) if roof_geometry is not None else float(np.median(values[:, 0]))
        label_y = float(roof_geometry.bounds[1]) - 6.0 if roof_geometry is not None else float(np.min(values[:, 1])) - 6.0
        axes[1].text(
            label_x,
            label_y,
            label,
            color="white",
            fontsize=8,
            ha="center",
            va="bottom",
            bbox={"facecolor": "#202020", "edgecolor": "none", "alpha": 0.42, "pad": 1.4},
            zorder=8,
        )

    axes[1].text(
        0.02,
        0.96,
        "(b) Height projection",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#333333", "linewidth": 0.4, "alpha": 0.90, "pad": 2.0},
        zorder=10,
    )
    axes[1].legend(loc="upper right", frameon=True, framealpha=0.84, borderpad=0.5)
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axes[1], fraction=0.040, pad=0.018)
        colorbar.set_label("Height / m")

    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return {
        "file": str(out_path),
        "dimensions_px": [3420, 1440],
        "selected": [{"fid": fid, "label": label, "points": int(points[fid].shape[0])} for fid, label in selected],
        "crop": {"min_col": min_c, "max_col": max_c, "min_row": min_r, "max_row": max_r},
        "height_range_m": [vmin, vmax],
        "pairwise_projected_overlap": overlap_report(selected, triangles),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    install_chinese_labels()
    radar_grid = parse_radar_grid(RSLC_DIR / f"{DATE}.rslc.par")
    rows = int(radar_grid["azimuth_lines"])
    cols = int(radar_grid["range_samples"])
    range_spacing_m = float(radar_grid["range_pixel_spacing"])
    azimuth_spacing_m = float(radar_grid["azimuth_pixel_spacing"])
    native_amplitude = read_amplitude(RSLC_DIR / f"{DATE}.rslc", rows, cols)
    amplitude = resample_amplitude_to_one_meter(
        native_amplitude,
        azimuth_spacing_m=azimuth_spacing_m,
        range_spacing_m=range_spacing_m,
    )
    triangles = load_triangles(range_spacing_m=range_spacing_m, azimuth_spacing_m=azimuth_spacing_m)
    points = load_points(range_spacing_m=range_spacing_m, azimuth_spacing_m=azimuth_spacing_m)

    resampling_metadata = {
        "date": DATE,
        "source": str(RSLC_DIR / f"{DATE}.rslc"),
        "source_value": "linear amplitude computed from big-endian complex int16 I/Q samples",
        "method": "bilinear interpolation of linear amplitude",
        "native_shape": [rows, cols],
        "native_range_spacing_m": range_spacing_m,
        "native_azimuth_spacing_m": azimuth_spacing_m,
        "target_range_spacing_m": 1.0,
        "target_azimuth_spacing_m": 1.0,
        "resampled_shape": [int(amplitude.shape[0]), int(amplitude.shape[1])],
        "coordinate_system": "radar range/azimuth grid; not map-georeferenced",
        "preview_stretch": "global positive-amplitude 2nd-98th percentile, clipped, gamma 0.65",
    }
    raster_outputs = save_resampled_scene(amplitude, resampling_metadata)
    comparison_path = plot_native_vs_resampled(
        native_amplitude,
        amplitude,
        range_spacing_m=range_spacing_m,
        azimuth_spacing_m=azimuth_spacing_m,
    )
    raster_outputs["comparison_png"] = str(comparison_path)

    reports = {
        "date": DATE,
        "input_triangles": str(TRIANGLES),
        "input_points": str(POINTS),
        "projection_basis": "globally and conservatively locally registered strict bottom/wall/roof triangles",
        "sar_resampling": {
            **resampling_metadata,
            "overlay_transform": "col_1m = native_col * native_range_spacing_m; row_1m = native_row * native_azimuth_spacing_m",
            "outputs": raster_outputs,
        },
        "figures": [
            reproduce(
                filename="图件_309304838608.png",
                selected=NON_OVERLAP,
                triangles=triangles,
                points=points,
                amplitude=amplitude,
                pad_x=75,
                pad_y=105,
            ),
            reproduce(
                filename="图件_873994219036.png",
                selected=LAYOVER,
                triangles=triangles,
                points=points,
                amplitude=amplitude,
                pad_x=100,
                pad_y=115,
            ),
        ],
    }
    (OUT_DIR / "复现元数据.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
