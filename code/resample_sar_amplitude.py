from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sar-resampling")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import zoom

from chinese_matplotlib import install_chinese_labels


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RSLC = PROJECT_DIR / "data" / "RE_SLAVES" / "20200708.rslc"
DEFAULT_OUT_DIR = PROJECT_DIR / "results" / "picall" / "注册复现" / "雷达重采样与论文图件"


def parse_radar_grid(path: Path) -> dict[str, float | int]:
    values: dict[str, float | int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        key = key.strip()
        if key in {"range_samples", "azimuth_lines"}:
            values[key] = int(float(rest.split()[0]))
        elif key in {"range_pixel_spacing", "azimuth_pixel_spacing"}:
            values[key] = float(rest.split()[0])
    required = {"range_samples", "azimuth_lines", "range_pixel_spacing", "azimuth_pixel_spacing"}
    missing = required.difference(values)
    if missing:
        raise ValueError(f"Missing radar-grid fields: {sorted(missing)}")
    return values


def read_linear_amplitude(path: Path, rows: int, cols: int) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=">i2")
    expected = rows * cols * 2
    if raw.size != expected:
        raise ValueError(f"Unexpected RSLC size: got {raw.size}, expected {expected} int16 values")
    parts = raw.reshape(rows, cols, 2).astype(np.float32)
    return np.hypot(parts[:, :, 0], parts[:, :, 1]).astype(np.float32)


def stretch_limits(native: np.ndarray) -> tuple[float, float]:
    positive = native[np.isfinite(native) & (native > 0)]
    if not positive.size:
        return 0.0, 1.0
    lo, hi = np.percentile(positive, (2.0, 98.0))
    return float(lo), float(hi)


def display_array(amplitude: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scaled = np.clip((amplitude - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return np.round(255.0 * scaled**0.65).astype(np.uint8)


def resolution_tag(spacing_m: float) -> str:
    text = f"{spacing_m:g}".replace(".", "p")
    return f"{text}m"


def save_comparison(
    native: np.ndarray,
    resampled: np.ndarray,
    *,
    native_range_spacing_m: float,
    native_azimuth_spacing_m: float,
    target_spacing_m: float,
    out_path: Path,
    lo: float,
    hi: float,
) -> None:
    physical_width_m = float(native.shape[1]) * native_range_spacing_m
    physical_height_m = float(native.shape[0]) * native_azimuth_spacing_m
    extent = [0.0, physical_width_m, physical_height_m, 0.0]
    native_display = display_array(native, lo, hi)
    resampled_display = display_array(resampled, lo, hi)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.7,
            "savefig.dpi": 300,
        }
    )
    install_chinese_labels()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 7.0), constrained_layout=True, sharex=True, sharey=True)
    axes[0].imshow(native_display, cmap="gray", extent=extent, interpolation="nearest", aspect="equal")
    axes[1].imshow(resampled_display, cmap="gray", extent=extent, interpolation="nearest", aspect="equal")
    axes[0].set_title(
        f"(a) Native SAR\n{native_range_spacing_m:.3f} m × {native_azimuth_spacing_m:.3f} m; "
        f"{native.shape[1]} × {native.shape[0]}",
        fontsize=10,
    )
    axes[1].set_title(
        f"(b) Resampled SAR\n{target_spacing_m:.3f} m × {target_spacing_m:.3f} m; "
        f"{resampled.shape[1]} × {resampled.shape[0]}",
        fontsize=10,
    )
    for ax in axes:
        ax.set_xlabel("Range distance / m")
        ax.set_ylabel("Azimuth distance / m")
        ax.tick_params(width=0.7, length=3)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run(rslc: Path, target_spacing_m: float, out_dir: Path) -> dict:
    par_path = Path(str(rslc) + ".par")
    grid = parse_radar_grid(par_path)
    rows = int(grid["azimuth_lines"])
    cols = int(grid["range_samples"])
    range_spacing_m = float(grid["range_pixel_spacing"])
    azimuth_spacing_m = float(grid["azimuth_pixel_spacing"])
    native = read_linear_amplitude(rslc, rows, cols)
    resampled = zoom(
        native,
        zoom=(azimuth_spacing_m / target_spacing_m, range_spacing_m / target_spacing_m),
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = resolution_tag(target_spacing_m)
    stem = rslc.stem
    names = {
        "0p3m": ("图件_971747229267.tif", "图件_494937630487.png", "图件_256193733709.png", "零点三米重采样元数据.json"),
        "1m": ("图件_621563374170.tif", "图件_444307131627.png", "图件_43812585258.png", "一米重采样元数据.json"),
    }
    suffixes = names.get(tag, (f"重采样_{tag}.tif", f"重采样_{tag}_预览.png", f"原始与重采样_{tag}_对比.png", f"重采样_{tag}_元数据.json"))
    tif_path, png_path, comparison_path, metadata_path = (out_dir / f"{stem}_{name}" for name in suffixes)
    lo, hi = stretch_limits(native)

    tifffile.imwrite(
        tif_path,
        resampled,
        compression="deflate",
        metadata={
            "axes": "YX",
            "coordinate_system": "radar range/azimuth grid",
            "range_pixel_spacing_m": target_spacing_m,
            "azimuth_pixel_spacing_m": target_spacing_m,
            "value": "linear SAR amplitude",
        },
    )
    Image.fromarray(display_array(resampled, lo, hi), mode="L").save(png_path, optimize=True)
    save_comparison(
        native,
        resampled,
        native_range_spacing_m=range_spacing_m,
        native_azimuth_spacing_m=azimuth_spacing_m,
        target_spacing_m=target_spacing_m,
        out_path=comparison_path,
        lo=lo,
        hi=hi,
    )
    metadata = {
        "source": str(rslc),
        "source_value": "linear amplitude computed from big-endian complex int16 I/Q samples",
        "resampling_method": "bilinear interpolation of linear amplitude",
        "native_shape": [rows, cols],
        "native_range_spacing_m": range_spacing_m,
        "native_azimuth_spacing_m": azimuth_spacing_m,
        "target_range_spacing_m": target_spacing_m,
        "target_azimuth_spacing_m": target_spacing_m,
        "resampled_shape": [int(resampled.shape[0]), int(resampled.shape[1])],
        "physical_extent_m": [cols * range_spacing_m, rows * azimuth_spacing_m],
        "coordinate_system": "radar range/azimuth grid; not map-georeferenced",
        "preview_stretch": {"lower_percentile": 2.0, "upper_percentile": 98.0, "gamma": 0.65},
        "outputs": {
            "float32_tiff": str(tif_path),
            "preview_png": str(png_path),
            "comparison_png": str(comparison_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["outputs"]["metadata_json"] = str(metadata_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample TerraSAR-X radar-grid amplitude to an isotropic target spacing.")
    parser.add_argument("--rslc", type=Path, default=DEFAULT_RSLC)
    parser.add_argument("--target-spacing", type=float, default=0.3)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    if not math.isfinite(args.target_spacing) or args.target_spacing <= 0:
        raise ValueError("--target-spacing must be a positive finite value")
    print(json.dumps(run(args.rslc, args.target_spacing, args.out_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
