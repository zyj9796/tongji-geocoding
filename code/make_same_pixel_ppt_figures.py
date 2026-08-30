from __future__ import annotations

import csv
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image

from sar_display import stretch_sar_grayscale

from io_paths import (
    FULL_AREA_DIR,
    FULL_AREA_GEOJSON_DIR,
    PPT_CSV_DIR as CSV_DIR,
    PPT_DIR as OUT_DIR,
    PPT_DOC_DIR as DOC_DIR,
    PPT_IMAGE_DIR as FIG_DIR,
    PPT_ZIP as ZIP_PATH,
    PROJECT_DIR,
    REPO_ROOT,
    RESULTS_DIR as RESULT_DIR,
    RSLC_DIR,
    SAME_PIXEL_DIR,
    SAME_PIXEL_IMAGE_DIR,
    TRASH_DIR,
)

REPO_ROOT = PROJECT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR / "code"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from compare_same_ps_pixel_3d import load_buildings
from geocode_gamma_rslc_with_buildings import make_orbit, parse_gamma_par
from reproduce_thesis_tongji_tsx import rasterize_building

MATCH_CSV = SAME_PIXEL_DIR / "same_ps_pixel_ps_gamma_vs_bc_points.csv"
STATS_CSV = SAME_PIXEL_DIR / "same_ps_pixel_selected_buildings_stats.csv"
BUILDINGS_GEOJSON = FULL_AREA_GEOJSON_DIR / "20200708_all_valid_geocoded_buildings.geojson"
RSLC_PAR = RSLC_DIR / "20220601.rslc.par"
SAR_BMP = RSLC_DIR / "20220601.crop.bmp"

OLD_RESULT_DIRS = [
    RESULT_DIR / "psinsar_vs_shizhan_3d",
    RESULT_DIR / "psinsar_vs_bc_3d",
    RESULT_DIR / "psinsar_gamma_vs_shizhan_3d_relative_height",
    RESULT_DIR / "psinsar_gamma_vs_bc_3d_relative_height",
    RESULT_DIR / "psinsar_gamma_vs_shizhan_3d_visualization",
    RESULT_DIR / "psinsar_gamma_vs_bc_3d_visualization",
]

COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#d97706",
    "#0891b2",
    "#65a30d",
    "#e11d48",
    "#0f766e",
    "#4f46e5",
]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def selected_fids(stats_rows: list[dict]) -> list[int]:
    return [int(row["fid"]) for row in stats_rows]


def height_valid_matches(match_rows: list[dict], fids: list[int]) -> list[dict]:
    fid_set = set(fids)
    return [r for r in match_rows if int(r["fid"]) in fid_set and int(float(r["height_valid"])) == 1]


def load_sar_image() -> np.ndarray:
    arr = np.asarray(Image.open(SAR_BMP).convert("L"), dtype=np.float32)
    return stretch_sar_grayscale(
        arr,
        lower_percentile=1.0,
        upper_percentile=99.4,
        gamma=1.0,
        power_input=False,
    )


def build_models(fids: list[int]) -> tuple[dict[int, dict], dict[int, dict]]:
    buildings = load_buildings(BUILDINGS_GEOJSON)
    par = parse_gamma_par(RSLC_PAR)
    orbit = make_orbit(par)
    shape = (int(par["azimuth_lines"]), int(par["range_samples"]))
    models = {fid: rasterize_building(buildings[fid], par, orbit, shape) for fid in fids}
    return buildings, models


def add_model_edges(ax, model: dict, color: str, lw: float = 0.6, alpha: float = 0.72) -> None:
    rc = model["projected_rc"]
    for tri in model["triangles"]:
        pts = np.column_stack([rc[tri, 1], rc[tri, 0]])
        ax.add_patch(MplPolygon(pts, closed=True, fill=False, edgecolor=color, linewidth=lw, alpha=alpha))


def set_extent_from_models(ax, rows: list[dict], models: dict[int, dict], pad: int = 80) -> None:
    coords = []
    for row in rows:
        coords.append([float(row["sar_row_0based"]), float(row["sar_col_0based"])])
    for model in models.values():
        coords.extend(model["projected_rc"].tolist())
    arr = np.asarray(coords, dtype=np.float64)
    r0 = max(0, int(np.nanmin(arr[:, 0])) - pad)
    r1 = int(np.nanmax(arr[:, 0])) + pad
    c0 = max(0, int(np.nanmin(arr[:, 1])) - pad)
    c1 = int(np.nanmax(arr[:, 1])) + pad
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)


def fig_sar_overview(sar: np.ndarray, rows: list[dict], fids: list[int], models: dict[int, dict]) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 8.0), dpi=300)
    ax.imshow(sar, cmap="gray", origin="upper")
    for i, fid in enumerate(fids):
        fid_rows = [r for r in rows if int(r["fid"]) == fid]
        color = COLORS[i % len(COLORS)]
        add_model_edges(ax, models[fid], color, lw=0.65, alpha=0.8)
        x = [float(r["sar_col_0based"]) for r in fid_rows]
        y = [float(r["sar_row_0based"]) for r in fid_rows]
        ax.scatter(x, y, s=7, color=color, alpha=0.68, linewidths=0)
        center = np.nanmean(models[fid]["projected_rc"], axis=0)
        ax.text(center[1], center[0], str(fid), color="white", fontsize=8, weight="bold", ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.18", "fc": color, "ec": "white", "lw": 0.35, "alpha": 0.9})
    set_extent_from_models(ax, rows, models, pad=130)
    ax.set_title("Selected Buildings and Same PS Pixels on SAR Amplitude Image", fontsize=15, weight="bold")
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    ax.grid(color="white", alpha=0.18, linewidth=0.35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "图_01_雷达总览与选定建筑像素.png")
    plt.close(fig)


def fig_per_building_crops(sar: np.ndarray, rows: list[dict], fids: list[int], models: dict[int, dict]) -> None:
    ncols = 5
    nrows = math.ceil(len(fids) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.25 * nrows), dpi=300, squeeze=False)
    for i, fid in enumerate(fids):
        ax = axes.ravel()[i]
        color = COLORS[i % len(COLORS)]
        ax.imshow(sar, cmap="gray", origin="upper")
        fid_rows = [r for r in rows if int(r["fid"]) == fid]
        add_model_edges(ax, models[fid], color, lw=0.72, alpha=0.86)
        ax.scatter([float(r["sar_col_0based"]) for r in fid_rows], [float(r["sar_row_0based"]) for r in fid_rows],
                   s=8, color="#f97316", alpha=0.78, linewidths=0)
        set_extent_from_models(ax, fid_rows, {fid: models[fid]}, pad=28)
        ax.set_title(f"FID {fid} | n={len(fid_rows)}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(color="white", alpha=0.18, linewidth=0.25)
    for ax in axes.ravel()[len(fids):]:
        ax.axis("off")
    fig.suptitle("Per-Building SAR Crops: Projection Mesh and Exact PS Pixels", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "图_02_逐建筑雷达裁剪图.png")
    plt.close(fig)


def fig_pixel_density(rows: list[dict]) -> None:
    x = np.asarray([float(r["sar_col_0based"]) for r in rows], dtype=np.float64)
    y = np.asarray([float(r["sar_row_0based"]) for r in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9.2, 7.2), dpi=300)
    hb = ax.hexbin(x, y, gridsize=55, mincnt=1, cmap="magma", norm=LogNorm())
    ax.scatter(x, y, s=2, color="white", alpha=0.22, linewidths=0)
    ax.set_ylim(float(np.max(y)) + 16, float(np.min(y)) - 16)
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    ax.set_title("Same-Pixel PS Density in SAR Coordinate Space", fontsize=14, weight="bold")
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("PS count per hexbin")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_03_sar_pixel_density.png")
    plt.close(fig)


def fig_geographic_map(rows: list[dict], buildings: dict[int, dict], fids: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 8.2), dpi=300)
    for i, fid in enumerate(fids):
        b = buildings[fid]
        ring = b["ring_lonlat"]
        color = COLORS[i % len(COLORS)]
        ax.add_patch(MplPolygon(ring, closed=True, facecolor=color, edgecolor="#111827", linewidth=0.55, alpha=0.14))
        ax.plot(ring[:, 0], ring[:, 1], color=color, linewidth=1.0)
        fid_rows = [r for r in rows if int(r["fid"]) == fid]
        ax.scatter([float(r["ps_lon"]) for r in fid_rows], [float(r["ps_lat"]) for r in fid_rows],
                   s=6, color="#f97316", alpha=0.42, linewidths=0, label="PS/GAMMA" if i == 0 else None)
        ax.scatter([float(r["shizhan_lon_same_pixel"]) for r in fid_rows],
                   [float(r["shizhan_lat_same_pixel"]) for r in fid_rows],
                   s=5, color=color, alpha=0.75, linewidths=0, label="Building-constrained" if i == 0 else None)
        cx, cy = np.mean(ring[:, 0]), np.mean(ring[:, 1])
        ax.text(cx, cy, str(fid), fontsize=7, ha="center", va="center", color="#111827")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Geographic Comparison: PS/GAMMA vs. Building-Constrained Points", fontsize=14, weight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.22, linewidth=0.35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "图_04_同像素地理定位对比.png")
    plt.close(fig)


def fig_error_hist(rows: list[dict]) -> None:
    h = np.asarray([float(r["same_pixel_horizontal_m"]) for r in rows], dtype=np.float64)
    z = np.asarray([float(r["ps_minus_shizhan_height_m"]) for r in rows], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=300)
    axes[0].hist(h, bins=42, color="#2563eb", alpha=0.82, edgecolor="white", linewidth=0.35)
    axes[0].axvline(np.median(h), color="#111827", linewidth=1.2, label=f"median={np.median(h):.2f} m")
    axes[0].set_xlabel("Horizontal difference / m")
    axes[0].set_ylabel("PS count")
    axes[0].legend(fontsize=8)
    axes[1].hist(z, bins=42, color="#f97316", alpha=0.82, edgecolor="white", linewidth=0.35)
    axes[1].axvline(np.median(z), color="#111827", linewidth=1.2, label=f"median={np.median(z):.2f} m")
    axes[1].set_xlabel("PS height minus Shizhan height / m")
    axes[1].legend(fontsize=8)
    fig.suptitle("Same-Pixel Difference Distributions", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_05_difference_histograms.png")
    plt.close(fig)


def fig_boxplot(rows: list[dict], fids: list[int]) -> None:
    data = [[float(r["same_pixel_horizontal_m"]) for r in rows if int(r["fid"]) == fid] for fid in fids]
    fig, ax = plt.subplots(figsize=(11.6, 5.2), dpi=300)
    bp = ax.boxplot(data, labels=[str(fid) for fid in fids], patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.38)
    ax.set_xlabel("Building FID")
    ax.set_ylabel("Horizontal difference / m")
    ax.set_title("Per-Building Same-Pixel Horizontal Difference", fontsize=14, weight="bold")
    ax.grid(axis="y", alpha=0.24, linewidth=0.35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_06_per_building_horizontal_boxplot.png")
    plt.close(fig)


def fig_scatter_diagnostics(rows: list[dict]) -> None:
    h = np.asarray([float(r["same_pixel_horizontal_m"]) for r in rows], dtype=np.float64)
    z = np.asarray([float(r["ps_minus_shizhan_height_m"]) for r in rows], dtype=np.float64)
    coh = np.asarray([float(r["coherence"]) for r in rows], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=300)
    s0 = axes[0].scatter(h, z, c=coh, cmap="viridis", s=8, alpha=0.58, linewidths=0)
    axes[0].axhline(0, color="#111827", linewidth=0.8)
    axes[0].set_xlabel("Horizontal difference / m")
    axes[0].set_ylabel("PS height minus Shizhan height / m")
    axes[0].set_title("Horizontal vs. Vertical Difference")
    fig.colorbar(s0, ax=axes[0], label="Coherence")
    axes[1].scatter(coh, h, c="#2563eb", s=8, alpha=0.48, linewidths=0)
    axes[1].set_xlabel("Coherence")
    axes[1].set_ylabel("Horizontal difference / m")
    axes[1].set_title("Coherence Diagnostic")
    for ax in axes:
        ax.grid(alpha=0.22, linewidth=0.35)
    fig.suptitle("Same-Pixel Diagnostic Scatter Plots", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_07_same_pixel_diagnostics.png")
    plt.close(fig)


def fig_height_profiles(rows: list[dict], fids: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(11.6, 5.4), dpi=300)
    positions = np.arange(len(fids))
    for i, fid in enumerate(fids):
        vals = np.asarray([float(r["shizhan_relative_height_m"]) for r in rows if int(r["fid"]) == fid], dtype=np.float64)
        if vals.size == 0:
            continue
        jitter = np.linspace(-0.22, 0.22, min(vals.size, 180))
        sample = vals[:: max(1, math.ceil(vals.size / 180))]
        jitter = jitter[: sample.size]
        ax.scatter(np.full(sample.size, positions[i]) + jitter, sample, s=8, color=COLORS[i % len(COLORS)], alpha=0.55, linewidths=0)
        ax.plot([positions[i] - 0.25, positions[i] + 0.25], [np.median(vals), np.median(vals)], color="#111827", linewidth=1.1)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(fid) for fid in fids], rotation=0)
    ax.set_xlabel("Building FID")
    ax.set_ylabel("Shizhan relative height / m")
    ax.set_title("Recovered Building-Constrained Heights by Building", fontsize=14, weight="bold")
    ax.grid(axis="y", alpha=0.23, linewidth=0.35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_08_building_constrained_height_profiles.png")
    plt.close(fig)


def fig_stats_table(stats_rows: list[dict]) -> None:
    cols = [
        "fid",
        "same_pixel_ps_count",
        "height_valid_ps_count",
        "height_valid_ratio",
        "horizontal_m_median",
        "horizontal_m_p90",
        "abs_height_diff_m_median",
    ]
    labels = ["FID", "PS", "valid PS", "valid ratio", "H med/m", "H P90/m", "|Z| med/m"]
    cell_text = []
    for row in stats_rows:
        cell_text.append([
            row["fid"],
            row["same_pixel_ps_count"],
            row["height_valid_ps_count"],
            f"{float(row['height_valid_ratio']):.2f}",
            f"{float(row['horizontal_m_median']):.2f}",
            f"{float(row['horizontal_m_p90']):.2f}",
            f"{float(row['abs_height_diff_m_median']):.2f}",
        ])
    fig, ax = plt.subplots(figsize=(12.4, 4.6), dpi=300)
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.45)
    for (r, _c), cell in table.get_celld().items():
        cell.set_linewidth(0.3)
        if r == 0:
            cell.set_facecolor("#111827")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f3f4f6")
    ax.set_title("Selected Building Same-Pixel Statistics", fontsize=14, weight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_09_selected_building_statistics_table.png")
    plt.close(fig)


def copy_base_outputs() -> None:
    for src in [
        MATCH_CSV,
        STATS_CSV,
        SAME_PIXEL_IMAGE_DIR / "图_同像素永久散射体三维建筑.png",
        SAME_PIXEL_IMAGE_DIR / "图_同像素永久散射体雷达像素.png",
    ]:
        if src.exists():
            dst = (CSV_DIR if src.suffix.lower() == ".csv" else FIG_DIR) / src.name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)


def move_old_results_to_trash() -> list[str]:
    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    moved = []
    for src in OLD_RESULT_DIRS:
        if not src.exists():
            continue
        dst = TRASH_DIR / src.name
        if dst.exists():
            suffix = 1
            while (TRASH_DIR / f"{src.name}_{suffix}").exists():
                suffix += 1
            dst = TRASH_DIR / f"{src.name}_{suffix}"
        shutil.move(str(src), str(dst))
        moved.append(str(dst.relative_to(PROJECT_DIR)))
    return moved


def write_markdown(stats_rows: list[dict], fids: list[int], moved: list[str]) -> None:
    total = sum(int(r["height_valid_ps_count"]) for r in stats_rows)
    med = np.median([float(r["horizontal_m_median"]) for r in stats_rows])
    p90 = np.median([float(r["horizontal_m_p90"]) for r in stats_rows])
    lines = [
        "# PS-InSAR Same-Pixel PPT Figure Package",
        "",
        "This package is the current recommended material for slides. It uses the strict same-SAR-pixel comparison, not nearest-neighbor matching.",
        "",
        "## Current SAR Basis",
        "",
        "- Geographic background figures use the DSM terrain-geocoded GAMMA RSLC products, not a GCP/TPS or building-warped raster.",
        "- Radar-coordinate overview/crop figures retain the original RSLC amplitude geometry.",
        "- SAR grayscale uses amplitude-domain linear clipping at the valid-pixel 2nd and 98th percentiles; this affects display only.",
        "",
        "## Core Rule",
        "",
        "- PS `line/pixel` are converted from 1-based STAMPS coordinates to 0-based SAR row/column.",
        "- A PS point is kept only when the same SAR pixel falls inside a selected building projection triangle mask.",
        "- The Shizhan/building-constrained coordinate is recomputed on that exact projected building triangle by barycentric inversion.",
        "- 3D plots use relative building height, with base = 0 m and roof = vector `height_m`.",
        "",
        "## Selected Buildings",
        "",
        f"- FIDs: {', '.join(str(fid) for fid in fids)}",
        f"- Height-valid same-pixel PS used in figures: {total}",
        f"- Median of per-building horizontal medians: {med:.2f} m",
        f"- Median of per-building horizontal P90 values: {p90:.2f} m",
        "",
        "## PPT Figure Index",
        "",
        "- `图_01_雷达总览与选定建筑像素.png`: SAR amplitude overview with selected building projection meshes and exact PS pixels.",
        "- `图_02_逐建筑雷达裁剪图.png`: per-building SAR close-ups for slide callouts.",
        "- `fig_03_sar_pixel_density.png`: density of exact PS pixels in SAR coordinates.",
        "- `图_04_同像素地理定位对比.png`: geographic PS/GAMMA versus building-constrained points with footprints.",
        "- `fig_05_difference_histograms.png`: horizontal and vertical difference distributions.",
        "- `fig_06_per_building_horizontal_boxplot.png`: per-building horizontal difference boxplot.",
        "- `fig_07_same_pixel_diagnostics.png`: horizontal/vertical/coherence diagnostics.",
        "- `fig_08_building_constrained_height_profiles.png`: recovered relative-height profiles by building.",
        "- `fig_09_selected_building_statistics_table.png`: slide-ready statistics table.",
        "- `图_同像素永久散射体三维建筑.png`: original strict same-pixel 3D comparison.",
        "- `图_同像素永久散射体雷达像素.png`: original strict same-pixel radar-coordinate mesh plot.",
        "",
        "## Interpretation",
        "",
        "The figures mark the selected buildings directly in SAR row/column space and show the PS pixels used for the final comparison. The horizontal difference remains around 30-35 m for the typical buildings even under the strict same-pixel rule, so this result should be explained as the difference between the PS/GAMMA 3D coordinate surface and the building-constrained surface for the same SAR observation.",
        "",
        "## Old Results Moved To Trash",
        "",
    ]
    if moved:
        lines.extend([f"- `{p}`" for p in moved])
    else:
        lines.append("- No old result directory needed moving in this run.")
    lines.extend([
        "",
        "## Files",
        "",
        "- `../../pic_all/`: PNG figures for PPT.",
        "- `csv/`: copied same-pixel comparison tables.",
        "- `docs/README.md`: this detailed explanation.",
        "",
    ])
    text = "\n".join(lines)
    (DOC_DIR / "README.md").write_text(text, encoding="utf-8")
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")
    SAME_PIXEL_DIR.joinpath("README.md").write_text(text, encoding="utf-8")


def make_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in OUT_DIR.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(RESULT_DIR))
        for path in FIG_DIR.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(RESULT_DIR))
        for path in TRASH_DIR.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(RESULT_DIR))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    match_rows = read_csv(MATCH_CSV)
    stats_rows = read_csv(STATS_CSV)
    fids = selected_fids(stats_rows)
    rows = height_valid_matches(match_rows, fids)
    sar = load_sar_image()
    buildings, models = build_models(fids)

    copy_base_outputs()
    fig_sar_overview(sar, rows, fids, models)
    fig_per_building_crops(sar, rows, fids, models)
    fig_pixel_density(rows)
    fig_geographic_map(rows, buildings, fids)
    fig_error_hist(rows)
    fig_boxplot(rows, fids)
    fig_scatter_diagnostics(rows)
    fig_height_profiles(rows, fids)
    fig_stats_table(stats_rows)
    write_csv(CSV_DIR / "height_valid_same_pixel_selected_points.csv", rows)

    moved = move_old_results_to_trash()
    write_markdown(stats_rows, fids, moved)
    make_zip()
    print(f"figures={len(list(FIG_DIR.glob('*.png')))}")
    print(f"out_dir={OUT_DIR}")
    print(f"zip={ZIP_PATH}")
    print(f"trash={TRASH_DIR}")


if __name__ == "__main__":
    main()
