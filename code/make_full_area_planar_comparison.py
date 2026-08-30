from __future__ import annotations

import csv
import json
import math
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon

from io_paths import FULL_AREA_DIR, FULL_AREA_GEOJSON_DIR, FULL_AREA_IMAGE_DIR, PPT_DIR, PPT_IMAGE_DIR, PPT_ZIP, PROJECT_DIR, RESULTS_DIR, TRASH_DIR

PPT_FIG_DIR = PPT_IMAGE_DIR
ZIP_PATH = PPT_ZIP

POINTS_CSV = FULL_AREA_DIR / "20200708_all_buildings_method_vs_gamma_points.csv"
BUILDINGS_GEOJSON = FULL_AREA_GEOJSON_DIR / "20200708_all_valid_geocoded_buildings.geojson"
STATS_CSV = FULL_AREA_DIR / "20200708_all_buildings_fig5_4_like_stats.csv"
OUT_PNG = FULL_AREA_IMAGE_DIR / "20200708_图件_1045917436903_版本2.png"
PPT_COPY = PPT_FIG_DIR / OUT_PNG.name


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_building_patches(path: Path) -> list[Polygon]:
    data = json.loads(path.read_text(encoding="utf-8"))
    patches: list[Polygon] = []
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue
        coords = geom.get("coordinates", [])
        if not coords:
            continue
        ring = np.asarray(coords[0], dtype=np.float64)
        if ring.shape[0] < 3:
            continue
        patches.append(Polygon(ring[:, :2], closed=True))
    return patches


def point_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = [r for r in rows if int(float(r.get("gamma_dsm_ok", 0))) == 1]
    method = np.asarray([[float(r["method_lon"]), float(r["method_lat"])] for r in valid], dtype=np.float64)
    gamma = np.asarray([[float(r["gamma_dsm_lon"]), float(r["gamma_dsm_lat"])] for r in valid], dtype=np.float64)
    fid = np.asarray([int(r["fid"]) for r in valid], dtype=np.int64)
    return method, gamma, fid


def horizontal_distance_m(method: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    lat0 = np.deg2rad((method[:, 1] + gamma[:, 1]) / 2.0)
    dx = (method[:, 0] - gamma[:, 0]) * math.pi / 180.0 * 6378137.0 * np.cos(lat0)
    dy = (method[:, 1] - gamma[:, 1]) * math.pi / 180.0 * 6378137.0
    return np.hypot(dx, dy)


def add_buildings(ax, patches: list[Polygon]) -> None:
    coll = PatchCollection(patches, facecolor="#e5e7eb", edgecolor="#6b7280", linewidth=0.12, alpha=0.42)
    ax.add_collection(coll)


def set_equal_lonlat(ax, extent: tuple[float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect(1.0 / max(math.cos(math.radians((ymin + ymax) / 2.0)), 0.1), adjustable="box")
    ax.grid(color="#9ca3af", linewidth=0.25, alpha=0.28)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)


def sample_indices(n: int, target: int) -> np.ndarray:
    if n <= target:
        return np.arange(n)
    return np.linspace(0, n - 1, target, dtype=np.int64)


def summarize_text(rows: list[dict], distances: np.ndarray, fids: np.ndarray) -> str:
    stats_rows = read_csv(STATS_CSV) if STATS_CSV.exists() else []
    valid_buildings = len(set(fids.tolist()))
    mean_method = ""
    mean_gamma = ""
    if stats_rows:
        method_keys = [k for k in stats_rows[0] if "method" in k.lower() and "mean" in k.lower()]
        gamma_keys = [k for k in stats_rows[0] if "gamma" in k.lower() and "mean" in k.lower()]
        if method_keys:
            vals = [float(r[method_keys[0]]) for r in stats_rows if r.get(method_keys[0], "")]
            mean_method = f"\nMethod boundary mean: {np.mean(vals):.2f} m"
        if gamma_keys:
            vals = [float(r[gamma_keys[0]]) for r in stats_rows if r.get(gamma_keys[0], "")]
            mean_gamma = f"\nGAMMA/DSM boundary mean: {np.mean(vals):.2f} m"
    return (
        f"Date: 20200708\n"
        f"Buildings: {valid_buildings}\n"
        f"Point pairs: {len(rows):,}\n"
        f"Pair distance median: {np.median(distances):.2f} m\n"
        f"Pair distance P90: {np.percentile(distances, 90):.2f} m"
        f"{mean_method}{mean_gamma}"
    )


def make_figure() -> None:
    FULL_AREA_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_csv(POINTS_CSV)
    method, gamma, fid = point_arrays(rows)
    patches = load_building_patches(BUILDINGS_GEOJSON)
    dist = horizontal_distance_m(method, gamma)

    all_xy = np.vstack([method, gamma])
    pad_x = (np.max(all_xy[:, 0]) - np.min(all_xy[:, 0])) * 0.035
    pad_y = (np.max(all_xy[:, 1]) - np.min(all_xy[:, 1])) * 0.035
    extent = (
        float(np.min(all_xy[:, 0]) - pad_x),
        float(np.max(all_xy[:, 0]) + pad_x),
        float(np.min(all_xy[:, 1]) - pad_y),
        float(np.max(all_xy[:, 1]) + pad_y),
    )

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.2), dpi=300)
    sampled_points = sample_indices(method.shape[0], 24000)
    sampled_lines = sample_indices(method.shape[0], 1800)

    add_buildings(axes[0], patches)
    axes[0].scatter(method[sampled_points, 0], method[sampled_points, 1], s=1.2, c="#2563eb", alpha=0.52, linewidths=0)
    axes[0].set_title("Building-Constrained Method", fontsize=12, weight="bold")

    add_buildings(axes[1], patches)
    axes[1].scatter(gamma[sampled_points, 0], gamma[sampled_points, 1], s=1.2, c="#f97316", alpha=0.52, linewidths=0)
    axes[1].set_title("Traditional GAMMA/DSM", fontsize=12, weight="bold")

    add_buildings(axes[2], patches)
    segments = np.stack([method[sampled_lines], gamma[sampled_lines]], axis=1)
    line_coll = LineCollection(segments, colors="#6b7280", linewidths=0.22, alpha=0.22)
    axes[2].add_collection(line_coll)
    axes[2].scatter(gamma[sampled_points, 0], gamma[sampled_points, 1], s=1.0, c="#f97316", alpha=0.34, linewidths=0, label="GAMMA/DSM")
    axes[2].scatter(method[sampled_points, 0], method[sampled_points, 1], s=1.0, c="#2563eb", alpha=0.52, linewidths=0, label="Building-constrained")
    axes[2].legend(loc="lower left", fontsize=8, markerscale=4, frameon=True)
    axes[2].text(
        0.02,
        0.98,
        summarize_text(rows, dist, fid),
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#9ca3af", "lw": 0.45, "alpha": 0.92},
    )
    axes[2].set_title("Planar Overlay and Sampled Pair Links", fontsize=12, weight="bold")

    for ax in axes:
        set_equal_lonlat(ax, extent)

    fig.suptitle("Full-Area Planar Comparison of SAR Building Geocoding Methods", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_PNG)
    plt.close(fig)

    PPT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    if PPT_FIG_DIR.exists():
        import shutil

        if OUT_PNG.resolve() != PPT_COPY.resolve():
            shutil.copy2(OUT_PNG, PPT_COPY)


def update_zip() -> None:
    if not ZIP_PATH.exists() or not PPT_DIR.exists():
        return
    tmp = ZIP_PATH.with_suffix(".tmp.zip")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root in [PPT_DIR, PPT_IMAGE_DIR, TRASH_DIR]:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(RESULTS_DIR))
    tmp.replace(ZIP_PATH)


def main() -> None:
    make_figure()
    update_zip()
    print(f"figure={OUT_PNG}")
    if PPT_COPY.exists():
        print(f"ppt_copy={PPT_COPY}")
    if ZIP_PATH.exists():
        print(f"zip_updated={ZIP_PATH}")


if __name__ == "__main__":
    main()
