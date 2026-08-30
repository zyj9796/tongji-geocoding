from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-paper-projection")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon


ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = ROOT / "results" / "outputs" / "work" / "gamma_dsm_geocode"


def par_value(path: Path, key: str) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(rf"^{re.escape(key)}:\s+([^\s]+)", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"Missing {key} in {path}")
    return int(float(match.group(1)))


def read_background(date: str) -> np.ndarray:
    work = WORK_ROOT / date
    par = work / f"{date}.mli.par"
    rows = par_value(par, "azimuth_lines")
    cols = par_value(par, "range_samples")
    intensity = np.fromfile(work / f"{date}.mli", dtype=">f4").reshape(rows, cols)
    amp = np.sqrt(np.maximum(intensity.astype(np.float32), 0.0))
    valid = np.isfinite(amp) & (amp > 0)
    p2, p98 = np.percentile(amp[valid], [2, 98])
    return np.clip((amp - p2) / max(float(p98 - p2), 1e-6), 0.0, 1.0)


def load_polygons(path: Path) -> list[tuple[str, np.ndarray]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    polygons: list[tuple[str, np.ndarray]] = []
    for feature in data.get("features", []):
        geometry = feature.get("geometry", {})
        if geometry.get("type") != "Polygon" or not geometry.get("coordinates"):
            continue
        xy = np.asarray(geometry["coordinates"][0], dtype=np.float64)
        if xy.shape[0] > 1 and np.allclose(xy[0], xy[-1]):
            xy = xy[:-1]
        if xy.shape[0] >= 3 and np.all(np.isfinite(xy)):
            polygons.append((str(feature.get("properties", {}).get("surface", "")), xy))
    return polygons


def draw(ax, background: np.ndarray, polygons: list[tuple[str, np.ndarray]], title: str) -> None:
    ax.imshow(background, cmap="gray", vmin=0, vmax=1)
    for surface, xy in polygons:
        color = "#ffb000" if surface == "roof" else "#00d4ff"
        width = 0.24 if surface == "roof" else 0.20
        alpha = 0.68 if surface == "roof" else 0.38
        ax.add_patch(MplPolygon(xy, closed=True, fill=False, edgecolor=color, linewidth=width, alpha=alpha))
    ax.set_xlim(0, background.shape[1])
    ax.set_ylim(background.shape[0], 0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot initial and corrected paper-method building projections.")
    parser.add_argument("--date", default="20200708")
    parser.add_argument("--initial-geojson", required=True)
    parser.add_argument("--corrected-geojson", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--row-shift", type=float, required=True)
    parser.add_argument("--col-shift", type=float, required=True)
    args = parser.parse_args()

    background = read_background(args.date)
    initial = load_polygons(Path(args.initial_geojson))
    corrected = load_polygons(Path(args.corrected_geojson))
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.2), dpi=260, sharex=True, sharey=True)
    draw(axes[0], background, initial, "(a) Initial zero-Doppler 3D model projection")
    draw(
        axes[1],
        background,
        corrected,
        f"(b) Corrected projection (row {args.row_shift:+.3f}, col {args.col_shift:+.3f})",
    )
    fig.suptitle(f"Building vectors projected onto SAR amplitude ({args.date})", fontsize=14, y=0.995)
    fig.text(0.5, 0.955, "Cyan: building bottom; amber: building roof", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.01, 1, 0.93), pad=0.7)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
