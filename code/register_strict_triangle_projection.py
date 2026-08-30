from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-strict-registration")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MplPath
from scipy.ndimage import binary_dilation


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR / "code"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from geocode_gamma_rslc_with_buildings import parse_gamma_par, read_rslc_amplitude  # noqa: E402
from io_paths import RSLC_DIR  # noqa: E402
from run_strict_triangle_projection import refine_triangle_mask  # noqa: E402


def load_triangles(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    triangles: list[dict] = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry", {})
        if geometry.get("type") != "Polygon" or not geometry.get("coordinates"):
            continue
        xy = np.asarray(geometry["coordinates"][0], dtype=np.float64)
        if xy.shape[0] > 1 and np.allclose(xy[0], xy[-1], atol=1e-9, rtol=0):
            xy = xy[:-1]
        if xy.shape[0] != 3 or not np.all(np.isfinite(xy)):
            continue
        triangles.append({"feature": feature, "xy": xy})
    return payload, triangles


def rasterize_triangles(triangles: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    mask = np.zeros(shape, dtype=bool)
    for xy in triangles:
        c0 = max(0, int(math.floor(np.min(xy[:, 0]))) - 1)
        c1 = min(cols - 1, int(math.ceil(np.max(xy[:, 0]))) + 1)
        r0 = max(0, int(math.floor(np.min(xy[:, 1]))) - 1)
        r1 = min(rows - 1, int(math.ceil(np.max(xy[:, 1]))) + 1)
        if c1 < c0 or r1 < r0:
            continue
        yy, xx = np.mgrid[r0 : r1 + 1, c0 : c1 + 1]
        inside = MplPath(xy).contains_points(
            np.column_stack([xx.ravel(), yy.ravel()]), radius=1e-9
        ).reshape(yy.shape)
        mask[r0 : r1 + 1, c0 : c1 + 1] |= inside
    return mask


def shift_mask(mask: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
    out = np.zeros_like(mask)
    rows, cols = mask.shape
    src_r0 = max(0, -row_shift)
    src_r1 = min(rows, rows - row_shift)
    src_c0 = max(0, -col_shift)
    src_c1 = min(cols, cols - col_shift)
    dst_r0 = max(0, row_shift)
    dst_r1 = min(rows, rows + row_shift)
    dst_c0 = max(0, col_shift)
    dst_c1 = min(cols, cols + col_shift)
    if src_r1 > src_r0 and src_c1 > src_c0:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1]
    return out


def edge_map(amplitude: np.ndarray) -> np.ndarray:
    image = amplitude.astype(np.float32) / 255.0
    gy, gx = np.gradient(image)
    edges = np.hypot(gx, gy)
    positive = edges[np.isfinite(edges) & (edges > 0)]
    p98 = float(np.percentile(positive, 98)) if positive.size else 1.0
    return np.clip(edges / max(p98, 1e-6), 0.0, 1.0)


def score_shift(mask: np.ndarray, amplitude01: np.ndarray, edges: np.ndarray, dr: int, dc: int) -> dict:
    shifted = shift_mask(mask, dr, dc)
    pixels = int(shifted.sum())
    if pixels < 100:
        return {"score": -1e9, "inside_amp": 0.0, "ring_amp": 0.0, "inside_edge": 0.0, "ring_edge": 0.0, "pixels": pixels}
    inner_dilation = binary_dilation(shifted, iterations=1)
    ring = binary_dilation(shifted, iterations=5) & ~inner_dilation
    inside_amp = float(np.mean(amplitude01[shifted]))
    ring_amp = float(np.mean(amplitude01[ring])) if np.any(ring) else float(np.mean(amplitude01))
    inside_edge = float(np.mean(edges[shifted]))
    ring_edge = float(np.mean(edges[ring])) if np.any(ring) else float(np.mean(edges))
    penalty = 0.0008 * (dr * dr + dc * dc)
    score = 100.0 * (inside_amp - ring_amp) + 45.0 * (inside_edge - ring_edge) - penalty
    return {
        "score": score,
        "inside_amp": inside_amp,
        "ring_amp": ring_amp,
        "inside_edge": inside_edge,
        "ring_edge": ring_edge,
        "pixels": pixels,
    }


def search(mask: np.ndarray, amplitude01: np.ndarray, edges: np.ndarray, max_shift: int, coarse_step: int) -> tuple[int, int, dict, list[dict]]:
    records: list[dict] = []
    best = (-1e18, 0, 0, {})
    for dr in range(-max_shift, max_shift + 1, coarse_step):
        for dc in range(-max_shift, max_shift + 1, coarse_step):
            result = score_shift(mask, amplitude01, edges, dr, dc)
            records.append({"stage": "coarse", "row_shift": dr, "col_shift": dc, **result})
            if result["score"] > best[0]:
                best = (result["score"], dr, dc, result)
    _, coarse_r, coarse_c, _ = best
    for dr in range(coarse_r - coarse_step, coarse_r + coarse_step + 1):
        for dc in range(coarse_c - coarse_step, coarse_c + coarse_step + 1):
            result = score_shift(mask, amplitude01, edges, dr, dc)
            records.append({"stage": "fine", "row_shift": dr, "col_shift": dc, **result})
            if result["score"] > best[0]:
                best = (result["score"], dr, dc, result)
    return best[1], best[2], best[3], records


def shift_payload(payload: dict, row_shift: int, col_shift: int) -> dict:
    for feature in payload.get("features", []):
        geometry = feature.get("geometry", {})
        if geometry.get("type") != "Polygon":
            continue
        geometry["coordinates"] = [
            [[float(x) + col_shift, float(y) + row_shift] for x, y, *rest in ring]
            for ring in geometry.get("coordinates", [])
        ]
        properties = feature.setdefault("properties", {})
        properties["registration_row_shift"] = row_shift
        properties["registration_col_shift"] = col_shift
    payload["registration"] = {
        "row_shift": row_shift,
        "col_shift": col_shift,
        "coordinate_system": "x=range column, y=azimuth row",
        "method": "global union-of-all-surface-triangles registration against real SAR amplitude and edge contrast",
    }
    return payload


def triangle_segments(triangles: list[dict], dr: int, dc: int) -> dict[str, list[np.ndarray]]:
    segments: dict[str, list[np.ndarray]] = {"bottom": [], "wall": [], "roof": []}
    offset = np.asarray([dc, dr], dtype=np.float64)
    for item in triangles:
        surface = str(item["feature"].get("properties", {}).get("surface", "wall"))
        xy = item["xy"] + offset
        segments.setdefault(surface, []).append(np.vstack([xy, xy[0]]))
    return segments


def plot_projection(path: Path, amplitude: np.ndarray, segments: dict[str, list[np.ndarray]], date: str, dr: int, dc: int) -> None:
    colors = {"bottom": "#00d4ff", "wall": "#ff4fd8", "roof": "#ffb000"}
    widths = {"bottom": 0.18, "wall": 0.13, "roof": 0.20}
    alphas = {"bottom": 0.38, "wall": 0.27, "roof": 0.58}
    fig, ax = plt.subplots(figsize=(11.0, 8.0), dpi=300)
    ax.imshow(amplitude.astype(np.float32) / 255.0, cmap="gray", vmin=0, vmax=1)
    for surface in ("bottom", "wall", "roof"):
        ax.add_collection(LineCollection(segments[surface], colors=colors[surface], linewidths=widths[surface], alpha=alphas[surface]))
    ax.set_xlim(0, amplitude.shape[1])
    ax.set_ylim(amplitude.shape[0], 0)
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    ax.set_title(f"Registered strict building triangle projection ({date})")
    ax.text(
        0.012,
        0.988,
        f"row shift: {dr:+d}\ncol shift: {dc:+d}\ncyan: bottom, magenta: wall, amber: roof",
        transform=ax.transAxes,
        va="top",
        color="white",
        fontsize=8,
        bbox={"facecolor": "black", "alpha": 0.58, "edgecolor": "none", "pad": 3},
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_refined(path: Path, amplitude: np.ndarray, refined: np.ndarray, roof_segments: list[np.ndarray], date: str, dr: int, dc: int) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 8.0), dpi=300)
    ax.imshow(amplitude.astype(np.float32) / 255.0, cmap="gray", vmin=0, vmax=1)
    overlay = np.zeros((*refined.shape, 4), dtype=np.float32)
    overlay[refined] = (0.15, 1.0, 0.25, 0.62)
    ax.imshow(overlay)
    ax.add_collection(LineCollection(roof_segments, colors="#ffb000", linewidths=0.16, alpha=0.45))
    ax.set_xlim(0, amplitude.shape[1])
    ax.set_ylim(amplitude.shape[0], 0)
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    ax.set_title(f"Registered triangle-constrained SAR amplitude refinement ({date})")
    ax.text(
        0.012,
        0.988,
        f"row shift: {dr:+d}\ncol shift: {dc:+d}\ngreen: refined pixels, amber: roof triangles",
        transform=ax.transAxes,
        va="top",
        color="white",
        fontsize=8,
        bbox={"facecolor": "black", "alpha": 0.58, "edgecolor": "none", "pad": 3},
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    pic_dir = Path(args.pic_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pic_dir.mkdir(parents=True, exist_ok=True)
    par = parse_gamma_par(RSLC_DIR / f"{args.date}.rslc.par")
    amplitude = read_rslc_amplitude(
        RSLC_DIR / f"{args.date}.rslc", int(par["azimuth_lines"]), int(par["range_samples"])
    )
    amplitude01 = amplitude.astype(np.float32) / 255.0
    edges = edge_map(amplitude)
    payload, triangles = load_triangles(Path(args.triangle_geojson))
    registration_triangles = [
        item["xy"]
        for item in triangles
        if args.registration_surface == "all"
        or str(item["feature"].get("properties", {}).get("surface", "")) == args.registration_surface
    ]
    if not registration_triangles:
        raise RuntimeError(f"No {args.registration_surface!r} triangles available for registration")
    # The literature workflow registers the roof projection against the real
    # SAR bright/edge response, then applies the resulting shift to every
    # bottom/wall/roof triangle.  Mixing wall and bottom interiors into the
    # global score biases the range direction for layover-rich buildings.
    initial_mask = rasterize_triangles(registration_triangles, amplitude.shape)
    base = score_shift(initial_mask, amplitude01, edges, 0, 0)
    dr, dc, best, search_records = search(initial_mask, amplitude01, edges, args.max_shift, args.coarse_step)

    corrected_geojson = out_dir / f"{args.date}_registered_strict_sar_surface_triangles.geojson"
    corrected_geojson.write_text(json.dumps(shift_payload(payload, dr, dc), ensure_ascii=False), encoding="utf-8")
    search_csv = out_dir / f"{args.date}_strict_triangle_registration_search.csv"
    with search_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(search_records[0].keys()))
        writer.writeheader()
        writer.writerows(search_records)

    by_fid: dict[int, list[np.ndarray]] = {}
    offset = np.asarray([dc, dr], dtype=np.float64)
    for item in triangles:
        fid = int(item["feature"].get("properties", {}).get("fid", -1))
        by_fid.setdefault(fid, []).append(item["xy"] + offset)
    refined_union = np.zeros(amplitude.shape, dtype=bool)
    refinement_rows: list[dict] = []
    for fid, fid_triangles in by_fid.items():
        mask0 = rasterize_triangles(fid_triangles, amplitude.shape)
        refined, threshold = refine_triangle_mask(mask0, amplitude, args.kappa, args.min_component)
        refined_union |= refined
        refinement_rows.append(
            {
                "fid": fid,
                "registered_mask_pixels": int(mask0.sum()),
                "registered_refined_pixels": int(refined.sum()),
                "amplitude_threshold": threshold,
                "registration_row_shift": dr,
                "registration_col_shift": dc,
            }
        )
    refinement_csv = out_dir / f"{args.date}_registered_triangle_refinement_metrics.csv"
    with refinement_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(refinement_rows[0].keys()))
        writer.writeheader()
        writer.writerows(refinement_rows)

    segments = triangle_segments(triangles, dr, dc)
    projection_png = pic_dir / f"{args.date}_registered_strict_triangle_projection.png"
    refinement_png = pic_dir / f"{args.date}_registered_strict_triangle_refined_mask.png"
    plot_projection(projection_png, amplitude, segments, args.date, dr, dc)
    plot_refined(refinement_png, amplitude, refined_union, segments["roof"], args.date, dr, dc)

    summary = {
        "date": args.date,
        "triangles": len(triangles),
        "registration_surface": args.registration_surface,
        "registration_triangles": len(registration_triangles),
        "buildings": len(by_fid),
        "base_score": base["score"],
        "registered_score": best["score"],
        "score_gain": best["score"] - base["score"],
        "row_shift": dr,
        "col_shift": dc,
        "registered_inside_amplitude": best["inside_amp"],
        "registered_ring_amplitude": best["ring_amp"],
        "registered_inside_edge": best["inside_edge"],
        "registered_union_mask_pixels": best["pixels"],
        "registered_refined_union_pixels": int(refined_union.sum()),
        "corrected_triangle_geojson": str(corrected_geojson),
        "registration_search_csv": str(search_csv),
        "refinement_metrics_csv": str(refinement_csv),
        "projection_png": str(projection_png),
        "refinement_png": str(refinement_png),
        "combined_figure_created": False,
    }
    summary_path = out_dir / f"{args.date}_strict_triangle_registration_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Register strict building surface triangles to real SAR amplitude.")
    parser.add_argument("--date", default="20200708")
    parser.add_argument(
        "--triangle-geojson",
        default=str(PROJECT_DIR / "results" / "outputs" / "strict_triangle_projection" / "20200708_strict_sar_surface_triangles.geojson"),
    )
    parser.add_argument("--out-dir", default=str(PROJECT_DIR / "results" / "outputs" / "strict_triangle_registration"))
    parser.add_argument("--pic-dir", default=str(PROJECT_DIR / "results" / "pic_all"))
    parser.add_argument("--max-shift", type=int, default=60)
    parser.add_argument("--coarse-step", type=int, default=3)
    parser.add_argument("--kappa", type=float, default=0.25)
    parser.add_argument("--min-component", type=int, default=2)
    parser.add_argument("--registration-surface", choices=["roof", "bottom", "wall", "all"], default="roof")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
