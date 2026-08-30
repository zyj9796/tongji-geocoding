from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-strict-triangles")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MplPath
from osgeo import gdal, ogr
from scipy.ndimage import binary_closing, label


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR / "code"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from geocode_gamma_rslc_with_buildings import llh_to_ecef, make_orbit, parse_gamma_par, read_rslc_amplitude  # noqa: E402
from io_paths import BUILDINGS_SHP, DSM_TIF, RSLC_DIR, TIF_DIR  # noqa: E402
from raster_height import RasterHeightSampler  # noqa: E402
from reproduce_thesis_tongji_tsx import ecef_to_llh, first_exterior_ring, project_llh_to_radar  # noqa: E402


def geotiff_bounds(path: Path) -> tuple[float, float, float, float]:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(path)
    gt = ds.GetGeoTransform()
    xs = [gt[0], gt[0] + ds.RasterXSize * gt[1]]
    ys = [gt[3], gt[3] + ds.RasterYSize * gt[5]]
    ds = None
    return min(xs), min(ys), max(xs), max(ys)


def clean_ring(ring: np.ndarray) -> np.ndarray:
    ring = np.asarray(ring, dtype=np.float64)[:, :2]
    if ring.shape[0] > 1 and np.allclose(ring[0], ring[-1], atol=1e-12, rtol=0):
        ring = ring[:-1]
    keep = [0]
    for i in range(1, ring.shape[0]):
        if not np.allclose(ring[i], ring[keep[-1]], atol=1e-12, rtol=0):
            keep.append(i)
    ring = ring[keep]
    if ring.shape[0] < 3:
        raise ValueError("Footprint has fewer than three unique vertices")
    return ring


def signed_area(xy: np.ndarray) -> float:
    return 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))


def point_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-12) -> bool:
    cross_ab = float(np.cross(b - a, p - a))
    cross_bc = float(np.cross(c - b, p - b))
    cross_ca = float(np.cross(a - c, p - c))
    return cross_ab >= -eps and cross_bc >= -eps and cross_ca >= -eps


def ear_clip_triangulation(ring_lonlat: np.ndarray) -> np.ndarray:
    ring = clean_ring(ring_lonlat)
    lat0 = float(np.mean(ring[:, 1]))
    xy = np.column_stack([ring[:, 0] * math.cos(math.radians(lat0)), ring[:, 1]])
    xy -= np.mean(xy, axis=0)
    order = list(range(ring.shape[0]))
    if signed_area(xy) < 0:
        order.reverse()
    triangles: list[tuple[int, int, int]] = []
    guard = 0
    while len(order) > 3:
        found = False
        scale = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-12)
        eps = scale * scale * 1e-10
        for k, curr in enumerate(order):
            prev = order[(k - 1) % len(order)]
            nxt = order[(k + 1) % len(order)]
            a, b, c = xy[prev], xy[curr], xy[nxt]
            cross = float(np.cross(b - a, c - b))
            if cross <= eps:
                continue
            if any(
                point_in_triangle(xy[idx], a, b, c, eps=eps)
                for idx in order
                if idx not in (prev, curr, nxt)
            ):
                continue
            triangles.append((prev, curr, nxt))
            del order[k]
            found = True
            break
        guard += 1
        if not found or guard > ring.shape[0] * ring.shape[0]:
            raise ValueError("Constrained footprint triangulation failed")
    triangles.append(tuple(order))
    out = np.asarray(triangles, dtype=np.int32)
    if out.shape[0] != ring.shape[0] - 2:
        raise ValueError("Unexpected footprint triangle count")
    return out


def load_buildings(path: Path, bounds: tuple[float, float, float, float], dsm: RasterHeightSampler) -> list[dict]:
    ds = ogr.Open(str(path))
    if ds is None:
        raise FileNotFoundError(path)
    layer = ds.GetLayer(0)
    layer.SetSpatialFilterRect(*bounds)
    buildings: list[dict] = []
    for feature in layer:
        ring = first_exterior_ring(feature.GetGeometryRef())
        if ring is None:
            continue
        ring = clean_ring(ring)
        floor = float(feature.GetField("Floor") or 0.0)
        height = float(feature.GetField("height") or 0.0)
        if height <= 0 and floor > 0:
            height = floor * 3.0
        if height <= 0:
            continue
        try:
            top = float(dsm.building_surface_height(ring))
        except Exception:
            continue
        buildings.append(
            {
                "fid": int(feature.GetFID()),
                "clean_id": int(feature.GetField("clean_id") or feature.GetFID()),
                "floor": int(floor),
                "height_m": height,
                "base_height_m": max(0.0, top - height),
                "top_height_m": top,
                "ring_lonlat": ring,
            }
        )
    return buildings


def build_strict_model(building: dict, par: dict, orbit) -> dict:
    ring = building["ring_lonlat"]
    n = ring.shape[0]
    bottom_h = float(building["base_height_m"])
    top_h = float(building["top_height_m"])
    vertices = np.asarray(
        [llh_to_ecef(float(lon), float(lat), bottom_h) for lon, lat in ring]
        + [llh_to_ecef(float(lon), float(lat), top_h) for lon, lat in ring],
        dtype=np.float64,
    )
    footprint_tris = ear_clip_triangulation(ring)
    triangles: list[tuple[int, int, int]] = []
    surfaces: list[str] = []
    for i in range(n):
        j = (i + 1) % n
        triangles.extend([(i, j, n + j), (i, n + j, n + i)])
        surfaces.extend(["wall", "wall"])
    for a, b, c in footprint_tris:
        triangles.append((n + int(a), n + int(b), n + int(c)))
        surfaces.append("roof")
        triangles.append((int(c), int(b), int(a)))
        surfaces.append("bottom")
    projected = []
    for xyz in vertices:
        lon, lat, height = ecef_to_llh(xyz)
        projected.append(project_llh_to_radar(lon, lat, height, par, orbit))
    return {
        "vertices_ecef": vertices,
        "projected_rc": np.asarray(projected, dtype=np.float64),
        "triangles": np.asarray(triangles, dtype=np.int32),
        "surfaces": np.asarray(surfaces),
        "footprint_triangles": int(footprint_tris.shape[0]),
    }


def rasterize_model(model: dict, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = shape
    mask = np.zeros(shape, dtype=bool)
    tri_index = np.full(shape, -1, dtype=np.int32)
    projected = model["projected_rc"]
    for ti, tri in enumerate(model["triangles"]):
        rc = projected[tri]
        xy = np.column_stack([rc[:, 1], rc[:, 0]])
        if not np.all(np.isfinite(xy)):
            continue
        c0 = max(0, int(math.floor(np.min(xy[:, 0]))) - 1)
        c1 = min(cols - 1, int(math.ceil(np.max(xy[:, 0]))) + 1)
        r0 = max(0, int(math.floor(np.min(xy[:, 1]))) - 1)
        r1 = min(rows - 1, int(math.ceil(np.max(xy[:, 1]))) + 1)
        if c1 < c0 or r1 < r0:
            continue
        yy, xx = np.mgrid[r0 : r1 + 1, c0 : c1 + 1]
        inside = MplPath(xy).contains_points(np.column_stack([xx.ravel(), yy.ravel()]), radius=1e-9).reshape(yy.shape)
        sub_mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        sub_tri = tri_index[r0 : r1 + 1, c0 : c1 + 1]
        new = inside & ~sub_mask
        sub_mask[inside] = True
        sub_tri[new] = ti
    return mask, tri_index


def refine_triangle_mask(mask0: np.ndarray, amplitude: np.ndarray, kappa: float, min_component: int) -> tuple[np.ndarray, float]:
    values = amplitude[mask0].astype(np.float64)
    if values.size == 0:
        return mask0.copy(), float("nan")
    threshold = float(values.mean() + kappa * values.std())
    refined = mask0 & (amplitude > threshold)
    refined = binary_closing(refined, structure=np.ones((3, 3), dtype=bool)) & mask0
    labels, count = label(refined)
    if count:
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_component
        keep[0] = False
        refined = keep[labels] & mask0
    return refined, threshold


def triangle_feature(building: dict, model: dict, ti: int) -> dict:
    tri = model["triangles"][ti]
    rc = model["projected_rc"][tri]
    xy = np.column_stack([rc[:, 1], rc[:, 0]])
    coords = xy.tolist() + [xy[0].tolist()]
    return {
        "type": "Feature",
        "properties": {
            "fid": building["fid"],
            "clean_id": building["clean_id"],
            "surface": str(model["surfaces"][ti]),
            "triangle_index": ti,
            "height_m": building["height_m"],
            "base_height_m": building["base_height_m"],
            "top_height_m": building["top_height_m"],
        },
        "geometry": {"type": "Polygon", "coordinates": [coords]},
    }


def write_triangle_geojson(path: Path, features: list[dict]) -> None:
    payload = {
        "type": "FeatureCollection",
        "name": "strict_building_surface_triangles_in_sar_coordinates",
        "coordinate_system": "x=range column, y=azimuth row",
        "method": "ECEF extruded building model; zero-Doppler/range projection of every triangle vertex; no global pixel shift",
        "features": features,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def plot_results(
    projection_path: Path,
    refinement_path: Path,
    amplitude: np.ndarray,
    segments: dict[str, list[np.ndarray]],
    refined_union: np.ndarray,
    date: str,
) -> None:
    show = amplitude.astype(np.float32) / 255.0
    colors = {"bottom": "#00d4ff", "wall": "#ff4fd8", "roof": "#ffb000"}
    widths = {"bottom": 0.16, "wall": 0.12, "roof": 0.18}
    alphas = {"bottom": 0.35, "wall": 0.24, "roof": 0.55}

    fig, ax = plt.subplots(figsize=(11.0, 8.0), dpi=300)
    ax.imshow(show, cmap="gray", vmin=0, vmax=1)
    for surface in ("bottom", "wall", "roof"):
        if segments[surface]:
            ax.add_collection(
                LineCollection(segments[surface], colors=colors[surface], linewidths=widths[surface], alpha=alphas[surface])
            )
    ax.set_xlim(0, amplitude.shape[1])
    ax.set_ylim(amplitude.shape[0], 0)
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    ax.set_title(f"Strict bottom-wall-roof triangle projection ({date})")
    fig.tight_layout()
    fig.savefig(projection_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.0, 8.0), dpi=300)
    ax.imshow(show, cmap="gray", vmin=0, vmax=1)
    overlay = np.zeros((*refined_union.shape, 4), dtype=np.float32)
    overlay[refined_union] = (0.15, 1.0, 0.25, 0.60)
    ax.imshow(overlay)
    if segments["roof"]:
        ax.add_collection(LineCollection(segments["roof"], colors=colors["roof"], linewidths=0.15, alpha=0.45))
    ax.set_xlim(0, amplitude.shape[1])
    ax.set_ylim(amplitude.shape[0], 0)
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    ax.set_title(f"Triangle-constrained SAR amplitude refinement ({date})")
    fig.tight_layout()
    fig.savefig(refinement_path)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    pic_dir = Path(args.pic_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pic_dir.mkdir(parents=True, exist_ok=True)
    gamma_tif = Path(args.gamma_tif) if args.gamma_tif else TIF_DIR / f"{args.date}_gamma_dem_geocoded_wgs84.tif"
    par = parse_gamma_par(RSLC_DIR / f"{args.date}.rslc.par")
    orbit = make_orbit(par)
    amplitude = read_rslc_amplitude(
        RSLC_DIR / f"{args.date}.rslc", int(par["azimuth_lines"]), int(par["range_samples"])
    )
    dsm = RasterHeightSampler(Path(args.dsm))
    buildings = load_buildings(Path(args.buildings_shp), geotiff_bounds(gamma_tif), dsm)
    if args.max_buildings > 0:
        buildings = buildings[: args.max_buildings]
    if not buildings:
        raise RuntimeError("No updated-SHP buildings intersect the SAR extent")

    refined_union = np.zeros(amplitude.shape, dtype=bool)
    features: list[dict] = []
    metrics: list[dict] = []
    skipped: list[dict] = []
    segments: dict[str, list[np.ndarray]] = {"bottom": [], "wall": [], "roof": []}
    for index, building in enumerate(buildings, start=1):
        try:
            model = build_strict_model(building, par, orbit)
            mask0, _ = rasterize_model(model, amplitude.shape)
            refined, threshold = refine_triangle_mask(mask0, amplitude, args.kappa, args.min_component)
            refined_union |= refined
            counts = {surface: int(np.sum(model["surfaces"] == surface)) for surface in segments}
            for ti, tri in enumerate(model["triangles"]):
                rc = model["projected_rc"][tri]
                xy = np.column_stack([rc[:, 1], rc[:, 0]])
                if np.all(np.isfinite(xy)):
                    closed = np.vstack([xy, xy[0]])
                    segments[str(model["surfaces"][ti])].append(closed)
                    features.append(triangle_feature(building, model, ti))
            metrics.append(
                {
                    "date": args.date,
                    "fid": building["fid"],
                    "clean_id": building["clean_id"],
                    "vertices": int(building["ring_lonlat"].shape[0]),
                    "height_m": building["height_m"],
                    "base_height_m": building["base_height_m"],
                    "top_height_m": building["top_height_m"],
                    "wall_triangles": counts["wall"],
                    "roof_triangles": counts["roof"],
                    "bottom_triangles": counts["bottom"],
                    "total_triangles": int(model["triangles"].shape[0]),
                    "mask0_pixels": int(mask0.sum()),
                    "refined_pixels": int(refined.sum()),
                    "amplitude_threshold": threshold,
                }
            )
        except Exception as exc:
            skipped.append({"fid": building["fid"], "clean_id": building["clean_id"], "reason": str(exc)})
        if index % 100 == 0:
            print(f"processed {index}/{len(buildings)} valid={len(metrics)} skipped={len(skipped)}", flush=True)

    triangle_path = out_dir / f"{args.date}_strict_sar_surface_triangles.geojson"
    metrics_path = out_dir / f"{args.date}_strict_triangle_projection_metrics.csv"
    skipped_path = out_dir / f"{args.date}_strict_triangle_projection_skipped.csv"
    projection_path = pic_dir / f"{args.date}_strict_triangle_projection.png"
    refinement_path = pic_dir / f"{args.date}_strict_triangle_refined_mask.png"
    write_triangle_geojson(triangle_path, features)
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)
    with skipped_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fid", "clean_id", "reason"])
        writer.writeheader()
        writer.writerows(skipped)
    plot_results(projection_path, refinement_path, amplitude, segments, refined_union, args.date)

    summary = {
        "date": args.date,
        "buildings_shp": str(Path(args.buildings_shp)),
        "input_buildings": len(buildings),
        "projected_buildings": len(metrics),
        "skipped_buildings": len(skipped),
        "surface_triangles": len(features),
        "wall_triangles": int(sum(row["wall_triangles"] for row in metrics)),
        "roof_triangles": int(sum(row["roof_triangles"] for row in metrics)),
        "bottom_triangles": int(sum(row["bottom_triangles"] for row in metrics)),
        "initial_mask_pixels_sum": int(sum(row["mask0_pixels"] for row in metrics)),
        "refined_mask_pixels_sum": int(sum(row["refined_pixels"] for row in metrics)),
        "refinement_kappa": args.kappa,
        "geometry_shift_applied": False,
        "paper_basis": "Sections 2.4 and 3.3-3.5: extruded bottom/wall/roof triangle mesh, per-vertex zero-Doppler/range projection, triangle rasterization, local amplitude refinement constrained by M0.",
        "triangle_geojson": str(triangle_path),
        "metrics_csv": str(metrics_path),
        "projection_png": str(projection_path),
        "refinement_png": str(refinement_path),
        "combined_figure_created": False,
    }
    summary_path = out_dir / f"{args.date}_strict_triangle_projection_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict literature-method building surface triangle projection to SAR coordinates.")
    parser.add_argument("--date", default="20200708")
    parser.add_argument("--buildings-shp", default=str(BUILDINGS_SHP))
    parser.add_argument("--dsm", default=str(DSM_TIF))
    parser.add_argument("--gamma-tif", default="")
    parser.add_argument("--out-dir", default=str(PROJECT_DIR / "results" / "outputs" / "strict_triangle_projection"))
    parser.add_argument("--pic-dir", default=str(PROJECT_DIR / "results" / "pic_all"))
    parser.add_argument("--kappa", type=float, default=0.25)
    parser.add_argument("--min-component", type=int, default=2)
    parser.add_argument("--max-buildings", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
