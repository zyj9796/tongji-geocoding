from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("FONTCONFIG_PATH", "/etc/fonts")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon as MplPolygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from io_paths import FULL_AREA_GEOJSON_DIR, PS_POINTS_CSV, REPO_ROOT, RSLC_DIR, SAME_PIXEL_DIR, SAME_PIXEL_IMAGE_DIR, TRASH_DIR

sys.path.insert(0, str(REPO_ROOT / "src"))

from geocode_gamma_rslc_with_buildings import make_orbit, parse_gamma_par
from reproduce_thesis_tongji_tsx import barycentric, ecef_to_llh, local_en, rasterize_building


DEFAULT_PS_CSV = PS_POINTS_CSV
DEFAULT_BUILDINGS = FULL_AREA_GEOJSON_DIR / "20200708_all_valid_geocoded_buildings.geojson"
DEFAULT_BY_BUILDING = TRASH_DIR / "psinsar_vs_bc_3d" / "psinsar_3dlut_vs_bc_by_building.csv"
DEFAULT_RSLC_PAR = RSLC_DIR / "20220601.rslc.par"
DEFAULT_OUT_DIR = SAME_PIXEL_DIR
EARTH_R = 6378137.0


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def value(row: dict, *names: str, default: str | None = None) -> str:
    for name in names:
        if name in row and row[name] not in ("", None):
            return row[name]
    if default is not None:
        return default
    raise KeyError(f"missing any of columns: {', '.join(names)}")


def load_buildings(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    buildings = {}
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        fid = int(props["fid"])
        ring = np.asarray(feat["geometry"]["coordinates"][0], dtype=np.float64)
        if ring.shape[0] > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        buildings[fid] = {
            "fid": fid,
            "floor": int(props.get("floor", 0)),
            "height_m": float(props.get("height_m", 0.0)),
            "base_height_m": float(props.get("base_height_m", 0.0)),
            "top_height_m": float(props.get("top_height_m", 0.0)),
            "ring_lonlat": ring[:, :2],
        }
    return buildings


def candidate_fids(by_building_csv: Path, buildings: dict[int, dict], count: int) -> list[int]:
    if not by_building_csv.exists():
        fallback_paths = [
            TRASH_DIR / "psinsar_vs_bc_3d" / "psinsar_3dlut_vs_bc_by_building.csv",
            TRASH_DIR / "psinsar_vs_shizhan_3d" / "psinsar_3dlut_vs_shizhan_by_building.csv",
        ]
        for trash_csv in fallback_paths:
            if trash_csv.exists():
                by_building_csv = trash_csv
                break
    rows = read_csv(by_building_csv)
    rows.sort(key=lambda r: int(r["matched_ps_count"]), reverse=True)
    out = []
    for row in rows:
        fid = int(row["nearest_shizhan_fid"])
        if fid in buildings:
            out.append(fid)
        if len(out) >= count:
            break
    return out


def ps_pixel(row: dict) -> tuple[float, float, int, int]:
    # STAMPS candidate files are 1-based; numpy/SAR arrays are 0-based.
    row_f = float(value(row, "line", "azimuth_pixel")) - 1.0
    col_f = float(value(row, "pixel", "range_pixel")) - 1.0
    return row_f, col_f, int(round(row_f)), int(round(col_f))


def invert_same_pixel(model: dict, row_f: float, col_f: float, row_i: int, col_i: int) -> tuple[float, float, float, int] | None:
    tri_idx = model["tri_idx"]
    if row_i < 0 or col_i < 0 or row_i >= tri_idx.shape[0] or col_i >= tri_idx.shape[1]:
        return None
    ti = int(tri_idx[row_i, col_i])
    if ti < 0:
        return None
    tri = model["triangles"][ti]
    proj_xy = np.column_stack([model["projected_rc"][:, 1], model["projected_rc"][:, 0]])
    bc = barycentric(np.asarray([col_f, row_f], dtype=np.float64), proj_xy[tri])
    if bc is None:
        return None
    u, v, w = bc
    if min(u, v, w) < -0.08:
        return None
    xyz = u * model["vertices_ecef"][tri[0]] + v * model["vertices_ecef"][tri[1]] + w * model["vertices_ecef"][tri[2]]
    lon, lat, h = ecef_to_llh(xyz)
    return float(lon), float(lat), float(h), ti


def local_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lat0 = (lat1 + lat2) / 2.0
    dx = (lon1 - lon2) * math.pi / 180.0 * EARTH_R * math.cos(math.radians(lat0))
    dy = (lat1 - lat2) * math.pi / 180.0 * EARTH_R
    return float(math.hypot(dx, dy))


def build_exact_matches(ps_rows: list[dict], buildings: dict[int, dict], models: dict[int, dict], height_tolerance_m: float) -> list[dict]:
    matches = []
    for ps in ps_rows:
        row_f, col_f, row_i, col_i = ps_pixel(ps)
        for fid, model in models.items():
            inv = invert_same_pixel(model, row_f, col_f, row_i, col_i)
            if inv is None:
                continue
            b = buildings[fid]
            sh_lon, sh_lat, sh_h, tri_idx = inv
            ps_h = float(value(ps, "lut_z", "height_m", "z_dsm_m", "height"))
            ps_rel = ps_h - float(b["base_height_m"])
            sh_rel = sh_h - float(b["base_height_m"])
            ps_lon = float(value(ps, "lut_lon", "longitude", "lon"))
            ps_lat = float(value(ps, "lut_lat", "latitude", "lat"))
            horizontal_m = local_distance_m(ps_lon, ps_lat, sh_lon, sh_lat)
            height_diff = ps_h - sh_h
            valid_h = -height_tolerance_m <= ps_rel <= float(b["height_m"]) + height_tolerance_m
            matches.append(
                {
                    "ps_id": int(float(ps["ps_id"])),
                    "fid": fid,
                    "ps_line_1based": float(value(ps, "line", "azimuth_pixel")),
                    "ps_pixel_1based": float(value(ps, "pixel", "range_pixel")),
                    "sar_row_0based": row_f,
                    "sar_col_0based": col_f,
                    "ps_lon": ps_lon,
                    "ps_lat": ps_lat,
                    "ps_height_m": ps_h,
                    "ps_relative_height_m": ps_rel,
                    "shizhan_lon_same_pixel": sh_lon,
                    "shizhan_lat_same_pixel": sh_lat,
                    "shizhan_height_m": sh_h,
                    "shizhan_relative_height_m": sh_rel,
                    "same_pixel_horizontal_m": horizontal_m,
                    "ps_minus_shizhan_height_m": height_diff,
                    "same_pixel_3d_m": float(math.hypot(horizontal_m, height_diff)),
                    "triangle_index": tri_idx,
                    "height_valid": int(valid_h),
                    "building_floor": int(b["floor"]),
                    "building_height_m": float(b["height_m"]),
                    "building_base_height_m": float(b["base_height_m"]),
                    "building_top_height_m": float(b["top_height_m"]),
                    "coherence": float(ps.get("coherence", "nan")),
                }
            )
            break
    return matches


def choose_plot_fids(matches: list[dict], count: int) -> list[int]:
    counts: dict[int, int] = {}
    for row in matches:
        if int(row["height_valid"]):
            counts[int(row["fid"])] = counts.get(int(row["fid"]), 0) + 1
    return [fid for fid, _n in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:count]]


def extruded_faces(building: dict, lon0: float, lat0: float) -> list[np.ndarray]:
    ring = building["ring_lonlat"]
    east, north = local_en(ring[:, 0], ring[:, 1], lon0, lat0)
    bottom = np.column_stack([east, north, np.zeros_like(east)])
    roof = np.column_stack([east, north, np.full_like(east, float(building["height_m"]))])
    faces: list[np.ndarray] = [roof, bottom[::-1]]
    for i in range(ring.shape[0]):
        j = (i + 1) % ring.shape[0]
        faces.append(np.asarray([bottom[i], bottom[j], roof[j], roof[i]], dtype=np.float64))
    return faces


def sample_rows(rows: list[dict], max_points: int) -> list[dict]:
    if len(rows) <= max_points:
        return rows
    step = int(math.ceil(len(rows) / max_points))
    return rows[::step]


def plot_3d(out_png: Path, matches: list[dict], buildings: dict[int, dict], fids: list[int]) -> None:
    ncols = 5 if len(fids) > 6 else min(3, len(fids))
    nrows = int(math.ceil(len(fids) / ncols))
    fig = plt.figure(figsize=(4.0 * ncols, 4.1 * nrows), dpi=300)
    colors = ["#3b82f6", "#ef4444", "#22c55e", "#a855f7", "#f59e0b", "#06b6d4", "#84cc16", "#f43f5e", "#14b8a6", "#6366f1"]
    valid = [r for r in matches if int(r["height_valid"]) and int(r["fid"]) in set(fids)]
    norm = Normalize(vmin=0.0, vmax=max(float(buildings[fid]["height_m"]) for fid in fids))
    scatter = None
    for i, fid in enumerate(fids):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        b = buildings[fid]
        rows = sample_rows([r for r in valid if int(r["fid"]) == fid], 260)
        lon0 = float(np.mean(b["ring_lonlat"][:, 0]))
        lat0 = float(np.mean(b["ring_lonlat"][:, 1]))
        faces = extruded_faces(b, lon0, lat0)
        ax.add_collection3d(Poly3DCollection(faces, facecolor=colors[i % len(colors)], edgecolor="#111111", linewidth=0.35, alpha=0.16))
        ps = np.asarray([[float(r["ps_lon"]), float(r["ps_lat"]), float(r["ps_relative_height_m"])] for r in rows], dtype=np.float64)
        sh = np.asarray([[float(r["shizhan_lon_same_pixel"]), float(r["shizhan_lat_same_pixel"]), float(r["shizhan_relative_height_m"])] for r in rows], dtype=np.float64)
        pe, pn = local_en(ps[:, 0], ps[:, 1], lon0, lat0)
        se, sn = local_en(sh[:, 0], sh[:, 1], lon0, lat0)
        ax.scatter(pe, pn, ps[:, 2], s=7, c="#f97316", alpha=0.48, depthshade=False, label="PS/GAMMA same pixel")
        scatter = ax.scatter(se, sn, sh[:, 2], s=8, c=sh[:, 2], cmap="viridis", norm=norm, alpha=0.92, depthshade=False, label="Shizhan same pixel")
        for k in range(0, ps.shape[0], max(1, ps.shape[0] // 36)):
            ax.plot([pe[k], se[k]], [pn[k], sn[k]], [ps[k, 2], sh[k, 2]], color="#666666", linewidth=0.35, alpha=0.42)
        xy = np.vstack([np.column_stack([face[:, 0], face[:, 1]]) for face in faces])
        all_x = np.concatenate([xy[:, 0], pe, se])
        all_y = np.concatenate([xy[:, 1], pn, sn])
        radius = max(float(np.ptp(all_x)), float(np.ptp(all_y)), 20.0) / 2 * 1.18
        ax.set_xlim(float((np.min(all_x) + np.max(all_x)) / 2) - radius, float((np.min(all_x) + np.max(all_x)) / 2) + radius)
        ax.set_ylim(float((np.min(all_y) + np.max(all_y)) / 2) - radius, float((np.min(all_y) + np.max(all_y)) / 2) + radius)
        ax.set_zlim(-3.0, float(b["height_m"]) + 4.0)
        ax.set_xlabel("East / m", fontsize=8)
        ax.set_ylabel("North / m", fontsize=8)
        ax.set_zlabel("Height above base / m", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=25, azim=-52)
        ax.set_title(f"FID {fid}: {len(rows)} same PS pixels, {b['floor']}F/{b['height_m']:.0f}m", fontsize=8.5)
    fig.suptitle("Same PS Pixel Comparison: PS/GAMMA 3D vs. Shizhan Building-Constrained Inversion", fontsize=13)
    fig.subplots_adjust(left=0.025, right=0.91, bottom=0.04, top=0.90, wspace=0.04, hspace=0.23)
    if scatter is not None:
        cax = fig.add_axes([0.93, 0.20, 0.012, 0.58])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label("Shizhan height above base / m")
    fig.savefig(out_png)
    plt.close(fig)


def plot_radar(out_png: Path, matches: list[dict], models: dict[int, dict], fids: list[int]) -> None:
    ncols = 5 if len(fids) > 6 else min(3, len(fids))
    nrows = int(math.ceil(len(fids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 3.35 * nrows), dpi=300, squeeze=False)
    colors = ["#3b82f6", "#ef4444", "#22c55e", "#a855f7", "#f59e0b", "#06b6d4", "#84cc16", "#f43f5e", "#14b8a6", "#6366f1"]
    valid = [r for r in matches if int(r["height_valid"]) and int(r["fid"]) in set(fids)]
    for i, fid in enumerate(fids):
        ax = axes.ravel()[i]
        rows = [r for r in valid if int(r["fid"]) == fid]
        model = models[fid]
        rc = np.asarray([[float(r["sar_row_0based"]), float(r["sar_col_0based"])] for r in rows], dtype=np.float64)
        all_rc = np.vstack([rc, model["projected_rc"]])
        r0 = max(0, int(np.nanmin(all_rc[:, 0])) - 18)
        r1 = min(model["tri_idx"].shape[0] - 1, int(np.nanmax(all_rc[:, 0])) + 18)
        c0 = max(0, int(np.nanmin(all_rc[:, 1])) - 18)
        c1 = min(model["tri_idx"].shape[1] - 1, int(np.nanmax(all_rc[:, 1])) + 18)
        ax.set_facecolor("#050505")
        for tri in model["triangles"]:
            pts = np.column_stack([model["projected_rc"][tri, 1], model["projected_rc"][tri, 0]])
            ax.add_patch(MplPolygon(pts, closed=True, fill=False, edgecolor=colors[i % len(colors)], linewidth=0.36, alpha=0.58))
        ax.scatter(rc[:, 1], rc[:, 0], s=6, c="#f97316", alpha=0.72, linewidths=0)
        ax.set_xlim(c0, c1)
        ax.set_ylim(r1, r0)
        ax.set_xlabel("Range column")
        ax.set_ylabel("Azimuth row")
        ax.set_title(f"FID {fid}: exact PS pixels", fontsize=8.5)
        ax.grid(color="white", linewidth=0.22, alpha=0.18)
    for ax in axes.ravel()[len(fids) :]:
        ax.axis("off")
    fig.suptitle("Same PS Pixels Inside Building Projection Masks", fontsize=13)
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.07, top=0.86, wspace=0.28, hspace=0.55)
    fig.savefig(out_png)
    plt.close(fig)


def summarize(matches: list[dict], fids: list[int]) -> list[dict]:
    out = []
    for fid in fids:
        rows = [r for r in matches if int(r["fid"]) == fid]
        valid = [r for r in rows if int(r["height_valid"])]
        if not rows:
            continue
        for_stats = valid if valid else rows
        h = np.asarray([float(r["same_pixel_horizontal_m"]) for r in for_stats], dtype=np.float64)
        z = np.asarray([float(r["ps_minus_shizhan_height_m"]) for r in for_stats], dtype=np.float64)
        out.append(
            {
                "fid": fid,
                "same_pixel_ps_count": len(rows),
                "height_valid_ps_count": len(valid),
                "height_valid_ratio": float(len(valid) / len(rows)),
                "horizontal_m_median": float(np.percentile(h, 50)),
                "horizontal_m_p90": float(np.percentile(h, 90)),
                "ps_minus_shizhan_height_m_median": float(np.percentile(z, 50)),
                "abs_height_diff_m_median": float(np.percentile(np.abs(z), 50)),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    SAME_PIXEL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    buildings = load_buildings(Path(args.buildings_geojson))
    fids = candidate_fids(Path(args.by_building_csv), buildings, int(args.candidate_buildings))
    par = parse_gamma_par(Path(args.rslc_par))
    orbit = make_orbit(par)
    shape = (int(par["azimuth_lines"]), int(par["range_samples"]))
    models = {fid: rasterize_building(buildings[fid], par, orbit, shape) for fid in fids}
    matches = build_exact_matches(read_csv(Path(args.ps_csv)), buildings, models, float(args.height_tolerance_m))
    selected = choose_plot_fids(matches, int(args.count))
    selected_buildings = {fid: buildings[fid] for fid in selected}
    selected_models = {fid: models[fid] for fid in selected}

    write_csv(out_dir / "same_ps_pixel_ps_gamma_vs_bc_points.csv", matches)
    write_csv(out_dir / "same_ps_pixel_selected_buildings_stats.csv", summarize(matches, selected))
    plot_3d(SAME_PIXEL_IMAGE_DIR / "图_同像素永久散射体三维建筑.png", matches, selected_buildings, selected)
    plot_radar(SAME_PIXEL_IMAGE_DIR / "图_同像素永久散射体雷达像素.png", matches, selected_models, selected)
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Same PS Pixel Comparison",
                "",
                "This result starts from the same PS radar pixel. STAMPS `line/pixel` are converted from 1-based to 0-based SAR row/column.",
                f"Building projection and shizhan inversion use the PS master RSLC parameter file: `{Path(args.rslc_par)}`.",
                "For each PS pixel, shizhan coordinates are recomputed by barycentric inversion on the building projection triangle containing that exact pixel.",
                "The 3D figure uses relative height per building: base = 0 m, roof = vector `height_m`.",
                f"Height-valid PS filter for plotting: `[-{float(args.height_tolerance_m):g}, height_m + {float(args.height_tolerance_m):g}]` m.",
                "",
                "Outputs:",
                "- `same_ps_pixel_ps_gamma_vs_bc_points.csv`: per-PS same-pixel comparison.",
                "- `same_ps_pixel_selected_buildings_stats.csv`: selected building statistics.",
                "- `../pic_all/图_同像素永久散射体三维建筑.png`: strict same-pixel 3D comparison.",
                "- `../pic_all/图_同像素永久散射体雷达像素.png`: exact PS pixels in SAR coordinates.",
                "",
                f"Selected FIDs: {', '.join(str(x) for x in selected)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"same_pixel_matches={len(matches)}")
    print(f"selected_fids={selected}")
    print(f"out_dir={out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ps-csv", default=str(DEFAULT_PS_CSV))
    parser.add_argument("--buildings-geojson", default=str(DEFAULT_BUILDINGS))
    parser.add_argument("--by-building-csv", default=str(DEFAULT_BY_BUILDING))
    parser.add_argument("--rslc-par", default=str(DEFAULT_RSLC_PAR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--candidate-buildings", type=int, default=120)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--height-tolerance-m", type=float, default=3.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
