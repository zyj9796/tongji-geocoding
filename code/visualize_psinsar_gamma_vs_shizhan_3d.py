from __future__ import annotations

import argparse
import csv
import json
import math
import os
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

from io_paths import FULL_AREA_DIR as FULL_DIR, FULL_AREA_GEOJSON_DIR, LEGACY_IMAGE_DIR, OUTPUTS_DIR, PROJECT_DIR


COMPARE_DIR = OUTPUTS_DIR / "psinsar_vs_bc_3d"
DEFAULT_MATCH_CSV = COMPARE_DIR / "psinsar_3dlut_vs_bc_nearest_points.csv"
DEFAULT_BY_BUILDING_CSV = COMPARE_DIR / "psinsar_3dlut_vs_bc_by_building.csv"
DEFAULT_BUILDINGS = FULL_AREA_GEOJSON_DIR / "20200708_all_valid_geocoded_buildings.geojson"
DEFAULT_OUT_DIR = OUTPUTS_DIR / "psinsar_gamma_vs_bc_3d_relative_height"
EARTH_R = 6378137.0


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def local_en(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    east = (lon - lon0) * math.pi / 180.0 * EARTH_R * math.cos(math.radians(lat0))
    north = (lat - lat0) * math.pi / 180.0 * EARTH_R
    return east, north


def load_buildings(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        fid = int(props["fid"])
        ring = np.asarray(feat["geometry"]["coordinates"][0], dtype=np.float64)
        if ring.shape[0] > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        out[fid] = {
            "fid": fid,
            "floor": int(props.get("floor", 0)),
            "height_m": float(props.get("height_m", 0.0)),
            "base_height_m": float(props.get("base_height_m", 0.0)),
            "top_height_m": float(props.get("top_height_m", 0.0)),
            "ring_lonlat": ring[:, :2],
        }
    return out


def extruded_faces(building: dict, lon0: float, lat0: float, relative_height: bool = True) -> list[np.ndarray]:
    ring = building["ring_lonlat"]
    east, north = local_en(ring[:, 0], ring[:, 1], lon0, lat0)
    if relative_height:
        base = 0.0
        top = float(building["height_m"])
    else:
        base = float(building["base_height_m"])
        top = float(building["top_height_m"])
    bottom = np.column_stack([east, north, np.full_like(east, base)])
    roof = np.column_stack([east, north, np.full_like(east, top)])
    faces: list[np.ndarray] = [roof, bottom[::-1]]
    for i in range(ring.shape[0]):
        j = (i + 1) % ring.shape[0]
        faces.append(np.asarray([bottom[i], bottom[j], roof[j], roof[i]], dtype=np.float64))
    return faces


def select_fids(by_building_rows: list[dict], buildings: dict[int, dict], count: int) -> list[int]:
    rows = [r for r in by_building_rows if int(r["nearest_shizhan_fid"]) in buildings]
    rows.sort(key=lambda r: int(r["matched_ps_count"]), reverse=True)
    return [int(r["nearest_shizhan_fid"]) for r in rows[:count]]


def group_rows(match_rows: list[dict], fids: list[int], buildings: dict[int, dict], max_horizontal_m: float, height_tolerance_m: float) -> dict[int, list[dict]]:
    wanted = set(fids)
    grouped = {fid: [] for fid in fids}
    for row in match_rows:
        fid = int(row["nearest_shizhan_fid"])
        if fid not in wanted or fid not in buildings:
            continue
        if not int(float(row["within_horizontal_threshold"])):
            continue
        if float(row["ps3dlut_to_shizhan_horizontal_m"]) > max_horizontal_m:
            continue
        building = buildings[fid]
        ps_rel_h = float(row["ps_3dlut_height_m"]) - float(building["base_height_m"])
        if -height_tolerance_m <= ps_rel_h <= float(building["height_m"]) + height_tolerance_m:
            grouped[fid].append(row)
    return grouped


def sample_rows(rows: list[dict], max_points: int) -> list[dict]:
    if len(rows) <= max_points:
        return rows
    step = int(math.ceil(len(rows) / max_points))
    return rows[::step]


def make_3d_panels(out_png: Path, buildings: dict[int, dict], grouped: dict[int, list[dict]]) -> None:
    fids = list(grouped)
    ncols = 5 if len(fids) > 6 else min(3, len(fids))
    nrows = int(math.ceil(len(fids) / ncols))
    fig = plt.figure(figsize=(4.0 * ncols, 4.15 * nrows), dpi=300)
    all_sh_h = [float(r["nearest_shizhan_height_m"]) - float(buildings[fid]["base_height_m"]) for fid, rows in grouped.items() for r in rows]
    norm = Normalize(vmin=float(np.min(all_sh_h)), vmax=float(np.max(all_sh_h)))
    scatter_for_colorbar = None
    colors = ["#3b82f6", "#ef4444", "#22c55e", "#a855f7", "#f59e0b", "#06b6d4", "#84cc16", "#f43f5e", "#14b8a6", "#6366f1"]

    for i, fid in enumerate(fids):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        b = buildings[fid]
        rows = sample_rows(grouped[fid], 240)
        lon0 = float(np.mean(b["ring_lonlat"][:, 0]))
        lat0 = float(np.mean(b["ring_lonlat"][:, 1]))
        faces = extruded_faces(b, lon0, lat0, relative_height=True)
        ax.add_collection3d(Poly3DCollection(faces, facecolor=colors[i % len(colors)], edgecolor="#111111", linewidth=0.35, alpha=0.16))

        ps = np.asarray([[float(r["ps_3dlut_lon"]), float(r["ps_3dlut_lat"]), float(r["ps_3dlut_height_m"])] for r in rows], dtype=np.float64)
        sh = np.asarray([[float(r["nearest_shizhan_lon"]), float(r["nearest_shizhan_lat"]), float(r["nearest_shizhan_height_m"])] for r in rows], dtype=np.float64)
        ps[:, 2] -= float(b["base_height_m"])
        sh[:, 2] -= float(b["base_height_m"])
        pe, pn = local_en(ps[:, 0], ps[:, 1], lon0, lat0)
        se, sn = local_en(sh[:, 0], sh[:, 1], lon0, lat0)

        ax.scatter(pe, pn, ps[:, 2], s=7, c="#f97316", alpha=0.46, depthshade=False, label="PS 3D / GAMMA")
        sc = ax.scatter(se, sn, sh[:, 2], s=8, c=sh[:, 2], cmap="viridis", norm=norm, alpha=0.92, depthshade=False, label="Shizhan")
        scatter_for_colorbar = sc
        for k in range(0, ps.shape[0], max(1, ps.shape[0] // 36)):
            ax.plot([pe[k], se[k]], [pn[k], sn[k]], [ps[k, 2], sh[k, 2]], color="#666666", linewidth=0.35, alpha=0.42)

        xy = np.vstack([np.column_stack([face[:, 0], face[:, 1]]) for face in faces])
        all_x = np.concatenate([xy[:, 0], pe, se])
        all_y = np.concatenate([xy[:, 1], pn, sn])
        all_z = np.concatenate([np.asarray([p[2] for face in faces for p in face]), ps[:, 2], sh[:, 2]])
        xmid = float((np.min(all_x) + np.max(all_x)) / 2)
        ymid = float((np.min(all_y) + np.max(all_y)) / 2)
        radius = max(float(np.ptp(all_x)), float(np.ptp(all_y)), 20.0) / 2 * 1.18
        ax.set_xlim(xmid - radius, xmid + radius)
        ax.set_ylim(ymid - radius, ymid + radius)
        ax.set_zlim(min(-3.0, float(np.min(all_z)) - 3.0), max(float(b["height_m"]) + 4.0, float(np.max(all_z)) + 4.0))
        ax.set_xlabel("East / m", fontsize=8)
        ax.set_ylabel("North / m", fontsize=8)
        ax.set_zlabel("Height above building base / m", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=25, azim=-52)
        ax.set_title(f"FID {fid}: {len(grouped[fid])} valid PS, {b['floor']}F/{b['height_m']:.0f}m", fontsize=8.5)

    fig.suptitle("Relative-Height Comparison: PS 3D/GAMMA vs. Shizhan Building-Constrained Method", fontsize=13)
    fig.subplots_adjust(left=0.025, right=0.91, bottom=0.04, top=0.90, wspace=0.04, hspace=0.23)
    if scatter_for_colorbar is not None:
        cax = fig.add_axes([0.93, 0.20, 0.012, 0.58])
        cbar = fig.colorbar(scatter_for_colorbar, cax=cax)
        cbar.set_label("Shizhan method height above base / m")
    fig.savefig(out_png)
    plt.close(fig)


def make_vector_map(out_png: Path, buildings: dict[int, dict], grouped: dict[int, list[dict]]) -> None:
    rings = [buildings[fid]["ring_lonlat"] for fid in grouped]
    all_ring = np.vstack(rings)
    xpad = max(float(np.ptp(all_ring[:, 0])) * 0.45, 0.0012)
    ypad = max(float(np.ptp(all_ring[:, 1])) * 0.45, 0.0012)
    xlim = (float(np.min(all_ring[:, 0]) - xpad), float(np.max(all_ring[:, 0]) + xpad))
    ylim = (float(np.min(all_ring[:, 1]) - ypad), float(np.max(all_ring[:, 1]) + ypad))
    fig, ax = plt.subplots(figsize=(9.2, 8.0), dpi=300)
    colors = ["#3b82f6", "#ef4444", "#22c55e", "#a855f7", "#f59e0b", "#06b6d4", "#84cc16", "#f43f5e", "#14b8a6", "#6366f1"]

    for i, fid in enumerate(grouped, start=1):
        b = buildings[fid]
        ring = b["ring_lonlat"]
        color = colors[(i - 1) % len(colors)]
        ax.add_patch(MplPolygon(ring, closed=True, fill=True, facecolor=color, edgecolor="#111111", linewidth=0.8, alpha=0.26, zorder=3))
        rows = sample_rows(grouped[fid], 500)
        ps_lon = np.asarray([float(r["ps_3dlut_lon"]) for r in rows], dtype=np.float64)
        ps_lat = np.asarray([float(r["ps_3dlut_lat"]) for r in rows], dtype=np.float64)
        sh_lon = np.asarray([float(r["nearest_shizhan_lon"]) for r in rows], dtype=np.float64)
        sh_lat = np.asarray([float(r["nearest_shizhan_lat"]) for r in rows], dtype=np.float64)
        ax.scatter(ps_lon, ps_lat, s=4, c="#f97316", alpha=0.38, linewidths=0, zorder=4)
        ax.scatter(sh_lon, sh_lat, s=3, c=color, alpha=0.52, linewidths=0, zorder=5)
        cx = float(np.mean(ring[:, 0]))
        cy = float(np.mean(ring[:, 1]))
        ax.text(cx, cy, f"{i}\nFID {fid}\nPS {len(grouped[fid])}", ha="center", va="center", fontsize=6.6, bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.88, "pad": 1.0}, zorder=6)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude / deg")
    ax.set_ylabel("Latitude / deg")
    ax.set_title("Selected Buildings: Height-Valid PS 3D/GAMMA Points and Shizhan Method Points")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(color="#dddddd", linewidth=0.3)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def write_summary(path: Path, grouped: dict[int, list[dict]], buildings: dict[int, dict]) -> None:
    fields = [
        "fid",
        "floor",
        "height_m",
        "matched_ps_count",
        "horizontal_m_median",
        "horizontal_m_p90",
        "ps_minus_shizhan_height_m_median",
        "abs_height_diff_m_median",
        "distance_3d_m_median",
        "ps_relative_height_m_median",
        "shizhan_relative_height_m_median",
    ]
    rows = []
    for fid, items in grouped.items():
        h = np.asarray([float(r["ps3dlut_to_shizhan_horizontal_m"]) for r in items], dtype=np.float64)
        z = np.asarray([float(r["ps3dlut_minus_shizhan_height_m"]) for r in items], dtype=np.float64)
        d3 = np.asarray([float(r["ps3dlut_to_shizhan_3d_m"]) for r in items], dtype=np.float64)
        b = buildings[fid]
        ps_rel = np.asarray([float(r["ps_3dlut_height_m"]) - float(b["base_height_m"]) for r in items], dtype=np.float64)
        sh_rel = np.asarray([float(r["nearest_shizhan_height_m"]) - float(b["base_height_m"]) for r in items], dtype=np.float64)
        rows.append(
            {
                "fid": fid,
                "floor": b["floor"],
                "height_m": b["height_m"],
                "matched_ps_count": len(items),
                "horizontal_m_median": float(np.percentile(h, 50)),
                "horizontal_m_p90": float(np.percentile(h, 90)),
                "ps_minus_shizhan_height_m_median": float(np.percentile(z, 50)),
                "abs_height_diff_m_median": float(np.percentile(np.abs(z), 50)),
                "distance_3d_m_median": float(np.percentile(d3, 50)),
                "ps_relative_height_m_median": float(np.percentile(ps_rel, 50)),
                "shizhan_relative_height_m_median": float(np.percentile(sh_rel, 50)),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    LEGACY_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    buildings = load_buildings(Path(args.buildings_geojson))
    by_building = read_csv(Path(args.by_building_csv))
    match_rows = read_csv(Path(args.match_csv))
    candidate_fids = select_fids(by_building, buildings, max(int(args.count) * 30, int(args.count)))
    grouped_all = group_rows(match_rows, candidate_fids, buildings, float(args.max_horizontal_m), float(args.height_tolerance_m))
    sorted_fids = sorted([fid for fid, rows in grouped_all.items() if rows], key=lambda fid: len(grouped_all[fid]), reverse=True)[: int(args.count)]
    grouped = {fid: grouped_all[fid] for fid in sorted_fids}
    selected_buildings = {fid: buildings[fid] for fid in grouped}

    make_3d_panels(LEGACY_IMAGE_DIR / "fig_psinsar_gamma_vs_bc_3d_buildings.png", selected_buildings, grouped)
    make_vector_map(LEGACY_IMAGE_DIR / "fig_psinsar_gamma_vs_bc_vector_map.png", selected_buildings, grouped)
    write_summary(out_dir / "psinsar_gamma_vs_bc_selected_buildings.csv", grouped, selected_buildings)
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Relative-Height PS-InSAR/GAMMA 3D Coordinates vs Shizhan Method",
                "",
                f"Matched point source: `{Path(args.match_csv)}`",
                f"Building source: `{Path(args.buildings_geojson)}`",
                "",
                "Each building is plotted in a local relative-height frame: building base = 0 m, roof = vector `height_m`.",
                "Orange points are PS-InSAR 3DLUT coordinates after subtracting that building's `base_height_m`; these are used here as the GAMMA/PS-InSAR geocoding result.",
                "Height-colored points are nearest shizhan building-constrained method points in the same relative-height frame.",
                f"PS points are filtered to `[-{float(args.height_tolerance_m):g}, height_m + {float(args.height_tolerance_m):g}]` m before plotting.",
                "Gray line segments connect each PS point to its nearest shizhan point in horizontal coordinates.",
                "",
                "Outputs:",
                "- `../pic_all/fig_psinsar_gamma_vs_bc_3d_buildings.png`: 3D building-vector extrusion with PS/GAMMA and building-constrained point comparison.",
                "- `../pic_all/fig_psinsar_gamma_vs_bc_vector_map.png`: footprint map showing selected buildings and both point sets.",
                "- `psinsar_gamma_vs_bc_selected_buildings.csv`: per-building counts and distance statistics.",
                "",
                f"Selected FIDs: {', '.join(str(fid) for fid in grouped)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"selected_fids={list(grouped)}")
    print(f"out_dir={out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-csv", default=str(DEFAULT_MATCH_CSV))
    parser.add_argument("--by-building-csv", default=str(DEFAULT_BY_BUILDING_CSV))
    parser.add_argument("--buildings-geojson", default=str(DEFAULT_BUILDINGS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--max-horizontal-m", type=float, default=15.0)
    parser.add_argument("--height-tolerance-m", type=float, default=3.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
