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
from osgeo import ogr

from io_paths import BUILDINGS_SHP, FULL_AREA_DIR as FULL_DIR, FULL_AREA_GEOJSON_DIR, PROJECT_DIR, REPO_ROOT, RESULTS_DIR, RSLC_DIR, TYPICAL_DIR, TYPICAL_IMAGE_DIR

sys.path.insert(0, str(REPO_ROOT / "src"))

from geocode_gamma_rslc_with_buildings import make_orbit, parse_gamma_par, read_rslc_amplitude
from reproduce_thesis_tongji_tsx import local_en, rasterize_building


DATE = "20200708"
OUT_DIR = TYPICAL_DIR


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_valid_buildings(path: Path) -> dict[int, dict]:
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


def load_all_building_rings(path: Path) -> list[np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rings = []
    for feat in data.get("features", []):
        ring = np.asarray(feat["geometry"]["coordinates"][0], dtype=np.float64)
        if ring.shape[0] > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        rings.append(ring[:, :2])
    return rings


def read_shp_attributes(path: Path) -> dict[int, dict]:
    ds = ogr.Open(str(path))
    if ds is None:
        raise FileNotFoundError(path)
    lyr = ds.GetLayer(0)
    attrs = {}
    for feat in lyr:
        fid = int(feat.GetFID())
        floor = int(feat.GetField("Floor") or 0)
        height = float(feat.GetField("height") or 0.0)
        height_source = "height"
        if height <= 0 and floor > 0:
            height = floor * 3.0
            height_source = "Floor*3m"
        attrs[fid] = {"floor": floor, "height_m": height, "height_source": height_source, "id": feat.GetField("Id")}
    return attrs


def choose_typical_buildings(stats: list[dict], count: int) -> list[int]:
    valid = [r for r in stats if int(float(r["sample_points"])) >= 30]
    valid.sort(key=lambda r: float(r["gamma_dsm_mean_boundary_distance_m"]), reverse=True)
    picked: list[int] = []
    height_bins = [(70, math.inf), (45, 70), (25, 45), (0, 25)]
    for lo, hi in height_bins:
        for row in valid:
            fid = int(row["fid"])
            h = float(row["height_m"])
            if fid not in picked and lo <= h < hi:
                picked.append(fid)
                break
        if len(picked) >= count:
            return picked[:count]
    for row in valid:
        fid = int(row["fid"])
        if fid not in picked:
            picked.append(fid)
        if len(picked) >= count:
            break
    return picked[:count]


def points_for_fids(points_csv: Path, fids: list[int]) -> dict[int, list[dict]]:
    wanted = set(fids)
    out = {fid: [] for fid in fids}
    with points_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fid = int(row["fid"])
            if fid in wanted:
                out[fid].append(row)
    return out


def extruded_faces(building: dict, lon0: float, lat0: float) -> list[np.ndarray]:
    ring = building["ring_lonlat"]
    east, north = local_en(ring[:, 0], ring[:, 1], lon0, lat0)
    base = float(building["base_height_m"])
    top = float(building["top_height_m"])
    bottom = np.column_stack([east, north, np.full_like(east, base)])
    roof = np.column_stack([east, north, np.full_like(east, top)])
    faces: list[np.ndarray] = [roof, bottom[::-1]]
    for i in range(ring.shape[0]):
        j = (i + 1) % ring.shape[0]
        faces.append(np.asarray([bottom[i], bottom[j], roof[j], roof[i]], dtype=np.float64))
    return faces


def make_3d_plot(out_png: Path, buildings: dict[int, dict], points: dict[int, list[dict]], stats_by_fid: dict[int, dict], attrs: dict[int, dict]) -> None:
    n = len(buildings)
    ncols = 5 if n > 6 else min(3, n)
    nrows = int(math.ceil(n / ncols))
    fig = plt.figure(figsize=(3.95 * ncols, 4.2 * nrows), dpi=300)
    colors = ["#3b82f6", "#ef4444", "#22c55e", "#a855f7", "#f59e0b", "#06b6d4", "#84cc16", "#f43f5e", "#14b8a6", "#6366f1"]
    scatter_for_colorbar = None
    all_method_heights = []
    for rows in points.values():
        all_method_heights.extend(float(r["method_height_m"]) for r in rows)
    norm = Normalize(vmin=float(np.min(all_method_heights)), vmax=float(np.max(all_method_heights)))
    for i, fid in enumerate(buildings):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        b = buildings[fid]
        color = colors[i % len(colors)]
        lon0 = float(np.mean(b["ring_lonlat"][:, 0]))
        lat0 = float(np.mean(b["ring_lonlat"][:, 1]))
        faces = extruded_faces(b, lon0, lat0)
        ax.add_collection3d(Poly3DCollection(faces, facecolor=color, edgecolor="#111111", linewidth=0.35, alpha=0.15))
        rows = points[fid]
        if not rows:
            continue
        method = np.asarray([[float(r["method_lon"]), float(r["method_lat"]), float(r["method_height_m"])] for r in rows], dtype=np.float64)
        gamma = np.asarray([[float(r["gamma_dsm_lon"]), float(r["gamma_dsm_lat"]), float(r["gamma_dsm_height_m"])] for r in rows], dtype=np.float64)
        step = max(1, method.shape[0] // 220)
        me, mn = local_en(method[::step, 0], method[::step, 1], lon0, lat0)
        ge, gn = local_en(gamma[::step, 0], gamma[::step, 1], lon0, lat0)
        ax.scatter(ge, gn, gamma[::step, 2], s=7, c="#f97316", alpha=0.38, depthshade=False)
        sc = ax.scatter(me, mn, method[::step, 2], s=8, c=method[::step, 2], cmap="viridis", norm=norm, alpha=0.9, depthshade=False)
        scatter_for_colorbar = sc
        for k in range(0, me.size, max(1, me.size // 28)):
            ax.plot([ge[k], me[k]], [gn[k], mn[k]], [gamma[::step, 2][k], method[::step, 2][k]], color="#777777", linewidth=0.35, alpha=0.38)
        stat = stats_by_fid[fid]
        shp = attrs.get(fid, {})
        xy = np.vstack([np.column_stack([face[:, 0], face[:, 1]]) for face in faces])
        all_x = np.concatenate([xy[:, 0], me, ge])
        all_y = np.concatenate([xy[:, 1], mn, gn])
        all_z = np.concatenate([np.asarray([p[2] for face in faces for p in face]), method[::step, 2], gamma[::step, 2]])
        xmid = float((np.min(all_x) + np.max(all_x)) / 2)
        ymid = float((np.min(all_y) + np.max(all_y)) / 2)
        radius = max(float(np.ptp(all_x)), float(np.ptp(all_y)), 20.0) / 2 * 1.15
        ax.set_xlim(xmid - radius, xmid + radius)
        ax.set_ylim(ymid - radius, ymid + radius)
        ax.set_zlim(max(0.0, float(np.min(all_z)) - 6.0), float(np.max(all_z)) + 8.0)
        ax.set_xlabel("East / m", fontsize=8)
        ax.set_ylabel("North / m", fontsize=8)
        ax.set_zlabel("Height / m", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=25, azim=-52)
        ax.set_title(
            f"FID {fid}: {int(shp.get('floor', b['floor']))}F/{float(shp.get('height_m', b['height_m'])):.0f}m, "
            f"GAMMA mean {float(stat['gamma_dsm_mean_boundary_distance_m']):.1f}m",
            fontsize=9,
        )
    fig.suptitle("Typical Tongji Buildings: 3D Vector Extrusion and Method-vs-GAMMA Geocoding", fontsize=13)
    fig.subplots_adjust(left=0.025, right=0.91, bottom=0.04, top=0.91, wspace=0.04, hspace=0.20)
    if scatter_for_colorbar is not None:
        cax = fig.add_axes([0.93, 0.20, 0.012, 0.58])
        cbar = fig.colorbar(scatter_for_colorbar, cax=cax)
        cbar.set_label("Literature-method height / m")
    fig.savefig(out_png)
    plt.close(fig)


def make_radar_projection_plot(out_png: Path, buildings: dict[int, dict], points: dict[int, list[dict]], amp: np.ndarray, par: dict, orbit) -> None:
    models = {fid: rasterize_building(b, par, orbit, amp.shape) for fid, b in buildings.items()}
    n = len(buildings)
    ncols = 5 if n > 6 else min(3, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 3.5 * nrows), dpi=300, squeeze=False)
    colors = ["#3b82f6", "#ef4444", "#22c55e", "#a855f7", "#f59e0b", "#06b6d4", "#84cc16", "#f43f5e", "#14b8a6", "#6366f1"]
    for i, fid in enumerate(buildings):
        ax = axes.ravel()[i]
        color = colors[i % len(colors)]
        model = models[fid]
        rows = points[fid]
        arr = np.asarray([[float(r["row"]), float(r["col"]), float(r["method_height_m"])] for r in rows], dtype=np.float64)
        rc = np.vstack([arr[:, :2], model["projected_rc"]])
        r0 = max(0, int(np.nanmin(rc[:, 0])) - 22)
        r1 = min(amp.shape[0] - 1, int(np.nanmax(rc[:, 0])) + 22)
        c0 = max(0, int(np.nanmin(rc[:, 1])) - 22)
        c1 = min(amp.shape[1] - 1, int(np.nanmax(rc[:, 1])) + 22)
        ax.imshow(amp[r0 : r1 + 1, c0 : c1 + 1], cmap="gray", extent=[c0, c1, r1, r0], origin="upper")
        for tri in model["triangles"]:
            pts = np.column_stack([model["projected_rc"][tri, 1], model["projected_rc"][tri, 0]])
            ax.add_patch(MplPolygon(pts, closed=True, fill=False, edgecolor=color, linewidth=0.36, alpha=0.55))
        step = max(1, arr.shape[0] // 260)
        ax.scatter(arr[::step, 1], arr[::step, 0], s=7, c=arr[::step, 2], cmap="viridis", alpha=0.78, linewidths=0)
        cy = float(np.nanmean(model["projected_rc"][:, 0]))
        cx = float(np.nanmean(model["projected_rc"][:, 1]))
        ax.text(cx, cy, f"FID {fid}", color="white", fontsize=7, ha="center", va="center", bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.55, "pad": 1.2})
        ax.set_xlim(c0, c1)
        ax.set_ylim(r1, r0)
        ax.set_xlabel("Range column")
        ax.set_ylabel("Azimuth row")
        ax.set_title(f"FID {fid}: projected model and pixels", fontsize=8.5)
        ax.grid(color="white", linewidth=0.22, alpha=0.18)
    for ax in axes.ravel()[len(buildings) :]:
        ax.axis("off")
    fig.suptitle("Projected Building Models and Selected Strong-Scatterer Pixels in SAR Coordinates", fontsize=13, y=0.98)
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.07, top=0.86, wspace=0.28, hspace=0.55)
    fig.savefig(out_png)
    plt.close(fig)


def make_vector_overview(out_png: Path, all_rings: list[np.ndarray], buildings: dict[int, dict], stats_by_fid: dict[int, dict], attrs: dict[int, dict]) -> None:
    colors = ["#3b82f6", "#ef4444", "#22c55e", "#a855f7", "#f59e0b", "#06b6d4", "#84cc16", "#f43f5e", "#14b8a6", "#6366f1"]
    selected_rings = [b["ring_lonlat"] for b in buildings.values()]
    all_selected = np.vstack(selected_rings)
    xpad = max(float(np.ptp(all_selected[:, 0])) * 0.45, 0.0012)
    ypad = max(float(np.ptp(all_selected[:, 1])) * 0.45, 0.0012)
    xlim = (float(np.min(all_selected[:, 0]) - xpad), float(np.max(all_selected[:, 0]) + xpad))
    ylim = (float(np.min(all_selected[:, 1]) - ypad), float(np.max(all_selected[:, 1]) + ypad))

    fig, ax = plt.subplots(figsize=(9.4, 8.0), dpi=300)
    for ring in all_rings:
        if np.max(ring[:, 0]) < xlim[0] or np.min(ring[:, 0]) > xlim[1] or np.max(ring[:, 1]) < ylim[0] or np.min(ring[:, 1]) > ylim[1]:
            continue
        ax.add_patch(MplPolygon(ring, closed=True, fill=False, edgecolor="#b8b8b8", linewidth=0.35, alpha=0.58, zorder=1))
    for i, (fid, b) in enumerate(buildings.items(), start=1):
        ring = b["ring_lonlat"]
        color = colors[(i - 1) % len(colors)]
        ax.add_patch(MplPolygon(ring, closed=True, fill=True, facecolor=color, edgecolor="#111111", linewidth=0.8, alpha=0.32, zorder=4))
        cx = float(np.mean(ring[:, 0]))
        cy = float(np.mean(ring[:, 1]))
        stat = stats_by_fid[fid]
        shp = attrs.get(fid, {})
        ax.text(
            cx,
            cy,
            f"{i}\nFID {fid}\n{int(shp.get('floor', b['floor']))}F",
            ha="center",
            va="center",
            fontsize=6.8,
            color="#111111",
            bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.86, "pad": 1.0},
            zorder=6,
        )
        ax.scatter([cx], [cy], s=12 + float(stat["gamma_dsm_mean_boundary_distance_m"]) * 2.0, c=color, edgecolors="#111111", linewidths=0.45, zorder=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude / deg")
    ax.set_ylabel("Latitude / deg")
    ax.set_title("Selected Typical Buildings on Building Vector Footprints")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(color="#dddddd", linewidth=0.3, alpha=0.65)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def make_sar_overview(out_png: Path, buildings: dict[int, dict], points: dict[int, list[dict]], amp: np.ndarray, par: dict, orbit) -> None:
    models = {fid: rasterize_building(b, par, orbit, amp.shape) for fid, b in buildings.items()}
    colors = ["#3b82f6", "#ef4444", "#22c55e", "#a855f7", "#f59e0b", "#06b6d4", "#84cc16", "#f43f5e", "#14b8a6", "#6366f1"]
    fig, ax = plt.subplots(figsize=(11.2, 7.8), dpi=300)
    ax.imshow(amp, cmap="gray", origin="upper")
    for i, fid in enumerate(buildings, start=1):
        color = colors[(i - 1) % len(colors)]
        model = models[fid]
        rc = model["projected_rc"]
        arr = np.asarray([[float(r["row"]), float(r["col"])] for r in points[fid]], dtype=np.float64)
        for tri in model["triangles"]:
            pts = np.column_stack([rc[tri, 1], rc[tri, 0]])
            ax.add_patch(MplPolygon(pts, closed=True, fill=False, edgecolor=color, linewidth=0.5, alpha=0.72, zorder=3))
        if arr.size:
            step = max(1, arr.shape[0] // 120)
            ax.scatter(arr[::step, 1], arr[::step, 0], s=5, c=color, alpha=0.75, linewidths=0, zorder=4)
        cy = float(np.nanmean(rc[:, 0]))
        cx = float(np.nanmean(rc[:, 1]))
        ax.scatter([cx], [cy], s=42, marker="o", c=color, edgecolors="white", linewidths=0.7, zorder=5)
        ax.text(
            cx + 5,
            cy - 5,
            f"{i}: FID {fid}",
            color="white",
            fontsize=7,
            ha="left",
            va="center",
            bbox={"facecolor": "black", "edgecolor": color, "alpha": 0.66, "pad": 1.3},
            zorder=6,
        )
    ax.set_xlim(0, amp.shape[1] - 1)
    ax.set_ylim(amp.shape[0] - 1, 0)
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    ax.set_title("Selected Typical Buildings on the Full SAR Intensity Image")
    ax.grid(color="white", linewidth=0.22, alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def write_summary(path: Path, fids: list[int], stats_by_fid: dict[int, dict], attrs: dict[int, dict], points: dict[int, list[dict]]) -> None:
    fields = [
        "fid",
        "shp_id",
        "floor",
        "height_m",
        "height_source",
        "sample_points",
        "method_mean_boundary_distance_m",
        "gamma_dsm_mean_boundary_distance_m",
        "gamma_minus_method_mean_m",
        "gamma_dsm_p90_boundary_distance_m",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for fid in fids:
            stat = stats_by_fid[fid]
            shp = attrs.get(fid, {})
            writer.writerow(
                {
                    "fid": fid,
                    "shp_id": shp.get("id", ""),
                    "floor": shp.get("floor", stat["floor"]),
                    "height_m": shp.get("height_m", stat["height_m"]),
                    "height_source": shp.get("height_source", "height"),
                    "sample_points": len(points[fid]),
                    "method_mean_boundary_distance_m": stat["method_mean_boundary_distance_m"],
                    "gamma_dsm_mean_boundary_distance_m": stat["gamma_dsm_mean_boundary_distance_m"],
                    "gamma_minus_method_mean_m": float(stat["gamma_dsm_mean_boundary_distance_m"]) - float(stat["method_mean_boundary_distance_m"]),
                    "gamma_dsm_p90_boundary_distance_m": stat["gamma_dsm_p90_boundary_distance_m"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=DATE)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    date = args.date
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    TYPICAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    stats_csv = FULL_DIR / f"{date}_all_buildings_fig5_4_like_stats.csv"
    points_csv = FULL_DIR / f"{date}_all_buildings_method_vs_gamma_points.csv"
    buildings_geojson = FULL_AREA_GEOJSON_DIR / f"{date}_all_valid_geocoded_buildings.geojson"
    stats = read_rows(stats_csv)
    stats_by_fid = {int(r["fid"]): r for r in stats}
    selected_fids = choose_typical_buildings(stats, args.count)
    valid_buildings = load_valid_buildings(buildings_geojson)
    all_rings = load_all_building_rings(buildings_geojson)
    buildings = {fid: valid_buildings[fid] for fid in selected_fids}
    attrs = read_shp_attributes(BUILDINGS_SHP)
    points = points_for_fids(points_csv, selected_fids)

    par = parse_gamma_par(RSLC_DIR / f"{date}.rslc.par")
    orbit = make_orbit(par)
    amp = read_rslc_amplitude(RSLC_DIR / f"{date}.rslc", int(par["azimuth_lines"]), int(par["range_samples"]))

    make_3d_plot(TYPICAL_IMAGE_DIR / f"{date}_typical_buildings_3d_method_vs_gamma.png", buildings, points, stats_by_fid, attrs)
    make_radar_projection_plot(TYPICAL_IMAGE_DIR / f"{date}_typical_buildings_projected_pixels.png", buildings, points, amp, par, orbit)
    make_vector_overview(TYPICAL_IMAGE_DIR / f"{date}_typical_buildings_vector_overview.png", all_rings, buildings, stats_by_fid, attrs)
    make_sar_overview(TYPICAL_IMAGE_DIR / f"{date}_typical_buildings_full_sar_overview.png", buildings, points, amp, par, orbit)
    write_summary(out_dir / f"{date}_typical_buildings_comparison_stats.csv", selected_fids, stats_by_fid, attrs, points)
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Typical Building 3D Comparison",
                "",
                f"Date: `{date}`",
                f"Selected FIDs: {', '.join(str(x) for x in selected_fids)}",
                "",
                "Selection prioritizes buildings with large GAMMA/DEM mean boundary error while keeping different height ranges.",
                "Heights come from `data/shp/tongji_clip.shp` field `height`; if height is missing, `Floor*3 m` is used.",
                "",
                "Outputs:",
                f"- `../pic_all/{date}_typical_buildings_3d_method_vs_gamma.png`: extruded building vectors, literature-method 3D points, GAMMA/DEM points, and displacement links.",
                f"- `../pic_all/{date}_typical_buildings_projected_pixels.png`: projected building model triangles and selected strong-scatterer pixels in SAR row/column coordinates.",
                f"- `../pic_all/{date}_typical_buildings_vector_overview.png`: selected buildings labeled on building vector footprints.",
                f"- `../pic_all/{date}_typical_buildings_full_sar_overview.png`: selected buildings labeled on the full SAR intensity image.",
                f"- `{date}_typical_buildings_comparison_stats.csv`: per-building height, floor, sample count, and method-vs-GAMMA error statistics.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"selected_fids={selected_fids}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
