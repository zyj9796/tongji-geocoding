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
from scipy.spatial import cKDTree

from io_paths import FULL_AREA_DIR, LEGACY_IMAGE_DIR, LEGACY_TABLE_DIR, PROJECT_DIR, PS_POINTS_CSV


REPO_ROOT = PROJECT_DIR.parents[1]
DEFAULT_PS_3D = PS_POINTS_CSV
DEFAULT_SHIZHAN = FULL_AREA_DIR / "20200708_all_buildings_method_vs_gamma_points.csv"
DEFAULT_OUT_DIR = LEGACY_TABLE_DIR / "psinsar_vs_bc_3d"
EARTH_R = 6378137.0


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def local_en(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    east = (lon - lon0) * math.pi / 180.0 * EARTH_R * math.cos(math.radians(lat0))
    north = (lat - lat0) * math.pi / 180.0 * EARTH_R
    return east, north


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values[np.isfinite(values)], q)) if np.any(np.isfinite(values)) else float("nan")


def value(row: dict, *names: str, default: str | None = None) -> str:
    for name in names:
        if name in row and row[name] not in ("", None):
            return row[name]
    if default is not None:
        return default
    raise KeyError(f"missing any of columns: {', '.join(names)}")


def load_ps(path: Path) -> dict[str, np.ndarray]:
    rows = read_csv(path)
    return {
        "rows": rows,
        "ps_id": np.asarray([int(float(r["ps_id"])) for r in rows], dtype=np.int64),
        "line": np.asarray([float(value(r, "line", "azimuth_pixel")) for r in rows], dtype=np.float64),
        "pixel": np.asarray([float(value(r, "pixel", "range_pixel")) for r in rows], dtype=np.float64),
        "lon": np.asarray([float(value(r, "lut_lon", "longitude", "lon")) for r in rows], dtype=np.float64),
        "lat": np.asarray([float(value(r, "lut_lat", "latitude", "lat")) for r in rows], dtype=np.float64),
        "h": np.asarray([float(value(r, "lut_z", "height_m", "z_dsm_m", "height")) for r in rows], dtype=np.float64),
        "stamps_lon": np.asarray([float(value(r, "stamps_lon", "longitude", "lon")) for r in rows], dtype=np.float64),
        "stamps_lat": np.asarray([float(value(r, "stamps_lat", "latitude", "lat")) for r in rows], dtype=np.float64),
        "stamps_h": np.asarray([float(value(r, "stamps_hgt", "height_m", "z_dsm_m", "height")) for r in rows], dtype=np.float64),
        "coherence": np.asarray([float(r.get("coherence", "nan")) for r in rows], dtype=np.float64),
        "interp_residual_px": np.asarray([float(r.get("interp_radar_residual_px", "nan")) for r in rows], dtype=np.float64),
        "nearest_distance_px": np.asarray([float(r.get("nearest_distance_px", "nan")) for r in rows], dtype=np.float64),
    }


def load_shizhan(path: Path) -> dict[str, np.ndarray]:
    rows = read_csv(path)
    return {
        "rows": rows,
        "fid": np.asarray([int(float(r["fid"])) for r in rows], dtype=np.int64),
        "row": np.asarray([float(r["row"]) for r in rows], dtype=np.float64),
        "col": np.asarray([float(r["col"]) for r in rows], dtype=np.float64),
        "lon": np.asarray([float(r["method_lon"]) for r in rows], dtype=np.float64),
        "lat": np.asarray([float(r["method_lat"]) for r in rows], dtype=np.float64),
        "h": np.asarray([float(r["method_height_m"]) for r in rows], dtype=np.float64),
        "gamma_lon": np.asarray([float(r["gamma_dsm_lon"]) for r in rows], dtype=np.float64),
        "gamma_lat": np.asarray([float(r["gamma_dsm_lat"]) for r in rows], dtype=np.float64),
        "gamma_h": np.asarray([float(r["gamma_dsm_height_m"]) for r in rows], dtype=np.float64),
    }


def compare(ps: dict[str, np.ndarray], sh: dict[str, np.ndarray], max_horizontal_m: float) -> tuple[list[dict], dict]:
    lon0 = float(np.mean(np.concatenate([ps["lon"], sh["lon"]])))
    lat0 = float(np.mean(np.concatenate([ps["lat"], sh["lat"]])))

    ps_e, ps_n = local_en(ps["lon"], ps["lat"], lon0, lat0)
    ps_st_e, ps_st_n = local_en(ps["stamps_lon"], ps["stamps_lat"], lon0, lat0)
    sh_e, sh_n = local_en(sh["lon"], sh["lat"], lon0, lat0)
    sh_gamma_e, sh_gamma_n = local_en(sh["gamma_lon"], sh["gamma_lat"], lon0, lat0)

    spatial_tree = cKDTree(np.column_stack([sh_e, sh_n]))
    horizontal_m, spatial_idx = spatial_tree.query(np.column_stack([ps_e, ps_n]), k=1)

    radar_tree = cKDTree(np.column_stack([sh["row"], sh["col"]]))
    radar_px, radar_idx = radar_tree.query(np.column_stack([ps["line"], ps["pixel"]]), k=1)

    rows: list[dict] = []
    for i in range(ps["ps_id"].size):
        j = int(spatial_idx[i])
        jr = int(radar_idx[i])
        dz = float(ps["h"][i] - sh["h"][j])
        d3 = float(math.hypot(float(horizontal_m[i]), dz))
        stamps_h = float(math.hypot(ps_st_e[i] - sh_e[j], ps_st_n[i] - sh_n[j]))
        gamma_h = float(math.hypot(sh_gamma_e[j] - sh_e[j], sh_gamma_n[j] - sh_n[j]))
        radar_spatial_h = float(math.hypot(ps_e[i] - sh_e[jr], ps_n[i] - sh_n[jr]))
        rows.append(
            {
                "ps_id": int(ps["ps_id"][i]),
                "ps_line": float(ps["line"][i]),
                "ps_pixel": float(ps["pixel"][i]),
                "ps_3dlut_lon": float(ps["lon"][i]),
                "ps_3dlut_lat": float(ps["lat"][i]),
                "ps_3dlut_height_m": float(ps["h"][i]),
                "ps_stamps_lon": float(ps["stamps_lon"][i]),
                "ps_stamps_lat": float(ps["stamps_lat"][i]),
                "ps_stamps_height_m": float(ps["stamps_h"][i]),
                "ps_coherence": float(ps["coherence"][i]),
                "ps_interp_residual_px": float(ps["interp_residual_px"][i]),
                "nearest_shizhan_fid": int(sh["fid"][j]),
                "nearest_shizhan_row": float(sh["row"][j]),
                "nearest_shizhan_col": float(sh["col"][j]),
                "nearest_shizhan_lon": float(sh["lon"][j]),
                "nearest_shizhan_lat": float(sh["lat"][j]),
                "nearest_shizhan_height_m": float(sh["h"][j]),
                "ps3dlut_to_shizhan_horizontal_m": float(horizontal_m[i]),
                "ps3dlut_minus_shizhan_height_m": dz,
                "ps3dlut_to_shizhan_3d_m": d3,
                "stamps_to_shizhan_horizontal_m": stamps_h,
                "shizhan_gamma_to_method_horizontal_m": gamma_h,
                "nearest_radar_shizhan_fid": int(sh["fid"][jr]),
                "nearest_radar_shizhan_row": float(sh["row"][jr]),
                "nearest_radar_shizhan_col": float(sh["col"][jr]),
                "ps_to_shizhan_radar_distance_px": float(radar_px[i]),
                "ps_to_radar_nearest_shizhan_horizontal_m": radar_spatial_h,
                "within_horizontal_threshold": int(float(horizontal_m[i]) <= max_horizontal_m),
            }
        )

    d = {k: np.asarray([float(r[k]) for r in rows], dtype=np.float64) for k in [
        "ps3dlut_to_shizhan_horizontal_m",
        "ps3dlut_minus_shizhan_height_m",
        "ps3dlut_to_shizhan_3d_m",
        "stamps_to_shizhan_horizontal_m",
        "ps_to_shizhan_radar_distance_px",
        "ps_to_radar_nearest_shizhan_horizontal_m",
    ]}
    close = d["ps3dlut_to_shizhan_horizontal_m"] <= max_horizontal_m
    summary = {
        "ps_source": str(DEFAULT_PS_3D),
        "shizhan_source": str(DEFAULT_SHIZHAN),
        "ps_count": int(ps["ps_id"].size),
        "shizhan_point_count": int(sh["fid"].size),
        "max_horizontal_match_m": float(max_horizontal_m),
        "matched_ps_count": int(np.sum(close)),
        "matched_ps_ratio": float(np.mean(close)),
    }
    for prefix, arr in d.items():
        summary[f"{prefix}_median"] = percentile(arr, 50)
        summary[f"{prefix}_p90"] = percentile(arr, 90)
        summary[f"{prefix}_mean"] = float(np.nanmean(arr))
    abs_height = np.abs(d["ps3dlut_minus_shizhan_height_m"])
    summary["ps3dlut_abs_height_diff_m_median"] = percentile(abs_height, 50)
    summary["ps3dlut_abs_height_diff_m_p90"] = percentile(abs_height, 90)
    summary["ps3dlut_abs_height_diff_m_mean"] = float(np.nanmean(abs_height))
    for prefix, arr in d.items():
        if np.any(close):
            summary[f"matched_{prefix}_median"] = percentile(arr[close], 50)
            summary[f"matched_{prefix}_p90"] = percentile(arr[close], 90)
            summary[f"matched_{prefix}_mean"] = float(np.nanmean(arr[close]))
    if np.any(close):
        matched_abs_height = abs_height[close]
        summary["matched_ps3dlut_abs_height_diff_m_median"] = percentile(matched_abs_height, 50)
        summary["matched_ps3dlut_abs_height_diff_m_p90"] = percentile(matched_abs_height, 90)
        summary["matched_ps3dlut_abs_height_diff_m_mean"] = float(np.nanmean(matched_abs_height))
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_fid(rows: list[dict], max_horizontal_m: float) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for r in rows:
        if float(r["ps3dlut_to_shizhan_horizontal_m"]) <= max_horizontal_m:
            grouped.setdefault(int(r["nearest_shizhan_fid"]), []).append(r)
    out = []
    for fid, items in sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True):
        h = np.asarray([float(x["ps3dlut_to_shizhan_horizontal_m"]) for x in items], dtype=np.float64)
        z = np.asarray([float(x["ps3dlut_minus_shizhan_height_m"]) for x in items], dtype=np.float64)
        rpx = np.asarray([float(x["ps_to_shizhan_radar_distance_px"]) for x in items], dtype=np.float64)
        out.append(
            {
                "nearest_shizhan_fid": fid,
                "matched_ps_count": len(items),
                "horizontal_m_median": percentile(h, 50),
                "horizontal_m_p90": percentile(h, 90),
                "height_diff_m_median": percentile(z, 50),
                "height_abs_diff_m_median": percentile(np.abs(z), 50),
                "radar_distance_px_median": percentile(rpx, 50),
            }
        )
    return out


def plot_map(path: Path, rows: list[dict], max_points: int = 12000) -> None:
    arr = rows[:: max(1, len(rows) // max_points)]
    ps_lon = np.asarray([float(r["ps_3dlut_lon"]) for r in arr])
    ps_lat = np.asarray([float(r["ps_3dlut_lat"]) for r in arr])
    sh_lon = np.asarray([float(r["nearest_shizhan_lon"]) for r in arr])
    sh_lat = np.asarray([float(r["nearest_shizhan_lat"]) for r in arr])
    d = np.asarray([float(r["ps3dlut_to_shizhan_horizontal_m"]) for r in arr])
    fig, ax = plt.subplots(figsize=(8.4, 7.2), dpi=300)
    ax.scatter(sh_lon, sh_lat, s=1.2, c="#2563eb", alpha=0.35, linewidths=0, label="Nearest shizhan method points")
    sc = ax.scatter(ps_lon, ps_lat, s=2.0, c=np.clip(d, 0, 50), cmap="magma", alpha=0.72, linewidths=0, label="PS-InSAR 3DLUT points")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude / deg")
    ax.set_ylabel("Latitude / deg")
    ax.set_title("PS-InSAR 3D Points vs. Shizhan Building-Constrained Points")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(color="#dddddd", linewidth=0.3)
    ax.legend(loc="upper right", fontsize=8)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.02)
    cbar.set_label("Horizontal nearest-neighbor distance / m")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_hist(path: Path, rows: list[dict], max_horizontal_m: float) -> None:
    h = np.asarray([float(r["ps3dlut_to_shizhan_horizontal_m"]) for r in rows])
    z = np.asarray([float(r["ps3dlut_minus_shizhan_height_m"]) for r in rows])
    rpx = np.asarray([float(r["ps_to_shizhan_radar_distance_px"]) for r in rows])
    close = h <= max_horizontal_m
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), dpi=300)
    axes[0].hist(np.clip(h, 0, 80), bins=60, color="#2563eb", alpha=0.75)
    axes[0].axvline(max_horizontal_m, color="#ef4444", linewidth=1.0)
    axes[0].set_xlabel("Horizontal distance / m")
    axes[0].set_ylabel("PS count")
    axes[0].set_title("All PS")
    axes[1].hist(np.clip(np.abs(z[close]), 0, 80), bins=50, color="#16a34a", alpha=0.75)
    axes[1].set_xlabel("|height difference| / m")
    axes[1].set_title(f"Matched PS <= {max_horizontal_m:g} m")
    axes[2].hist(np.clip(rpx, 0, 80), bins=60, color="#f97316", alpha=0.75)
    axes[2].set_xlabel("Radar pixel distance / px")
    axes[2].set_title("Radar nearest neighbor")
    for ax in axes:
        ax.grid(axis="y", color="#dddddd", linewidth=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    ps_path = Path(args.ps_3d_csv)
    sh_path = Path(args.shizhan_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    LEGACY_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    ps = load_ps(ps_path)
    sh = load_shizhan(sh_path)
    rows, summary = compare(ps, sh, float(args.max_horizontal_m))
    summary["ps_source"] = str(ps_path)
    summary["shizhan_source"] = str(sh_path)

    write_csv(out_dir / "psinsar_3dlut_vs_bc_nearest_points.csv", rows)
    write_csv(out_dir / "psinsar_3dlut_vs_bc_by_building.csv", summarize_by_fid(rows, float(args.max_horizontal_m)))
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_map(LEGACY_IMAGE_DIR / "图件_292635425203.png", rows)
    plot_hist(LEGACY_IMAGE_DIR / "fig_psinsar_3dlut_vs_bc_hist.png", rows, float(args.max_horizontal_m))
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# PS-InSAR 3D vs Shizhan Building-Constrained Geocoding",
                "",
                "This is a legacy nearest-neighbor comparison retained for reference. Current presentation material should use the strict same-SAR-pixel package under `results/outputs/tables/psinsar_same_pixel/`.",
                "Any geographic SAR backdrop used with current results must come from the GAMMA/DSM terrain-geocoded raster; display uses the shared amplitude-domain 2%-98% stretch.",
                "",
                f"PS 3D source: `{ps_path}`",
                f"Shizhan source: `{sh_path}`",
                "",
                "The PS 3D coordinates use `lut_lon/lut_lat/lut_z` from the optimized IDW 3DLUT geocoding output.",
                "For each PS point, the nearest shizhan building-constrained point is found in horizontal EN coordinates; a radar row/column nearest neighbor is also reported.",
                "",
                "Outputs:",
                "- `psinsar_3dlut_vs_bc_nearest_points.csv`: per-PS nearest-neighbor comparison.",
                "- `psinsar_3dlut_vs_bc_by_building.csv`: matched PS counts and distance statistics grouped by nearest building-constrained building FID.",
                "- `summary.json`: global counts and median/p90/mean metrics.",
                "- `../pic_all/图件_292635425203.png`: map overlay colored by horizontal difference.",
                "- `../pic_all/fig_psinsar_3dlut_vs_bc_hist.png`: distance histograms.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ps-3d-csv", default=str(DEFAULT_PS_3D))
    parser.add_argument("--shizhan-csv", default=str(DEFAULT_SHIZHAN))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-horizontal-m", type=float, default=15.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
