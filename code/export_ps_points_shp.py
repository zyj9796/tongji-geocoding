from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "data" / "ps_points_all.csv"
DEFAULT_OUT_DIR = PROJECT_DIR / "results" / "outputs" / "geodata" / "ps_points_shp"

FIELD_MAP = {
    "ps_id": "ps_id",
    "longitude": "longitude",
    "latitude": "latitude",
    "height_m": "height_m",
    "x_utm51n_m": "x_utm51n",
    "y_utm51n_m": "y_utm51n",
    "z_dsm_m": "z_dsm_m",
    "velocity_mm_yr": "vel_mm_yr",
    "coherence": "coherence",
    "azimuth_pixel": "az_pixel",
    "range_pixel": "rg_pixel",
}


def export(input_csv: Path, out_dir: Path) -> dict:
    frame = pd.read_csv(input_csv)
    missing = set(FIELD_MAP).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame[list(FIELD_MAP)].rename(columns=FIELD_MAP)
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
        raise ValueError("PS table contains non-finite values")
    if not frame["longitude"].between(-180.0, 180.0).all():
        raise ValueError("Longitude is outside EPSG:4979 range")
    if not frame["latitude"].between(-90.0, 90.0).all():
        raise ValueError("Latitude is outside EPSG:4979 range")

    geometry = gpd.points_from_xy(
        frame["longitude"],
        frame["latitude"],
        z=frame["height_m"],
        crs="EPSG:4979",
    )
    gdf = gpd.GeoDataFrame(frame, geometry=geometry, crs="EPSG:4979")
    out_dir.mkdir(parents=True, exist_ok=True)
    shp_path = out_dir / "ps_points_wgs84_3d.shp"
    gdf.to_file(shp_path, driver="ESRI Shapefile", encoding="UTF-8", engine="pyogrio")

    restored = gpd.read_file(shp_path, engine="pyogrio")
    if len(restored) != len(gdf):
        raise RuntimeError(f"Feature-count mismatch after write: {len(restored)} != {len(gdf)}")
    if not restored.geometry.has_z.all():
        raise RuntimeError("Written Shapefile is not PointZ")
    z_restored = np.asarray([geom.z for geom in restored.geometry], dtype=np.float64)
    if not np.allclose(z_restored, frame["height_m"].to_numpy(dtype=np.float64), atol=1e-7):
        raise RuntimeError("Geometry Z values do not match height_m")

    components = sorted(out_dir.glob("ps_points_wgs84_3d.*"))
    zip_path = out_dir / "ps_points_wgs84_3d_shapefile.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in components:
            if path != zip_path:
                archive.write(path, arcname=path.name)

    bounds = gdf.total_bounds.tolist()
    summary = {
        "source_csv": str(input_csv),
        "feature_count": int(len(gdf)),
        "geometry_type": "3D PointZ",
        "crs": "EPSG:4979 (WGS 84 three-dimensional)",
        "bounds_lon_lat": {
            "min_longitude": float(bounds[0]),
            "min_latitude": float(bounds[1]),
            "max_longitude": float(bounds[2]),
            "max_latitude": float(bounds[3]),
        },
        "height_m": {
            "min": float(frame["height_m"].min()),
            "max": float(frame["height_m"].max()),
        },
        "attribute_fields": list(frame.columns),
        "shapefile": str(shp_path),
        "zip": str(zip_path),
        "components": [str(path) for path in components if path != zip_path],
    }
    (out_dir / "ps_points_wgs84_3d_metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PS point coordinates to a WGS84 3D PointZ Shapefile.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(export(args.input, args.out_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
