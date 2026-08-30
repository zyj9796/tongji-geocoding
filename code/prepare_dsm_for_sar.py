from __future__ import annotations

import json
from pathlib import Path

from osgeo import gdal, osr

from io_paths import DATA_DIR, DSM_SAR_EXTENT_TIF, RASTER_ROOT, TIF_DIR


def _raster_bounds_wgs84(path: Path) -> tuple[float, float, float, float]:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(path)
    gt = ds.GetGeoTransform()
    xs = [gt[0], gt[0] + ds.RasterXSize * gt[1]]
    ys = [gt[3], gt[3] + ds.RasterYSize * gt[5]]
    return min(xs), min(ys), max(xs), max(ys)


def sar_bounds_wgs84() -> tuple[float, float, float, float]:
    rasters = sorted(TIF_DIR.glob("*_gamma_dem_geocoded_wgs84.tif"))
    if not rasters:
        rasters = sorted(RASTER_ROOT.rglob("*_gamma_dem_geocoded_wgs84.tif"))
    if not rasters:
        raise FileNotFoundError("No *_gamma_dem_geocoded_wgs84.tif found under results/outputs/rasters")
    bounds = [_raster_bounds_wgs84(path) for path in rasters]
    return (
        min(b[0] for b in bounds),
        min(b[1] for b in bounds),
        max(b[2] for b in bounds),
        max(b[3] for b in bounds),
    )


def source_dsm() -> Path:
    preferred = DATA_DIR / "tongji_dsm.tif"
    if preferred.exists():
        return preferred
    candidates = []
    for pattern in ["*dsm*.tif", "*DSM*.tif", "*.tif"]:
        candidates.extend(DATA_DIR.glob(pattern))
    for path in sorted(set(candidates)):
        if path.is_file() and path.name != DSM_SAR_EXTENT_TIF.name:
            return path
    raise FileNotFoundError(f"No source DSM found in {DATA_DIR}")


def transform_bounds_to_dataset_srs(bounds_wgs84: tuple[float, float, float, float], dsm_path: Path, margin_m: float) -> tuple[float, float, float, float]:
    ds = gdal.Open(str(dsm_path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(dsm_path)
    dst = osr.SpatialReference(wkt=ds.GetProjection())
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    src = osr.SpatialReference()
    src.ImportFromEPSG(4326)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tx = osr.CoordinateTransformation(src, dst)
    min_lon, min_lat, max_lon, max_lat = bounds_wgs84
    corners = [
        tx.TransformPoint(min_lon, min_lat),
        tx.TransformPoint(min_lon, max_lat),
        tx.TransformPoint(max_lon, min_lat),
        tx.TransformPoint(max_lon, max_lat),
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return min(xs) - margin_m, min(ys) - margin_m, max(xs) + margin_m, max(ys) + margin_m


def crop_dsm(margin_m: float = 30.0) -> Path:
    src = source_dsm()
    bounds_wgs84 = sar_bounds_wgs84()
    bounds_native = transform_bounds_to_dataset_srs(bounds_wgs84, src, margin_m)
    opts = gdal.WarpOptions(
        format="GTiff",
        outputBounds=bounds_native,
        resampleAlg="bilinear",
        creationOptions=["COMPRESS=LZW", "TILED=YES"],
    )
    out_tmp = DSM_SAR_EXTENT_TIF.with_suffix(".tmp.tif")
    ds = gdal.Warp(str(out_tmp), str(src), options=opts)
    if ds is None:
        raise RuntimeError(f"Failed to crop DSM {src}")
    ds = None
    out_tmp.replace(DSM_SAR_EXTENT_TIF)
    meta = {
        "source_dsm": str(src),
        "output_dsm": str(DSM_SAR_EXTENT_TIF),
        "sar_bounds_wgs84": bounds_wgs84,
        "crop_bounds_native": bounds_native,
        "margin_m": margin_m,
    }
    DSM_SAR_EXTENT_TIF.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return DSM_SAR_EXTENT_TIF


def main() -> None:
    out = crop_dsm()
    print(f"dsm_for_sar={out}")


if __name__ == "__main__":
    main()
