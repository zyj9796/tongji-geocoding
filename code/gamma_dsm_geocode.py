from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
from osgeo import gdal


def _par_value(path: Path, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s+([^\s]+)", path.read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise ValueError(f"Missing {key} in {path}")
    return match.group(1)


def _gamma_environment(work_dir: Path) -> dict[str, str]:
    """Provide the ABI-compatible GDAL soname required by this GAMMA build."""
    env = os.environ.copy()
    compat_dir = work_dir / "lib"
    compat_dir.mkdir(parents=True, exist_ok=True)
    compat = compat_dir / "libgdal.so.26"
    if not compat.exists():
        candidates = [Path("/usr/lib/libgdal.so.30"), Path("/usr/lib/libgdal.so")]
        source = next((path for path in candidates if path.exists()), None)
        if source is None:
            raise RuntimeError("GAMMA requires libgdal.so.26 and no compatible system GDAL library was found")
        compat.symlink_to(source)
    env["LD_LIBRARY_PATH"] = f"{compat_dir}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    return env


def _run(args: list[str], env: dict[str, str], log) -> None:
    log.write("$ " + " ".join(args) + "\n")
    log.flush()
    subprocess.run(args, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)


def geocode_rslc_with_dsm(
    date: str,
    rslc_dir: Path,
    dsm_tif: Path,
    output_dir: Path,
    work_root: Path,
) -> tuple[Path, Path, dict]:
    """Terrain-geocode one RSLC intensity image with GAMMA and the supplied DSM."""
    output_dir.mkdir(parents=True, exist_ok=True)
    work = work_root / date
    work.mkdir(parents=True, exist_ok=True)
    env = _gamma_environment(work_root)

    rslc = rslc_dir / f"{date}.rslc"
    rslc_par = rslc_dir / f"{date}.rslc.par"
    if not rslc.exists() or not rslc_par.exists():
        raise FileNotFoundError(f"Missing RSLC input for {date}")
    if not dsm_tif.exists():
        raise FileNotFoundError(dsm_tif)

    dem = work_root / "tongji_dsm.dem"
    dem_par = work_root / "tongji_dsm.dem_par"
    mli = work / f"{date}.mli"
    mli_par = work / f"{date}.mli.par"
    dem_seg = work / f"{date}.dem_seg"
    dem_seg_par = work / f"{date}.dem_seg.par"
    lookup = work / f"{date}.lt"
    geocoded = work / f"{date}.mli.geo"
    native_tif = work / f"{date}_gamma_dsm_geocoded_native.tif"
    gamma_tif = output_dir / f"{date}_gamma_dem_geocoded_wgs84.tif"
    radar_tif = output_dir / f"{date}_sar_intensity_radar.tif"
    log_path = work / "gamma_geocode.log"

    with log_path.open("w", encoding="utf-8") as log:
        if not dem.exists() or not dem_par.exists() or dsm_tif.stat().st_mtime > dem.stat().st_mtime:
            _run(["dem_import", str(dsm_tif), str(dem), str(dem_par), "0", "1", "-", "-", "-", "-", "-", "-", "-9999"], env, log)
        _run(["multi_look", str(rslc), str(rslc_par), str(mli), str(mli_par), "1", "1"], env, log)
        _run([
            "gc_map2", str(mli_par), str(dem_par), str(dem), str(dem_seg_par), str(dem_seg), str(lookup),
            "1", "1", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "0", "8",
        ], env, log)
        width_in = _par_value(mli_par, "range_samples")
        width_out = _par_value(dem_seg_par, "width")
        nlines_out = _par_value(dem_seg_par, "nlines")
        _run(["geocode_back", str(mli), width_in, str(lookup), str(geocoded), width_out, nlines_out, "2", "0"], env, log)
        _run(["data2geotiff", str(dem_seg_par), str(geocoded), "2", str(native_tif), "0", "1"], env, log)

    warp_options = gdal.WarpOptions(
        format="GTiff",
        dstSRS="EPSG:4326",
        xRes=0.0000025,
        yRes=0.0000025,
        resampleAlg="bilinear",
        srcNodata=0,
        dstNodata=0,
        creationOptions=["COMPRESS=LZW", "TILED=YES"],
    )
    ds = gdal.Warp(str(gamma_tif), str(native_tif), options=warp_options)
    if ds is None:
        raise RuntimeError(f"Failed to reproject GAMMA GeoTIFF {native_tif}")
    ds = None
    radar_width = int(width_in)
    radar_height = int(_par_value(mli_par, "azimuth_lines"))
    radar = np.fromfile(mli, dtype=">f4").reshape(radar_height, radar_width)
    driver = gdal.GetDriverByName("GTiff")
    radar_ds = driver.Create(
        str(radar_tif), radar_width, radar_height, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    radar_ds.GetRasterBand(1).WriteArray(radar)
    radar_ds.FlushCache()
    radar_ds = None

    metadata = {
        "date": date,
        "source_rslc": str(rslc),
        "source_dsm": str(dsm_tif),
        "output_tif": str(gamma_tif),
        "gamma_commands": ["dem_import", "multi_look", "gc_map2", "geocode_back", "data2geotiff"],
        "dem_projection": _par_value(dem_seg_par, "DEM_projection"),
        "width": int(width_out),
        "nlines": int(nlines_out),
        "log": str(log_path),
        "note": "RSLC intensity terrain-geocoded by GAMMA using the source DSM; GDAL is used only for final CRS reprojection.",
    }
    (output_dir / f"{date}_gamma_dsm_geocode.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return radar_tif, gamma_tif, metadata
