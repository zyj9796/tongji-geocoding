"""GAMMA parameter conversion and direct LLH-to-SAR projection helpers."""

from __future__ import annotations

import os
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
INPUT = ROOT / "inputs/image1"
PICALL = ROOT / "PICALL"
WORK = ROOT / "work"
BUILDINGS = REPO / "data/shp/tongji_clip_rslc_extent_equal_height_clean.shp"
GAMMA = Path("/usr/local/GAMMA/DIFF/bin/coord_to_sarpix")
GAMMA_LIB = REPO / "results/outputs/work/gamma_dsm_geocode/lib"
# Heights written in the project DSM and reported to users use the local Wusong
# datum. RPC and GAMMA coord_to_sarpix require heights in the WGS84 map datum.
# Until a surveyed local vertical transformation is supplied, GAMMA EGM96 is
# used explicitly as a provisional proxy; the vendor sceneAverageHeight is not
# a Wusong-to-WGS84 conversion.
BASE_WUSONG_M = 4.0
SCENE_REFERENCE_ELLIPSOID_M = 15.994634628295898
GAMMA_EGM96 = Path("/usr/local/GAMMA/DIFF/scripts/egm96.dem")
PROJECTION_DATUM_VERSION = "wusong4_plus_gamma_egm96_proxy_v2"
X_OFFSET = 22351
Y_OFFSET = 0


def _one(root: ET.Element, path: str) -> str:
    node = root.find(path)
    if node is None or node.text is None:
        raise ValueError(f"XML字段缺失: {path}")
    return node.text.strip()


def _seconds(text: str) -> float:
    # Vendor UTC strings carry nanoseconds while datetime accepts at most six
    # fractional digits.  Orbit timing here only needs microsecond precision.
    match = re.match(r"^(.*\.\d{6})\d+(.*)$", text)
    value = datetime.fromisoformat((match.group(1) + match.group(2)) if match else text)
    return value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6


_EGM96_GRID = None


def egm96_undulation(latitude, longitude):
    """Bilinearly sample GAMMA's 0.25-degree EGM96 geoid grid."""
    global _EGM96_GRID
    if _EGM96_GRID is None:
        values = np.fromfile(GAMMA_EGM96, dtype=">f4")
        if values.size != 720 * 1440:
            raise ValueError(f"unexpected GAMMA EGM96 size: {values.size}")
        _EGM96_GRID = values.reshape(720, 1440).astype(np.float64)
    latitude, longitude = np.broadcast_arrays(
        np.asarray(latitude, dtype=np.float64), np.asarray(longitude, dtype=np.float64)
    )
    row = (latitude - 89.875) / -0.25
    col = (longitude + 179.875) / 0.25
    r0 = np.clip(np.floor(row).astype(int), 0, 718)
    c0 = np.clip(np.floor(col).astype(int), 0, 1438)
    dr, dc = row - r0, col - c0
    return (
        _EGM96_GRID[r0, c0] * (1 - dr) * (1 - dc)
        + _EGM96_GRID[r0 + 1, c0] * dr * (1 - dc)
        + _EGM96_GRID[r0, c0 + 1] * (1 - dr) * dc
        + _EGM96_GRID[r0 + 1, c0 + 1] * dr * dc
    )


def wusong_to_ellipsoid(height_m, latitude, longitude):
    """Provisional conversion using GAMMA EGM96, not a surveyed Wusong grid."""
    return np.asarray(height_m, dtype=np.float64) + egm96_undulation(latitude, longitude)


def ellipsoid_to_wusong(height_m, latitude, longitude):
    """Inverse of the provisional GAMMA-EGM96 conversion."""
    return np.asarray(height_m, dtype=np.float64) - egm96_undulation(latitude, longitude)


def write_gamma_par(xml_path: Path, par_path: Path) -> None:
    root = ET.parse(xml_path).getroot()
    rows = int(_one(root, ".//imageRaster/numberOfRows"))
    cols = int(_one(root, ".//imageRaster/numberOfColumns"))
    first_range_time = float(_one(root, ".//sceneInfo/rangeTime/firstPixel"))
    last_range_time = float(_one(root, ".//sceneInfo/rangeTime/lastPixel"))
    line_time = float(_one(root, ".//imageRaster/columnSpacing"))
    start = _seconds(_one(root, ".//sceneInfo/start/timeUTC"))
    center = _seconds(_one(root, ".//sceneCenterCoord/azimuthTimeUTC"))
    stop = _seconds(_one(root, ".//sceneInfo/stop/timeUTC"))
    speed_of_light = 299792458.0
    near_range = 0.5 * speed_of_light * first_range_time
    far_range = 0.5 * speed_of_light * last_range_time
    center_range = 0.5 * (near_range + far_range)
    range_spacing = (far_range - near_range) / (cols - 1)
    vectors = root.findall(".//platform/orbit/stateVec")
    first_vector_time = _seconds(_one(root, ".//orbitHeader/firstStateTime/firstStateTimeUTC"))
    vector_interval = float(_one(root, ".//orbitHeader/stateVectorTimeSpacing"))
    frequency = float(_one(root, ".//instrument/radarParameters/centerFrequency"))
    lines = [
        "Gamma Interferometric SAR Processor (ISP) - Image Parameter File", "",
        "title: SVN2-03 image 6062500184900001 vendor XML conversion", "sensor: SVN2-03",
        "date: 2026 6 24 21 56 35.806686", f"start_time: {start:.9f} s",
        f"center_time: {center:.9f} s", f"end_time: {stop:.9f} s",
        f"azimuth_line_time: {line_time:.15e} s", "line_header_size: 0",
        f"range_samples: {cols}", f"azimuth_lines: {rows}", "range_looks: 1", "azimuth_looks: 1",
        "image_format: SCOMPLEX", "image_geometry: SLANT_RANGE", "range_scale_factor: 1.0",
        "azimuth_scale_factor: 1.0",
        f"center_latitude: {float(_one(root, './/sceneCenterCoord/lat')):.10f} degrees",
        f"center_longitude: {float(_one(root, './/sceneCenterCoord/lon')):.10f} degrees",
        f"heading: {float(_one(root, './/sceneInfo/headingAngle')):.10f} degrees",
        f"range_pixel_spacing: {range_spacing:.12f} m",
        f"azimuth_pixel_spacing: {7378.157193 * line_time:.12f} m",
        f"near_range_slc: {near_range:.12f} m", f"center_range_slc: {center_range:.12f} m",
        f"far_range_slc: {far_range:.12f} m",
        "first_slant_range_polynomial: 0 0 0 0 0 0 s m 1 m^-1 m^-2 m^-3",
        "center_slant_range_polynomial: 0 0 0 0 0 0 s m 1 m^-1 m^-2 m^-3",
        "last_slant_range_polynomial: 0 0 0 0 0 0 s m 1 m^-1 m^-2 m^-3",
        f"incidence_angle: {float(_one(root, './/sceneCenterCoord/incidenceAngle')):.10f} degrees",
        "azimuth_deskew: ON", "azimuth_angle: 90.0 degrees",
        f"radar_frequency: {frequency:.7e} Hz", "adc_sampling_rate: 8.0e+08 Hz",
        "chirp_bandwidth: 6.0e+08 Hz", f"prf: {1.0 / line_time:.10f} Hz",
        "azimuth_proc_bandwidth: 3.9586494e+03 Hz",
        "doppler_polynomial: 0 0 0 0 Hz Hz/m Hz/m^2 Hz/m^3",
        "doppler_poly_dot: 0 0 0 0 Hz/s Hz/s/m Hz/s/m^2 Hz/s/m^3",
        "doppler_poly_ddot: 0 0 0 0 Hz/s^2 Hz/s^2/m Hz/s^2/m^2 Hz/s^2/m^3",
        "earth_semi_major_axis: 6378137.0 m", "earth_semi_minor_axis: 6356752.3141 m",
        f"number_of_state_vectors: {len(vectors)}",
        f"time_of_first_state_vector: {first_vector_time:.9f} s",
        f"state_vector_interval: {vector_interval:.9f} s",
    ]
    for index, node in enumerate(vectors, 1):
        position = [float(_one(node, key)) for key in ("posX", "posY", "posZ")]
        velocity = [float(_one(node, key)) for key in ("velX", "velY", "velZ")]
        lines.append(f"state_vector_position_{index}: {' '.join(f'{v:.10f}' for v in position)} m m m")
        lines.append(f"state_vector_velocity_{index}: {' '.join(f'{v:.10f}' for v in velocity)} m/s m/s m/s")
    par_path.parent.mkdir(parents=True, exist_ok=True)
    par_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def gamma_project(par: Path, lat: float, lon: float, height: float) -> tuple[float, float]:
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(GAMMA_LIB)
    result = subprocess.run(
        [str(GAMMA), str(par), "-", "-", f"{lat:.12f}", f"{lon:.12f}", f"{height:.6f}"],
        check=True, capture_output=True, text=True, env=environment,
    )
    match = re.search(r"(?:SAR image coordinate|SLC/MLI range, azimuth pixel):\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)", result.stdout)
    if not match:
        raise RuntimeError(f"无法解析GAMMA输出:\n{result.stdout}\n{result.stderr}")
    return float(match.group(1)) - X_OFFSET, float(match.group(2)) - Y_OFFSET
