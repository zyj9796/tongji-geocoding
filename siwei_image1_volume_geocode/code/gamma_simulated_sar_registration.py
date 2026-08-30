#!/usr/bin/env python3
"""Register GAMMA gc_map2/sim_sar output to the cropped observed SAR image."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.registration import phase_cross_correlation

from gamma_projection_core import WORK, X_OFFSET, Y_OFFSET


# This LUT was generated from the Wusong-height DSM with the local
# Wusong-to-WGS84 ellipsoid offset applied in DEM_hgt_offset.  The legacy
# gamma_simulated_sar directory treated 4 m Wusong as 4 m ellipsoid and forced
# phase correlation to absorb a roughly 50-pixel range error.
SIM_WORK = WORK / "gamma_simulated_sar_ellipsoid"


def _par_int(path: Path, key: str) -> int:
    match = re.search(rf"^{re.escape(key)}:\s+([0-9]+)", path.read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise ValueError(f"GAMMA参数缺失: {key}")
    return int(match.group(1))


def _robust01(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image) & (image > 0)]
    if not finite.size:
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(finite, [2.0, 98.0])
    return np.clip((image - low) / max(float(high - low), 1e-6), 0, 1).astype(np.float32)


def gamma_simulated_radar(shape: tuple[int, int], factor: int) -> np.ndarray:
    """Splat GAMMA's map-geometry sim_sar through its lookup table into crop pixels."""
    parameter = SIM_WORK / "dem_seg.par"
    width = _par_int(parameter, "width")
    lines = _par_int(parameter, "nlines")
    lut = np.fromfile(SIM_WORK / "image1.lt", dtype=">f4")
    simulated = np.fromfile(SIM_WORK / "sim_sar", dtype=">f4")
    if lut.size != width * lines * 2 or simulated.size != width * lines:
        raise ValueError("GAMMA模拟SAR或查找表尺寸与dem_seg.par不一致")
    lut = lut.reshape(lines, width, 2).astype(np.float32)
    simulated = simulated.reshape(lines, width).astype(np.float32)
    rows = (shape[0] + factor - 1) // factor
    cols = (shape[1] + factor - 1) // factor
    col = np.rint((lut[:, :, 0] - X_OFFSET) / factor).astype(np.int32)
    row = np.rint((lut[:, :, 1] - Y_OFFSET) / factor).astype(np.int32)
    valid = np.isfinite(simulated) & (simulated > 0)
    valid &= (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
    linear = row[valid].astype(np.int64) * cols + col[valid]
    total = np.bincount(linear, weights=simulated[valid], minlength=rows * cols)
    count = np.bincount(linear, minlength=rows * cols)
    radar = (total / np.maximum(count, 1)).reshape(rows, cols)
    return _robust01(radar)


def _feature_image(image: np.ndarray) -> np.ndarray:
    normalized = _robust01(image)
    # Retain building-scale structure and suppress both speckle and broad radiometric trends.
    feature = gaussian_filter(normalized, 0.8) - gaussian_filter(normalized, 5.0)
    window = np.hanning(feature.shape[0])[:, None] * np.hanning(feature.shape[1])[None, :]
    return ((feature - float(feature.mean())) * window).astype(np.float32)


def simulated_sar_registration(amplitude: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    """Estimate the translation applied to GAMMA projections using simulated/real SAR features."""
    results = []
    displays: dict[int, np.ndarray] = {}
    for factor in (2, 4):
        simulated = gamma_simulated_radar(amplitude.shape, factor)
        displays[factor] = simulated
        observed = _robust01(np.log1p(np.maximum(amplitude[::factor, ::factor], 0)))
        observed = observed[: simulated.shape[0], : simulated.shape[1]]
        shift, error, phase = phase_cross_correlation(
            _feature_image(observed), _feature_image(simulated), upsample_factor=4, normalization="phase"
        )
        results.append((factor, float(shift[1] * factor), float(shift[0] * factor), float(error), float(phase)))
    primary = results[0]
    check = results[1]
    disagreement = float(np.hypot(primary[1] - check[1], primary[2] - check[2]))
    if abs(primary[1]) > 96 or abs(primary[2]) > 96 or disagreement > 8:
        raise RuntimeError(f"GAMMA模拟SAR配准未通过限制检查: {results}")
    registration = {
        "method": "GAMMA_gc_map2_sim_sar_to_observed_SAR_multiscale_phase_correlation",
        "col_shift_px": primary[1],
        "row_shift_px": primary[2],
        "factor2_col_shift_px": primary[1],
        "factor2_row_shift_px": primary[2],
        "factor4_col_shift_px": check[1],
        "factor4_row_shift_px": check[2],
        "cross_scale_disagreement_px": disagreement,
        "search_limit_px": 96.0,
    }
    return registration, displays[2]


def simulated_sar_spatial_registration(amplitude: np.ndarray) -> tuple[dict, np.ndarray]:
    """Fit a smooth spatial correction from reliable simulated/observed SAR windows."""
    global_registration, simulated_factor2 = simulated_sar_registration(amplitude)
    factor = 4
    simulated = gamma_simulated_radar(amplitude.shape, factor)
    observed = _robust01(np.log1p(np.maximum(amplitude[::factor, ::factor], 0)))
    observed = observed[:simulated.shape[0], :simulated.shape[1]]
    reference = _feature_image(observed)
    moving = _feature_image(simulated)
    rows, cols = reference.shape
    base_dx = global_registration["col_shift_px"] / factor
    base_dy = global_registration["row_shift_px"] / factor
    records = []
    for grid_row in range(1, 5):
        center_row = grid_row * rows / 5.0
        for grid_col in range(1, 6):
            center_col = grid_col * cols / 6.0
            half_rows, half_cols = 180, 210
            r0, r1 = max(20, int(center_row - half_rows)), min(rows - 20, int(center_row + half_rows))
            c0, c1 = max(20, int(center_col - half_cols)), min(cols - 20, int(center_col + half_cols))
            rr, cc = np.mgrid[r0:r1, c0:c1]
            ref_values = reference[r0:r1, c0:c1].ravel()
            candidates = []
            for dy in np.arange(base_dy - 4, base_dy + 4.01, 1.0):
                for dx in np.arange(base_dx - 4, base_dx + 4.01, 1.0):
                    mov_values = map_coordinates(moving, np.asarray([rr - dy, cc - dx]), order=1, mode="constant").ravel()
                    denominator = np.sqrt(np.sum(ref_values * ref_values) * np.sum(mov_values * mov_values))
                    score = float(np.sum(ref_values * mov_values) / denominator) if denominator else -1.0
                    candidates.append((score, float(dx), float(dy)))
            candidates.sort(reverse=True)
            best = candidates[0]
            separated = [item[0] for item in candidates if np.hypot(item[1] - best[1], item[2] - best[2]) >= 2.0]
            margin = best[0] - max(separated)
            records.append({
                "col_px": center_col * factor, "row_px": center_row * factor,
                "col_shift_px": best[1] * factor, "row_shift_px": best[2] * factor,
                "score": best[0], "peak_margin": margin,
            })
    accepted = [item for item in records if item["score"] >= 0.055 and item["peak_margin"] >= 0.008]
    if len(accepted) < 10:
        raise RuntimeError(f"可靠模拟SAR局部配准窗口不足: {len(accepted)}")
    x = np.asarray([(item["col_px"] - amplitude.shape[1] / 2) / (amplitude.shape[1] / 2) for item in accepted])
    y = np.asarray([(item["row_px"] - amplitude.shape[0] / 2) / (amplitude.shape[0] / 2) for item in accepted])
    design = np.column_stack([np.ones(len(accepted)), x, y, x * y])
    col_values = np.asarray([item["col_shift_px"] for item in accepted])
    row_values = np.asarray([item["row_shift_px"] for item in accepted])
    col_coefficients = np.linalg.lstsq(design, col_values, rcond=None)[0]
    row_coefficients = np.linalg.lstsq(design, row_values, rcond=None)[0]
    prediction_col = design @ col_coefficients
    prediction_row = design @ row_coefficients
    residual = np.hypot(prediction_col - col_values, prediction_row - row_values)
    # Bounds must follow the current datum-correct simulation.  The legacy LUT
    # needed a -58..-30 px range correction because its heights were wrong;
    # retaining those constants would silently reintroduce that error here.
    col_bounds = [float(np.min(col_values) - factor), float(np.max(col_values) + factor)]
    row_bounds = [float(np.min(row_values) - factor), float(np.max(row_values) + factor)]
    summary = {
        **global_registration,
        "method": "GAMMA_sim_sar_to_observed_SAR_reliable_window_bilinear_residual_field",
        "spatial_model": "shift = b0 + b1*x + b2*y + b3*x*y; x/y normalized about crop center",
        "col_coefficients_px": col_coefficients.tolist(),
        "row_coefficients_px": row_coefficients.tolist(),
        "col_shift_bounds_px": col_bounds,
        "row_shift_bounds_px": row_bounds,
        "accepted_windows": len(accepted),
        "total_windows": len(records),
        "fit_residual_median_px": float(np.median(residual)),
        "fit_residual_p90_px": float(np.percentile(residual, 90)),
        "windows": records,
    }
    return summary, simulated_factor2


def evaluate_spatial_shift(registration: dict, col: np.ndarray, row: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    x = (col - shape[1] / 2) / (shape[1] / 2)
    y = (row - shape[0] / 2) / (shape[0] / 2)
    design = np.stack([np.ones_like(x), x, y, x * y], axis=0)
    col_shift = np.tensordot(np.asarray(registration["col_coefficients_px"]), design, axes=(0, 0))
    row_shift = np.tensordot(np.asarray(registration["row_coefficients_px"]), design, axes=(0, 0))
    # Guard against extrapolation outside the accepted-window correction range.
    col_bounds = registration["col_shift_bounds_px"]
    row_bounds = registration["row_shift_bounds_px"]
    col_shift = np.clip(col_shift, col_bounds[0], col_bounds[1])
    row_shift = np.clip(row_shift, row_bounds[0], row_bounds[1])
    return col_shift.astype(np.float32), row_shift.astype(np.float32)
