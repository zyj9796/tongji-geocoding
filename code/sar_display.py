from __future__ import annotations

import numpy as np


def stretch_sar_grayscale(
    values: np.ndarray,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    gamma: float = 1.0,
    power_input: bool = True,
) -> np.ndarray:
    """Match the reference RSLC amplitude stretch while keeping no-data black."""
    data = np.asarray(values, dtype=np.float32)
    valid_mask = np.isfinite(data) & (data > 0)
    out = np.zeros(data.shape, dtype=np.float32)
    if not np.any(valid_mask):
        return out

    display_values = np.sqrt(data[valid_mask]) if power_input else data[valid_mask]
    low, high = np.percentile(display_values, [lower_percentile, upper_percentile])
    if high <= low:
        return out
    scaled = np.clip((display_values - low) / (high - low), 0.0, 1.0)
    out[valid_mask] = np.power(scaled, gamma)
    return out
