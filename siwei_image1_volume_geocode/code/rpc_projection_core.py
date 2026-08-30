#!/usr/bin/env python3
"""Forward projection with the vendor RPC model in cropped SAR coordinates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gamma_projection_core import INPUT, X_OFFSET, Y_OFFSET


RPC_TERMS = 20


@dataclass(frozen=True)
class RPCModel:
    line_offset: float
    sample_offset: float
    latitude_offset: float
    longitude_offset: float
    height_offset: float
    line_scale: float
    sample_scale: float
    latitude_scale: float
    longitude_scale: float
    height_scale: float
    line_numerator: np.ndarray
    line_denominator: np.ndarray
    sample_numerator: np.ndarray
    sample_denominator: np.ndarray

    @staticmethod
    def read(path: Path = INPUT / "source.rpb") -> "RPCModel":
        text = path.read_text(encoding="utf-8")

        def scalar(key: str) -> float:
            match = re.search(rf"\b{re.escape(key)}\s*=\s*([-+0-9.eE]+)", text)
            if match is None:
                raise ValueError(f"RPC scalar missing: {key}")
            return float(match.group(1))

        def coefficients(key: str) -> np.ndarray:
            match = re.search(rf"\b{re.escape(key)}\s*=\s*\((.*?)\);", text, re.DOTALL)
            if match is None:
                raise ValueError(f"RPC coefficients missing: {key}")
            values = np.asarray(
                [float(item) for item in re.findall(r"[-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?", match.group(1))],
                dtype=np.float64,
            )
            if len(values) != RPC_TERMS:
                raise ValueError(f"RPC {key} has {len(values)} terms, expected {RPC_TERMS}")
            return values

        return RPCModel(
            line_offset=scalar("lineOffset"), sample_offset=scalar("sampOffset"),
            latitude_offset=scalar("latOffset"), longitude_offset=scalar("longOffset"),
            height_offset=scalar("heightOffset"), line_scale=scalar("lineScale"),
            sample_scale=scalar("sampScale"), latitude_scale=scalar("latScale"),
            longitude_scale=scalar("longScale"), height_scale=scalar("heightScale"),
            line_numerator=coefficients("lineNumCoef"), line_denominator=coefficients("lineDenCoef"),
            sample_numerator=coefficients("sampNumCoef"), sample_denominator=coefficients("sampDenCoef"),
        )

    def project(self, latitude, longitude, ellipsoid_height_m) -> tuple[np.ndarray, np.ndarray]:
        """Return zero-based (crop column, crop row) arrays."""
        latitude, longitude, height = np.broadcast_arrays(
            np.asarray(latitude, dtype=np.float64),
            np.asarray(longitude, dtype=np.float64),
            np.asarray(ellipsoid_height_m, dtype=np.float64),
        )
        p = (latitude - self.latitude_offset) / self.latitude_scale
        l = (longitude - self.longitude_offset) / self.longitude_scale
        h = (height - self.height_offset) / self.height_scale
        terms = np.stack([
            np.ones_like(l), l, p, h, l*p, l*h, p*h, l*l, p*p, h*h,
            l*p*h, l**3, l*p*p, l*h*h, l*l*p, p**3, p*h*h, l*l*h, p*p*h, h**3,
        ], axis=-1)
        line = self.line_offset + self.line_scale * (
            terms @ self.line_numerator / (terms @ self.line_denominator)
        )
        sample = self.sample_offset + self.sample_scale * (
            terms @ self.sample_numerator / (terms @ self.sample_denominator)
        )
        # Product coordinates and scene corner metadata are one-based.
        return sample - 1.0 - X_OFFSET, line - 1.0 - Y_OFFSET

