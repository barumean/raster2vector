"""Sparse Pixel Vectorization — stroke-width estimation.

Estimates the visual thickness of each detected line segment and polyline
contour from the binary source image.  The approach follows the SPV principle:
scan perpendicular cross-sections at several points along each primitive and
average the run lengths, rather than thinning or inspecting every pixel.

Extracted widths are used to assign DXF lineweights so that fine dimension
lines (thin) and main boundary contours (thick) are distinguished in the
output drawing.
"""
from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np


# Standard DXF lineweight values (1/100 mm).  ezdxf accepts these integers on
# the `lineweight` dxfattrib.  0 means "1/100 mm" (hairline) and -3 is default.
# Mapping: thin < 2 px → 13 (0.13 mm), medium 2-4 px → 25, thick > 4 px → 50.
_DXF_WEIGHTS = [13, 18, 25, 35, 50, 70, 100]  # in 1/100 mm


def _nearest_dxf_weight(width_mm: float) -> int:
    """Round a stroke width in mm to the nearest standard DXF lineweight."""
    w_hundredths = int(round(width_mm * 100))
    return min(_DXF_WEIGHTS, key=lambda v: abs(v - w_hundredths))


def _perp_direction(dx: float, dy: float) -> tuple[float, float]:
    """Unit vector perpendicular to (dx, dy)."""
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return 0.0, 1.0
    return -dy / length, dx / length


def _measure_run_at(binary: np.ndarray, cx: float, cy: float,
                    pdx: float, pdy: float, max_half: int = 30) -> float:
    """Measure the foreground run length through (cx, cy) along (pdx, pdy).

    Casts rays from (cx, cy) in both ±(pdx, pdy) directions and returns
    the total number of foreground (255) pixels hit before a background pixel
    is encountered in each direction.
    """
    h, w = binary.shape
    total = 1  # the centre pixel itself
    for sign in (1, -1):
        for step in range(1, max_half + 1):
            xi = int(round(cx + sign * step * pdx))
            yi = int(round(cy + sign * step * pdy))
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                break
            if binary[yi, xi] == 0:
                break
            total += 1
    return float(total)


def estimate_line_widths(
    binary: np.ndarray,
    lines: Sequence[tuple],
    dpi: float = 96.0,
    n_samples: int = 5,
) -> list[int]:
    """Return a DXF lineweight (1/100 mm) for each line segment.

    Samples n_samples perpendicular cross-sections evenly spaced along each
    segment and averages the foreground run lengths.

    Args:
        binary: Binary image (strokes = 255).
        lines: List of (x1, y1, x2, y2) pixel tuples.
        dpi: Source image resolution for pixel→mm conversion.
        n_samples: Number of cross-sections per segment.

    Returns:
        List of DXF lineweight integers, one per line.
    """
    mm_per_px = 25.4 / dpi
    weights = []
    for x1, y1, x2, y2 in lines:
        dx, dy = float(x2 - x1), float(y2 - y1)
        pdx, pdy = _perp_direction(dx, dy)
        seg_len = math.hypot(dx, dy)
        if seg_len < 1.0:
            weights.append(-3)
            continue
        samples = []
        for k in range(n_samples):
            t = (k + 0.5) / n_samples
            cx, cy = x1 + t * dx, y1 + t * dy
            run = _measure_run_at(binary, cx, cy, pdx, pdy)
            samples.append(run)
        avg_px = float(np.median(samples))
        weights.append(_nearest_dxf_weight(avg_px * mm_per_px))
    return weights


def estimate_contour_widths(
    binary: np.ndarray,
    contours: Sequence[np.ndarray],
    dpi: float = 96.0,
    n_samples: int = 5,
) -> list[int]:
    """Return a DXF lineweight for each contour polyline.

    Samples perpendicular cross-sections at several vertices.

    Args:
        binary: Binary image (strokes = 255).
        contours: List of (N, 2) vertex arrays.
        dpi: Source image resolution.
        n_samples: Max cross-sections to sample per contour.

    Returns:
        List of DXF lineweight integers, one per contour.
    """
    mm_per_px = 25.4 / dpi
    weights = []
    for pts in contours:
        if len(pts) < 2:
            weights.append(-3)
            continue
        step = max(1, (len(pts) - 1) // n_samples)
        indices = list(range(0, len(pts) - 1, step))[:n_samples]
        samples = []
        for i in indices:
            p0, p1 = pts[i].astype(float), pts[min(i + 1, len(pts) - 1)].astype(float)
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            pdx, pdy = _perp_direction(dx, dy)
            cx, cy = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
            samples.append(_measure_run_at(binary, cx, cy, pdx, pdy))
        avg_px = float(np.median(samples))
        weights.append(_nearest_dxf_weight(avg_px * mm_per_px))
    return weights
