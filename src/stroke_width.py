"""Stroke-width estimation via medial-axis distance transform (SPV principle).

Two approaches are provided:

1. **medial_axis** (default, more accurate):
   `skimage.morphology.medial_axis(binary, return_distance=True)` computes the
   skeleton *and* the distance transform simultaneously.  The distance value at
   each skeleton pixel equals the **local half-width** of the stroke (the radius
   of the largest fitting circle at that point).  This is the skeleton-based
   equivalent of SPV's "width-run" measurement and requires no extra scanning.

2. **Cross-section sampling** (fallback when skimage is unavailable):
   Casts perpendicular rays from sample points along each primitive and averages
   the foreground run length (the original SPV principle from Dori & Liu 1999).

Extracted widths are mapped to standard DXF lineweights (1/100 mm, group code
370) so that thin dimension lines and thick boundary contours land on different
weights in the output drawing.
"""
from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

try:
    from skimage.morphology import medial_axis as _ski_medial_axis
    _SKIMAGE_MEDIAL = True
except ImportError:
    _SKIMAGE_MEDIAL = False


# Standard DXF lineweight values (1/100 mm). ezdxf accepts these integers on
# the `lineweight` dxfattrib.
_DXF_WEIGHTS = [13, 18, 25, 35, 50, 70, 100]  # in 1/100 mm


def _nearest_dxf_weight(width_mm: float) -> int:
    """Round a stroke width in mm to the nearest standard DXF lineweight."""
    w_hundredths = max(1, int(round(width_mm * 100)))
    return min(_DXF_WEIGHTS, key=lambda v: abs(v - w_hundredths))


# ── medial_axis-based width map ───────────────────────────────────────────────

def _build_width_map(binary: np.ndarray) -> np.ndarray | None:
    """Return a float32 array where each pixel holds the local stroke half-width.

    Uses `skimage.morphology.medial_axis(..., return_distance=True)`.  Each
    skeleton pixel carries the radius of the largest disc that fits at that
    point, i.e. the local half-width of the stroke.  Non-skeleton pixels are 0.

    Returns None if scikit-image is not installed (cross-section fallback used).
    """
    if not _SKIMAGE_MEDIAL:
        return None
    _, dist = _ski_medial_axis(binary > 0, return_distance=True)
    return dist.astype(np.float32)


def _sample_width_medial(
    width_map: np.ndarray,
    xs: list[float],
    ys: list[float],
    search_r: int = 4,
) -> float:
    """Sample the width_map near a list of (x, y) points.

    Looks up the maximum distance value in a small disc around each point,
    then returns twice the median (diameter = full stroke width).
    """
    h, w = width_map.shape
    samples = []
    for x, y in zip(xs, ys):
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - search_r), min(w, xi + search_r + 1)
        y0, y1 = max(0, yi - search_r), min(h, yi + search_r + 1)
        patch = width_map[y0:y1, x0:x1]
        if patch.size:
            samples.append(float(patch.max()))
    return 2.0 * float(np.median(samples)) if samples else 1.0


# ── Cross-section sampling fallback (SPV-style) ───────────────────────────────

def _perp_direction(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return 0.0, 1.0
    return -dy / length, dx / length


def _measure_run_at(binary: np.ndarray, cx: float, cy: float,
                    pdx: float, pdy: float, max_half: int = 30) -> float:
    h, w = binary.shape
    total = 1
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


# ── Public API ────────────────────────────────────────────────────────────────

def estimate_line_widths(
    binary: np.ndarray,
    lines: Sequence[tuple],
    dpi: float = 96.0,
    n_samples: int = 5,
    width_map: np.ndarray | None = None,
) -> list[int]:
    """Return a DXF lineweight (1/100 mm) for each line segment.

    Prefers the medial-axis distance-transform approach (more accurate than
    cross-section sampling) when scikit-image is available; falls back to
    perpendicular run-length scanning otherwise.

    Args:
        binary: Binary image (strokes = 255).
        lines: List of (x1, y1, x2, y2) pixel tuples.
        dpi: Source image resolution for pixel→mm conversion.
        n_samples: Number of cross-sections / sample points per segment.
        width_map: Pre-computed medial_axis distance map; if None, computed
            from binary when scikit-image is available.

    Returns:
        List of DXF lineweight integers, one per line.
    """
    mm_per_px = 25.4 / dpi
    if width_map is None and _SKIMAGE_MEDIAL:
        width_map = _build_width_map(binary)

    weights = []
    for x1, y1, x2, y2 in lines:
        dx, dy = float(x2 - x1), float(y2 - y1)
        seg_len = math.hypot(dx, dy)
        if seg_len < 1.0:
            weights.append(-3)
            continue

        # Sample points evenly spaced along the segment.
        xs = [x1 + (k + 0.5) / n_samples * dx for k in range(n_samples)]
        ys = [y1 + (k + 0.5) / n_samples * dy for k in range(n_samples)]

        if width_map is not None:
            avg_px = _sample_width_medial(width_map, xs, ys)
        else:
            pdx, pdy = _perp_direction(dx, dy)
            runs = [_measure_run_at(binary, x, y, pdx, pdy) for x, y in zip(xs, ys)]
            avg_px = float(np.median(runs))

        weights.append(_nearest_dxf_weight(avg_px * mm_per_px))
    return weights


def estimate_contour_widths(
    binary: np.ndarray,
    contours: Sequence[np.ndarray],
    dpi: float = 96.0,
    n_samples: int = 5,
    width_map: np.ndarray | None = None,
) -> list[int]:
    """Return a DXF lineweight for each contour polyline.

    Args:
        binary: Binary image (strokes = 255).
        contours: List of (N, 2) vertex arrays.
        dpi: Source image resolution.
        n_samples: Max sample points per contour.
        width_map: Pre-computed medial_axis distance map.

    Returns:
        List of DXF lineweight integers, one per contour.
    """
    mm_per_px = 25.4 / dpi
    if width_map is None and _SKIMAGE_MEDIAL:
        width_map = _build_width_map(binary)

    weights = []
    for pts in contours:
        if len(pts) < 2:
            weights.append(-3)
            continue
        step = max(1, (len(pts) - 1) // n_samples)
        indices = list(range(0, len(pts) - 1, step))[:n_samples]

        if width_map is not None:
            xs = [(pts[i][0] + pts[min(i + 1, len(pts) - 1)][0]) / 2.0
                  for i in indices]
            ys = [(pts[i][1] + pts[min(i + 1, len(pts) - 1)][1]) / 2.0
                  for i in indices]
            avg_px = _sample_width_medial(width_map, xs, ys)
        else:
            samples = []
            for i in indices:
                p0 = pts[i].astype(float)
                p1 = pts[min(i + 1, len(pts) - 1)].astype(float)
                dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                pdx, pdy = _perp_direction(dx, dy)
                cx, cy = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
                samples.append(_measure_run_at(binary, cx, cy, pdx, pdy))
            avg_px = float(np.median(samples))

        weights.append(_nearest_dxf_weight(avg_px * mm_per_px))
    return weights
