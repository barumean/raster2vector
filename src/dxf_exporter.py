import math
import os

import ezdxf
import numpy as np

# DXF layer definitions: (name, ACI colour)
_LAYERS = [
    ("LINES",            7),   # white/black — Hough straight segments
    ("CONTOURS",         3),   # green       — curved / complex polylines
    ("ARCS",             5),   # blue        — fitted circles / arcs
    ("TEXT_CANDIDATES",  2),   # yellow      — blobs classified as text
    ("NOISE_REJECTED",   8),   # dark grey   — entities below noise threshold
]


def _bulge_from_3pts(s, m, e) -> float:
    """DXF bulge b = tan(theta/4) for the arc s→e passing through m.

    The bulge is signed: positive = counter-clockwise, negative = clockwise,
    matching the DXF group-code-42 convention.  Points are taken in the target
    (already Y-flipped) coordinate space so orientation is correct on output.
    """
    (x1, y1), (x2, y2), (x3, y3) = s, m, e
    # Circumcircle centre via perpendicular-bisector determinant.
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-9:
        return 0.0  # collinear → straight segment
    ux = ((x1 ** 2 + y1 ** 2) * (y2 - y3) + (x2 ** 2 + y2 ** 2) * (y3 - y1)
          + (x3 ** 2 + y3 ** 2) * (y1 - y2)) / d
    uy = ((x1 ** 2 + y1 ** 2) * (x3 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x3)
          + (x3 ** 2 + y3 ** 2) * (x2 - x1)) / d
    a0 = math.atan2(y1 - uy, x1 - ux)
    a1 = math.atan2(y2 - uy, x2 - ux)
    a2 = math.atan2(y3 - uy, x3 - ux)
    two_pi = 2 * math.pi
    sweep_ccw = (a2 - a0) % two_pi          # CCW sweep start→end
    mid_ccw = (a1 - a0) % two_pi            # CCW position of the mid point
    if mid_ccw <= sweep_ccw:
        theta = sweep_ccw                    # arc runs CCW (positive)
    else:
        theta = -(two_pi - sweep_ccw)        # arc runs CW (negative)
    return math.tan(theta / 4.0)


def export_to_dxf(
    lines: list,
    contours: list,
    output_path: str,
    image_height: int,
    dpi: float = 96.0,
    units_mm: bool = True,
    text_mask_contours: list | None = None,
    arcs: list | None = None,
) -> int:
    """Export detected geometry to a DXF file.

    Coordinate mapping
    ------------------
    pixel (x, y)  →  DXF (x * scale,  (image_height - y) * scale)
    where scale = 25.4 / dpi  (mm/px)  when units_mm is True,
          scale = 1.0  / dpi  (in/px)  otherwise.

    Entity rules
    ------------
    * Straight segments from Hough → LINE on layer LINES.
    * Curved polylines             → LWPOLYLINE (closed when start≈end) on CONTOURS.
    * Text candidates              → LWPOLYLINE on TEXT_CANDIDATES.

    Args:
        lines: List of (x1, y1, x2, y2) pixel tuples.
        contours: List of np.ndarray (N, 2) pixel arrays.
        output_path: Destination .dxf file path.
        image_height: Source image height in pixels (for Y-flip).
        dpi: Source image resolution used for pixel→unit conversion.
        units_mm: Produce millimetre coordinates when True, inches otherwise.
        text_mask_contours: Optional contours from the text-candidate mask.

    Returns:
        Total number of DXF entities written.
    """
    scale = 25.4 / dpi if units_mm else 1.0 / dpi

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    doc = ezdxf.new(dxfversion="R2010")
    doc.units = 4 if units_mm else 1  # 4 = mm, 1 = inches

    msp = doc.modelspace()

    for name, colour in _LAYERS:
        doc.layers.add(name, color=colour)

    def px(x: float, y: float) -> tuple[float, float]:
        return x * scale, (image_height - y) * scale

    entity_count = 0

    # ── Straight lines ────────────────────────────────────────────────────────
    for x1, y1, x2, y2 in lines:
        msp.add_line(px(x1, y1), px(x2, y2), dxfattribs={"layer": "LINES"})
        entity_count += 1

    # ── Curved / complex polylines ────────────────────────────────────────────
    for contour in contours:
        pts = [px(float(p[0]), float(p[1])) for p in contour]
        first, last = contour[0].astype(float), contour[-1].astype(float)
        closed = bool(np.linalg.norm(first - last) < 2.0)
        msp.add_lwpolyline(pts, close=closed, dxfattribs={"layer": "CONTOURS"})
        entity_count += 1

    # ── Circles / arcs (compact CAD primitives, DXF Section 5) ────────────────
    for arc in (arcs or []):
        if arc.get("type") == "circle":
            cx, cy = arc["center"]
            r = float(arc["r"])
            cxf, cyf = px(cx, cy)
            msp.add_circle((cxf, cyf), r * scale, dxfattribs={"layer": "ARCS"})
            entity_count += 1
        elif arc.get("type") == "arc":
            s = px(*arc["start"])
            m = px(*arc["mid"])
            e = px(*arc["end"])
            bulge = _bulge_from_3pts(s, m, e)
            # Two-vertex LWPOLYLINE: start carries the bulge, then the end.
            msp.add_lwpolyline(
                [(s[0], s[1], 0.0, 0.0, bulge), (e[0], e[1])],
                format="xyseb",
                dxfattribs={"layer": "ARCS"},
            )
            entity_count += 1

    # ── Text candidates ───────────────────────────────────────────────────────
    for tc in (text_mask_contours or []):
        pts = [px(float(p[0]), float(p[1])) for p in tc]
        if len(pts) >= 2:
            first, last = tc[0].astype(float), tc[-1].astype(float)
            closed = bool(np.linalg.norm(first - last) < 2.0)
            msp.add_lwpolyline(pts, close=closed,
                               dxfattribs={"layer": "TEXT_CANDIDATES"})
            entity_count += 1

    doc.saveas(output_path)
    return entity_count
