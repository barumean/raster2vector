import os

import ezdxf
import numpy as np

# DXF layer definitions: (name, ACI colour)
_LAYERS = [
    ("LINES",            7),   # white/black — Hough straight segments
    ("CONTOURS",         3),   # green       — curved / complex polylines
    ("TEXT_CANDIDATES",  2),   # yellow      — blobs classified as text
    ("NOISE_REJECTED",   8),   # dark grey   — entities below noise threshold
]


def export_to_dxf(
    lines: list,
    contours: list,
    output_path: str,
    image_height: int,
    dpi: float = 96.0,
    units_mm: bool = True,
    text_mask_contours: list | None = None,
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
