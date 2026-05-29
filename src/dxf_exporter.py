import os

import ezdxf
import numpy as np


def export_to_dxf(
    lines: list,
    contours: list,
    output_path: str,
    image_height: int,
    dpi: float = 96.0,
    units_mm: bool = True,
) -> int:
    """Export detected lines and contours to a DXF file.

    Coordinate conversion: pixel -> mm (or inches when units_mm=False).
    DXF Y-axis is flipped relative to image pixel Y:
        dxf_y = (image_height - pixel_y) * scale

    Args:
        lines: List of (x1, y1, x2, y2) tuples in pixel space.
        contours: List of np.ndarray of shape (N, 2) in pixel space.
        output_path: Destination .dxf file path.
        image_height: Height of the source image in pixels.
        dpi: Dots per inch of the source image (used for unit conversion).
        units_mm: When True produce millimetre coordinates; otherwise inches.

    Returns:
        Total number of DXF entities written.
    """
    pixel_to_unit = 25.4 / dpi if units_mm else 1.0 / dpi

    # Create parent directories if they don't exist
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    doc = ezdxf.new(dxfversion="R2010")
    doc.units = 4 if units_mm else 1  # 4 = mm, 1 = inches

    msp = doc.modelspace()

    # Create named layers
    doc.layers.add("lines", color=7)     # white / black depending on background
    doc.layers.add("contours", color=3)  # green

    def px_to_dxf(x: float, y: float) -> tuple[float, float]:
        return x * pixel_to_unit, (image_height - y) * pixel_to_unit

    entity_count = 0

    # Add LINE entities for Hough-detected straight segments
    for x1, y1, x2, y2 in lines:
        dx1, dy1 = px_to_dxf(x1, y1)
        dx2, dy2 = px_to_dxf(x2, y2)
        msp.add_line((dx1, dy1), (dx2, dy2), dxfattribs={"layer": "lines"})
        entity_count += 1

    # Add LWPOLYLINE entities for remaining contour shapes
    for contour in contours:
        dxf_points = [px_to_dxf(float(pt[0]), float(pt[1])) for pt in contour]
        first, last = contour[0], contour[-1]
        is_closed = bool(np.linalg.norm(first.astype(float) - last.astype(float)) < 2.0)
        msp.add_lwpolyline(dxf_points, close=is_closed, dxfattribs={"layer": "contours"})
        entity_count += 1

    doc.saveas(output_path)
    return entity_count
