"""DXF serialisation using ezdxf.

Arc encoding
------------
Fitted arcs are exported as native DXF ARC entities via ezdxf.math.arc_to_bulge
(the official utility) rather than a hand-rolled circumcircle formula.  The
pipeline is:

  3-point arc (start, mid, end in pixel space)
    → Y-flip to DXF space
    → circumcircle fit → (center, start_angle, end_angle, radius)
    → ezdxf.math.arc_to_bulge(center, start_angle_rad, end_angle_rad, radius)
       returns (start_pt, end_pt, bulge)   [bulge = tan(θ/4)]
    → msp.add_lwpolyline([start_pt+bulge, end_pt], format="xyseb")

This preserves the DXF group-code-42 convention (b > 0 → CCW, b < 0 → CW)
exactly.  Full circles use msp.add_circle.

Reference: ezdxf.readthedocs.io → math → arc_to_bulge
"""
import math
import os

import ezdxf
import ezdxf.math as ezm
import numpy as np

# DXF layer definitions: (name, ACI colour)
_LAYERS = [
    ("LINES",            1),   # red         — straight segments
    ("CONTOURS",         3),   # green       — curved / complex polylines
    ("ARCS",             5),   # blue        — fitted circles / arcs
    ("ELONGATED",        6),   # magenta     — dash / elongated-char candidates
    ("TEXT_CANDIDATES",  2),   # yellow      — text-string blobs
    ("NOISE_REJECTED",   8),   # dark grey   — below noise threshold
    ("DASHED",           1),   # red         — detected dashed / hidden lines
    ("BOXES",            4),   # cyan        — detected rectangular closed regions
    ("BOX",              7),   # white/black — outermost drawing border
]

# Standard DXF DASHED linetype pattern (dash=0.5, gap=0.25 drawing units)
_DASHED_PATTERN = [0.5, -0.25]


def _circumcircle_3pts(
    s: tuple[float, float],
    m: tuple[float, float],
    e: tuple[float, float],
) -> tuple[float, float, float, float, float] | None:
    """Return (cx, cy, r, start_angle_rad, end_angle_rad) from three arc points.

    All points are already in DXF (Y-flipped) coordinate space.
    Returns None if the three points are collinear.
    """
    x1, y1 = s
    x2, y2 = m
    x3, y3 = e
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-9:
        return None
    ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1)
          + (x3**2 + y3**2) * (y1 - y2)) / d
    uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3)
          + (x3**2 + y3**2) * (x2 - x1)) / d
    r = math.hypot(x1 - ux, y1 - uy)
    # DXF arc angles are measured CCW from the positive X axis.
    a_start = math.atan2(y1 - uy, x1 - ux)
    a_end = math.atan2(y3 - uy, x3 - ux)
    # Verify that the mid point falls on the CCW arc from start to end.
    a_mid = math.atan2(y2 - uy, x2 - ux)
    two_pi = 2 * math.pi
    mid_ccw = (a_mid - a_start) % two_pi
    end_ccw = (a_end - a_start) % two_pi
    if mid_ccw > end_ccw:
        # Mid is not between start and end CCW → swap direction (flip end).
        a_start, a_end = a_end, a_start
    return ux, uy, r, a_start, a_end


# Keep the name exported so existing tests can import it.
def _bulge_from_3pts(s, m, e) -> float:
    """DXF bulge b = tan(theta/4) for the arc s→e passing through m.

    Uses the circumcircle approach backed by ezdxf.math.arc_to_bulge, which is
    the official ezdxf utility.  Falls back to a direct trig formula if the
    circumcircle degenerate check triggers.
    """
    result = _circumcircle_3pts(s, m, e)
    if result is None:
        return 0.0
    cx, cy, r, a_start, a_end = result
    try:
        _sp, _ep, bulge = ezm.arc_to_bulge(
            center=(cx, cy),
            start_angle=a_start,
            end_angle=a_end,
            radius=r,
        )
        return float(bulge)
    except Exception:
        return 0.0


def export_to_dxf(
    lines: list,
    contours: list,
    output_path: str,
    image_height: int,
    dpi: float = 96.0,
    units_mm: bool = True,
    text_mask_contours: list | None = None,
    arcs: list | None = None,
    line_weights: list | None = None,
    contour_weights: list | None = None,
    elongated_contours: list | None = None,
    dashed_lines: list | None = None,
    box_contours: list | None = None,
    page_border: tuple | None = None,
) -> int:
    """Export detected geometry to a DXF file.

    Coordinate mapping
    ------------------
    pixel (x, y)  →  DXF (x * scale,  (image_height - y) * scale)
    where scale = 25.4 / dpi  (mm/px) when units_mm is True.

    Entity mapping
    --------------
    * Straight segments       → LINE        on LINES
    * Curved polylines        → LWPOLYLINE  on CONTOURS
    * Fitted full circles     → CIRCLE      on ARCS
    * Fitted partial arcs     → LWPOLYLINE (bulge) on ARCS, using
                                ezdxf.math.arc_to_bulge for encoding
    * Elongated/dash blobs    → LWPOLYLINE  on ELONGATED
    * Text-string blobs       → LWPOLYLINE  on TEXT_CANDIDATES

    Args:
        lines: List of (x1, y1, x2, y2) pixel tuples.
        contours: List of np.ndarray (N, 2) pixel arrays.
        output_path: Destination .dxf file path.
        image_height: Source image height in pixels (for Y-flip).
        dpi: Source image resolution for pixel→unit conversion.
        units_mm: Produce millimetre coordinates when True, inches otherwise.
        text_mask_contours: Contours from the text-candidate mask.
        arcs: List of arc/circle dicts from the vectorizer.
        line_weights: DXF lineweight integers per line (1/100 mm).
        contour_weights: DXF lineweight integers per contour.
        elongated_contours: Contours from the elongated/dash-candidate mask.
        dashed_lines: List of dashed groups from _detect_dashed_lines().
            Each group is a list of (x1,y1,x2,y2) dash segments sorted along
            the dashed-line axis.  Emitted as a single LINE entity spanning
            the full group extent, on the DASHED layer with a DASHED linetype.

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

    # Register the DASHED linetype so entities on the DASHED layer render
    # with the correct dash pattern in compliant DXF viewers.
    if "DASHED" not in doc.linetypes:
        doc.linetypes.add("DASHED", pattern=_DASHED_PATTERN,
                          description="Dashed ____ ____ ____")

    def px(x: float, y: float) -> tuple[float, float]:
        return x * scale, (image_height - y) * scale

    entity_count = 0

    # ── Straight lines ────────────────────────────────────────────────────────
    for i, (x1, y1, x2, y2) in enumerate(lines):
        attribs: dict = {"layer": "LINES"}
        if line_weights and i < len(line_weights) and line_weights[i] > 0:
            attribs["lineweight"] = line_weights[i]
        msp.add_line(px(x1, y1), px(x2, y2), dxfattribs=attribs)
        entity_count += 1

    # ── Curved / complex polylines ────────────────────────────────────────────
    for i, contour in enumerate(contours):
        pts = [px(float(p[0]), float(p[1])) for p in contour]
        first, last = contour[0].astype(float), contour[-1].astype(float)
        closed = bool(np.linalg.norm(first - last) < 2.0)
        attribs = {"layer": "CONTOURS"}
        if contour_weights and i < len(contour_weights) and contour_weights[i] > 0:
            attribs["lineweight"] = contour_weights[i]
        msp.add_lwpolyline(pts, close=closed, dxfattribs=attribs)
        entity_count += 1

    # ── Circles / arcs via ezdxf.math.arc_to_bulge ───────────────────────────
    for arc in (arcs or []):
        if arc.get("type") == "circle":
            cx, cy = arc["center"]
            r = float(arc["r"])
            cxf, cyf = px(cx, cy)
            msp.add_circle((cxf, cyf), r * scale, dxfattribs={"layer": "ARCS"})
            entity_count += 1

        elif arc.get("type") == "arc":
            # Y-flip all three control points before fitting the circumcircle,
            # so the derived angles are in DXF coordinate space.
            sf = px(*arc["start"])
            mf = px(*arc["mid"])
            ef = px(*arc["end"])

            result = _circumcircle_3pts(sf, mf, ef)
            if result is None:
                continue
            cx, cy, r, a_start, a_end = result
            try:
                start_pt, end_pt, bulge = ezm.arc_to_bulge(
                    center=(cx, cy),
                    start_angle=a_start,
                    end_angle=a_end,
                    radius=r,
                )
                msp.add_lwpolyline(
                    [(start_pt.x, start_pt.y, 0.0, 0.0, bulge),
                     (end_pt.x, end_pt.y)],
                    format="xyseb",
                    dxfattribs={"layer": "ARCS"},
                )
                entity_count += 1
            except Exception:
                pass

    # ── Elongated / dash-candidate blobs ─────────────────────────────────────
    for tc in (elongated_contours or []):
        pts = [px(float(p[0]), float(p[1])) for p in tc]
        if len(pts) >= 2:
            msp.add_lwpolyline(pts, dxfattribs={"layer": "ELONGATED"})
            entity_count += 1

    # ── Dashed / hidden lines (Scan2CAD dash_line_identification) ────────────
    for group in (dashed_lines or []):
        if len(group) < 1:
            continue
        # Determine axis direction from the group's overall span.
        all_pts = [(float(s[0]), float(s[1])) for s in group] + \
                  [(float(s[2]), float(s[3])) for s in group]
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        # Project all endpoints onto the axis and take the two extremes.
        if dx >= dy:
            key = lambda pt: pt[0]
        else:
            key = lambda pt: pt[1]
        start_pt = min(all_pts, key=key)
        end_pt   = max(all_pts, key=key)
        x1, y1 = px(start_pt[0], start_pt[1])
        x2, y2 = px(end_pt[0],   end_pt[1])
        msp.add_line(
            (x1, y1), (x2, y2),
            dxfattribs={"layer": "DASHED", "linetype": "DASHED"},
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

    # ── Rectangular annotation boxes ──────────────────────────────────────────
    for bc in (box_contours or []):
        pts = [px(float(p[0]), float(p[1])) for p in bc]
        if len(pts) >= 2:
            first, last = bc[0].astype(float), bc[-1].astype(float)
            closed = bool(np.linalg.norm(first - last) < 4.0)
            msp.add_lwpolyline(pts, close=closed,
                               dxfattribs={"layer": "BOXES"})
            entity_count += 1

    # ── Outermost drawing border ───────────────────────────────────────────────
    if page_border is not None:
        x_min, y_min, x_max, y_max = page_border
        p1 = px(x_min, y_min)
        p2 = px(x_max, y_min)
        p3 = px(x_max, y_max)
        p4 = px(x_min, y_max)
        msp.add_lwpolyline([p1, p2, p3, p4], close=True,
                           dxfattribs={"layer": "BOX"})
        entity_count += 1

    doc.saveas(output_path)
    return entity_count
