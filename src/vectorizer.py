"""Contour-first vectorisation pipeline.

Design rationale
----------------
The primary goal is "faithfully reproduce the visible ink on the drawing as
clean CAD polylines", not to extract centre-lines.  Skeleton / thinning
approaches lose stroke width context and are kept only as a hidden fallback.

Pipeline
--------
1. Build a 1-px edge image:
   - Canny on *grayscale* so subtle colour/tone boundaries are preserved
     even when binarisation merges them.
   - Thin (skeletonise) the Canny output so a thick stroke produces ONE
     edge path instead of two parallel edges.
   - Suppress text-candidate regions (optional).
2. findContours on the 1-px edge image → primary geometry (all shapes,
   open/closed, curved, complex).
3. Filter contours by arc length and bounding-box area.
4. Simplify each contour with Douglas-Peucker (adaptive ε).
5. Closed / straight classification per contour.
6. Supplemental HoughLinesP only for long straight features that
   findContours did not cover.
7. Merge collinear Hough fragments → snap near endpoints.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize as _ski_skel
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False


# ── Thin edge images to 1 px ──────────────────────────────────────────────────

def _zhang_suen_thin(img: np.ndarray) -> np.ndarray:
    """Zhang-Suen parallel thinning algorithm (1984).

    Two-pass iterative removal of border pixels while preserving 8-connectivity
    and line endpoints.  Each pass applies the deletion conditions:

    Common to both passes:
      (1) 2 ≤ B(P1) ≤ 6   (non-isolated, non-fully-surrounded)
      (2) A(P1) = 1        (exactly one 0→1 transition in the 3×3 ring)

    Pass 1: P2·P4·P6 = 0  AND  P4·P6·P8 = 0
    Pass 2: P2·P4·P8 = 0  AND  P2·P6·P8 = 0

    Returns a 1-pixel-wide binary image (foreground = 255).
    """
    binary = (img > 0).astype(np.uint8)
    while True:
        changed = False
        for odd_pass in (True, False):
            h, w = binary.shape
            # Pad for neighbour lookup
            p = np.pad(binary, 1, constant_values=0)
            P2 = p[0:h,   1:w+1]
            P3 = p[0:h,   2:w+2]
            P4 = p[1:h+1, 2:w+2]
            P5 = p[2:h+2, 2:w+2]
            P6 = p[2:h+2, 1:w+1]
            P7 = p[2:h+2, 0:w]
            P8 = p[1:h+1, 0:w]
            P9 = p[0:h,   0:w]

            # B(P1): count of foreground neighbours
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
            # A(P1): 0→1 transitions in ring P2,P3,...,P9,P2
            ring = np.stack([P2, P3, P4, P5, P6, P7, P8, P9], axis=2)
            shifted = np.roll(ring, -1, axis=2)
            A = ((ring == 0) & (shifted == 1)).sum(axis=2)

            cond12 = (binary == 1) & (B >= 2) & (B <= 6) & (A == 1)
            if odd_pass:
                cond34 = (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                cond34 = (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            delete = cond12 & cond34
            if delete.any():
                binary[delete] = 0
                changed = True
        if not changed:
            break
    return binary * 255


def _thin(img: np.ndarray) -> np.ndarray:
    """Return a 1-pixel-wide version of a binary image (foreground = 255)."""
    if _SKIMAGE:
        return (_ski_skel(img > 0).astype(np.uint8)) * 255
    return _zhang_suen_thin(img)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _perp_dist(pt: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    d = p2 - p1
    n = np.linalg.norm(d)
    if n < 1e-9:
        return float(np.linalg.norm(pt - p1))
    return float(abs(d[0] * (pt[1] - p1[1]) - d[1] * (pt[0] - p1[0])) / n)


def _is_straight(pts: np.ndarray, max_dev: float) -> bool:
    if len(pts) <= 2:
        return True
    p1, p2 = pts[0].astype(float), pts[-1].astype(float)
    return max(_perp_dist(p.astype(float), p1, p2) for p in pts) <= max_dev


def _adaptive_epsilon(contour_cv2: np.ndarray, image_diag: float,
                      base_fraction: float = 0.003) -> float:
    """DP ε scaled to the larger of the image diagonal and contour bbox diagonal."""
    x, y, w, h = cv2.boundingRect(contour_cv2)
    bbox_diag = max(float(np.hypot(w, h)), 1.0)
    return max(1.5, image_diag * base_fraction, bbox_diag * 0.01)


def _is_closed(contour_cv2: np.ndarray, tol_px: float = 4.0) -> bool:
    """True when the first and last points are within tol_px of each other."""
    pts = contour_cv2.reshape(-1, 2).astype(float)
    if len(pts) < 3:
        return False
    return float(np.linalg.norm(pts[0] - pts[-1])) <= tol_px


# ── Contour filtering and simplification ─────────────────────────────────────

def _contour_arc_length(contour_cv2: np.ndarray) -> float:
    return float(cv2.arcLength(contour_cv2, closed=False))


def _filter_contours(
    contours: list[np.ndarray],
    min_length: float,
    min_area: float,
) -> list[np.ndarray]:
    """Remove contours that are too short or span a negligibly small area."""
    kept = []
    for c in contours:
        if _contour_arc_length(c) < min_length:
            continue
        _, _, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        kept.append(c)
    return kept


def _simplify(contour_cv2: np.ndarray, epsilon: float) -> np.ndarray:
    """Douglas-Peucker simplification; returns (N,2) int array."""
    approx = cv2.approxPolyDP(contour_cv2, epsilon, closed=False)
    sq = approx.squeeze()
    if sq.ndim == 1:
        sq = sq.reshape(1, 2)
    return sq.astype(np.int32)


# ── Right-angle corner enhancement (imagetracerjs §internodes) ───────────────

def _snap_right_angles(pts: np.ndarray, tol_deg: float = 10.0) -> np.ndarray:
    """Snap near-90° corners to exact right angles.

    Implements the imagetracerjs right-angle enhancement heuristic: for each
    interior vertex B in polyline A–B–C, if the turn angle at B is within
    *tol_deg* of 90°, snap the outgoing segment direction to be exactly
    perpendicular to the incoming segment direction while preserving the
    outgoing segment length.

    This produces clean axis-aligned corners in architectural drawings without
    modifying vertices that are not near-orthogonal turns.

    Args:
        pts:     (N, 2) integer vertex array from DP simplification.
        tol_deg: Tolerance in degrees around 90°.  A corner qualifies when
                 |angle − 90°| ≤ tol_deg.  Default 10°.

    Returns:
        (N, 2) integer vertex array with snapped corners.
    """
    if len(pts) < 3:
        return pts
    result = pts.astype(float).copy()
    # cos_tol: maximum |cos(angle)| to qualify as near-90°
    cos_tol = math.cos(math.radians(90.0 - tol_deg))
    for i in range(1, len(result) - 1):
        a, b, c = result[i - 1], result[i], result[i + 1]
        v1 = b - a          # incoming direction (A→B)
        v2 = c - b          # outgoing direction (B→C)
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        if abs(cos_a) > cos_tol:
            continue
        # Near-90°: snap C so that B→C is exactly perpendicular to A→B.
        u = v1 / n1                               # unit vector along A→B
        perp = np.array([-u[1], u[0]])            # 90° CCW rotation
        if (v2[0] * perp[0] + v2[1] * perp[1]) < 0:
            perp = -perp                          # choose closest 90° direction
        result[i + 1] = b + perp * n2
    return np.round(result).astype(np.int32)


# ── Circle / arc fitting (DXF Section 5: native ARC/CIRCLE + bulge) ────────────

def _fit_circle(pts: np.ndarray) -> tuple[float, float, float, float]:
    """Algebraic (Kåsa) least-squares circle fit.

    Returns (cx, cy, r, rms_residual).  rms_residual is the root-mean-square
    distance of the points from the fitted circle, in pixels.
    """
    x = pts[:, 0].astype(float)
    y = pts[:, 1].astype(float)
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(x))])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        return cx, cy, 0.0, float("inf")
    r = float(np.sqrt(r2))
    resid = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return float(cx), float(cy), r, resid


def _angular_span(pts: np.ndarray, cx: float, cy: float) -> float:
    """Total angular coverage (radians, 0..2π) of points around a centre."""
    ang = np.sort(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    if len(ang) < 2:
        return 0.0
    gaps = np.diff(ang)
    wrap = (ang[0] + 2 * np.pi) - ang[-1]
    largest_gap = max(float(gaps.max()), float(wrap))
    return 2 * np.pi - largest_gap


def _try_fit_arc(
    contour_pts: np.ndarray,
    closed: bool,
    arc_tol: float,
    min_radius: float = 3.0,
    max_radius_factor: float = 5.0,
    image_diag: float = 1e9,
) -> Optional[dict]:
    """Attempt to model a contour as a circle or circular arc.

    Returns a dict describing the primitive, or None if the points do not fit
    a circle within ``arc_tol`` RMS pixels.

    dict forms:
      {"type": "circle", "center": (cx, cy), "r": r}
      {"type": "arc",    "start": (x,y), "mid": (x,y), "end": (x,y)}

    The arc form carries three pixel points; the DXF exporter derives the
    bulge ``b = tan(theta/4)`` from them after the Y-flip, so arc orientation
    survives the coordinate transform.
    """
    pts = contour_pts.reshape(-1, 2).astype(float)
    if len(pts) < 5:
        return None
    cx, cy, r, resid = _fit_circle(pts)
    if not np.isfinite(resid) or resid > arc_tol:
        return None
    if r < min_radius or r > image_diag * max_radius_factor:
        return None

    span = _angular_span(pts, cx, cy)
    # Full circle: angular coverage near the whole 2π (a traced loop's
    # endpoints rarely coincide exactly, so rely on span, not the closed flag).
    if span >= np.deg2rad(300):
        return {"type": "circle", "center": (cx, cy), "r": r}
    # Partial arc: need a meaningful sweep, else a near-straight chord.
    if span < np.deg2rad(20):
        return None

    # Endpoints must be the two ends of the *angular* sweep, NOT pts[0]/pts[-1]:
    # a contour traced from a thinned 2-px stroke runs out along the arc and
    # loops back, so its first and last points nearly coincide.  Order points by
    # angle about the centre, find the largest angular gap (the uncovered side),
    # and rotate so the covered sweep is contiguous; its ends are the endpoints
    # and its middle element is the mid point.
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    order = np.argsort(ang)
    a_s = ang[order]
    gaps = np.append(np.diff(a_s), (a_s[0] + 2 * np.pi) - a_s[-1])
    g = int(np.argmax(gaps))
    rot = np.concatenate([order[g + 1:], order[: g + 1]])
    start = pts[rot[0]]
    end = pts[rot[-1]]
    mid = pts[rot[len(rot) // 2]]
    # Guard: if the resolved endpoints are still nearly coincident, the shape is
    # not a clean arc — let it fall back to a polyline.
    if float(np.hypot(start[0] - end[0], start[1] - end[1])) < 3.0:
        return None
    return {
        "type": "arc",
        "start": (float(start[0]), float(start[1])),
        "mid": (float(mid[0]), float(mid[1])),
        "end": (float(end[0]), float(end[1])),
    }


# ── Collinear Hough merge ─────────────────────────────────────────────────────

def _segs_mergeable(a, b, angle_tol_deg, perp_tol, gap_tol) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    da = np.array([ax2 - ax1, ay2 - ay1], dtype=float)
    db = np.array([bx2 - bx1, by2 - by1], dtype=float)
    na, nb = np.linalg.norm(da), np.linalg.norm(db)
    if na < 1e-9 or nb < 1e-9:
        return False
    ang_a = np.arctan2(da[1], da[0]) % np.pi
    ang_b = np.arctan2(db[1], db[0]) % np.pi
    if min(abs(ang_a - ang_b), np.pi - abs(ang_a - ang_b)) > np.deg2rad(angle_tol_deg):
        return False
    p1, p2 = np.array([ax1, ay1], float), np.array([ax2, ay2], float)
    if max(_perp_dist(np.array([bx1, by1], float), p1, p2),
           _perp_dist(np.array([bx2, by2], float), p1, p2)) > perp_tol:
        return False
    u = da / na
    a_lo, a_hi = sorted([p1 @ u, p2 @ u])
    b_lo, b_hi = sorted([np.array([bx1, by1], float) @ u,
                          np.array([bx2, by2], float) @ u])
    return max(a_lo - b_hi, b_lo - a_hi, 0.0) <= gap_tol


def _merge_collinear_lines(
    lines: list,
    angle_tol_deg: float = 2.0,
    perp_tol: float = 2.0,
    gap_tol: float = 20.0,
) -> list:
    """Keep for backward-compat; also used internally for Hough results."""
    if not lines:
        return lines
    segs = [tuple(float(v) for v in s) for s in lines]
    changed = True
    while changed:
        changed = False
        used = [False] * len(segs)
        result = []
        for i in range(len(segs)):
            if used[i]:
                continue
            ax1, ay1, ax2, ay2 = segs[i]
            for j in range(i + 1, len(segs)):
                if used[j]:
                    continue
                if _segs_mergeable((ax1, ay1, ax2, ay2), segs[j],
                                   angle_tol_deg, perp_tol, gap_tol):
                    d = np.array([ax2 - ax1, ay2 - ay1], float)
                    nn = np.linalg.norm(d)
                    if nn < 1e-9:
                        continue
                    u = d / nn
                    pts = np.array([[ax1, ay1], [ax2, ay2],
                                    [segs[j][0], segs[j][1]],
                                    [segs[j][2], segs[j][3]]], float)
                    proj = pts @ u
                    lo, hi = pts[proj.argmin()], pts[proj.argmax()]
                    ax1, ay1, ax2, ay2 = lo[0], lo[1], hi[0], hi[1]
                    used[j] = True
                    changed = True
            result.append((round(ax1), round(ay1), round(ax2), round(ay2)))
            used[i] = True
        segs = [tuple(float(v) for v in s) for s in result]
    return [tuple(int(v) for v in s) for s in segs]


def _snap_endpoints(lines: list, radius: float = 4.0) -> list:
    """Snap near-touching line endpoints to a shared centroid."""
    if not lines:
        return lines
    pts_list, tags = [], []
    for i, (x1, y1, x2, y2) in enumerate(lines):
        pts_list += [[float(x1), float(y1)], [float(x2), float(y2)]]
        tags += [(i, 0), (i, 1)]
    arr = np.array(pts_list)
    parent = list(range(len(arr)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if np.linalg.norm(arr[i] - arr[j]) <= radius:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    from collections import defaultdict
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(len(arr)):
        clusters[find(i)].append(i)
    for members in clusters.values():
        if len(members) > 1:
            arr[members] = arr[members].mean(axis=0)

    result = list(lines)
    for idx, ((li, which), pt) in enumerate(zip(tags, arr)):
        x1, y1, x2, y2 = result[li]
        if which == 0:
            result[li] = (int(round(pt[0])), int(round(pt[1])), x2, y2)
        else:
            result[li] = (x1, y1, int(round(pt[0])), int(round(pt[1])))
    return result


# ── Main extraction pipeline ──────────────────────────────────────────────────

def extract_lines_and_contours(
    binary: np.ndarray,
    gray: Optional[np.ndarray] = None,
    text_mask: Optional[np.ndarray] = None,
    # Contour quality
    min_contour_length: float = 15.0,
    min_contour_area: float = 10.0,
    approx_epsilon: Optional[float] = None,
    max_line_deviation: float = 2.0,
    # Hough supplement
    min_line_length: int = 80,
    max_gap: int = 15,
    hough_threshold: int = 30,
    use_hough: bool = True,
    # Canny
    canny_low: int = 50,
    canny_high: int = 150,
    # Arc / circle detection (DXF Section 5)
    detect_arcs: bool = True,
    arc_tol: Optional[float] = None,
    return_arcs: bool = False,
    # Legacy / advanced
    mode: str = "edge",
    merge_lines: bool = True,
    snap_radius: float = 4.0,
    pre_close_kernel: int = 0,
    # Right-angle corner enhancement (imagetracerjs §internodes)
    right_angle_enhance: bool = False,
    right_angle_tol: float = 10.0,
):
    """Extract LINE segments and LWPOLYLINE contours from a drawing image.

    Contour extraction is the primary path (all visible geometry).
    HoughLinesP is supplemental and only picks up long straight features
    that remain after contour detection.

    Args:
        binary       : White-foreground binary image (strokes = 255).
        gray         : Original 8-bit grayscale.  When given, Canny runs on
                       this (captures subtle tone boundaries not in binary).
        text_mask    : Optional 255-regions to suppress before detection.
        min_contour_length: Arc-length threshold; shorter contours are noise.
        min_contour_area: Bounding-box area threshold for noise removal.
        approx_epsilon: DP simplification tolerance.  None = auto (0.3 % of
                        image diagonal, min 1.5 px).
        max_line_deviation: A simplified contour is classified as a straight
                        LINE when every point is within this many pixels of
                        the chord.
        min_line_length: Minimum length for supplemental Hough lines (px).
                        Set high because contours already cover short geometry.
        max_gap      : Max gap inside a Hough segment (px).
        hough_threshold: Accumulator threshold for HoughLinesP.
        use_hough    : Toggle supplemental Hough pass.
        canny_low / canny_high: Canny hysteresis thresholds.
        mode         : 'edge' (default).  'skeleton' is deprecated and treated
                       as 'edge' with a warning.
        merge_lines  : Merge collinear Hough fragments.
        snap_radius  : Endpoint snap distance (px).  0 = disabled.
        pre_close_kernel: Closing before edge detection (0 = off).

        detect_arcs  : Try to model curved contours as circles/arcs.
        arc_tol      : Max RMS pixel residual for a circle fit.  None = auto
                       (max(2.0, 0.5 %% of image diagonal)).
        return_arcs  : When True, return a 3-tuple (lines, contours, arcs) and
                       divert circle/arc-shaped contours into ``arcs``.  When
                       False (default), behaves exactly as the 2-tuple API and
                       does not divert any geometry.
        right_angle_enhance: After DP simplification, snap corners whose angle
                       is within right_angle_tol degrees of 90° to exact right
                       angles.  Improves output quality for architectural and
                       mechanical drawings with orthogonal geometry.
                       (imagetracerjs: rightangleenhance, default false here.)
        right_angle_tol: Tolerance in degrees around 90° for the enhancement.
                       Default 10°.

    Returns:
        (lines, contours)              when return_arcs is False
        (lines, contours, arcs)        when return_arcs is True
        lines    : list of (x1, y1, x2, y2) int tuples — straight segments.
        contours : list of np.ndarray shape (N, 2) — polyline vertex arrays.
        arcs     : list of dicts — {"type":"circle"|"arc", ...} primitives.
    """
    if mode == "skeleton":
        warnings.warn(
            "--mode skeleton is deprecated; using edge mode instead. "
            "Skeleton-based extraction is no longer recommended for drawing "
            "reproduction because it loses stroke-width and shape context.",
            DeprecationWarning, stacklevel=2,
        )

    h, w = binary.shape[:2]
    image_diag = float(np.hypot(h, w))
    eps = approx_epsilon if approx_epsilon is not None else max(1.5, image_diag * 0.003)

    # ── Step 1: Build 1-px edge image ────────────────────────────────────────
    # Canny runs on grayscale when available (preserves subtle tone boundaries),
    # otherwise on the binary.  pre_close_kernel must be applied to whichever
    # source actually feeds Canny, not to an unused copy.
    canny_src = gray if gray is not None else binary
    if pre_close_kernel > 0:
        kc = np.ones((pre_close_kernel, pre_close_kernel), np.uint8)
        canny_src = cv2.morphologyEx(canny_src, cv2.MORPH_CLOSE, kc, iterations=1)

    edges = cv2.Canny(canny_src, canny_low, canny_high, apertureSize=3)
    thin_edges = _thin(edges)

    if text_mask is not None:
        dm = cv2.dilate(text_mask, np.ones((5, 5), np.uint8), iterations=1)
        thin_edges = cv2.bitwise_and(thin_edges, cv2.bitwise_not(dm))

    # ── Step 2: findContours (primary geometry extraction) ───────────────────
    raw_contours, hierarchy = cv2.findContours(
        thin_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1
    )

    # ── Step 3: Filter ───────────────────────────────────────────────────────
    filtered = _filter_contours(raw_contours, min_contour_length, min_contour_area)

    # ── Step 4-5: Simplify and classify each contour ─────────────────────────
    lines: list = []
    contours: list = []
    arcs: list = []
    contour_mask = np.zeros_like(thin_edges)
    arc_tolerance = arc_tol if arc_tol is not None else max(2.0, image_diag * 0.005)

    for c in filtered:
        pts = _simplify(c, eps)
        if len(pts) < 2:
            continue

        if right_angle_enhance and len(pts) >= 3:
            pts = _snap_right_angles(pts, tol_deg=right_angle_tol)

        closed = _is_closed(c, tol_px=max(4.0, eps * 2))

        if _is_straight(pts, max_line_deviation):
            # Entire simplified contour is straight → LINE entity
            lines.append((int(pts[0, 0]), int(pts[0, 1]),
                          int(pts[-1, 0]), int(pts[-1, 1])))
        elif return_arcs and detect_arcs and (
            arc := _try_fit_arc(c, closed, arc_tolerance, image_diag=image_diag)
        ):
            # Curved contour that fits a circle/arc → compact CAD primitive
            arcs.append(arc)
        else:
            contours.append(pts)

        # Paint onto mask so Hough skips covered regions
        cv2.polylines(contour_mask,
                      [pts.reshape(-1, 1, 2)],
                      isClosed=closed, color=255, thickness=3)

    # ── Step 6: Supplemental Hough (long straight features only) ─────────────
    if use_hough:
        # Subtract contour-covered regions from the edge image
        cover = cv2.dilate(contour_mask, np.ones((7, 7), np.uint8), iterations=1)
        residual = cv2.bitwise_and(thin_edges, cv2.bitwise_not(cover))

        hough_result = cv2.HoughLinesP(
            residual, rho=1, theta=np.pi / 180,
            threshold=hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_gap,
        )
        hough_lines: list = []
        if hough_result is not None:
            for seg in hough_result:
                x1, y1, x2, y2 = seg[0]
                hough_lines.append((int(x1), int(y1), int(x2), int(y2)))

        if merge_lines and hough_lines:
            hough_lines = _merge_collinear_lines(hough_lines)
        lines.extend(hough_lines)

    # ── Step 7: Snap near-touching line endpoints ────────────────────────────
    if snap_radius > 0 and lines:
        lines = _snap_endpoints(lines, radius=snap_radius)

    if return_arcs:
        arcs = _dedup_circles(arcs)
        return lines, contours, arcs
    return lines, contours


def _dedup_circles(arcs: list, center_tol: float = 5.0, r_tol: float = 5.0,
                   endpoint_tol: float = 6.0) -> list:
    """Collapse near-identical primitives (the two edges of a thick stroke).

    Circles are matched by centre + radius; arcs by their start/end endpoints
    (in either orientation).  Duplicates are dropped, keeping the first.
    """
    kept: list = []
    for a in arcs:
        if a.get("type") == "circle":
            cx, cy = a["center"]
            r = a["r"]
            dup = False
            for b in kept:
                if b.get("type") != "circle":
                    continue
                bx, by = b["center"]
                if abs(cx - bx) <= center_tol and abs(cy - by) <= center_tol \
                        and abs(r - b["r"]) <= r_tol:
                    b["center"] = ((cx + bx) / 2.0, (cy + by) / 2.0)
                    b["r"] = (r + b["r"]) / 2.0
                    dup = True
                    break
            if not dup:
                kept.append(a)
        elif a.get("type") == "arc":
            s, e = np.array(a["start"]), np.array(a["end"])
            dup = False
            for b in kept:
                if b.get("type") != "arc":
                    continue
                bs, be = np.array(b["start"]), np.array(b["end"])
                same = (np.linalg.norm(s - bs) <= endpoint_tol
                        and np.linalg.norm(e - be) <= endpoint_tol)
                flipped = (np.linalg.norm(s - be) <= endpoint_tol
                           and np.linalg.norm(e - bs) <= endpoint_tol)
                if same or flipped:
                    dup = True
                    break
            if not dup:
                kept.append(a)
        else:
            kept.append(a)
    return kept
