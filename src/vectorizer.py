import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize as ski_skeletonize
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False


# ── Skeletonisation ───────────────────────────────────────────────────────────

def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """Return a 1-pixel-wide skeleton of the foreground (nonzero pixels)."""
    if _SKIMAGE_AVAILABLE:
        skel = ski_skeletonize(binary > 0)
        return skel.astype(np.uint8) * 255
    # Fallback: iterative morphological thinning
    skel = np.zeros_like(binary)
    tmp = binary.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(tmp, kernel)
        dilated = cv2.dilate(eroded, kernel)
        diff = cv2.subtract(tmp, dilated)
        skel = cv2.bitwise_or(skel, diff)
        tmp = eroded.copy()
        if cv2.countNonZero(tmp) == 0:
            break
    return skel


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _perp_distance(pt: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Perpendicular distance from pt to the infinite line through p1–p2."""
    d = p2 - p1
    n = np.linalg.norm(d)
    if n < 1e-9:
        return float(np.linalg.norm(pt - p1))
    return float(abs(d[0] * (pt[1] - p1[1]) - d[1] * (pt[0] - p1[0])) / n)


def _line_is_straight(pts: np.ndarray, max_deviation: float = 2.0) -> bool:
    """True when every point lies within max_deviation px of the chord."""
    if len(pts) <= 2:
        return True
    p1, p2 = pts[0].astype(float), pts[-1].astype(float)
    worst = max(_perp_distance(p.astype(float), p1, p2) for p in pts)
    return worst <= max_deviation


# ── Collinear segment merging ─────────────────────────────────────────────────

def _segments_mergeable(a, b, angle_tol_deg, perp_tol, gap_tol) -> bool:
    """True when a and b are collinear, nearly-parallel, and adjacent/overlapping.

    Parallel lines that are separated perpendicularly (e.g. double wall lines)
    are explicitly rejected via the perp_tol check.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    da = np.array([ax2 - ax1, ay2 - ay1], dtype=float)
    db = np.array([bx2 - bx1, by2 - by1], dtype=float)
    na, nb = np.linalg.norm(da), np.linalg.norm(db)
    if na < 1e-9 or nb < 1e-9:
        return False

    ang_a = np.arctan2(da[1], da[0]) % np.pi
    ang_b = np.arctan2(db[1], db[0]) % np.pi
    angle_diff = min(abs(ang_a - ang_b), np.pi - abs(ang_a - ang_b))
    if angle_diff > np.deg2rad(angle_tol_deg):
        return False

    p1 = np.array([ax1, ay1], dtype=float)
    p2 = np.array([ax2, ay2], dtype=float)
    perp_b1 = _perp_distance(np.array([bx1, by1], float), p1, p2)
    perp_b2 = _perp_distance(np.array([bx2, by2], float), p1, p2)
    if max(perp_b1, perp_b2) > perp_tol:
        return False

    u = da / na
    a_lo = min(p1 @ u, p2 @ u)
    a_hi = max(p1 @ u, p2 @ u)
    b_lo = min(np.array([bx1, by1], float) @ u, np.array([bx2, by2], float) @ u)
    b_hi = max(np.array([bx1, by1], float) @ u, np.array([bx2, by2], float) @ u)
    gap = max(a_lo - b_hi, b_lo - a_hi, 0.0)
    return gap <= gap_tol


def _merge_collinear_lines(
    lines: list,
    angle_tol_deg: float = 2.0,
    perp_tol: float = 2.0,
    gap_tol: float = 20.0,
) -> list:
    """Merge collinear, adjacent/overlapping Hough segments into longer ones."""
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
                if _segments_mergeable((ax1, ay1, ax2, ay2), segs[j],
                                       angle_tol_deg, perp_tol, gap_tol):
                    d = np.array([ax2 - ax1, ay2 - ay1], dtype=float)
                    n = np.linalg.norm(d)
                    if n < 1e-9:
                        continue
                    u = d / n
                    pts = np.array([
                        [ax1, ay1], [ax2, ay2],
                        [segs[j][0], segs[j][1]], [segs[j][2], segs[j][3]],
                    ], dtype=float)
                    proj = pts @ u
                    lo, hi = pts[proj.argmin()], pts[proj.argmax()]
                    ax1, ay1, ax2, ay2 = lo[0], lo[1], hi[0], hi[1]
                    used[j] = True
                    changed = True
            result.append((round(ax1), round(ay1), round(ax2), round(ay2)))
            used[i] = True
        segs = [tuple(float(v) for v in s) for s in result]

    return [tuple(int(v) for v in s) for s in segs]


# ── Endpoint snapping ─────────────────────────────────────────────────────────

def _snap_endpoints(lines: list, radius: float = 4.0) -> list:
    """Snap near-touching line endpoints to a shared midpoint.

    When two endpoint are within `radius` pixels of each other they are
    both moved to their midpoint.  This closes small gaps that survive
    the merge step (e.g. lines that are not collinear but nearly meet at
    a corner), keeping the DXF topology clean.
    """
    if not lines:
        return lines

    # Collect all endpoints with their (line_index, which_end) tag
    pts = []   # list of [x, y]  — mutable
    tags = []  # (line_idx, 0=start|1=end)
    for i, (x1, y1, x2, y2) in enumerate(lines):
        pts.append([float(x1), float(y1)])
        tags.append((i, 0))
        pts.append([float(x2), float(y2)])
        tags.append((i, 1))

    pts_arr = np.array(pts)
    n = len(pts_arr)

    # Union-find: group endpoints that are within radius of each other
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pts_arr[i] - pts_arr[j]) <= radius:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    # For each cluster compute the centroid and replace all members
    from collections import defaultdict
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    for members in clusters.values():
        if len(members) < 2:
            continue
        centroid = pts_arr[members].mean(axis=0)
        for m in members:
            pts_arr[m] = centroid

    # Rebuild lines
    result = list(lines)
    for idx, ((li, which), pt) in enumerate(zip(tags, pts_arr)):
        x1, y1, x2, y2 = result[li]
        if which == 0:
            result[li] = (int(round(pt[0])), int(round(pt[1])), x2, y2)
        else:
            result[li] = (x1, y1, int(round(pt[0])), int(round(pt[1])))
    return result


# ── Main extraction pipeline ──────────────────────────────────────────────────

def extract_lines_and_contours(
    binary: np.ndarray,
    min_line_length: int = 30,
    max_gap: int = 20,
    hough_threshold: int = 30,
    canny_low: int = 50,
    canny_high: int = 150,
    approx_epsilon: float = 1.5,
    mode: str = "skeleton",
    merge_lines: bool = True,
    snap_radius: float = 4.0,
    pre_close_kernel: int = 0,
) -> tuple[list, list]:
    """Extract straight lines and contour polylines from a binary image.

    Pipeline
    --------
    1. Optional pre-close: dilate+erode to fill gaps in thick strokes so that
       a solid thick line skeletonises to a single centre-line (not two edges).
    2. Source image: skeleton (1-px centre-lines) or Canny edges.
    3. HoughLinesP on the 1-px source → straight-line segments.
    4. Merge collinear fragments → longer lines.
    5. Snap near-touching endpoints → clean topology.
    6. findContours on remaining pixels → curved / complex polylines.

    The input binary must have foreground (strokes) = 255, as produced by
    load_and_preprocess (which normalises polarity automatically).

    Args:
        binary: White-foreground binary image, uint8.
        min_line_length: Minimum Hough segment length (pixels).
        max_gap: Max gap (pixels) bridged inside a single Hough segment.
        hough_threshold: Accumulator threshold for HoughLinesP.
        canny_low / canny_high: Hysteresis thresholds (edge mode only).
        approx_epsilon: Douglas-Peucker tolerance for polyline vertices.
        mode: 'skeleton' (default, drawing-optimised) or 'edge' (Canny).
        merge_lines: Merge collinear adjacent fragments.
        snap_radius: Max distance (px) between endpoints that get snapped.
        pre_close_kernel: If > 0, apply a closing of this kernel size before
            skeletonising; helps join the two Canny edges of a thick stroke
            into a solid filled region that skeletonises to one centre-line.

    Returns:
        (hough_lines, contours)
        hough_lines : list of (x1, y1, x2, y2) int tuples.
        contours    : list of np.ndarray shape (N, 2).
    """
    work = binary.copy()

    # Step 1: Pre-close — fill interior of thick strokes so skeleton
    # produces ONE centre-line instead of two edge-lines.
    if pre_close_kernel > 0:
        k = np.ones((pre_close_kernel, pre_close_kernel), np.uint8)
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, k, iterations=1)

    # Step 2: Build 1-px source image.
    if mode == "skeleton":
        source = _skeletonize(work)
    else:
        # edge mode: also thin the Canny output so thick strokes don't
        # produce double edges entering Hough.
        edges = cv2.Canny(work, canny_low, canny_high, apertureSize=3)
        source = _skeletonize(edges)  # thin even Canny output

    # Step 3: Hough line detection.
    hough_result = cv2.HoughLinesP(
        source, rho=1, theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_gap,
    )

    lines: list = []
    mask = np.zeros_like(source)
    if hough_result is not None:
        for seg in hough_result:
            x1, y1, x2, y2 = seg[0]
            lines.append((int(x1), int(y1), int(x2), int(y2)))
            cv2.line(mask, (x1, y1), (x2, y2), 255, 3)

    # Step 4: Merge collinear fragments.
    if merge_lines and lines:
        lines = _merge_collinear_lines(lines)

    # Step 5: Snap near-touching endpoints.
    if snap_radius > 0 and lines:
        lines = _snap_endpoints(lines, radius=snap_radius)

    # Step 6: Remaining pixels → contours / curved polylines.
    dilated_mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    remaining = cv2.bitwise_and(source, cv2.bitwise_not(dilated_mask))

    raw_contours, _ = cv2.findContours(
        remaining, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    contours: list = []
    for c in raw_contours:
        approx = cv2.approxPolyDP(c, approx_epsilon, closed=False)
        squeezed = approx.squeeze()
        if squeezed.ndim != 2 or len(squeezed) < 2:
            continue
        if _line_is_straight(squeezed):
            x1, y1 = int(squeezed[0, 0]), int(squeezed[0, 1])
            x2, y2 = int(squeezed[-1, 0]), int(squeezed[-1, 1])
            lines.append((x1, y1, x2, y2))
        else:
            contours.append(squeezed)

    return lines, contours
