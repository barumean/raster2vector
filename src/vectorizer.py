import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize as ski_skeletonize
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """Return a 1-pixel-wide skeleton of the foreground (nonzero pixels)."""
    if _SKIMAGE_AVAILABLE:
        skel = ski_skeletonize(binary > 0)
        return skel.astype(np.uint8) * 255
    # Fallback: iterative morphological thinning via OpenCV
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


def _perp_distance(pt: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Perpendicular distance from pt to the infinite line through p1, p2."""
    d = p2 - p1
    n = np.linalg.norm(d)
    if n < 1e-9:
        return float(np.linalg.norm(pt - p1))
    # 2D cross product (scalar); avoids deprecated np.cross on 2-vectors
    return float(abs(d[0] * (pt[1] - p1[1]) - d[1] * (pt[0] - p1[0])) / n)


def _line_is_straight(pts: np.ndarray, max_deviation: float = 2.0) -> bool:
    """True when every point lies within max_deviation px of the end-to-end chord."""
    if len(pts) <= 2:
        return True
    p1 = pts[0].astype(float)
    p2 = pts[-1].astype(float)
    fpts = pts.astype(float)
    worst = max(_perp_distance(p, p1, p2) for p in fpts)
    return worst <= max_deviation


def _segments_mergeable(a, b, angle_tol_deg, perp_tol, gap_tol) -> bool:
    """Two segments merge only if they are genuinely collinear and adjacent.

    Requires: similar angle, small perpendicular offset (so distinct parallel
    lines are NOT merged), and overlapping/adjacent projections along the axis.
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
    # Perpendicular offset: reject distinct parallel lines.
    if max(_perp_distance(np.array([bx1, by1], float), p1, p2),
           _perp_distance(np.array([bx2, by2], float), p1, p2)) > perp_tol:
        return False

    # Projection intervals along segment a's direction.
    u = da / na
    a_proj = np.array([p1 @ u, p2 @ u])
    b_proj = np.array([np.array([bx1, by1], float) @ u,
                       np.array([bx2, by2], float) @ u])
    a_lo, a_hi = a_proj.min(), a_proj.max()
    b_lo, b_hi = b_proj.min(), b_proj.max()
    gap = max(a_lo - b_hi, b_lo - a_hi, 0.0)
    return gap <= gap_tol


def _merge_collinear_lines(
    lines: list,
    angle_tol_deg: float = 2.0,
    perp_tol: float = 2.0,
    gap_tol: float = 10.0,
) -> list:
    """Merge collinear, adjacent Hough segments into longer single segments."""
    if not lines:
        return lines

    merged = [tuple(float(v) for v in seg) for seg in lines]
    changed = True
    while changed:
        changed = False
        used = [False] * len(merged)
        result = []
        for i in range(len(merged)):
            if used[i]:
                continue
            ax1, ay1, ax2, ay2 = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if _segments_mergeable((ax1, ay1, ax2, ay2), merged[j],
                                       angle_tol_deg, perp_tol, gap_tol):
                    # Extend by projecting all four endpoints onto the axis.
                    d = np.array([ax2 - ax1, ay2 - ay1], dtype=float)
                    n = np.linalg.norm(d)
                    if n < 1e-9:
                        continue
                    u = d / n
                    pts = np.array([
                        [ax1, ay1], [ax2, ay2],
                        [merged[j][0], merged[j][1]], [merged[j][2], merged[j][3]],
                    ], dtype=float)
                    proj = pts @ u
                    lo, hi = pts[proj.argmin()], pts[proj.argmax()]
                    ax1, ay1, ax2, ay2 = lo[0], lo[1], hi[0], hi[1]
                    used[j] = True
                    changed = True
            result.append((round(ax1), round(ay1), round(ax2), round(ay2)))
            used[i] = True
        merged = [tuple(float(v) for v in seg) for seg in result]

    return [tuple(int(v) for v in seg) for seg in merged]


def extract_lines_and_contours(
    binary: np.ndarray,
    min_line_length: int = 50,
    max_gap: int = 10,
    hough_threshold: int = 50,
    canny_low: int = 50,
    canny_high: int = 150,
    approx_epsilon: float = 1.5,
    mode: str = "edge",
    merge_lines: bool = True,
) -> tuple[list, list]:
    """Extract straight lines and contour polylines from a binary image.

    The input binary is expected to have foreground (strokes) = 255, as
    produced by load_and_preprocess (which normalises polarity).

    Returns:
        (hough_lines, contours)
        hough_lines: list of (x1, y1, x2, y2).
        contours: list of np.ndarray shape (N, 2).
    """
    if mode == "skeleton":
        source = _skeletonize(binary)
    else:
        source = cv2.Canny(binary, canny_low, canny_high, apertureSize=3)

    hough_result = cv2.HoughLinesP(
        source, rho=1, theta=np.pi / 180, threshold=hough_threshold,
        minLineLength=min_line_length, maxLineGap=max_gap,
    )

    lines = []
    mask = np.zeros_like(source)
    if hough_result is not None:
        for seg in hough_result:
            x1, y1, x2, y2 = seg[0]
            lines.append((int(x1), int(y1), int(x2), int(y2)))
            cv2.line(mask, (x1, y1), (x2, y2), 255, 3)

    if merge_lines:
        lines = _merge_collinear_lines(lines)

    dilated_mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    remaining = cv2.bitwise_and(source, cv2.bitwise_not(dilated_mask))

    raw_contours, _ = cv2.findContours(remaining, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = []
    for c in raw_contours:
        approx = cv2.approxPolyDP(c, approx_epsilon, closed=False)
        squeezed = approx.squeeze()
        if squeezed.ndim != 2 or len(squeezed) < 2:
            continue
        if _line_is_straight(squeezed):
            x1, y1 = squeezed[0]
            x2, y2 = squeezed[-1]
            lines.append((int(x1), int(y1), int(x2), int(y2)))
        else:
            contours.append(squeezed)

    return lines, contours
