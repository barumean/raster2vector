import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize as ski_skeletonize
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """Return a 1-pixel-wide skeleton of the foreground."""
    if _SKIMAGE_AVAILABLE:
        bool_img = binary > 0
        skel = ski_skeletonize(bool_img)
        return (skel.astype(np.uint8)) * 255
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


def _line_is_straight(pts: np.ndarray, max_deviation: float = 2.0) -> bool:
    """Return True when all points lie within max_deviation px of the chord."""
    if len(pts) <= 2:
        return True
    p1, p2 = pts[0].astype(float), pts[-1].astype(float)
    d = p2 - p1
    norm = np.linalg.norm(d)
    if norm < 1e-6:
        return True
    # Perpendicular distance from each interior point to the line
    perp = np.abs(np.cross(d, pts.astype(float) - p1)) / norm
    return float(perp.max()) <= max_deviation


def _merge_collinear_lines(
    lines: list[tuple],
    angle_tol_deg: float = 2.0,
    dist_tol: float = 5.0,
) -> list[tuple]:
    """Merge nearby, nearly-parallel Hough line segments into longer ones."""
    if not lines:
        return lines

    def seg_angle(x1, y1, x2, y2):
        return np.arctan2(y2 - y1, x2 - x1) % np.pi

    def seg_midpoint(x1, y1, x2, y2):
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    angle_tol = np.deg2rad(angle_tol_deg)
    merged = list(lines)
    changed = True
    while changed:
        changed = False
        used = [False] * len(merged)
        result = []
        for i in range(len(merged)):
            if used[i]:
                continue
            x1, y1, x2, y2 = merged[i]
            ai = seg_angle(x1, y1, x2, y2)
            mi = seg_midpoint(x1, y1, x2, y2)
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                ax1, ay1, ax2, ay2 = merged[j]
                aj = seg_angle(ax1, ay1, ax2, ay2)
                mj = seg_midpoint(ax1, ay1, ax2, ay2)
                angle_diff = min(abs(ai - aj), np.pi - abs(ai - aj))
                centre_dist = np.hypot(mi[0] - mj[0], mi[1] - mj[1])
                if angle_diff <= angle_tol and centre_dist <= dist_tol:
                    # Merge: take extreme endpoints along the shared direction
                    pts = np.array([[x1, y1], [x2, y2], [ax1, ay1], [ax2, ay2]])
                    dx, dy = np.cos(ai), np.sin(ai)
                    proj = pts @ np.array([dx, dy])
                    idx_min, idx_max = proj.argmin(), proj.argmax()
                    x1, y1 = pts[idx_min]
                    x2, y2 = pts[idx_max]
                    used[j] = True
                    changed = True
            result.append((int(x1), int(y1), int(x2), int(y2)))
            used[i] = True
        merged = result
    return merged


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
    """Extract straight lines and contours/polylines from a binary image.

    Args:
        binary: Binary uint8 image (white foreground on black).
        min_line_length: Minimum Hough line segment length (pixels).
        max_gap: Maximum gap to bridge between collinear segments.
        hough_threshold: Accumulator threshold for HoughLinesP.
        canny_low: Lower hysteresis threshold for Canny.
        canny_high: Upper hysteresis threshold for Canny.
        approx_epsilon: Douglas-Peucker tolerance for polyline simplification.
        mode: 'edge' (Canny-based) or 'skeleton' (centre-line thinning).
        merge_lines: Merge nearby collinear Hough segments after detection.

    Returns:
        (hough_lines, contours)
        hough_lines: list of (x1, y1, x2, y2).
        contours: list of np.ndarray shape (N, 2).
    """
    if mode == "skeleton":
        source = _skeletonize(binary)
    else:
        source = cv2.Canny(binary, canny_low, canny_high, apertureSize=3)

    # ── Hough line detection ──────────────────────────────────────────────────
    hough_result = cv2.HoughLinesP(
        source,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_gap,
    )

    lines = []
    mask = np.zeros_like(source)

    if hough_result is not None:
        for seg in hough_result:
            x1, y1, x2, y2 = seg[0]
            lines.append((x1, y1, x2, y2))
            cv2.line(mask, (x1, y1), (x2, y2), 255, 3)

    if merge_lines:
        lines = _merge_collinear_lines(lines)

    # ── Contour extraction on remaining edges ─────────────────────────────────
    dilated_mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    remaining = cv2.bitwise_and(source, cv2.bitwise_not(dilated_mask))

    raw_contours, _ = cv2.findContours(remaining, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    contours = []
    for c in raw_contours:
        approx = cv2.approxPolyDP(c, approx_epsilon, closed=False)
        squeezed = approx.squeeze()
        if squeezed.ndim != 2 or len(squeezed) < 2:
            continue
        # Demote nearly-straight short contours to lines
        if _line_is_straight(squeezed):
            x1, y1 = squeezed[0]
            x2, y2 = squeezed[-1]
            lines.append((int(x1), int(y1), int(x2), int(y2)))
        else:
            contours.append(squeezed)

    return lines, contours
