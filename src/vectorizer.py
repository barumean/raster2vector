import cv2
import numpy as np


def extract_lines_and_contours(
    binary: np.ndarray,
    min_line_length: int = 50,
    max_gap: int = 10,
) -> tuple[list, list]:
    """Extract straight lines (via HoughLinesP) and remaining contours.

    Args:
        binary: Binary (thresholded) image, uint8.
        min_line_length: Minimum length of a Hough line segment (pixels).
        max_gap: Maximum gap between collinear segments to bridge (pixels).

    Returns:
        (hough_lines, contours)
        hough_lines: list of (x1, y1, x2, y2) tuples.
        contours: list of np.ndarray with shape (N, 2) — squeezed contour points.
    """
    # Step 1: Canny edge detection on binary image
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # Step 2: HoughLinesP for straight line segments
    hough_result = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=max_gap,
    )

    lines = []
    mask = np.zeros_like(edges)

    if hough_result is not None:
        for line in hough_result:
            x1, y1, x2, y2 = line[0]
            lines.append((x1, y1, x2, y2))
            # Draw detected lines on mask so they can be excluded from contours
            cv2.line(mask, (x1, y1), (x2, y2), 255, 3)

    # Step 3: Dilate mask and subtract from edges to avoid duplicates
    dilated_mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    remaining_edges = cv2.bitwise_and(edges, cv2.bitwise_not(dilated_mask))

    # Step 4: findContours on remaining edges
    contour_result, _ = cv2.findContours(
        remaining_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = []
    for c in contour_result:
        # Reduce vertices with approxPolyDP for a lighter DXF
        epsilon = 1.5
        approx = cv2.approxPolyDP(c, epsilon, closed=False)
        squeezed = approx.squeeze()
        if squeezed.ndim == 2 and len(squeezed) >= 2:
            contours.append(squeezed)

    return lines, contours
