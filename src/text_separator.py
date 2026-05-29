from collections import defaultdict

import cv2
import numpy as np


def _hough_collinear_groups(
    centres: np.ndarray,
    min_count: int,
    angle_step_deg: float,
    rho_tol: float,
    max_gap: float,
) -> list[list[int]]:
    """Group candidate centroids that lie on a common line (Fletcher-Kasturi).

    Each centroid votes for the family of lines passing through it at the
    quantised angles θ via ρ = x·cosθ + y·sinθ.  Centroids sharing the same
    (θ, ρ) cell are collinear.  Within each collinear cell the members are then
    split into runs whose consecutive spacing (along the line) is ≤ max_gap, so
    that a long page-spanning coincidental alignment is not treated as one
    string.  Runs of ≥ min_count members are returned.

    Args:
        centres: (N, 2) array of (cx, cy) centroids.
        min_count: Minimum members for a run to qualify as a text string.
        angle_step_deg: Angular quantisation of the Hough sweep (degrees).
        rho_tol: ρ bin width (px); ~half the char height keeps a baseline together.
        max_gap: Maximum spacing between consecutive characters along the line.

    Returns:
        List of index lists (each a collinear, well-spaced run of centroids).
    """
    n = len(centres)
    if n < min_count:
        return []
    cx = centres[:, 0]
    cy = centres[:, 1]
    angles = np.deg2rad(np.arange(0.0, 180.0, angle_step_deg))
    cos = np.cos(angles)
    sin = np.sin(angles)
    rho = np.outer(cx, cos) + np.outer(cy, sin)  # (N, A)

    groups: list[list[int]] = []
    seen: set[frozenset] = set()
    for a in range(len(angles)):
        bins: dict[int, list[int]] = defaultdict(list)
        for k in range(n):
            bins[int(round(rho[k, a] / rho_tol))].append(k)
        # Line direction (perpendicular to the normal) for spacing checks.
        dvec = np.array([-sin[a], cos[a]])
        for members in bins.values():
            if len(members) < min_count:
                continue
            proj = centres[members] @ dvec
            order = np.argsort(proj)
            ms = [members[i] for i in order]
            pj = proj[order]
            run = [ms[0]]
            for i in range(1, len(ms)):
                if pj[i] - pj[i - 1] <= max_gap:
                    run.append(ms[i])
                else:
                    if len(run) >= min_count:
                        key = frozenset(run)
                        if key not in seen:
                            seen.add(key)
                            groups.append(run)
                    run = [ms[i]]
            if len(run) >= min_count:
                key = frozenset(run)
                if key not in seen:
                    seen.add(key)
                    groups.append(run)
    return groups


def separate_text_and_graphics(
    binary: np.ndarray,
    min_char_area: int = 10,
    max_char_area: int = 2000,
    min_aspect: float = 0.1,
    max_aspect: float = 10.0,
    min_string_count: int = 3,
    search_radius_factor: float = 3.0,
    angle_step_deg: float = 3.0,
    rho_tol_factor: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate text blobs from graphics using connected-component analysis.

    Implements a Fletcher-Kasturi-style heuristic:
    1. Extract connected components.
    2. Filter by bounding-box area and aspect ratio to find character candidates.
    3. Group candidates whose centroids are *collinear* (lie on a shared text
       baseline) using a Hough transform on the centroids, with a consecutive
       spacing constraint so unrelated alignments are not merged.
    4. Any collinear run of >= min_string_count characters is treated as text.

    Args:
        binary: Binary image (white foreground on black background), uint8.
        min_char_area: Minimum CC area to consider as a character.
        max_char_area: Maximum CC area to consider as a character (larger = graphics).
        min_aspect: Minimum bounding-box aspect ratio (w/h) for character candidates.
        max_aspect: Maximum bounding-box aspect ratio (w/h) for character candidates.
        min_string_count: Minimum characters in a collinear run to label as text.
        search_radius_factor: Multiplier on median char height for the maximum
            inter-character spacing along a baseline.
        angle_step_deg: Angular resolution of the Hough sweep (degrees).
        rho_tol_factor: ρ bin width as a fraction of median char height.

    Returns:
        (text_mask, graphics_mask) — both uint8 binary images.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    char_indices = []
    char_centres = []  # (cx, cy)
    char_heights = []

    for i in range(1, n_labels):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if w == 0 or h == 0:
            continue
        aspect = w / h
        if (min_char_area <= area <= max_char_area
                and min_aspect <= aspect <= max_aspect):
            char_indices.append(i)
            char_centres.append((x + w / 2.0, y + h / 2.0))
            char_heights.append(h)

    text_mask = np.zeros_like(binary)

    if len(char_indices) >= min_string_count:
        centres = np.array(char_centres, dtype=np.float64)
        avg_h = float(np.median(char_heights))
        max_gap = avg_h * search_radius_factor
        rho_tol = max(2.0, avg_h * rho_tol_factor)

        groups = _hough_collinear_groups(
            centres, min_string_count, angle_step_deg, rho_tol, max_gap,
        )
        for run in groups:
            for local_idx in run:
                text_mask[labels == char_indices[local_idx]] = 255

    graphics_mask = cv2.bitwise_and(binary, cv2.bitwise_not(text_mask))
    return text_mask, graphics_mask
