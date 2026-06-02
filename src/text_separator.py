"""Text/graphics separation — Fletcher-Kasturi (1988) with Tombre (2002) refinements.

Three-layer output
------------------
Following Tombre et al. "Text/Graphics Separation Revisited" (DAS 2002, LNCS 2423,
pp. 200–211), components are split into three layers rather than two:

  text_mask      — small, roughly square blobs that form collinear "string" runs.
  graphics_mask  — large blobs that exceed T1 (main line geometry).
  elongated_mask — small but highly elongated blobs (T4 test); these are dash
                   candidates or isolated characters such as "I", "l", "1".
                   Feed to a downstream dashed-line detector rather than the
                   graphics layer.

Key thresholds (Tombre et al. §3, table of "good results"):
  T1 = 1.5 × max(A_mode, A_mean)   — upper area bound for text candidates.
  T2 = 20                            — max elongation (max_dim / min_dim).
  T3 = 0.5                           — minimum density (area / bbox_area).
  T4 = 2                             — minimum elongation to be a dash candidate.
"""
from collections import defaultdict
from typing import Optional

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
        rho_tol: ρ bin width (px); R ≈ 0.2 × H_avg per FK.
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
    max_char_area: Optional[int] = None,
    size_threshold_n: float = 1.5,
    t2_max_elongation: float = 20.0,
    t3_min_density: float = 0.5,
    t4_dash_elongation: float = 2.0,
    min_aspect: float = 0.1,
    max_aspect: float = 10.0,
    min_string_count: int = 3,
    search_radius_factor: float = 3.0,
    angle_step_deg: float = 3.0,
    rho_tol_factor: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separate text, graphics, and dash/elongated blobs (3-layer FK+Tombre).

    Implements Fletcher & Kasturi (IEEE TPAMI, 1988) with the three-layer
    refinement from Tombre et al. (DAS 2002):

    1. Connected-component extraction with area, density, and elongation filters.
    2. Dynamic upper area threshold T1 = size_threshold_n × max(A_mode, A_mean).
    3. Density filter T3: components whose foreground area / bbox area < T3 are
       noisy/sparse and are not reliable text candidates.
    4. Elongation filter T4: small but elongated components (max_dim/min_dim ≥ T4)
       form a separate elongated (dash-candidate) layer.
    5. Hough-transform collinear grouping of surviving text candidates (FK §3,
       Tombre µ=2.5 word-gap heuristic, ρ resolution R=0.2×H_avg).

    Tombre et al. recommended threshold values (§3, "good results"):
      size_threshold_n = 1.5   T2 = 20   T3 = 0.5   T4 = 2

    Args:
        binary: Binary image (white foreground on black background), uint8.
        min_char_area: Minimum CC area to consider as a character.
        max_char_area: Hard upper area cap; None = use dynamic T1 only.
        size_threshold_n: Multiplier n in T1 = n × max(A_mode, A_mean).
        t2_max_elongation: Maximum elongation (max_dim/min_dim) for a text
            candidate.  Larger → graphics (Tombre: 20).
        t3_min_density: Minimum foreground density (area / bbox_area) for a
            text candidate.  Sparser components are noisy outlines.
            (Tombre: 0.5).
        t4_dash_elongation: Elongation threshold above which a *small* component
            is classified into the elongated/dash layer rather than text.
            (Tombre: 2).
        min_aspect: Minimum bbox aspect ratio (w/h) — legacy guard.
        max_aspect: Maximum bbox aspect ratio (w/h) — legacy guard.
        min_string_count: Minimum characters in a collinear run to label as text.
        search_radius_factor: Max inter-character gap along the baseline
            (multiple of median char height).
        angle_step_deg: Angular resolution of the Hough sweep (degrees).
        rho_tol_factor: ρ bin width as a fraction of H_avg (FK R=0.2×H_avg).

    Returns:
        (text_mask, graphics_mask, elongated_mask) — three uint8 binary images.
        elongated_mask: small elongated blobs that are dash or "I"/"l" candidates;
            feed to dashed-line detection, not to the main graphic layer.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # ── Compute dynamic T1 (FK area threshold) ────────────────────────────────
    all_areas = []
    for i in range(1, n_labels):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a > 0:
            all_areas.append(a)

    if all_areas:
        a_mean = float(np.mean(all_areas))
        counts, edges = np.histogram(all_areas, bins=min(50, max(1, len(all_areas))))
        a_mode = float(edges[int(np.argmax(counts))])
        t1_dynamic = size_threshold_n * max(a_mode, a_mean)
    else:
        t1_dynamic = float(max_char_area or 2000)

    effective_max_area = int(min(
        max_char_area if max_char_area is not None else t1_dynamic,
        t1_dynamic,
    ))

    # ── Classify each component ───────────────────────────────────────────────
    text_indices: list[int] = []
    text_centres: list[tuple[float, float]] = []
    text_heights: list[float] = []
    elongated_indices: list[int] = []  # Tombre 3rd layer

    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if w == 0 or h == 0:
            continue

        # T1: size filter — large components are graphics, not text.
        if area > effective_max_area or area < min_char_area:
            continue

        # Derived geometry
        bbox_area = w * h
        density = area / bbox_area if bbox_area > 0 else 0.0
        max_dim = max(w, h)
        min_dim = min(w, h)
        elongation = max_dim / min_dim if min_dim > 0 else float("inf")
        aspect = w / h

        # T3: density filter — sparse/noisy outlines are unreliable text.
        if density < t3_min_density:
            continue

        # T2: elongation upper bound — grossly elongated blobs are dashes or
        # thin lines, not text characters.
        if elongation > t2_max_elongation:
            continue

        # Legacy aspect-ratio guard.
        if not (min_aspect <= aspect <= max_aspect):
            continue

        # T4: small elongated blobs → 3rd (dash/elongated) layer.
        if elongation >= t4_dash_elongation:
            elongated_indices.append(i)
            continue

        text_indices.append(i)
        text_centres.append((x + w / 2.0, y + h / 2.0))
        text_heights.append(float(h))

    # ── Paint the elongated (dash-candidate) mask ─────────────────────────────
    elongated_mask = np.zeros_like(binary)
    for idx in elongated_indices:
        elongated_mask[labels == idx] = 255

    # ── Hough collinear grouping of text candidates ───────────────────────────
    text_mask = np.zeros_like(binary)

    if len(text_indices) >= min_string_count:
        centres = np.array(text_centres, dtype=np.float64)
        avg_h = float(np.median(text_heights))
        # Tombre µ=2.5 word-gap heuristic for max inter-character spacing.
        max_gap = avg_h * search_radius_factor
        # FK: ρ resolution R = rho_tol_factor × H_avg (paper: 0.4; Tombre: 0.2).
        rho_tol = max(2.0, avg_h * rho_tol_factor)

        groups = _hough_collinear_groups(
            centres, min_string_count, angle_step_deg, rho_tol, max_gap,
        )
        for run in groups:
            for local_idx in run:
                text_mask[labels == text_indices[local_idx]] = 255

    graphics_mask = cv2.bitwise_and(
        binary,
        cv2.bitwise_not(cv2.bitwise_or(text_mask, elongated_mask)),
    )
    return text_mask, graphics_mask, elongated_mask
