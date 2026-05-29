import cv2
import numpy as np


def separate_text_and_graphics(
    binary: np.ndarray,
    min_char_area: int = 10,
    max_char_area: int = 2000,
    min_aspect: float = 0.1,
    max_aspect: float = 10.0,
    min_string_count: int = 2,
    search_radius_factor: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate text blobs from graphics using connected-component analysis.

    Implements a simplified Fletcher-Kasturi-style heuristic:
    1. Extract connected components.
    2. Filter by bounding-box area and aspect ratio to find character candidates.
    3. Group spatially adjacent candidates into "string" clusters.
    4. Any cluster with >= min_string_count characters is treated as text.

    Args:
        binary: Binary image (white foreground on black background), uint8.
        min_char_area: Minimum CC area to consider as a character.
        max_char_area: Maximum CC area to consider as a character (larger = graphics).
        min_aspect: Minimum bounding-box aspect ratio (w/h) for character candidates.
        max_aspect: Maximum bounding-box aspect ratio (w/h) for character candidates.
        min_string_count: Minimum characters in a cluster to label as text.
        search_radius_factor: Multiplier on average char height used as grouping radius.

    Returns:
        (text_mask, graphics_mask) — both uint8 binary images.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    char_indices = []
    char_boxes = []  # (x, y, w, h, cx, cy)

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
            char_boxes.append((x, y, w, h, x + w / 2, y + h / 2))

    text_mask = np.zeros_like(binary)

    if len(char_boxes) >= min_string_count:
        boxes = np.array(char_boxes, dtype=np.float32)  # (N, 6)
        avg_h = float(np.median(boxes[:, 3]))
        radius = avg_h * search_radius_factor

        # Simple union-find grouping by proximity of bounding-box centres
        parent = list(range(len(char_indices)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        cx = boxes[:, 4]
        cy = boxes[:, 5]
        for i in range(len(char_indices)):
            for j in range(i + 1, len(char_indices)):
                dist = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                if dist <= radius:
                    union(i, j)

        # Collect clusters
        from collections import defaultdict
        clusters: dict[int, list[int]] = defaultdict(list)
        for i, ci in enumerate(char_indices):
            clusters[find(i)].append(ci)

        # Paint text mask for clusters large enough
        for members in clusters.values():
            if len(members) >= min_string_count:
                for label_idx in members:
                    text_mask[labels == label_idx] = 255

    graphics_mask = cv2.bitwise_and(binary, cv2.bitwise_not(text_mask))
    return text_mask, graphics_mask
