from typing import Optional

import cv2
import numpy as np


def _despeckle(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Remove isolated foreground components smaller than min_area.

    Unlike a morphological open, this is thin-line safe: a continuous 1px line
    has an area equal to its length, so only genuine isolated specks are removed.
    Assumes foreground = nonzero (white).
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for i in range(1, n_labels):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def load_and_preprocess(
    image_path: str,
    threshold_method: str = "otsu",
    invert: bool = False,
    manual_threshold: Optional[int] = None,
    morph: str = "none",
    morph_kernel: int = 2,
    adaptive_block_size: int = 51,
    adaptive_c: int = 9,
    despeckle: bool = True,
    min_speckle_area: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Load image and produce a binary image with foreground (strokes) = 255.

    The output polarity is normalised so that the drawing strokes are the
    white (255) minority foreground, regardless of whether the source is
    black-on-white or white-on-black. This keeps every downstream consumer
    (Canny, findContours, skeletonize, text separation) consistent.

    Args:
        image_path: Path to the raster image.
        threshold_method: 'otsu' or 'adaptive'.
        invert: Invert grayscale before thresholding (manual override).
        manual_threshold: Fixed 0-255 threshold; overrides threshold_method.
        morph: Explicit morphology — 'none', 'open', or 'close'.
        morph_kernel: Square kernel side length for the morphological op.
        adaptive_block_size: Block size for adaptive thresholding (auto-clamped odd).
        adaptive_c: Constant subtracted from the mean in adaptive thresholding.
        despeckle: Remove tiny isolated components (thin-line safe).
        min_speckle_area: Components smaller than this (px) are dropped.

    Returns:
        (original_bgr, gray_image, binary_image)
        gray_image  : 8-bit grayscale of the original (after optional invert).
                      Used by Canny for gradient-based edge detection so that
                      subtle colour/tone boundaries are not lost to binarisation.
        binary_image: Normalised binary with strokes = 255 (minority foreground).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if invert:
        gray = cv2.bitwise_not(gray)

    gray_out = gray.copy()   # preserve pre-threshold grayscale for Canny

    h, w = gray.shape[:2]

    if manual_threshold is not None:
        _, binary = cv2.threshold(gray, manual_threshold, 255, cv2.THRESH_BINARY)
    elif threshold_method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:  # adaptive
        # Clamp block size: must be odd, >= 3, and not exceed the image dimension.
        max_block = min(h, w)
        if max_block % 2 == 0:
            max_block -= 1
        block = adaptive_block_size | 1          # force odd
        block = max(3, min(block, max(3, max_block)))
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            block, adaptive_c,
        )

    # Normalise polarity: strokes should be the white (255) foreground.
    # If white pixels are the majority, the background is white → invert.
    if np.count_nonzero(binary) > binary.size / 2:
        binary = cv2.bitwise_not(binary)

    # Thin-line-safe noise removal (default).
    if despeckle and min_speckle_area > 0:
        binary = _despeckle(binary, min_speckle_area)

    # Explicit morphology (opt-in; can damage thin lines, hence not default).
    if morph != "none" and morph_kernel > 0:
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        if morph == "open":
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        elif morph == "close":
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return img, gray_out, binary
