import cv2
import numpy as np


def load_and_preprocess(
    image_path: str,
    threshold_method: str = "otsu",
    invert: bool = False,
    manual_threshold: int = None,
    morph: str = "open",
    morph_kernel: int = 2,
    adaptive_block_size: int = 51,
    adaptive_c: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Load image and produce a binary (thresholded) version.

    Args:
        image_path: Path to the raster image.
        threshold_method: 'otsu' or 'adaptive'.
        invert: Invert pixel values before thresholding.
        manual_threshold: Fixed 0-255 threshold; overrides threshold_method.
        morph: Morphological post-processing — 'none', 'open', or 'close'.
        morph_kernel: Square kernel side length for morphological op.
        adaptive_block_size: Block size for adaptive thresholding (must be odd ≥ 3).
        adaptive_c: Constant subtracted from mean in adaptive thresholding.

    Returns:
        (original_bgr, binary_image)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if invert:
        gray = cv2.bitwise_not(gray)

    if manual_threshold is not None:
        _, binary = cv2.threshold(gray, manual_threshold, 255, cv2.THRESH_BINARY)
    elif threshold_method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:  # adaptive
        block = adaptive_block_size | 1  # ensure odd
        block = max(block, 3)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            adaptive_c,
        )

    if morph != "none" and morph_kernel > 0:
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        if morph == "open":
            # Skip open when white-pixel ratio is very low to preserve thin strokes
            white_ratio = np.count_nonzero(binary) / binary.size
            if white_ratio > 0.02:
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        elif morph == "close":
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return img, binary
