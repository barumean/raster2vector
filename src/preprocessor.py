from typing import Optional

import cv2
import numpy as np


def _selective_blur(gray: np.ndarray, radius: int, delta: int = 20) -> np.ndarray:
    """Edge-preserving Gaussian blur (imagetracerjs selective blur, §preprocess).

    Applies a Gaussian blur of the given radius, then restores the original
    pixel value wherever the absolute difference between original and blurred
    exceeds *delta*.  This smooths flat (low-contrast) regions — reducing
    scanner noise and JPEG artefacts — while leaving edges sharp.

    Reference: imagetracerjs selectiveblur(), blurdelta default = 20.

    Args:
        gray:   8-bit grayscale image.
        radius: Gaussian kernel radius (ksize = 2*radius+1).  0 = no-op.
        delta:  Intensity threshold; pixels with |original − blurred| > delta
                are treated as edges and restored to the original value.
    Returns:
        Filtered grayscale image, same shape/dtype as input.
    """
    if radius <= 0:
        return gray
    ksize = 2 * radius + 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    diff = np.abs(gray.astype(np.int32) - blurred.astype(np.int32))
    result = blurred.copy()
    result[diff > delta] = gray[diff > delta]
    return result


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


def _deskew(gray: np.ndarray, min_angle_deg: float = 0.5) -> np.ndarray:
    """Correct rotational skew using the dominant angle of line blobs.

    Dilates the image into wide horizontal blobs, finds the median angle of
    their minimum-area bounding rectangles, and rotates the grayscale image
    to compensate.  Angles smaller than min_angle_deg are ignored (no-op).

    The median is used instead of the mean to be robust against outliers
    (e.g. isolated diagonal marks).

    Args:
        gray: 8-bit grayscale image.
        min_angle_deg: Skip correction if the estimated skew is below this.

    Returns:
        Deskewed grayscale image (same dtype/shape as input).
    """
    # Threshold into a coarse binary so we can find blob orientations.
    _, coarse = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(coarse, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angles = []
    for c in contours:
        if len(c) < 5:
            continue
        _, _, angle = cv2.minAreaRect(c)
        # OpenCV minAreaRect returns angles in [-90, 0).  A near-vertical
        # rectangle reports ≈-89° instead of ≈1°, so correct for this.
        if angle < -45:
            angle = 90 + angle
        angles.append(angle)

    if not angles:
        return gray

    skew = float(np.median(angles))
    if abs(skew) < min_angle_deg:
        return gray

    h, w = gray.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), skew, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def load_and_preprocess(
    image_path: str,
    threshold_method: str = "otsu",
    invert: bool = False,
    manual_threshold: Optional[int] = None,
    morph: str = "none",
    morph_kernel: int = 2,
    adaptive_block_size: int = 51,
    adaptive_c: int = 9,
    sauvola_window_size: int = 25,
    sauvola_k: float = 0.2,
    despeckle: bool = True,
    min_speckle_area: int = 3,
    deskew: bool = False,
    deskew_min_angle: float = 0.5,
    blur_radius: int = 0,
    blur_delta: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load image and produce a binary image with foreground (strokes) = 255.

    The output polarity is normalised so that the drawing strokes are the
    white (255) minority foreground, regardless of whether the source is
    black-on-white or white-on-black. This keeps every downstream consumer
    (Canny, findContours, skeletonize, text separation) consistent.

    Args:
        image_path: Path to the raster image.
        threshold_method: 'otsu', 'adaptive', or 'sauvola'.
            - otsu: global Otsu binarisation; best when illumination is uniform.
            - adaptive: local Gaussian-weighted adaptive threshold; handles mild
              illumination variation.
            - sauvola: Sauvola (1999) local threshold; best for scanned drawings
              with shading, fold shadows, or uneven exposure.  A refinement of
              Niblack with far less noise in blank regions.  Uses
              `skimage.filters.threshold_sauvola`.
        invert: Invert grayscale before thresholding (manual override).
        manual_threshold: Fixed 0-255 threshold; overrides threshold_method.
        morph: Explicit morphology — 'none', 'open', or 'close'.
        morph_kernel: Square kernel side length for the morphological op.
        adaptive_block_size: Block size for adaptive thresholding (auto-clamped odd).
        adaptive_c: Constant subtracted from the mean in adaptive thresholding.
        sauvola_window_size: Local window size for Sauvola (should be roughly
            on the order of the stroke/character size, default 25).
        sauvola_k: Sauvola sensitivity parameter (0.2–0.5 typical; lower = more
            foreground). Sauvola (1999) recommends 0.5; lower values work better
            on thin engineering lines.
        despeckle: Remove tiny isolated components (thin-line safe).
        min_speckle_area: Components smaller than this (px) are dropped.
        deskew: Correct rotational skew before binarisation using the dominant
            angle of horizontal line blobs (cv2.minAreaRect on dilated contours).
        deskew_min_angle: Skew corrections smaller than this (degrees) are
            skipped to avoid unnecessary resampling.
        blur_radius: Gaussian kernel radius for the edge-preserving selective
            blur applied to the grayscale image before thresholding.  0 = off
            (default).  Values 1–3 reduce scanner noise / JPEG ringing while
            keeping hard edges sharp via the blur_delta guard.
            (imagetracerjs: blurradius default 0, max recommended 5.)
        blur_delta: Pixel intensity delta threshold for the selective blur:
            pixels where |original − blurred| > blur_delta are edge pixels and
            are restored to the original value.  Default 20.
            (imagetracerjs: blurdelta default 20.)

    Returns:
        (original_bgr, gray_image, binary_image)
        gray_image  : 8-bit grayscale (after optional invert + deskew).
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

    # Optional deskew before binarisation so the threshold operates on a
    # correctly-oriented image.
    if deskew:
        gray = _deskew(gray, min_angle_deg=deskew_min_angle)

    # Edge-preserving selective blur (imagetracerjs §preprocess).  Applied
    # after deskew so we smooth the already-aligned image, and before
    # thresholding so the threshold benefits from reduced noise.
    if blur_radius > 0:
        gray = _selective_blur(gray, blur_radius, blur_delta)

    gray_out = gray.copy()   # preserve pre-threshold grayscale for Canny

    h, w = gray.shape[:2]

    if manual_threshold is not None:
        _, binary = cv2.threshold(gray, manual_threshold, 255, cv2.THRESH_BINARY)
    elif threshold_method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_method == "sauvola":
        # Sauvola local thresholding (Sauvola & Pietikäinen, 1999).
        # Preferred over adaptive Gaussian for scanned engineering drawings
        # because it adapts to local contrast without over-thresholding blank
        # regions (the main failure mode of Niblack / simple local mean).
        try:
            from skimage.filters import threshold_sauvola
        except ImportError as exc:
            raise ImportError(
                "Sauvola thresholding requires scikit-image: pip install scikit-image"
            ) from exc
        # Window size must be odd and not exceed image dimensions.
        ws = max(3, sauvola_window_size | 1)
        ws = min(ws, min(h, w) | 1)
        thresh = threshold_sauvola(gray, window_size=ws, k=sauvola_k)
        binary = (gray > thresh).astype(np.uint8) * 255
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
