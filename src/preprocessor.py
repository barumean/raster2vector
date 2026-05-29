import cv2
import numpy as np


def load_and_preprocess(
    image_path: str,
    threshold_method: str = "otsu",
    invert: bool = False,
    manual_threshold: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load image and produce a binary (thresholded) version.

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
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,
            9,
        )

    # Morphological open to remove small noise specks
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return img, binary
