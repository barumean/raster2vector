"""Smoke test for the raster2vector pipeline."""

import os
import tempfile

import cv2
import ezdxf
import numpy as np
import pytest

from src.preprocessor import load_and_preprocess
from src.vectorizer import extract_lines_and_contours
from src.dxf_exporter import export_to_dxf


def make_synthetic_image() -> np.ndarray:
    """Create a 200x200 black image with a white 100x100 square outline centred on it."""
    img = np.zeros((200, 200), dtype=np.uint8)
    # Draw a white rectangle outline (not filled)
    cv2.rectangle(img, (50, 50), (150, 150), 255, 2)
    return img


@pytest.fixture()
def synthetic_image_path(tmp_path):
    """Write the synthetic image to a temp PNG and return its path."""
    img = make_synthetic_image()
    path = str(tmp_path / "test_square.png")
    cv2.imwrite(path, img)
    return path


@pytest.fixture()
def dxf_output_path(tmp_path):
    return str(tmp_path / "output.dxf")


def test_full_pipeline(synthetic_image_path, dxf_output_path):
    """Full pipeline: load → preprocess → vectorise → export → validate."""

    # 1. Preprocess
    original_bgr, binary = load_and_preprocess(
        synthetic_image_path,
        threshold_method="otsu",
        invert=False,
        manual_threshold=None,
    )
    assert binary is not None
    assert binary.dtype == np.uint8
    height, width = binary.shape[:2]
    assert height == 200 and width == 200

    # 2. Vectorise — use short min_line_length so the 100 px sides are picked up
    lines, contours = extract_lines_and_contours(binary, min_line_length=30, max_gap=10)
    total_shapes = len(lines) + len(contours)
    assert total_shapes >= 1, (
        f"Expected at least 1 detected shape, got lines={len(lines)} contours={len(contours)}"
    )

    # 3. Export to DXF
    entity_count = export_to_dxf(
        lines,
        contours,
        dxf_output_path,
        image_height=height,
        dpi=96.0,
        units_mm=True,
    )
    assert entity_count >= 1, f"Expected at least 1 DXF entity, got {entity_count}"
    assert os.path.isfile(dxf_output_path), "DXF file was not created"

    # 4. Load the DXF and verify entity count
    doc = ezdxf.readfile(dxf_output_path)
    msp = doc.modelspace()
    entities = list(msp)
    assert len(entities) >= 1, f"DXF modelspace has no entities (entity_count={entity_count})"

    # 5. Run ezdxf auditor — must report zero violations
    auditor = doc.audit()
    violations = list(auditor.violations)
    assert len(violations) == 0, (
        f"DXF audit found {len(violations)} violation(s):\n"
        + "\n".join(str(v) for v in violations)
    )


def test_load_nonexistent_image():
    """load_and_preprocess raises ValueError for missing files."""
    with pytest.raises(ValueError, match="Cannot load image"):
        load_and_preprocess("/nonexistent/path/image.png")


def test_export_empty(tmp_path):
    """export_to_dxf with no lines/contours produces a valid empty DXF."""
    output = str(tmp_path / "empty.dxf")
    count = export_to_dxf([], [], output, image_height=100)
    assert count == 0
    doc = ezdxf.readfile(output)
    auditor = doc.audit()
    assert len(list(auditor.violations)) == 0
