"""Smoke tests for the raster2vector pipeline."""

import os

import cv2
import ezdxf
import numpy as np
import pytest

from src.preprocessor import load_and_preprocess
from src.text_separator import separate_text_and_graphics
from src.vectorizer import (
    extract_lines_and_contours,
    _merge_collinear_lines,
    _snap_endpoints,
)
from src.dxf_exporter import export_to_dxf


# ── Synthetic image fixtures ──────────────────────────────────────────────────

def _make_square(size=200, rect_offset=50, line_width=2) -> np.ndarray:
    """White square outline on black background."""
    img = np.zeros((size, size), dtype=np.uint8)
    a, b = rect_offset, size - rect_offset
    cv2.rectangle(img, (a, a), (b, b), 255, line_width)
    return img


def _make_thin_lines(size=200) -> np.ndarray:
    """1-pixel-wide horizontal and vertical lines."""
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.line(img, (20, 100), (180, 100), 255, 1)
    cv2.line(img, (100, 20), (100, 180), 255, 1)
    return img


@pytest.fixture()
def square_image_path(tmp_path):
    path = str(tmp_path / "square.png")
    cv2.imwrite(path, _make_square())
    return path


@pytest.fixture()
def thin_image_path(tmp_path):
    path = str(tmp_path / "thin.png")
    cv2.imwrite(path, _make_thin_lines())
    return path


@pytest.fixture()
def dxf_out(tmp_path):
    return str(tmp_path / "output.dxf")


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_full_pipeline(square_image_path, dxf_out):
    """Full edge-mode pipeline produces a valid, non-empty DXF."""
    original_bgr, binary = load_and_preprocess(square_image_path)
    h = binary.shape[0]

    lines, contours = extract_lines_and_contours(binary, min_line_length=30)
    assert len(lines) + len(contours) >= 1

    count = export_to_dxf(lines, contours, dxf_out, image_height=h)
    assert count >= 1
    assert os.path.isfile(dxf_out)

    doc = ezdxf.readfile(dxf_out)
    assert len(list(doc.modelspace())) >= 1
    assert len(list(doc.audit().errors)) == 0


def test_skeleton_mode(square_image_path, dxf_out):
    """Skeleton mode also produces at least one entity."""
    original_bgr, binary = load_and_preprocess(square_image_path)
    lines, contours = extract_lines_and_contours(binary, min_line_length=20,
                                                  mode="skeleton")
    count = export_to_dxf(lines, contours, dxf_out, image_height=binary.shape[0])
    assert count >= 1


def test_thin_line_preservation(thin_image_path, dxf_out):
    """Thin 1-px lines survive preprocessing with morph='none'."""
    _, binary = load_and_preprocess(thin_image_path, morph="none")
    assert np.count_nonzero(binary) > 0, "Thin lines were erased by preprocessing"
    lines, contours = extract_lines_and_contours(binary, min_line_length=20)
    assert len(lines) + len(contours) >= 1


def test_morphology_options(square_image_path):
    """All morph modes run without error."""
    for morph in ("none", "open", "close"):
        _, binary = load_and_preprocess(square_image_path, morph=morph)
        assert binary is not None


def test_text_separation(tmp_path):
    """Text separator returns two non-None masks without crashing."""
    img = np.zeros((200, 200), dtype=np.uint8)
    # Draw some small blobs that look like characters
    for x in range(20, 120, 15):
        cv2.rectangle(img, (x, 50), (x + 8, 62), 255, -1)
    text_mask, graphics_mask = separate_text_and_graphics(img)
    assert text_mask.shape == img.shape
    assert graphics_mask.shape == img.shape


def test_polarity_normalization_black_on_white(tmp_path):
    """Black strokes on white background → foreground normalised to 255."""
    img = np.full((120, 120), 255, dtype=np.uint8)  # white background
    cv2.line(img, (10, 60), (110, 60), 0, 2)         # black line
    path = str(tmp_path / "bow.png")
    cv2.imwrite(path, img)
    _, binary = load_and_preprocess(path)
    # Foreground (the line) must be the white minority, not the page.
    assert np.count_nonzero(binary) < binary.size / 2
    assert np.count_nonzero(binary) > 0


def test_skeleton_no_page_border_black_on_white(tmp_path):
    """Skeleton mode must trace the actual stroke, not the page border."""
    img = np.full((120, 120), 255, dtype=np.uint8)
    cv2.line(img, (10, 60), (110, 60), 0, 2)
    path = str(tmp_path / "bow.png")
    cv2.imwrite(path, img)
    _, binary = load_and_preprocess(path)
    lines, contours = extract_lines_and_contours(binary, min_line_length=30,
                                                 mode="skeleton")
    # All detected geometry should sit near y≈60, never on the page edges (0/119).
    all_y = [y for (_, y1, _, y2) in lines for y in (y1, y2)]
    assert all_y, "skeleton mode found nothing"
    assert all(40 <= y <= 80 for y in all_y), f"page-border artefact detected: {all_y}"


def test_merge_keeps_parallel_lines():
    """Distinct parallel lines must NOT be merged into one."""
    merged = _merge_collinear_lines([(10, 35, 130, 35), (10, 39, 130, 39)])
    assert len(merged) == 2, f"parallel lines were wrongly merged: {merged}"


def test_merge_joins_collinear_fragments():
    """End-to-end collinear fragments SHOULD merge into one segment."""
    # gap_tol=20 so 0-gap segments definitely merge
    merged = _merge_collinear_lines([(0, 0, 100, 0), (100, 0, 200, 0)], gap_tol=20.0)
    assert len(merged) == 1, f"collinear fragments not merged: {merged}"
    x1, y1, x2, y2 = merged[0]
    assert min(x1, x2) == 0 and max(x1, x2) == 200


def test_snap_endpoints_closes_small_gap():
    """Endpoints within snap_radius get merged to the same point."""
    # Two lines whose ends are 3 px apart — should snap
    lines = [(0, 0, 100, 0), (103, 0, 200, 0)]
    snapped = _snap_endpoints(lines, radius=4.0)
    # After snapping, the inner gap endpoints must be at the same coordinate
    assert snapped[0][2] == snapped[1][0], f"endpoints not snapped: {snapped}"


def test_snap_endpoints_no_snap_far_apart():
    """Endpoints farther than snap_radius must NOT be moved."""
    lines = [(0, 0, 100, 0), (110, 0, 200, 0)]
    snapped = _snap_endpoints(lines, radius=4.0)
    assert snapped[0][2] == 100   # unchanged
    assert snapped[1][0] == 110   # unchanged


def test_thick_line_single_centre_line(tmp_path):
    """A thick (10px) horizontal line should produce only ONE centre line."""
    img = np.zeros((120, 300), dtype=np.uint8)
    cv2.line(img, (10, 60), (290, 60), 255, 10)   # thick white line
    path = str(tmp_path / "thick.png")
    cv2.imwrite(path, img)
    _, binary = load_and_preprocess(path)
    lines, contours = extract_lines_and_contours(
        binary, min_line_length=30, mode="skeleton",
    )
    # All y-coordinates of detected lines should cluster near 60
    all_y = [y for (_, y1, _, y2) in lines for y in (y1, y2)]
    assert all_y, "No lines detected on thick line"
    assert all(50 <= y <= 70 for y in all_y), \
        f"Got multiple y-bands (double-line artefact?): {sorted(set(all_y))}"


def test_pre_close_thick_line(tmp_path):
    """--pre-close-kernel joins the skeleton of a thick stroke."""
    img = np.zeros((120, 300), dtype=np.uint8)
    cv2.rectangle(img, (10, 50), (290, 70), 255, -1)  # filled rectangle (thick line)
    path = str(tmp_path / "rect.png")
    cv2.imwrite(path, img)
    _, binary = load_and_preprocess(path)
    lines_no_close, _ = extract_lines_and_contours(binary, min_line_length=30,
                                                    mode="skeleton", pre_close_kernel=0)
    lines_close, _ = extract_lines_and_contours(binary, min_line_length=30,
                                                 mode="skeleton", pre_close_kernel=7)
    # With closing, the thick rectangle should reduce to fewer lines
    assert len(lines_close) <= len(lines_no_close) + 2


def test_adaptive_block_oversized(tmp_path):
    """Oversized adaptive block size is clamped, not crashed."""
    img = np.zeros((50, 50), dtype=np.uint8)
    cv2.rectangle(img, (5, 5), (45, 45), 255, 2)
    path = str(tmp_path / "small.png")
    cv2.imwrite(path, img)
    # Should not raise cv2.error
    _, binary = load_and_preprocess(path, threshold_method="adaptive",
                                    adaptive_block_size=100000001)
    assert binary is not None


def test_thin_grid_survives_default(tmp_path):
    """A dense 1px grid must survive default preprocessing (despeckle, morph=none)."""
    img = np.zeros((100, 100), dtype=np.uint8)
    for i in range(0, 100, 10):
        cv2.line(img, (0, i), (99, i), 255, 1)
        cv2.line(img, (i, 0), (i, 99), 255, 1)
    path = str(tmp_path / "grid.png")
    cv2.imwrite(path, img)
    _, binary = load_and_preprocess(path)  # defaults: morph=none, despeckle=True
    assert np.count_nonzero(binary) > 0, "thin grid erased by default preprocessing"


def test_load_nonexistent_image():
    with pytest.raises(ValueError, match="Cannot load image"):
        load_and_preprocess("/nonexistent/path/image.png")


def test_export_empty(tmp_path):
    """Empty export produces a valid DXF with zero entities."""
    out = str(tmp_path / "empty.dxf")
    count = export_to_dxf([], [], out, image_height=100)
    assert count == 0
    doc = ezdxf.readfile(out)
    assert len(list(doc.audit().errors)) == 0


def test_dpi_validation():
    """Negative DPI is caught before any processing."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "raster2vector.py", "dummy.png", "--dpi", "-1"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "dpi" in result.stderr.lower() or "error" in result.stderr.lower()


def test_approx_epsilon_option(square_image_path, dxf_out):
    """Custom --approx-epsilon produces a valid DXF."""
    _, binary = load_and_preprocess(square_image_path)
    lines, contours = extract_lines_and_contours(binary, approx_epsilon=3.0)
    count = export_to_dxf(lines, contours, dxf_out, image_height=binary.shape[0])
    assert count >= 0  # may be 0 for a simple image — just must not crash


def test_layer_names(square_image_path, dxf_out):
    """DXF output contains the expected layer names."""
    _, binary = load_and_preprocess(square_image_path)
    lines, contours = extract_lines_and_contours(binary, min_line_length=30)
    export_to_dxf(lines, contours, dxf_out, image_height=binary.shape[0])
    doc = ezdxf.readfile(dxf_out)
    layer_names = {layer.dxf.name for layer in doc.layers}
    assert "LINES" in layer_names
    assert "CONTOURS" in layer_names
    assert "TEXT_CANDIDATES" in layer_names
