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
    _fit_circle,
)
from src.dxf_exporter import _bulge_from_3pts, export_to_dxf as _export
from src.stroke_width import estimate_line_widths, estimate_contour_widths, _DXF_WEIGHTS
from src.vectorizer import _zhang_suen_thin
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
    original_bgr, gray, binary = load_and_preprocess(square_image_path)
    h = binary.shape[0]

    lines, contours = extract_lines_and_contours(binary, min_line_length=30)
    assert len(lines) + len(contours) >= 1

    count = export_to_dxf(lines, contours, dxf_out, image_height=h)
    assert count >= 1
    assert os.path.isfile(dxf_out)

    doc = ezdxf.readfile(dxf_out)
    assert len(list(doc.modelspace())) >= 1
    assert len(list(doc.audit().errors)) == 0


def test_skeleton_mode_deprecated_alias(square_image_path, dxf_out):
    """mode='skeleton' is a deprecated alias: it warns and behaves like edge."""
    original_bgr, gray, binary = load_and_preprocess(square_image_path)
    with pytest.warns(DeprecationWarning):
        skel = extract_lines_and_contours(binary, min_line_length=20,
                                          mode="skeleton")
    edge = extract_lines_and_contours(binary, min_line_length=20, mode="edge")
    # Deprecated alias must produce identical geometry to edge mode.
    assert skel[0] == edge[0]
    assert len(skel[1]) == len(edge[1])
    count = export_to_dxf(skel[0], skel[1], dxf_out, image_height=binary.shape[0])
    assert count >= 1


def test_thin_line_preservation(thin_image_path, dxf_out):
    """Thin 1-px lines survive preprocessing with morph='none'."""
    _, _gray, binary = load_and_preprocess(thin_image_path, morph="none")
    assert np.count_nonzero(binary) > 0, "Thin lines were erased by preprocessing"
    lines, contours = extract_lines_and_contours(binary, min_line_length=20)
    assert len(lines) + len(contours) >= 1


def test_morphology_options(square_image_path):
    """All morph modes run without error."""
    for morph in ("none", "open", "close"):
        _, _gray, binary = load_and_preprocess(square_image_path, morph=morph)
        assert binary is not None


def test_sauvola_threshold(tmp_path):
    """Sauvola binarisation produces a non-empty binary image."""
    img = _make_square()
    path = str(tmp_path / "sq.png")
    cv2.imwrite(path, img)
    _, _gray, binary = load_and_preprocess(path, threshold_method="sauvola")
    assert binary is not None
    assert np.count_nonzero(binary) > 0


def test_deskew_does_not_crash(square_image_path):
    """Deskew flag runs without error on a straight drawing."""
    _, _gray, binary = load_and_preprocess(square_image_path, deskew=True)
    assert binary is not None


def test_text_separation(tmp_path):
    """A collinear row of character-like blobs is detected as text (3-tuple)."""
    img = np.zeros((200, 200), dtype=np.uint8)
    for x in range(20, 120, 15):
        cv2.rectangle(img, (x, 50), (x + 8, 62), 255, -1)
    text_mask, graphics_mask, elongated_mask = separate_text_and_graphics(img)
    assert text_mask.shape == img.shape
    assert graphics_mask.shape == img.shape
    assert elongated_mask.shape == img.shape
    assert np.count_nonzero(text_mask) > 0, "collinear blob row not detected as text"


def test_text_separation_isolated_blobs_not_text(tmp_path):
    """Two far-apart blobs (no collinear run of 3) are NOT labelled text."""
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (18, 22), 255, -1)
    cv2.rectangle(img, (170, 170), (178, 182), 255, -1)
    text_mask, _, _ = separate_text_and_graphics(img)
    assert np.count_nonzero(text_mask) == 0


def test_text_separation_collinear_required(tmp_path):
    """Blobs spaced as a line are grouped; a stray off-line blob is excluded."""
    img = np.zeros((200, 200), dtype=np.uint8)
    for x in range(20, 140, 15):
        cv2.rectangle(img, (x, 50), (x + 8, 62), 255, -1)
    cv2.rectangle(img, (100, 150), (108, 162), 255, -1)   # off-line stray
    text_mask, _, _ = separate_text_and_graphics(img)
    assert text_mask[156, 104] == 0, "off-baseline blob wrongly grouped as text"
    assert np.count_nonzero(text_mask) > 0


def test_text_separation_elongated_layer():
    """Elongated small blobs go to the 3rd (elongated) layer, not text."""
    img = np.zeros((200, 200), dtype=np.uint8)
    # Draw 5 narrow horizontal dash-like blobs at the same y (elongation ≥ T4=2)
    for x in range(10, 110, 20):
        cv2.rectangle(img, (x, 100), (x + 14, 103), 255, -1)  # 15×4: ratio ~3.75
    _, _, elong = separate_text_and_graphics(img)
    assert np.count_nonzero(elong) > 0, "dash-like elongated blobs not in elongated layer"


def test_polarity_normalization_black_on_white(tmp_path):
    """Black strokes on white background → foreground normalised to 255."""
    img = np.full((120, 120), 255, dtype=np.uint8)  # white background
    cv2.line(img, (10, 60), (110, 60), 0, 2)         # black line
    path = str(tmp_path / "bow.png")
    cv2.imwrite(path, img)
    _, _gray, binary = load_and_preprocess(path)
    # Foreground (the line) must be the white minority, not the page.
    assert np.count_nonzero(binary) < binary.size / 2
    assert np.count_nonzero(binary) > 0


def test_skeleton_no_page_border_black_on_white(tmp_path):
    """Skeleton mode must trace the actual stroke, not the page border."""
    img = np.full((120, 120), 255, dtype=np.uint8)
    cv2.line(img, (10, 60), (110, 60), 0, 2)
    path = str(tmp_path / "bow.png")
    cv2.imwrite(path, img)
    _, _gray, binary = load_and_preprocess(path)
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
    """A thick (10px) horizontal line should produce geometry near y≈60."""
    img = np.zeros((120, 300), dtype=np.uint8)
    cv2.line(img, (10, 60), (290, 60), 255, 10)   # thick white line
    path = str(tmp_path / "thick.png")
    cv2.imwrite(path, img)
    _, _gray, binary = load_and_preprocess(path)
    lines, contours = extract_lines_and_contours(
        binary, min_line_length=30,
    )
    # Contour-first pipeline: thick line produces outline contours near y≈60
    all_y = [y for (_, y1, _, y2) in lines for y in (y1, y2)]
    all_y += [pt[1] for c in contours for pt in c]
    assert all_y, "No geometry detected on thick line"
    # All geometry should be in the stripe 50-70, not scattered across the image
    assert all(45 <= y <= 75 for y in all_y), \
        f"Geometry outside expected band: {sorted(set(all_y))}"


def test_pre_close_kernel_runs_and_produces_geometry(tmp_path):
    """--pre-close-kernel runs without error and still yields geometry."""
    img = np.zeros((120, 300), dtype=np.uint8)
    cv2.rectangle(img, (10, 50), (290, 70), 255, -1)  # filled rectangle (thick line)
    path = str(tmp_path / "rect.png")
    cv2.imwrite(path, img)
    _, gray, binary = load_and_preprocess(path)
    l0, c0 = extract_lines_and_contours(binary, gray=gray, min_line_length=30,
                                        pre_close_kernel=0)
    l7, c7 = extract_lines_and_contours(binary, gray=gray, min_line_length=30,
                                        pre_close_kernel=7)
    # Both settings must produce at least one entity and not crash.
    assert len(l0) + len(c0) >= 1
    assert len(l7) + len(c7) >= 1


def test_adaptive_block_oversized(tmp_path):
    """Oversized adaptive block size is clamped, not crashed."""
    img = np.zeros((50, 50), dtype=np.uint8)
    cv2.rectangle(img, (5, 5), (45, 45), 255, 2)
    path = str(tmp_path / "small.png")
    cv2.imwrite(path, img)
    # Should not raise cv2.error
    _, _gray, binary = load_and_preprocess(path, threshold_method="adaptive",
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
    _, _gray, binary = load_and_preprocess(path)  # defaults: morph=none, despeckle=True
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
    _, _gray, binary = load_and_preprocess(square_image_path)
    lines, contours = extract_lines_and_contours(binary, approx_epsilon=3.0)
    count = export_to_dxf(lines, contours, dxf_out, image_height=binary.shape[0])
    assert count >= 0  # may be 0 for a simple image — just must not crash


# ── Arc / circle (DXF Section 5) ───────────────────────────────────────────────

def test_fit_circle_recovers_known_circle():
    """Kåsa circle fit recovers a known centre and radius."""
    cx0, cy0, r0 = 100.0, 80.0, 40.0
    ang = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    pts = np.column_stack([cx0 + r0 * np.cos(ang), cy0 + r0 * np.sin(ang)])
    cx, cy, r, resid = _fit_circle(pts)
    assert abs(cx - cx0) < 1e-6 and abs(cy - cy0) < 1e-6
    assert abs(r - r0) < 1e-6
    assert resid < 1e-6


def test_bulge_quarter_circle():
    """A 90° CCW arc has bulge tan(90°/4) = tan(22.5°) ≈ 0.4142."""
    # start (1,0) → mid (cos45,sin45) → end (0,1), centred at origin, CCW
    s = (1.0, 0.0)
    m = (np.cos(np.pi / 4), np.sin(np.pi / 4))
    e = (0.0, 1.0)
    b = _bulge_from_3pts(s, m, e)
    assert abs(b - np.tan(np.pi / 8)) < 1e-6, b



def test_circle_detected_and_exported(tmp_path, dxf_out):
    """A drawn circle is detected as an arc primitive and exported as CIRCLE."""
    img = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(img, (150, 150), 80, 255, 2)
    path = str(tmp_path / "circle.png")
    cv2.imwrite(path, img)
    _, gray, binary = load_and_preprocess(path)
    lines, contours, arcs = extract_lines_and_contours(
        binary, gray=gray, return_arcs=True,
    )
    assert any(a["type"] == "circle" for a in arcs), f"no circle detected: {arcs}"
    count = export_to_dxf(lines, contours, dxf_out,
                          image_height=binary.shape[0], arcs=arcs)
    assert count >= 1
    doc = ezdxf.readfile(dxf_out)
    assert len(list(doc.audit().errors)) == 0
    assert any(e.dxftype() == "CIRCLE" for e in doc.modelspace())


# ── Zhang-Suen thinning ────────────────────────────────────────────────────────

def test_zhang_suen_reduces_thick_line_to_1px():
    """Zhang-Suen must thin a 10px-wide line to a 1px skeleton."""
    img = np.zeros((60, 200), dtype=np.uint8)
    cv2.rectangle(img, (10, 25), (190, 35), 255, -1)  # 11px thick
    thinned = _zhang_suen_thin(img)
    # Column widths across the thick region should be 1
    col = thinned[10:50, 100]  # vertical slice at centre
    assert col.sum() // 255 == 1, f"skeleton is wider than 1 px: {col.sum()//255}"


def test_zhang_suen_preserves_connectivity():
    """A closed rectangle skeleton must remain connected (no broken corners)."""
    img = np.zeros((80, 80), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (70, 70), 255, 3)
    thinned = _zhang_suen_thin(img)
    n_labels, *_ = cv2.connectedComponentsWithStats(thinned, connectivity=8)
    assert n_labels == 2, f"skeleton broke into {n_labels - 1} component(s)"


# ── SPV stroke-width estimation ───────────────────────────────────────────────

def test_stroke_width_thin_line():
    """A 1-px line must round to a lighter DXF weight than a 5-px line."""
    img1 = np.zeros((50, 200), dtype=np.uint8)
    cv2.line(img1, (5, 25), (195, 25), 255, 1)
    img5 = np.zeros((50, 200), dtype=np.uint8)
    cv2.line(img5, (5, 25), (195, 25), 255, 5)
    w1 = estimate_line_widths(img1, [(5, 25, 195, 25)], dpi=96.0)[0]
    w5 = estimate_line_widths(img5, [(5, 25, 195, 25)], dpi=96.0)[0]
    assert w1 < w5, f"1px ({w1}) not lighter than 5px ({w5})"
    assert w1 in _DXF_WEIGHTS


def test_stroke_width_thick_line_heavier():
    """A 10-px line must yield a heavier DXF weight than a 1-px line."""
    img = np.zeros((50, 200), dtype=np.uint8)
    cv2.line(img, (5, 25), (195, 25), 255, 10)
    w_thick = estimate_line_widths(img, [(5, 25, 195, 25)], dpi=96.0)[0]
    img2 = np.zeros((50, 200), dtype=np.uint8)
    cv2.line(img2, (5, 25), (195, 25), 255, 1)
    w_thin = estimate_line_widths(img2, [(5, 25, 195, 25)], dpi=96.0)[0]
    assert w_thick > w_thin, f"thick {w_thick} not heavier than thin {w_thin}"


def test_stroke_width_written_to_dxf(tmp_path):
    """Lineweight attribute is accepted by ezdxf without audit errors."""
    img = np.zeros((50, 200), dtype=np.uint8)
    cv2.line(img, (5, 25), (195, 25), 255, 5)
    lines = [(5, 25, 195, 25)]
    weights = estimate_line_widths(img, lines, dpi=96.0)
    out = str(tmp_path / "lw.dxf")
    export_to_dxf(lines, [], out, image_height=50, line_weights=weights)
    doc = ezdxf.readfile(out)
    assert len(list(doc.audit().errors)) == 0
    ent = list(doc.modelspace())[0]
    assert ent.dxf.lineweight == weights[0]


# ─────────────────────────────────────────────────────────────────────────────

def test_return_arcs_false_keeps_two_tuple():
    """Default 2-tuple API is preserved (no arc diversion)."""
    img = _make_square()
    binary = img  # already white-on-black foreground
    result = extract_lines_and_contours(binary)
    assert isinstance(result, tuple) and len(result) == 2


def test_layer_names(square_image_path, dxf_out):
    """DXF output contains the expected layer names."""
    _, _gray, binary = load_and_preprocess(square_image_path)
    lines, contours = extract_lines_and_contours(binary, min_line_length=30)
    export_to_dxf(lines, contours, dxf_out, image_height=binary.shape[0])
    doc = ezdxf.readfile(dxf_out)
    layer_names = {layer.dxf.name for layer in doc.layers}
    assert "LINES" in layer_names
    assert "CONTOURS" in layer_names
    assert "TEXT_CANDIDATES" in layer_names
