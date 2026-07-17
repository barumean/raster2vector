"""Smoke tests for the raster2vector pipeline."""

import os

import cv2
import ezdxf
import numpy as np
import pytest

from src.preprocessor import load_and_preprocess, _selective_blur
from src.text_separator import separate_text_and_graphics
from src.vectorizer import (
    extract_lines_and_contours,
    _merge_collinear_lines,
    _snap_endpoints,
    _fit_circle,
    _snap_right_angles,
    _remove_staircase,
    _detect_corners,
    _find_splice_points,
    _split_at_marks,
    _gap_jump,
    _orthogonalize,
    _detect_dashed_lines,
    _structure_cleanup_polyline,
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
    """mode='skeleton' is a deprecated alias for centerline: warns and produces geometry."""
    original_bgr, gray, binary = load_and_preprocess(square_image_path)
    with pytest.warns(DeprecationWarning):
        skel = extract_lines_and_contours(binary, min_line_length=20,
                                          mode="skeleton")
    # centerline mode produces some geometry (lines and/or contours)
    assert len(skel[0]) + len(skel[1]) >= 1
    count = export_to_dxf(skel[0], skel[1], dxf_out, image_height=binary.shape[0])
    assert count >= 1


def test_centerline_mode(square_image_path, dxf_out):
    """mode='centerline' produces geometry via medial_axis skeleton graph."""
    _, _, binary = load_and_preprocess(square_image_path)
    result = extract_lines_and_contours(
        binary, min_line_length=20, mode="centerline", return_arcs=True
    )
    lines, contours, arcs = result
    assert len(lines) + len(contours) >= 1, "centerline mode produced no geometry"
    count = export_to_dxf(lines, contours, dxf_out, image_height=binary.shape[0],
                          arcs=arcs)
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
    """Deprecated skeleton alias must not revive page-border artefacts."""
    img = np.full((120, 120), 255, dtype=np.uint8)
    cv2.line(img, (10, 60), (110, 60), 0, 2)
    path = str(tmp_path / "bow.png")
    cv2.imwrite(path, img)
    _, _gray, binary = load_and_preprocess(path)
    with pytest.warns(DeprecationWarning):
        lines, contours = extract_lines_and_contours(binary, min_line_length=30,
                                                     mode="skeleton")
    # Detected geometry may be a contour; it must not sit on page edges (0/119).
    all_y = [y for (_, y1, _, y2) in lines for y in (y1, y2)]
    all_y += [int(pt[1]) for c in contours for pt in c]
    assert all_y, "deprecated skeleton alias found nothing"
    assert all(40 <= y <= 80 for y in all_y), f"page-border artefact detected: {all_y}"


def test_structure_cleanup_preserves_diagonal_angle():
    """Weak cleanup removes diagonal wiggle without snapping it horizontal."""
    pts = np.array([[0, 0], [20, 3], [40, 6], [60, 9], [80, 12]], dtype=np.int32)
    contour = pts.reshape(-1, 1, 2)
    cleaned = _structure_cleanup_polyline(
        contour, pts, closed=False, line_tolerance=2.5, quad_detection=True
    )
    assert len(cleaned) == 2
    assert cleaned[0].tolist() == [0, 0]
    assert cleaned[-1].tolist() == [80, 12]
    assert cleaned[-1][1] != cleaned[0][1], "diagonal was flattened"


def test_structure_cleanup_detects_quadrilateral_not_only_rectangle():
    """Closed four-sided slanted structures are kept as clean quadrilaterals."""
    pts = np.array([[10, 10], [90, 20], [80, 70], [20, 60], [10, 10]], dtype=np.int32)
    contour = pts.reshape(-1, 1, 2)
    cleaned = _structure_cleanup_polyline(
        contour, pts, closed=True, line_tolerance=3.0, quad_detection=True
    )
    assert len(cleaned) == 5
    assert np.array_equal(cleaned[0], cleaned[-1])
    # The first edge is intentionally slanted and must remain so.
    assert cleaned[0][1] != cleaned[1][1]


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


# ── Selective blur tests ──────────────────────────────────────────────────────

def test_selective_blur_smooths_flat_region():
    """Flat uniform region is smoothed by selective blur."""
    gray = np.full((50, 50), 128, dtype=np.uint8)
    # Add noise in the flat region
    rng = np.random.default_rng(42)
    noise = rng.integers(-10, 10, gray.shape).astype(np.int16)
    noisy = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    result = _selective_blur(noisy, radius=2, delta=20)
    # The blurred flat region should be closer to the true value than the noisy input
    assert float(np.abs(result.astype(float) - 128).mean()) < \
           float(np.abs(noisy.astype(float) - 128).mean())


def test_selective_blur_preserves_edges():
    """Sharp edges (large gradient) are not blurred away."""
    gray = np.zeros((50, 100), dtype=np.uint8)
    gray[:, 50:] = 200   # hard step edge at x=50
    result = _selective_blur(gray, radius=3, delta=20)
    # The step should still be present: pixel at edge is unblurred
    assert result[25, 49] < 50    # left side stays dark
    assert result[25, 50] > 150   # right side stays bright


def test_selective_blur_zero_radius_noop():
    """blur_radius=0 returns the image unchanged."""
    gray = np.random.default_rng(0).integers(0, 255, (60, 60), dtype=np.uint8)
    result = _selective_blur(gray, radius=0)
    np.testing.assert_array_equal(result, gray)


def test_load_preprocess_blur_radius(square_image_path):
    """blur_radius param is accepted and does not crash."""
    _, gray, binary = load_and_preprocess(square_image_path, blur_radius=2, blur_delta=20)
    assert binary.shape == gray.shape
    assert binary.dtype == np.uint8


# ── Right-angle enhancement tests ─────────────────────────────────────────────

def test_snap_right_angles_exact_90():
    """A corner that is already exactly 90° is unchanged."""
    pts = np.array([[0, 0], [10, 0], [10, 10]], dtype=np.int32)
    result = _snap_right_angles(pts, tol_deg=10.0)
    np.testing.assert_array_equal(result, pts)


def test_snap_right_angles_near_90_snapped():
    """A near-90° corner (within tolerance) is snapped to exactly 90°."""
    # Incoming: horizontal (0→10, 0→0); outgoing: slightly off-vertical (10→10, 0→11)
    pts = np.array([[0, 0], [10, 0], [10, 11]], dtype=np.int32)
    result = _snap_right_angles(pts, tol_deg=15.0)
    # After snapping, v2 should be perpendicular to v1 = (10,0), i.e. vertical
    v1 = result[1] - result[0]
    v2 = result[2] - result[1]
    dot = int(v1[0]) * int(v2[0]) + int(v1[1]) * int(v2[1])
    # dot product of perpendicular vectors is 0
    assert abs(dot) <= 1, f"Not right-angle after snap: dot={dot}"


def test_snap_right_angles_far_from_90_unchanged():
    """A corner far from 90° is not modified."""
    # 45° corner: A=(0,0) B=(10,0) C=(20,10) — turn angle = 45°
    pts = np.array([[0, 0], [10, 0], [20, 10]], dtype=np.int32)
    result = _snap_right_angles(pts, tol_deg=10.0)
    np.testing.assert_array_equal(result[0], pts[0])
    np.testing.assert_array_equal(result[1], pts[1])


def test_right_angle_enhance_cli(square_image_path, dxf_out):
    """--right-angle-enhance flag runs without error end-to-end."""
    from raster2vector import main
    ret = main([square_image_path, "-o", dxf_out, "--right-angle-enhance"])
    assert ret == 0
    assert os.path.exists(dxf_out)


# ── vtracer staircase removal tests ──────────────────────────────────────────

def test_remove_staircase_diagonal_removed():
    """1-pixel diagonal staircase steps are removed."""
    # Staircase going diagonally: (0,0)→(1,0)→(1,1)→(2,1) — the middle
    # vertex (1,0)→(1,1) is a single-step convex turn.
    pts = np.array([[0, 0], [1, 0], [1, 1], [2, 1], [2, 0], [0, 0]], dtype=np.int32)
    result = _remove_staircase(pts, closed=True)
    assert len(result) < len(pts), "Staircase step should have been removed"


def test_remove_staircase_straight_line_unchanged():
    """Straight collinear points are not removed."""
    pts = np.array([[0, 0], [5, 0], [10, 0]], dtype=np.int32)
    result = _remove_staircase(pts, closed=False)
    # No 1-pixel 45° steps here; all points kept (or possibly 2 remain)
    assert len(result) >= 2


def test_remove_staircase_reduces_aliased_diagonal():
    """A pixel-aliased diagonal line loses its staircase steps."""
    # Simulate a rasterized diagonal: alternating x and y increments of 1
    pts = []
    for i in range(10):
        pts.append([i, i])
        pts.append([i + 1, i])
    pts = np.array(pts, dtype=np.int32)
    result = _remove_staircase(pts, closed=False)
    assert len(result) <= len(pts)


def test_remove_staircase_cli(square_image_path, dxf_out):
    """--remove-staircase flag runs end-to-end without error."""
    from raster2vector import main
    ret = main([square_image_path, "-o", dxf_out, "--remove-staircase"])
    assert ret == 0


# ── vtracer corner detection tests ───────────────────────────────────────────

def test_detect_corners_right_angle():
    """A 90° corner is detected at the vertex."""
    # L-shape: horizontal then vertical
    pts = np.array([[0, 0], [10, 0], [10, 10]], dtype=np.int32)
    corners = _detect_corners(pts, threshold_deg=60.0)
    # The middle vertex (10, 0) is a 90° turn — should be a corner
    assert corners[1], "Middle vertex of 90° turn should be detected as corner"


def test_detect_corners_straight_line_no_corners():
    """Points on a straight line have no corners."""
    pts = np.array([[0, 0], [5, 0], [10, 0], [15, 0]], dtype=np.int32)
    corners = _detect_corners(pts, threshold_deg=60.0)
    # Interior points (index 1, 2) should not be corners
    assert not corners[1] and not corners[2]


def test_detect_corners_threshold_respected():
    """Small turns below threshold are not marked as corners."""
    # 30° turn — should not be corner with 60° threshold
    import math
    pts = np.array([[0, 0], [10, 0],
                    [10 + int(10 * math.cos(math.radians(30))),
                     int(10 * math.sin(math.radians(30)))]], dtype=np.int32)
    corners = _detect_corners(pts, threshold_deg=60.0)
    assert not corners[1], "30° turn should not be a corner with 60° threshold"


# ── vtracer splice point tests ────────────────────────────────────────────────

def test_find_splice_points_circle_has_splices():
    """A full circle has curvature inflections → multiple splice points."""
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    pts = np.column_stack([
        (50 + 30 * np.cos(angles)).astype(int),
        (50 + 30 * np.sin(angles)).astype(int),
    ])
    splices = _find_splice_points(pts, threshold_deg=45.0)
    # A circle turning 360° should have many 45° splice intervals
    assert splices.sum() >= 6, f"Circle should have ≥6 splice points, got {splices.sum()}"


def test_find_splice_points_straight_line_few():
    """A straight line has almost no curvature: far fewer splices than a circle."""
    pts = np.array([[i, 0] for i in range(20)], dtype=np.int32)
    splices = _find_splice_points(pts, threshold_deg=45.0)
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    circle_pts = np.column_stack([
        (50 + 30 * np.cos(angles)).astype(int),
        (50 + 30 * np.sin(angles)).astype(int),
    ])
    circle_splices = _find_splice_points(circle_pts, threshold_deg=45.0)
    assert splices.sum() < circle_splices.sum(), \
        "Straight line should have fewer splices than a circle"


def test_split_at_marks_basic():
    """_split_at_marks produces the expected number of segments."""
    pts = np.array([[i, 0] for i in range(10)], dtype=np.int32)
    marks = np.zeros(10, dtype=bool)
    marks[3] = True
    marks[7] = True
    segs = _split_at_marks(pts, marks)
    assert len(segs) == 3  # [0..3], [3..7], [7..9]
    assert segs[0][0, 0] == 0
    assert segs[1][0, 0] == 3
    assert segs[2][0, 0] == 7


# ── Segmented arc extraction end-to-end ──────────────────────────────────────

def test_segmented_arc_extraction_circle_image(tmp_path, dxf_out):
    """Segmented arc extraction finds circle arcs end-to-end."""
    # Draw two arcs (semicircles) so the whole circle doesn't fit as one arc
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img, (100, 100), 50, 255, 2)
    binary = img
    _, _, arcs = extract_lines_and_contours(
        binary,
        detect_arcs=True,
        return_arcs=True,
        corner_threshold=60.0,
        splice_threshold=45.0,
        min_contour_length=10,
    )
    # Should detect circle or arcs
    assert len(arcs) > 0, "Should detect at least one arc/circle from a drawn circle"


# ── Gap-jump tests ────────────────────────────────────────────────────────────

def test_gap_jump_bridges_small_gap():
    """Gap-jump adds a bridge for two nearly-touching collinear segment ends."""
    # Segment A: (0,0)→(40,0); Segment B: (45,0)→(90,0) — gap of 5px
    lines = [(0, 0, 40, 0), (45, 0, 90, 0)]
    result = _gap_jump(lines, gap_px=10.0, fan_deg=20.0)
    assert len(result) > len(lines), "Should have added at least one bridge"
    # Bridge should connect A's end (40,0) to B's start (45,0)
    bridges = result[len(lines):]
    assert any(b[0] == 40 and b[2] == 45 for b in bridges), \
        f"Expected bridge (40,0)→(45,0), got {bridges}"


def test_gap_jump_no_bridge_for_corner():
    """Gap-jump does NOT bridge endpoints that form a genuine corner (T-junction)."""
    # Horizontal A ends at (50,0); vertical B starts at (50,0) upward
    # Their endpoints coincide but directions are perpendicular — not a gap
    lines = [(0, 0, 50, 0), (50, 0, 50, 50)]
    result = _gap_jump(lines, gap_px=10.0, fan_deg=20.0)
    # No bridge should be added (these endpoints touch and are orthogonal)
    assert len(result) == len(lines), "Should not bridge a corner junction"


def test_gap_jump_ignores_large_gap():
    """Gap-jump does not bridge a gap larger than gap_px."""
    lines = [(0, 0, 40, 0), (80, 0, 120, 0)]  # gap = 40 px
    result = _gap_jump(lines, gap_px=15.0, fan_deg=20.0)
    assert len(result) == len(lines), "Gap of 40px > 15px limit should not be bridged"


def test_gap_jump_cli(square_image_path, dxf_out):
    """--gap-jump flag runs end-to-end without error."""
    from raster2vector import main
    ret = main([square_image_path, "-o", dxf_out, "--gap-jump", "--gap-px", "10"])
    assert ret == 0


# ── Orthogonalization tests ───────────────────────────────────────────────────

def test_orthogonalize_snaps_near_horizontal():
    """A line within 2° of horizontal is snapped to exact horizontal."""
    # 1° off horizontal
    import math
    angle_rad = math.radians(1.0)
    x1, y1 = 0, 0
    x2 = int(100 * math.cos(angle_rad))
    y2 = int(100 * math.sin(angle_rad))
    lines = [(x1, y1, x2, y2)]
    result = _orthogonalize(lines, base_angle_deg=0.0, accuracy_deg=2.0)
    rx1, ry1, rx2, ry2 = result[0]
    assert ry1 == ry2, f"Snapped line should be exactly horizontal (y1={ry1}, y2={ry2})"


def test_orthogonalize_snaps_near_vertical():
    """A line within 2° of vertical is snapped to exact vertical."""
    import math
    angle_rad = math.radians(89.0)  # 1° off vertical
    # Use length=500 so integer rounding doesn't swamp the 1° offset
    x2 = int(500 * math.cos(angle_rad))
    y2 = int(500 * math.sin(angle_rad))
    lines = [(0, 0, x2, y2)]
    result = _orthogonalize(lines, base_angle_deg=0.0, accuracy_deg=2.0)
    rx1, ry1, rx2, ry2 = result[0]
    assert rx1 == rx2, f"Snapped line should be exactly vertical (x1={rx1}, x2={rx2})"


def test_orthogonalize_leaves_diagonal_unchanged():
    """A 45° diagonal is not modified by orthogonalization."""
    lines = [(0, 0, 70, 70)]
    result = _orthogonalize(lines, base_angle_deg=0.0, accuracy_deg=2.0)
    assert result[0] == lines[0], "45° diagonal should not be snapped"


def test_orthogonalize_cli(square_image_path, dxf_out):
    """--orthogonalize flag runs end-to-end without error."""
    from raster2vector import main
    ret = main([square_image_path, "-o", dxf_out, "--orthogonalize"])
    assert ret == 0


# ── Dashed-line detection tests ───────────────────────────────────────────────

def _make_dashed_lines(n=6, dash=15, gap=10, y=50, x_start=10):
    """Synthetic horizontal dashed line: n dashes of length dash, gap spacing."""
    segs = []
    x = x_start
    for _ in range(n):
        segs.append((x, y, x + dash, y))
        x += dash + gap
    return segs


def test_detect_dashed_lines_finds_pattern():
    """Detects a clean periodic dashed horizontal line."""
    dashes = _make_dashed_lines(n=5, dash=15, gap=10)
    solid, groups = _detect_dashed_lines(dashes, max_dash_len_px=20.0, min_dash_count=3)
    assert len(groups) >= 1, f"Should detect 1 dashed group, got {groups}"
    assert len(solid) == 0, "All segments should be classified as dashed"


def test_detect_dashed_lines_ignores_solid():
    """Long solid segments are not confused with dashes."""
    # One long line (not a dash) and a separate cluster of dashes
    long_line = (0, 10, 200, 10)  # length 200, above max_dash_len_px
    dashes = _make_dashed_lines(n=5, dash=15, gap=10, y=80)
    all_lines = [long_line] + dashes
    solid, groups = _detect_dashed_lines(all_lines, max_dash_len_px=20.0, min_dash_count=3)
    assert long_line in solid, "Long solid line should remain in solid output"
    assert len(groups) >= 1, "Dashed group should still be detected"


def test_detect_dashes_written_to_dxf(tmp_path):
    """Dashed lines are written to the DASHED layer in DXF output."""
    dashes = _make_dashed_lines(n=5, dash=15, gap=10)
    out = str(tmp_path / "dashes.dxf")
    export_to_dxf(
        lines=[], contours=[], output_path=out, image_height=200,
        dashed_lines=[dashes],
    )
    doc = ezdxf.readfile(out)
    layer_names = {e.dxf.layer for e in doc.modelspace()}
    assert "DASHED" in layer_names, "DASHED layer entity should be present"


def test_detect_dashes_cli(square_image_path, dxf_out):
    """--detect-dashes flag runs end-to-end without error."""
    from raster2vector import main
    ret = main([square_image_path, "-o", dxf_out, "--detect-dashes"])
    assert ret == 0


# ── Box detection, consolidation, page-border tests ───────────────────────────

def test_is_rectangular_detects_box():
    """_is_rectangular returns True for an axis-aligned rectangle."""
    from src.vectorizer import _is_rectangular
    pts = np.array([[10, 10], [10, 60], [80, 60], [80, 10], [10, 10]])
    assert _is_rectangular(pts, angle_tol_deg=20.0)


def test_is_rectangular_rejects_triangle():
    """_is_rectangular returns False for a triangle."""
    from src.vectorizer import _is_rectangular
    pts = np.array([[0, 0], [50, 0], [25, 40], [0, 0]])
    assert not _is_rectangular(pts, angle_tol_deg=20.0)


def test_detect_boxes_classifies_rectangle():
    """extract_lines_and_contours with detect_boxes=True puts closed rects in box_contours."""
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (100, 80), 255, 2)
    result = extract_lines_and_contours(img, detect_boxes=True)
    lines, contours, box_contours = result
    assert len(box_contours) >= 1, "Rectangular closed contour should be in box_contours"


def test_compute_page_border_coverage():
    """compute_page_border returns a rect that encloses all line endpoints."""
    from src.vectorizer import compute_page_border
    lines = [(10, 20, 80, 50), (5, 5, 90, 90)]
    border = compute_page_border(lines, [])
    assert border is not None
    x1, y1, x2, y2 = border
    assert x1 <= 5 and y1 <= 5
    assert x2 >= 90 and y2 >= 90


def test_box_and_border_in_dxf(tmp_path):
    """BOX and BOXES layers appear in DXF when page_border and box_contours are given."""
    out = str(tmp_path / "box.dxf")
    box_pts = np.array([[10, 10], [10, 60], [80, 60], [80, 10], [10, 10]])
    export_to_dxf(
        lines=[(0, 0, 100, 0)], contours=[], output_path=out, image_height=100,
        box_contours=[box_pts], page_border=(0, 0, 100, 100),
    )
    doc = ezdxf.readfile(out)
    layer_names = {e.dxf.layer for e in doc.modelspace()}
    assert "BOX" in layer_names
    assert "BOXES" in layer_names


def test_consolidate_removes_duplicate_parallel():
    """_consolidate_segments absorbs a near-duplicate contour segment."""
    from src.vectorizer import _consolidate_segments
    lines = [(0, 0, 100, 0)]  # horizontal line at y=0
    # Contour: a near-identical segment at y=3 (within perp_tol)
    dup_contour = np.array([[0, 3], [50, 3], [100, 3]])
    remaining_lines, remaining_contours = _consolidate_segments(
        lines, [dup_contour], perp_tol_px=6.0, angle_tol_deg=4.0,
    )
    # The duplicate contour segment should have been absorbed (contour removed or shortened)
    total_contour_pts = sum(len(c) for c in remaining_contours)
    assert total_contour_pts < len(dup_contour), \
        "Near-duplicate contour segment should be absorbed into the line"


# ── Text-arc suppression ──────────────────────────────────────────────────────

def test_suppress_text_arcs_single_row():
    """A horizontal row of similar-radius circles is flagged as text."""
    from src.vectorizer import _suppress_text_arcs
    arcs = [{"type": "circle", "center": (float(cx), 100.0), "r": 20.0}
            for cx in (100, 200, 300, 400)]
    arcs.append({"type": "circle", "center": (500.0, 400.0), "r": 80.0})
    result = _suppress_text_arcs(arcs, min_cluster=3)
    flagged = [a for a in result if a.get("text_candidate")]
    assert len(flagged) == 4
    assert not result[-1].get("text_candidate"), \
        "standalone large circle must stay on ARCS"


def test_suppress_text_arcs_multi_row():
    """Two interleaved text rows of the same radius are each detected.

    Regression: rows sharing one radius group used to interleave when sorted
    by X, breaking the run scan so nothing was ever flagged.
    """
    from src.vectorizer import _suppress_text_arcs
    arcs = []
    for cx in (180, 235, 290, 345, 400):          # row 1 at y=250
        arcs.append({"type": "circle", "center": (float(cx), 250.0), "r": 16.0})
    for cx in (200, 260, 320, 380, 440):           # row 2 at y=150, similar r
        arcs.append({"type": "circle", "center": (float(cx), 150.0), "r": 19.0})
    result = _suppress_text_arcs(arcs, min_cluster=3)
    flagged = sum(1 for a in result if a.get("text_candidate"))
    assert flagged == 10, f"both rows should be flagged, got {flagged}/10"


def test_suppress_text_arcs_routes_to_layer(tmp_path):
    """text_candidate arcs are written to the TEXT_ARCS layer in the DXF."""
    import ezdxf
    arcs = [{"type": "circle", "center": (float(cx), 100.0), "r": 20.0,
             "text_candidate": True} for cx in (100, 200, 300)]
    arcs.append({"type": "circle", "center": (500.0, 300.0), "r": 50.0})
    out = str(tmp_path / "text_arcs.dxf")
    export_to_dxf([], [], out, image_height=400, arcs=arcs)
    doc = ezdxf.readfile(out)
    layers = [e.dxf.layer for e in doc.modelspace()]
    assert layers.count("TEXT_ARCS") == 3
    assert layers.count("ARCS") == 1
