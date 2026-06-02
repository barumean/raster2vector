#!/usr/bin/env python3
"""raster2vector — Convert B&W raster images to DXF vector drawings."""

import argparse
import os
import sys
from typing import Optional

import cv2
import numpy as np

from src.preprocessor import load_and_preprocess
from src.text_separator import separate_text_and_graphics
from src.vectorizer import extract_lines_and_contours, compute_page_border
from src.dxf_exporter import export_to_dxf
from src.stroke_width import estimate_line_widths, estimate_contour_widths


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="raster2vector",
        description="Convert a black-and-white raster image to a DXF vector file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("image", help="Input raster image (PNG, JPG, BMP, …)")
    p.add_argument("-o", "--output", default=None,
                   help="Output DXF path. Defaults to <input_stem>.dxf")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    pre = p.add_argument_group("preprocessing")
    pre.add_argument("--threshold-method", choices=["otsu", "adaptive", "sauvola"],
                     default="otsu", metavar="METHOD",
                     help="Binarisation method: otsu, adaptive, or sauvola. "
                          "sauvola (Sauvola 1999) is best for scanned drawings "
                          "with uneven illumination or fold shadows.")
    pre.add_argument("--threshold", type=int, default=None, metavar="0-255",
                     help="Manual threshold value; overrides --threshold-method")
    pre.add_argument("--invert", action="store_true",
                     help="Invert image before processing (white-on-black drawings)")
    pre.add_argument("--morph", choices=["none", "open", "close"], default="none",
                     help="Explicit morphology (opt-in; open/close can damage thin lines)")
    pre.add_argument("--morph-kernel", type=int, default=2, metavar="N",
                     help="Square kernel size for morphological op (pixels)")
    pre.add_argument("--no-despeckle", action="store_true",
                     help="Disable thin-line-safe removal of tiny isolated specks")
    pre.add_argument("--min-speckle-area", type=int, default=3, metavar="N",
                     help="Drop isolated components smaller than this area (pixels)")
    pre.add_argument("--adaptive-block-size", type=int, default=51, metavar="N",
                     help="Block size for adaptive thresholding (auto-clamped odd ≥ 3)")
    pre.add_argument("--adaptive-c", type=int, default=9, metavar="N",
                     help="Constant subtracted from mean in adaptive thresholding")
    pre.add_argument("--sauvola-window", type=int, default=25, metavar="N",
                     help="Local window size for Sauvola thresholding (≈ stroke size)")
    pre.add_argument("--sauvola-k", type=float, default=0.2, metavar="F",
                     help="Sauvola k parameter (0.2–0.5; lower → more foreground)")
    pre.add_argument("--deskew", action="store_true",
                     help="Auto-correct rotational skew before binarisation using "
                          "the dominant angle of horizontal line blobs")
    pre.add_argument("--blur-radius", type=int, default=0, metavar="N",
                     help="Edge-preserving Gaussian blur radius before thresholding "
                          "(0=off). Reduces scanner noise / JPEG ringing while "
                          "keeping hard edges sharp. Try 1–3 for scanned drawings. "
                          "(imagetracerjs selective blur, blurradius)")
    pre.add_argument("--blur-delta", type=int, default=20, metavar="N",
                     help="Intensity delta threshold for selective blur: pixels "
                          "where |original−blurred|>N are restored as edges. "
                          "Default 20. (imagetracerjs blurdelta)")

    # ── Vectorisation ─────────────────────────────────────────────────────────
    vec = p.add_argument_group("vectorisation")
    vec.add_argument("--mode", choices=["edge", "skeleton"], default="edge",
                     help="edge (default): contour-first + Canny on grayscale; "
                          "skeleton: deprecated, treated as edge")
    vec.add_argument("--pre-close-kernel", type=int, default=0, metavar="N",
                     help="Morphological closing before edge detection to fill thick "
                          "stroke interiors. 0=off. Try 5-15 for thick drawings.")
    vec.add_argument("--min-contour-length", type=float, default=15.0, metavar="F",
                     help="Minimum contour arc length in pixels (shorter discarded)")
    vec.add_argument("--min-contour-area", type=float, default=10.0, metavar="F",
                     help="Minimum contour bounding-box area in pixels (smaller discarded)")
    vec.add_argument("--max-line-deviation", type=float, default=2.0, metavar="F",
                     help="Max perpendicular deviation (px) to classify a simplified "
                          "contour as a straight LINE vs. LWPOLYLINE")
    vec.add_argument("--structure-cleanup", action="store_true",
                     help="Clean structure-like straight edges without forcing "
                          "horizontal/vertical angles")
    vec.add_argument("--structure-line-tolerance", type=float, default=2.5, metavar="F",
                     help="Pixel tolerance for weak structural cleanup")
    vec.add_argument("--no-quad-detection", action="store_true",
                     help="Disable closed four-sided contour cleanup")
    vec.add_argument("--no-hough", action="store_true",
                     help="Disable supplemental Hough line detection")
    vec.add_argument("--no-arcs", action="store_true",
                     help="Disable circle/arc fitting (export curves as polylines only)")
    vec.add_argument("--arc-tol", type=float, default=None, metavar="F",
                     help="Max RMS pixel residual for circle/arc fitting "
                          "(default: auto, 0.5%% of image diagonal)")
    vec.add_argument("--min-arc-radius", type=float, default=0.0, metavar="PX",
                     help="Minimum arc/circle radius in pixels; smaller fits are "
                          "discarded (useful to suppress text-character arcs, "
                          "e.g. --min-arc-radius 15)")
    vec.add_argument("--no-dedup-lines", action="store_true",
                     help="Disable near-duplicate line removal (keeps doubled lines "
                          "from thick strokes)")
    vec.add_argument("--min-line-length", type=int, default=80, metavar="PX",
                     help="Minimum Hough line segment length (supplemental only)")
    vec.add_argument("--max-gap", type=int, default=15, metavar="PX",
                     help="Maximum gap bridged inside a Hough segment (px)")
    vec.add_argument("--hough-threshold", type=int, default=30, metavar="N",
                     help="Accumulator threshold for HoughLinesP (lower = more lines)")
    vec.add_argument("--canny-low", type=int, default=50, metavar="N",
                     help="Lower Canny hysteresis threshold (edge mode only)")
    vec.add_argument("--canny-high", type=int, default=150, metavar="N",
                     help="Upper Canny hysteresis threshold (edge mode only)")
    vec.add_argument("--approx-epsilon", type=float, default=None, metavar="F",
                     help="Douglas-Peucker tolerance for polyline simplification "
                          "(default: auto = max(1.5, 0.3%% of image diagonal))")
    vec.add_argument("--snap-radius", type=float, default=4.0, metavar="F",
                     help="Snap endpoints within this distance (px) to the same "
                          "point; 0 to disable")
    vec.add_argument("--no-merge-lines", action="store_true",
                     help="Disable collinear Hough segment merging")
    vec.add_argument("--text-separation", action="store_true",
                     help="Separate text-like blobs onto the TEXT_CANDIDATES layer "
                          "(off by default; may misclassify small symbols)")
    vec.add_argument("--stroke-width", action="store_true",
                     help="Estimate stroke width per entity (SPV) and write DXF "
                          "lineweights so dimension lines vs. boundary lines differ")
    vec.add_argument("--right-angle-enhance", action="store_true",
                     help="Snap near-90° corners to exact right angles after "
                          "simplification. Improves output for architectural and "
                          "mechanical drawings with orthogonal geometry. "
                          "(imagetracerjs rightangleenhance)")
    vec.add_argument("--right-angle-tol", type=float, default=10.0, metavar="DEG",
                     help="Tolerance in degrees around 90° for right-angle snapping "
                          "(default 10°)")
    vec.add_argument("--remove-staircase", action="store_true",
                     help="Remove 1-pixel 45° staircase artefacts from raw contour "
                          "points before Douglas-Peucker simplification. Useful on "
                          "low-DPI scans with heavy pixel aliasing. "
                          "(vtracer: remove_staircase)")
    vec.add_argument("--corner-threshold", type=float, default=60.0, metavar="DEG",
                     help="Turn-angle threshold (degrees) for corner detection used "
                          "in segmented arc extraction. When a contour cannot be "
                          "fitted as a single arc, it is split at corners and "
                          "curvature inflections and arc fitting is re-attempted "
                          "per segment. Set to 0 to disable. Default 60°. "
                          "(vtracer: corner_threshold)")
    vec.add_argument("--splice-threshold", type=float, default=45.0, metavar="DEG",
                     help="Maximum angular span per arc segment for splice-point "
                          "detection (degrees). Prevents a single fitted arc from "
                          "spanning more than this arc angle. Default 45°. "
                          "(vtracer: splice_threshold)")
    vec.add_argument("--gap-jump", action="store_true",
                     help="Bridge pixel-level breaks between nearly-touching line "
                          "endpoints. Adds synthetic connector segments for pairs "
                          "within --gap-px whose directions align within --fan-angle. "
                          "Dramatically reduces disconnected segments in scanned "
                          "drawings with faded ink. (Scan2CAD: gap_jump)")
    vec.add_argument("--gap-px", type=float, default=15.0, metavar="PX",
                     help="Maximum gap distance to bridge with gap-jump (pixels, "
                          "default 15). Try 10–25 at 300 dpi.")
    vec.add_argument("--fan-angle", type=float, default=20.0, metavar="DEG",
                     help="Half-angle of gap-jump directional search cone (degrees, "
                          "default 20°). Prevents bridging genuine T/L corners.")
    vec.add_argument("--orthogonalize", action="store_true",
                     help="Snap lines within --ortho-accuracy of horizontal or "
                          "vertical to exact H/V. Eliminates small scanner-tilt "
                          "angle errors that break CAD trim/fill operations. "
                          "(Scan2CAD: orthogonal_snap)")
    vec.add_argument("--ortho-accuracy", type=float, default=2.0, metavar="DEG",
                     help="Angular tolerance for orthogonalization snap (degrees, "
                          "default 2°). (Scan2CAD: accuracy)")
    vec.add_argument("--ortho-base-angle", type=float, default=0.0, metavar="DEG",
                     help="Primary axis angle for orthogonalization (default 0° = "
                          "horizontal). Use with --deskew for non-standard drawings.")
    vec.add_argument("--detect-dashes", action="store_true",
                     help="Identify runs of collinear short segments forming dashed "
                          "or hidden-line patterns and emit them on a separate DASHED "
                          "layer with the DXF DASHED linetype. "
                          "(Scan2CAD: dash_line_identification)")
    vec.add_argument("--max-dash-len", type=float, default=40.0, metavar="PX",
                     help="Maximum segment length (pixels) to consider as a dash "
                          "candidate for dashed-line detection (default 40).")
    vec.add_argument("--no-detect-boxes", action="store_true",
                     help="Disable rectangular closed-contour classification "
                          "(keep all contours on the CONTOURS layer)")
    vec.add_argument("--box-angle-tol", type=float, default=20.0, metavar="DEG",
                     help="Max angle deviation from 0°/90° for a contour segment "
                          "to be considered part of a rectangle (default 20°)")
    vec.add_argument("--no-consolidate", action="store_true",
                     help="Disable cross-contour segment consolidation "
                          "(skip merging near-duplicate parallel segments)")
    vec.add_argument("--consolidate-perp-tol", type=float, default=6.0, metavar="PX",
                     help="Perpendicular distance tolerance for segment consolidation "
                          "(default 6 px)")
    vec.add_argument("--consolidate-angle-tol", type=float, default=4.0, metavar="DEG",
                     help="Angle tolerance for segment consolidation (default 4°)")
    vec.add_argument("--no-page-border", action="store_true",
                     help="Do not emit the outermost bounding rectangle on the BOX layer")
    vec.add_argument("--suppress-text-arcs", action="store_true",
                     help="Detect arc/circle clusters that look like text characters "
                          "(similar size, horizontal row) and route them to the "
                          "TEXT_ARCS layer instead of ARCS. Keeps engineering arcs "
                          "clean when the drawing contains annotation text.")
    vec.add_argument("--text-arc-min-cluster", type=int, default=3, metavar="N",
                     help="Minimum arcs in a row to be classified as text (default 3)")
    vec.add_argument("--text-arc-r-tol", type=float, default=0.4, metavar="F",
                     help="Radius similarity tolerance for text-arc clustering "
                          "(fraction of median radius, default 0.4)")
    vec.add_argument("--text-arc-y-tol", type=float, default=1.5, metavar="F",
                     help="Vertical band half-width for text-arc clustering "
                          "(multiple of avg radius, default 1.5)")
    vec.add_argument("--text-arc-x-gap", type=float, default=6.0, metavar="F",
                     help="Maximum X gap between consecutive text-arc centres "
                          "(multiple of avg radius, default 4.0)")

    # ── Output ────────────────────────────────────────────────────────────────
    out = p.add_argument_group("output")
    out.add_argument("--dpi", type=float, default=96.0,
                     help="Source image DPI for pixel→mm conversion")
    out.add_argument("--preview", action="store_true",
                     help="Save a debug PNG showing detected geometry")
    out.add_argument("--output-preview", default=None, metavar="PATH",
                     help="Path for preview PNG; defaults to <input_stem>_preview.png")
    out.add_argument("--verbose", action="store_true",
                     help="Print processing statistics")

    return p


# ── Preview renderer ──────────────────────────────────────────────────────────

def save_preview(
    original_bgr: np.ndarray,
    lines: list,
    contours: list,
    text_contours: list,
    preview_path: str,
    arcs: list | None = None,
) -> bool:
    canvas = original_bgr.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)      # red
    for c in contours:
        pts = c.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(canvas, [pts], False, (0, 255, 0), 1)         # green
    for arc in (arcs or []):                                        # magenta
        if arc.get("type") == "circle":
            cx, cy = arc["center"]
            cv2.circle(canvas, (int(round(cx)), int(round(cy))),
                       int(round(arc["r"])), (255, 0, 255), 2)
        elif arc.get("type") == "arc":
            for p in (arc["start"], arc["mid"], arc["end"]):
                cv2.circle(canvas, (int(round(p[0])), int(round(p[1]))),
                           3, (255, 0, 255), -1)
    for c in text_contours:
        pts = c.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(canvas, [pts], False, (0, 255, 255), 1)       # yellow
    ok = cv2.imwrite(preview_path, canvas)
    if not ok:
        print(f"Warning: failed to save preview to '{preview_path}'.", file=sys.stderr)
    return bool(ok)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate
    if args.threshold is not None and not (0 <= args.threshold <= 255):
        parser.error("--threshold must be between 0 and 255.")
    if args.dpi <= 0:
        parser.error("--dpi must be a positive number.")
    if args.morph_kernel < 1:
        parser.error("--morph-kernel must be >= 1.")
    if args.approx_epsilon is not None and args.approx_epsilon <= 0:
        parser.error("--approx-epsilon must be positive.")
    if args.structure_line_tolerance <= 0:
        parser.error("--structure-line-tolerance must be positive.")
    if args.adaptive_block_size < 3:
        parser.error("--adaptive-block-size must be >= 3.")
    if args.min_speckle_area < 0:
        parser.error("--min-speckle-area must be >= 0.")

    # Output path
    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.image))[0]
        args.output = stem + ".dxf"
    if os.path.exists(args.output):
        print(f"Warning: '{args.output}' already exists and will be overwritten.",
              file=sys.stderr)

    # ── 1. Preprocess ──────────────────────────────────────────────────────────
    try:
        original_bgr, gray, binary = load_and_preprocess(
            args.image,
            threshold_method=args.threshold_method,
            invert=args.invert,
            manual_threshold=args.threshold,
            morph=args.morph,
            morph_kernel=args.morph_kernel,
            adaptive_block_size=args.adaptive_block_size,
            adaptive_c=args.adaptive_c,
            sauvola_window_size=args.sauvola_window,
            sauvola_k=args.sauvola_k,
            despeckle=not args.no_despeckle,
            min_speckle_area=args.min_speckle_area,
            deskew=args.deskew,
            blur_radius=args.blur_radius,
            blur_delta=args.blur_delta,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except cv2.error as exc:
        print(f"Error: image processing failed — {exc}", file=sys.stderr)
        return 1

    h, w = binary.shape[:2]
    if args.verbose:
        print(f"Image: {w}x{h} px  ({args.image})")

    # ── 2. Text / graphics separation ─────────────────────────────────────────
    text_contours: list = []
    elongated_contours: list = []
    text_mask_img: Optional[np.ndarray] = None

    if args.text_separation:
        text_mask_img, _, elong_mask = separate_text_and_graphics(binary)
        raw_tc, _ = cv2.findContours(text_mask_img, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        for c in raw_tc:
            sq = c.squeeze()
            if sq.ndim == 2 and len(sq) >= 2:
                text_contours.append(sq)
        raw_el, _ = cv2.findContours(elong_mask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        for c in raw_el:
            sq = c.squeeze()
            if sq.ndim == 2 and len(sq) >= 2:
                elongated_contours.append(sq)
        if args.verbose:
            print(f"Text candidates: {len(text_contours)} blobs  "
                  f"Elongated: {len(elongated_contours)} blobs")

    # ── 3. Vectorise ───────────────────────────────────────────────────────────
    dashed_lines: list = []
    box_contours: list = []
    _vec_result = extract_lines_and_contours(
        binary,
        gray=gray,
        text_mask=text_mask_img,
        min_contour_length=args.min_contour_length,
        min_contour_area=args.min_contour_area,
        max_line_deviation=args.max_line_deviation,
        approx_epsilon=args.approx_epsilon,
        use_hough=not args.no_hough,
        min_line_length=args.min_line_length,
        max_gap=args.max_gap,
        hough_threshold=args.hough_threshold,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        detect_arcs=not args.no_arcs,
        arc_tol=args.arc_tol,
        min_arc_radius_px=args.min_arc_radius,
        return_arcs=True,
        mode=args.mode,
        merge_lines=not args.no_merge_lines,
        dedup_lines=not args.no_dedup_lines,
        snap_radius=args.snap_radius,
        pre_close_kernel=args.pre_close_kernel,
        right_angle_enhance=args.right_angle_enhance,
        right_angle_tol=args.right_angle_tol,
        remove_staircase=args.remove_staircase,
        corner_threshold=args.corner_threshold,
        splice_threshold=args.splice_threshold,
        gap_jump=args.gap_jump,
        gap_px=args.gap_px,
        fan_angle_deg=args.fan_angle,
        orthogonalize=args.orthogonalize,
        ortho_base_angle=args.ortho_base_angle,
        ortho_accuracy_deg=args.ortho_accuracy,
        detect_dashes=args.detect_dashes,
        max_dash_len_px=args.max_dash_len,
        detect_boxes=not args.no_detect_boxes,
        box_angle_tol=args.box_angle_tol,
        consolidate=not args.no_consolidate,
        consolidate_perp_tol=args.consolidate_perp_tol,
        consolidate_angle_tol=args.consolidate_angle_tol,
        structure_cleanup=args.structure_cleanup,
        structure_line_tolerance=args.structure_line_tolerance,
        quad_detection=not args.no_quad_detection,
        suppress_text_arcs=args.suppress_text_arcs,
        text_arc_min_cluster=args.text_arc_min_cluster,
        text_arc_r_tol=args.text_arc_r_tol,
        text_arc_y_tol=args.text_arc_y_tol,
        text_arc_x_gap=args.text_arc_x_gap,
    )
    _detect_boxes = not args.no_detect_boxes
    if args.detect_dashes and _detect_boxes:
        lines, contours, arcs, dashed_lines, box_contours = _vec_result
    elif args.detect_dashes:
        lines, contours, arcs, dashed_lines = _vec_result
    elif _detect_boxes:
        lines, contours, arcs, box_contours = _vec_result
    else:
        lines, contours, arcs = _vec_result

    if args.verbose:
        print(f"Lines: {len(lines)}  Contours: {len(contours)}  "
              f"Arcs: {len(arcs)}  Dashes: {len(dashed_lines)}  "
              f"Boxes: {len(box_contours)}")

    # ── 3b. Stroke-width estimation (SPV) ──────────────────────────────────────
    line_weights: Optional[list] = None
    contour_weights: Optional[list] = None
    if args.stroke_width:
        # Build the medial_axis width map once and reuse for both lines and
        # contours; this avoids computing it twice inside each estimate_* call.
        from src.stroke_width import _build_width_map
        width_map = _build_width_map(binary)
        line_weights = estimate_line_widths(binary, lines, dpi=args.dpi,
                                            width_map=width_map)
        contour_weights = estimate_contour_widths(binary, contours, dpi=args.dpi,
                                                  width_map=width_map)
        if args.verbose:
            from collections import Counter
            lw_counts = Counter(line_weights)
            print(f"Line weights (1/100 mm): {dict(sorted(lw_counts.items()))}")

    # ── 4. Export DXF ──────────────────────────────────────────────────────────
    try:
        total = export_to_dxf(
            lines, contours, args.output,
            image_height=h,
            dpi=args.dpi,
            units_mm=True,
            text_mask_contours=text_contours,
            elongated_contours=elongated_contours,
            arcs=arcs,
            line_weights=line_weights,
            contour_weights=contour_weights,
            dashed_lines=dashed_lines,
            box_contours=box_contours,
            page_border=compute_page_border(lines, contours) if not args.no_page_border else None,
        )
    except (OSError, IOError) as exc:
        print(f"Error: could not write DXF to '{args.output}' — {exc}", file=sys.stderr)
        return 1

    if total == 0:
        print(
            "Warning: no entities written to the DXF.\n"
            "  Try: --invert | --threshold-method adaptive | "
            "--min-line-length 20 | --threshold 128",
            file=sys.stderr,
        )
    else:
        print(f"Saved {total} entit{'y' if total == 1 else 'ies'} → '{args.output}'")

    # ── 5. Preview ─────────────────────────────────────────────────────────────
    if args.preview:
        preview_path = args.output_preview or (
            os.path.splitext(os.path.basename(args.image))[0] + "_preview.png"
        )
        ok = save_preview(original_bgr, lines, contours, text_contours,
                          preview_path, arcs=arcs)
        if ok and args.verbose:
            print(f"Preview → '{preview_path}'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
