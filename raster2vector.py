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
from src.vectorizer import extract_lines_and_contours
from src.dxf_exporter import export_to_dxf


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
    pre.add_argument("--threshold-method", choices=["otsu", "adaptive"],
                     default="otsu", metavar="METHOD",
                     help="Binarisation method: otsu or adaptive")
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
    vec.add_argument("--no-hough", action="store_true",
                     help="Disable supplemental Hough line detection")
    vec.add_argument("--no-arcs", action="store_true",
                     help="Disable circle/arc fitting (export curves as polylines only)")
    vec.add_argument("--arc-tol", type=float, default=None, metavar="F",
                     help="Max RMS pixel residual for circle/arc fitting "
                          "(default: auto, 0.5%% of image diagonal)")
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
    vec.add_argument("--approx-epsilon", type=float, default=1.5, metavar="F",
                     help="Douglas-Peucker tolerance for polyline simplification")
    vec.add_argument("--snap-radius", type=float, default=4.0, metavar="F",
                     help="Snap endpoints within this distance (px) to the same "
                          "point; 0 to disable")
    vec.add_argument("--no-merge-lines", action="store_true",
                     help="Disable collinear Hough segment merging")
    vec.add_argument("--text-separation", action="store_true",
                     help="Separate text-like blobs onto the TEXT_CANDIDATES layer "
                          "(off by default; may misclassify small symbols)")

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
    if args.approx_epsilon <= 0:
        parser.error("--approx-epsilon must be positive.")
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
            despeckle=not args.no_despeckle,
            min_speckle_area=args.min_speckle_area,
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
    text_mask_img: Optional[np.ndarray] = None

    if args.text_separation:
        text_mask_img, _ = separate_text_and_graphics(binary)
        raw_tc, _ = cv2.findContours(text_mask_img, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        for c in raw_tc:
            sq = c.squeeze()
            if sq.ndim == 2 and len(sq) >= 2:
                text_contours.append(sq)
        if args.verbose:
            print(f"Text candidates: {len(text_contours)} blobs")

    # ── 3. Vectorise ───────────────────────────────────────────────────────────
    lines, contours, arcs = extract_lines_and_contours(
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
        return_arcs=True,
        mode=args.mode,
        merge_lines=not args.no_merge_lines,
        snap_radius=args.snap_radius,
        pre_close_kernel=args.pre_close_kernel,
    )

    if args.verbose:
        print(f"Lines: {len(lines)}  Contours: {len(contours)}  Arcs: {len(arcs)}")

    # ── 4. Export DXF ──────────────────────────────────────────────────────────
    try:
        total = export_to_dxf(
            lines, contours, args.output,
            image_height=h,
            dpi=args.dpi,
            units_mm=True,
            text_mask_contours=text_contours,
            arcs=arcs,
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
