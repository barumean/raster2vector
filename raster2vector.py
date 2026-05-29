#!/usr/bin/env python3
"""raster2vector — Convert B&W raster images to DXF vector drawings."""

import argparse
import os
import sys

import cv2
import numpy as np

from src.preprocessor import load_and_preprocess
from src.vectorizer import extract_lines_and_contours
from src.dxf_exporter import export_to_dxf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raster2vector",
        description="Convert a black-and-white raster image to a DXF vector file.",
    )
    parser.add_argument("image", help="Path to the input raster image (PNG, JPG, BMP, …)")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output DXF file path. Defaults to <input_stem>.dxf in the current directory.",
    )
    parser.add_argument(
        "--threshold-method",
        choices=["otsu", "adaptive"],
        default="otsu",
        help="Binarisation method (default: otsu).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        metavar="VALUE",
        help="Manual threshold value 0-255. Overrides --threshold-method when provided.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert the image before processing (useful for white-on-black drawings).",
    )
    parser.add_argument(
        "--min-line-length",
        type=int,
        default=50,
        help="Minimum length (pixels) for a Hough line segment to be kept (default: 50).",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=10,
        help="Maximum gap (pixels) to bridge between collinear Hough segments (default: 10).",
    )
    parser.add_argument(
        "--dpi",
        type=float,
        default=96.0,
        help="Source image resolution in DPI used for pixel→mm conversion (default: 96).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Save a debug PNG alongside the DXF showing detected lines and contours.",
    )
    parser.add_argument(
        "--output-preview",
        default=None,
        metavar="PATH",
        help="Path for the preview PNG (default: <input_stem>_preview.png).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print processing statistics to stdout.",
    )
    return parser


def save_preview(
    original_bgr: np.ndarray,
    lines: list,
    contours: list,
    preview_path: str,
) -> None:
    """Render a debug PNG with detected lines (red) and contours (green)."""
    canvas = original_bgr.copy()

    for x1, y1, x2, y2 in lines:
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for contour in contours:
        pts = contour.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=False, color=(0, 255, 0), thickness=1)

    cv2.imwrite(preview_path, canvas)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate manual threshold range
    if args.threshold is not None and not (0 <= args.threshold <= 255):
        parser.error("--threshold must be between 0 and 255.")

    # Determine output path
    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.image))[0]
        args.output = stem + ".dxf"

    # Warn if output already exists
    if os.path.exists(args.output):
        print(f"Warning: output file '{args.output}' already exists and will be overwritten.",
              file=sys.stderr)

    # ── Preprocessing ────────────────────────────────────────────────────────
    try:
        original_bgr, binary = load_and_preprocess(
            args.image,
            threshold_method=args.threshold_method,
            invert=args.invert,
            manual_threshold=args.threshold,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    image_height, image_width = binary.shape[:2]

    if args.verbose:
        print(f"Image loaded: {image_width}x{image_height} px  ({args.image})")

    # ── Vectorisation ─────────────────────────────────────────────────────────
    lines, contours = extract_lines_and_contours(
        binary,
        min_line_length=args.min_line_length,
        max_gap=args.max_gap,
    )

    if args.verbose:
        print(f"Detected {len(lines)} Hough line segment(s) and {len(contours)} contour(s).")

    # ── DXF export ────────────────────────────────────────────────────────────
    entity_count = export_to_dxf(
        lines,
        contours,
        args.output,
        image_height=image_height,
        dpi=args.dpi,
        units_mm=True,
    )

    if entity_count == 0:
        print(
            "Warning: no entities were written to the DXF.\n"
            "  Try one or more of:\n"
            "    --invert              (if drawing is white-on-black)\n"
            "    --threshold-method adaptive\n"
            "    --min-line-length 20  (detect shorter lines)\n"
            "    --threshold 128       (manual binarisation)\n",
            file=sys.stderr,
        )
    else:
        print(f"Saved {entity_count} entit{'y' if entity_count == 1 else 'ies'} to '{args.output}'.")

    # ── Optional preview ──────────────────────────────────────────────────────
    if args.preview:
        if args.output_preview:
            preview_path = args.output_preview
        else:
            stem = os.path.splitext(os.path.basename(args.image))[0]
            preview_path = stem + "_preview.png"
        save_preview(original_bgr, lines, contours, preview_path)
        if args.verbose:
            print(f"Preview saved to '{preview_path}'.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
