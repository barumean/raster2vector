# raster2vector

Convert black-and-white raster images (PNG, JPG, BMP, …) to DXF vector drawings.

Straight line segments are detected with the Probabilistic Hough Transform and written as DXF `LINE` entities. Remaining edge geometry is traced as contours and written as `LWPOLYLINE` entities. Pixel coordinates are converted to millimetres using the configured DPI value.

## Requirements

- Python 3.10+
- See `requirements.txt`

```bash
pip install -r requirements.txt
```

## Usage

```
python raster2vector.py <image> [options]
```

### Positional argument

| Argument | Description |
|----------|-------------|
| `image`  | Path to the input raster image |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output PATH` | `<stem>.dxf` | Output DXF file path |
| `--threshold-method {otsu,adaptive}` | `otsu` | Binarisation method |
| `--threshold VALUE` | — | Manual threshold 0-255 (overrides `--threshold-method`) |
| `--invert` | off | Invert image before processing (white-on-black drawings) |
| `--min-line-length N` | `50` | Minimum Hough line segment length in pixels |
| `--max-gap N` | `10` | Maximum gap to bridge between collinear Hough segments |
| `--dpi FLOAT` | `96.0` | Source image resolution for pixel→mm conversion |
| `--preview` | off | Save a debug PNG highlighting detected lines and contours |
| `--output-preview PATH` | `<stem>_preview.png` | Path for the preview PNG |
| `--verbose` | off | Print processing statistics |

### Examples

```bash
# Basic conversion
python raster2vector.py drawing.png

# Adaptive threshold, custom output, verbose
python raster2vector.py sketch.jpg -o sketch.dxf --threshold-method adaptive --verbose

# White-on-black drawing with preview
python raster2vector.py blueprint.png --invert --preview --dpi 150
```

## Running tests

```bash
pytest tests/test_smoke.py -v
```

## Architecture

```
raster2vector.py        CLI entry point (argparse, orchestration)
src/
  __init__.py
  preprocessor.py       Image loading, binarisation, morphological clean-up
  vectorizer.py         Canny edge detection, HoughLinesP, contour extraction
  dxf_exporter.py       ezdxf document creation, layer setup, DXF export
tests/
  test_smoke.py         Synthetic-image smoke tests
```
