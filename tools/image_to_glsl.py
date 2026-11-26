"""Bounded texture-free image approximation using explicit GLSL arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


MAX_GRID = 8
MAX_INPUT_PIXELS = 4194304
MAX_INPUT_BYTES = 67108864
VERTEX_SOURCE = """#version 100
attribute vec4 a_position;
attribute vec2 a_uv;
varying vec2 v_uv;
void main() {
    gl_Position = a_position;
    v_uv = a_uv;
}
"""


def checked_integer(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    return value


def checked_grid(grid):
    values = np.asarray(grid, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 4:
        raise ValueError("color grid must have shape rows x columns x 4")
    checked_integer(values.shape[0], "grid rows", MAX_GRID)
    checked_integer(values.shape[1], "grid columns", MAX_GRID)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("grid colors must be finite RGBA values between zero and one")
    return values


def glsl_float(value):
    text = format(float(value), ".9g")
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return text


def axis_coefficients(source_size, nodes):
    checked_integer(source_size, "source dimension", MAX_INPUT_PIXELS)
    checked_integer(nodes, "grid dimension", MAX_GRID)
    if nodes > source_size:
        raise ValueError("grid dimension cannot exceed the source dimension")
    if nodes == 1:
        return 0.0, 0.0
    spacing = (nodes - 1) / (source_size - 1)
    return source_size * spacing, -0.5 * spacing


def fragment_source(grid, width, height):
    colors = checked_grid(grid)
    rows, columns = colors.shape[:2]
    scale_x, offset_x = axis_coefficients(width, columns)
    scale_y, offset_y = axis_coefficients(height, rows)
    lines = ["#version 100", "precision mediump float;", "varying vec2 v_uv;", "void main() {"]
    for axis, count, scale, offset in (("x", columns, scale_x, offset_x),
                                      ("y", rows, scale_y, offset_y)):
        if count == 1:
            lines.append(f"    float weight_{axis}0 = 1.0;")
        else:
            lines.append(f"    float coordinate_{axis} = clamp(v_uv.{axis} * {glsl_float(scale)}"
                         f" + ({glsl_float(offset)}), 0.0, {glsl_float(count - 1)});")
            for index in range(count):
                lines.append(f"    float weight_{axis}{index} = max(1.0 - abs(coordinate_{axis}"
                             f" - {glsl_float(index)}), 0.0);")
    for row in range(rows):
        terms = []
        for column in range(columns):
            color = ", ".join(glsl_float(value) for value in colors[rows - 1 - row, column])
            terms.append(f"vec4({color}) * weight_x{column}")
        lines.append(f"    vec4 row_{row} = " + "\n        + ".join(terms) + ";")
    lines.append("    vec4 color = " + "\n        + ".join(
        f"row_{row} * weight_y{row}" for row in range(rows)) + ";")
    lines.extend(["    gl_FragColor = clamp(color, 0.0, 1.0);", "}", ""])
    return "\n".join(lines)


def evaluate(grid, width, height, horizontal, vertical):
    colors = checked_grid(grid)
    rows, columns = colors.shape[:2]
    scale_x, offset_x = axis_coefficients(width, columns)
    scale_y, offset_y = axis_coefficients(height, rows)
    horizontal, vertical = np.broadcast_arrays(np.asarray(horizontal, dtype=np.float64),
                                               np.asarray(vertical, dtype=np.float64))
    if not np.isfinite(horizontal).all() or not np.isfinite(vertical).all():
        raise ValueError("UV coordinates must be finite")
    coordinate_x = np.clip(horizontal * scale_x + offset_x, 0, columns - 1)
    coordinate_y = np.clip(vertical * scale_y + offset_y, 0, rows - 1)
    left = np.floor(coordinate_x).astype(np.int64)
    bottom = np.floor(coordinate_y).astype(np.int64)
    right = np.minimum(left + 1, columns - 1)
    top = np.minimum(bottom + 1, rows - 1)
    amount_x = (coordinate_x - left)[..., None]
    amount_y = (coordinate_y - bottom)[..., None]
    lower = colors[rows - 1 - bottom, left] * (1 - amount_x) + colors[rows - 1 - bottom, right] * amount_x
    upper = colors[rows - 1 - top, left] * (1 - amount_x) + colors[rows - 1 - top, right] * amount_x
    return np.clip(lower * (1 - amount_y) + upper * amount_y, 0, 1)


def generate(image_path, output, *, grid_size=8, compile_vector=False):
    grid_size = checked_integer(grid_size, "grid size", MAX_GRID)
    image_path, output = Path(image_path).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("output already exists; choose a fresh directory")
    if type(compile_vector) is not bool:
        raise ValueError("compile_vector must be a boolean")
    with image_path.open("rb") as source:
        contents = source.read(MAX_INPUT_BYTES + 1)
    if len(contents) > MAX_INPUT_BYTES:
        raise ValueError(f"input file exceeds {MAX_INPUT_BYTES} bytes")
    with Image.open(io.BytesIO(contents)) as opened:
        if opened.width * opened.height > MAX_INPUT_PIXELS:
            raise ValueError(f"input image exceeds {MAX_INPUT_PIXELS} pixels")
        if getattr(opened, "n_frames", 1) != 1:
            raise ValueError("animated or multi-frame input is not supported")
        image = ImageOps.exif_transpose(opened).convert("RGBA")
    width, height = image.size
    columns, rows = min(width, grid_size), min(height, grid_size)
    grid = np.asarray(image.resize((columns, rows), Image.Resampling.BOX), dtype=np.float64) / 255
    horizontal = (np.arange(width, dtype=np.float64) + 0.5) / width
    vertical = 1 - (np.arange(height, dtype=np.float64) + 0.5) / height
    original = np.asarray(image, dtype=np.uint8)
    preview = np.empty_like(original)
    squared_error, maximum_error = 0.0, 0.0
    for first_row in range(0, height, 256):
        row_slice = slice(first_row, first_row + 256)
        for first_column in range(0, width, 256):
            column_slice = slice(first_column, first_column + 256)
            reconstructed = evaluate(grid, width, height, horizontal[None, column_slice], vertical[row_slice, None])
            error = reconstructed - original[row_slice, column_slice].astype(np.float64) / 255
            squared_error += float(np.sum(error * error))
            maximum_error = max(maximum_error, float(np.max(np.abs(error))))
            preview[row_slice, column_slice] = np.rint(reconstructed * 255).astype(np.uint8)
    output.mkdir(parents=True)
    (output / "vert.glsl").write_text(VERTEX_SOURCE, encoding="utf-8")
    (output / "frag.glsl").write_text(fragment_source(grid, width, height), encoding="utf-8")
    image.save(output / "input.png")
    Image.fromarray(preview).save(output / "preview.png")
    report = {
        "version": 1, "method": "bounded separable piecewise-bilinear color grid",
        "input": str(image_path), "input_sha256": hashlib.sha256(contents).hexdigest(),
        "width": width, "height": height, "grid_columns": columns, "grid_rows": rows,
        "grid_rgba_top_left": grid.tolist(), "grid_downsampling": "Pillow BOX",
        "requires_texture": False, "runtime_image_dependency": False,
        "uv_origin": "bottom-left; source and preview row zero at top",
        "uv_sampling": "pixel centers; clamp to edge",
        "color_space": "encoded input RGBA values; no lighting or linear-color reconstruction",
        "max_absolute_channel_error": maximum_error,
        "root_mean_square_channel_error": float(np.sqrt(squared_error / (width * height * 4))),
        "preview_backend": "independent float64 Python bilinear evaluation, not GPU rendering",
        "machine_code_emitted": False, "hardware_reference_executed": False,
        "fpga_execution": False, "board_upload_ready": False,
        "source_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                          for name in ("vert.glsl", "frag.glsl")},
    }
    report_path = output / "image-shader.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if compile_vector:
        from tools.run_vector_shader import compile_vector_glsl
        compiled = compile_vector_glsl(output, output / "vector")
        report["machine_code_emitted"] = True
        report["compiler_target"] = compiled["manifest"]["compiler_target"]
        report["compiler_manifest"] = compiled["manifest"]
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=8, help="maximum rows/columns, 1..8")
    parser.add_argument("--compile-vector", action="store_true", help="also compile hardware-format vector words")
    arguments = parser.parse_args(argv)
    try:
        report = generate(arguments.image, arguments.output, grid_size=arguments.grid,
                          compile_vector=arguments.compile_vector)
    except (ValueError, OSError, Image.DecompressionBombError) as error:
        print(f"image_to_glsl: {error}", file=sys.stderr)
        return 1
    print(json.dumps({name: report[name] for name in (
        "grid_columns", "grid_rows", "root_mean_square_channel_error", "machine_code_emitted", "fpga_execution")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
