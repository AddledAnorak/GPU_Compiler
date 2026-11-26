import hashlib
import json
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from tools import image_to_glsl as image_shader
from tools.run_shader import ROOT, compile_glsl
from tools.run_vector_shader import compile_vector_glsl, execute_vector_pair


def saved_image(path, pixels):
    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(path)
    return path


@pytest.mark.parametrize("width,height", [(1, 1), (1, 5), (6, 1), (2, 2), (3, 5), (8, 8)])
def test_full_grid_reconstruction_matches_source_at_pixel_centers(tmp_path, width, height):
    pixels = np.random.default_rng(42).integers(0, 256, (height, width, 4), dtype=np.uint8)
    source = saved_image(tmp_path / "input.png", pixels)
    output = tmp_path / "generated"
    report = image_shader.generate(source, output)
    assert np.array_equal(np.asarray(Image.open(output / "preview.png")), pixels)
    assert report["root_mean_square_channel_error"] < 1e-14
    assert report["grid_columns"] == width and report["grid_rows"] == height
    assert report["machine_code_emitted"] is False
    assert report["hardware_reference_executed"] is False
    assert report["runtime_image_dependency"] is False
    assert report["fpga_execution"] is False
    assert report == json.loads((output / "image-shader.json").read_text())


def test_shader_uses_math_and_embedded_colors_not_textures_or_image_path(tmp_path):
    source = saved_image(tmp_path / "private-photo.png", [[[255, 0, 0, 255], [0, 255, 0, 255]]])
    output = tmp_path / "generated"
    report = image_shader.generate(source, output)
    fragment = (output / "frag.glsl").read_text()
    assert "max(1.0 - abs(" in fragment
    assert "vec4(1.0, 0.0, 0.0, 1.0)" in fragment
    for unsupported in ("texture", "sampler", "for (", "while (", "private-photo", "uniform", "/"):
        assert unsupported not in fragment
    for name, checksum in report["source_sha256"].items():
        assert checksum == hashlib.sha256((output / name).read_bytes()).hexdigest()


@pytest.mark.parametrize("coordinate,expected", [((0, 1), [1, 0, 0, 1]),
                                                  ((1, 1), [0, 1, 0, 1]),
                                                  ((0, 0), [0, 0, 1, 0]),
                                                  ((1, 0), [1, 1, 1, 1]),
                                                  ((0.5, 0.5), [0.5, 0.5, 0.5, 0.75]),
                                                  ((-2, 3), [1, 0, 0, 1])])
def test_bilinear_oracle_orientation_edges_and_transparency(coordinate, expected):
    grid = [[[1, 0, 0, 1], [0, 1, 0, 1]], [[0, 0, 1, 0], [1, 1, 1, 1]]]
    assert np.allclose(image_shader.evaluate(grid, 2, 2, *coordinate), expected)


@pytest.mark.parametrize("value", [0, 9, -1, True, 2.5, "2", None])
def test_invalid_grid_size_fails_before_output_creation(tmp_path, value):
    output = tmp_path / "generated"
    with pytest.raises(ValueError, match="grid size"):
        image_shader.generate(tmp_path / "missing.png", output, grid_size=value)
    assert not output.exists()


@pytest.mark.parametrize("grid", [[], [[[0, 0, 0]]], [[[float("nan"), 0, 0, 1]]],
                                     [[[float("inf"), 0, 0, 1]]], [[[-1, 0, 0, 1]]],
                                     [[[2, 0, 0, 1]]], np.zeros((9, 1, 4))])
def test_invalid_color_grids_fail_closed(grid):
    with pytest.raises(ValueError):
        image_shader.fragment_source(grid, 8, 8)


def test_input_limits_animation_and_output_preservation(tmp_path, monkeypatch):
    source = saved_image(tmp_path / "input.png", np.zeros((2, 2, 4), dtype=np.uint8))
    output = tmp_path / "generated"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing work")
    with pytest.raises(ValueError, match="already exists"):
        image_shader.generate(source, output)
    assert marker.read_text() == "existing work"
    monkeypatch.setattr(image_shader, "MAX_INPUT_PIXELS", 3)
    with pytest.raises(ValueError, match="exceeds"):
        image_shader.generate(source, tmp_path / "oversize")
    assert not (tmp_path / "oversize").exists()
    frames = [Image.new("RGB", (1, 1), color) for color in ("red", "blue")]
    frames[0].save(tmp_path / "animated.gif", save_all=True, append_images=frames[1:])
    with pytest.raises(ValueError, match="multi-frame"):
        image_shader.generate(tmp_path / "animated.gif", tmp_path / "animated")


def test_nonconstant_downsample_reports_actual_approximation_error(tmp_path):
    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    pixels[::2, ::2, :3] = 255
    source = saved_image(tmp_path / "input.png", pixels)
    report = image_shader.generate(source, tmp_path / "generated", grid_size=2)
    assert report["grid_rows"] == report["grid_columns"] == 2
    assert report["root_mean_square_channel_error"] > 0.1
    assert report["max_absolute_channel_error"] > 0.5


def test_cli_success_and_error_without_traceback(tmp_path):
    source = saved_image(tmp_path / "input.png", [[[12, 34, 56, 255]]])
    command = [sys.executable, "-m", "tools.image_to_glsl", str(source), "--output", str(tmp_path / "generated")]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["machine_code_emitted"] is False
    failed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=10)
    assert failed.returncode == 1 and "fresh directory" in failed.stderr
    assert "Traceback" not in failed.stderr


@pytest.fixture(scope="module")
def compiled_image(tmp_path_factory):
    output = tmp_path_factory.mktemp("image-vector")
    pixels = np.array([[[255, 0, 0, 255], [0, 255, 0, 128]],
                       [[0, 0, 255, 0], [32, 64, 128, 255]]], dtype=np.uint8)
    source = saved_image(output / "input.png", pixels)
    report = image_shader.generate(source, output / "generated", compile_vector=True)
    directory = output / "generated" / "vector"
    compiled = {"assembly": {stage: (directory / f"{stage}.asm").read_text() for stage in ("vs", "fs")},
                "layouts": {stage: json.loads((directory / f"{stage}.layout.json").read_text()) for stage in ("vs", "fs")}}
    return report, compiled


@pytest.mark.parametrize("horizontal,vertical", [(0.25, 0.75), (0.75, 0.75), (0.25, 0.25),
                                                   (0.75, 0.25), (0.5, 0.5), (0.375, 0.625)])
def test_fresh_image_glsl_hardware_words_execute_against_independent_oracle(compiled_image, horizontal, vertical):
    report, compiled = compiled_image
    layout = compiled["layouts"]["fs"]
    pair = execute_vector_pair(compiled, vertex_inputs(compiled, horizontal, vertical), {}, {},
                               allow_approximate_fp16=True)
    result = pair["fragment"]
    actual = [result["outputs"][entry["word_offset"]] for entry in layout["outputs"]]
    expected = image_shader.evaluate(report["grid_rgba_top_left"], 2, 2, horizontal, vertical)
    assert np.allclose(actual, expected, atol=0.002, rtol=0)
    assert result["status"] == "committed" and result["fpga_execution"] is False
    assert report["machine_code_emitted"] is True and report["hardware_reference_executed"] is False


def vertex_inputs(compiled, horizontal, vertical):
    entries = compiled["layouts"]["vs"]["inputs"]
    assert 4 <= len(entries) <= 6
    position_location = next(location for location in {entry["location"] for entry in entries}
                             if sum(entry["location"] == location for entry in entries) == 4)
    values = {entry["word_offset"]: float((0, 0, 0, 1)[entry["component"]]
                                         if entry["location"] == position_location else (horizontal, vertical)[entry["component"]])
              for entry in entries}
    return values


def test_image_pair_uses_reflected_vertex_to_fragment_bridge(compiled_image):
    report, compiled = compiled_image
    result = execute_vector_pair(compiled, vertex_inputs(compiled, 0.375, 0.625), {}, {},
                                 allow_approximate_fp16=True)
    expected = image_shader.evaluate(report["grid_rgba_top_left"], 2, 2, 0.375, 0.625)
    fragment = compiled["layouts"]["fs"]
    actual = [result["fragment"]["outputs"][entry["word_offset"]] for entry in fragment["outputs"]]
    assert np.allclose(actual, expected, atol=0.002, rtol=0)
    assert result["varying_map"] and result["rasterization"] is False


def test_same_generated_glsl_compiles_on_legacy_target(tmp_path):
    source = saved_image(tmp_path / "input.png", [[[128, 0, 255, 255], [0, 128, 255, 255]]])
    image_shader.generate(source, tmp_path / "generated")
    assembly = compile_glsl(tmp_path / "generated", ROOT / "mesa/build", tmp_path / "legacy")
    assert all("mgc legacy-simulator backend" in source for source in assembly.values())


@pytest.mark.parametrize("width,height", [(1, 1), (1, 3), (3, 1), (8, 8)])
def test_degenerate_and_maximum_grids_fresh_vector_compile(tmp_path, width, height):
    pixels = np.random.default_rng(14).integers(0, 256, (height, width, 4), dtype=np.uint8)
    source = saved_image(tmp_path / "input.png", pixels)
    report = image_shader.generate(source, tmp_path / "generated")
    compiled = compile_vector_glsl(tmp_path / "generated", tmp_path / "vector")
    assert compiled["manifest"]["machine_code_emitted"] is True
    assert compiled["manifest"]["board_upload_ready"] is False
    result = execute_vector_pair(compiled, vertex_inputs(compiled, 0.375, 0.625), {}, {},
                                 allow_approximate_fp16=True)
    expected = image_shader.evaluate(report["grid_rgba_top_left"], width, height, 0.375, 0.625)
    actual = [result["fragment"]["outputs"][entry["word_offset"]]
              for entry in compiled["layouts"]["fs"]["outputs"]]
    assert np.allclose(actual, expected, atol=0.002, rtol=0)


@pytest.mark.parametrize("mode", ["RGB", "L", "P", "RGBA"])
def test_supported_input_color_modes_convert_to_standalone_rgba(tmp_path, mode):
    image = Image.new("RGB", (2, 1), (33, 77, 123)).convert(mode)
    image.save(tmp_path / "input.png")
    image_shader.generate(tmp_path / "input.png", tmp_path / "generated")
    assert np.array_equal(np.asarray(Image.open(tmp_path / "generated" / "preview.png")),
                          np.asarray(image.convert("RGBA")))


def test_exif_orientation_is_applied_before_grid_and_preview(tmp_path):
    pixels = np.random.default_rng(42).integers(0, 256, (3, 2, 4), dtype=np.uint8)
    image = Image.fromarray(pixels)
    exif = Image.Exif()
    exif[274] = 6
    image.save(tmp_path / "input.png", exif=exif)
    report = image_shader.generate(tmp_path / "input.png", tmp_path / "generated")
    assert (report["width"], report["height"]) == (3, 2)
    assert np.array_equal(np.asarray(Image.open(tmp_path / "generated" / "preview.png")),
                          np.rot90(pixels, -1))


def test_encoded_file_size_is_bounded_before_decode(tmp_path, monkeypatch):
    source = tmp_path / "large.png"
    source.write_bytes(b"not an image")
    monkeypatch.setattr(image_shader, "MAX_INPUT_BYTES", 3)
    with pytest.raises(ValueError, match="bytes"):
        image_shader.generate(source, tmp_path / "generated")
    assert not (tmp_path / "generated").exists()


def test_tiled_preview_and_error_metrics_match_full_oracle(tmp_path):
    pixels = np.random.default_rng(42).integers(0, 256, (260, 270, 4), dtype=np.uint8)
    source = saved_image(tmp_path / "input.png", pixels)
    report = image_shader.generate(source, tmp_path / "generated")
    horizontal = (np.arange(270) + 0.5) / 270
    vertical = 1 - (np.arange(260) + 0.5) / 260
    expected = image_shader.evaluate(report["grid_rgba_top_left"], 270, 260,
                                     horizontal[None, :], vertical[:, None])
    error = expected - pixels.astype(np.float64) / 255
    assert report["root_mean_square_channel_error"] == pytest.approx(np.sqrt(np.mean(error * error)))
    assert report["max_absolute_channel_error"] == pytest.approx(np.max(np.abs(error)))
    assert np.array_equal(np.asarray(Image.open(tmp_path / "generated" / "preview.png")),
                          np.rint(expected * 255).astype(np.uint8))
