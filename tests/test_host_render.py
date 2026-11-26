import copy
import json
import struct

import numpy as np
import pytest

from tools.host_render import ShaderExecutor, clip_triangle, fetch_vertices, pack_constants, raster_samples, render_artifact
from tools.mgcdriver import ProgramFormat
from tools.run_render import capture_draw, compare_images, softpipe_reference
from tools.run_shader import ROOT
from tools.shader_layout import ShaderLayout, link_varyings
from tools.teapot_mesh import generate_mesh


BASES = {"input": 0x40000, "uniform": 0x80000, "vertex": 0xC0000, "fragment": 0x400000}


def layout(stage, inputs=None, outputs=None, uniforms=None):
    return {"version": 1, "supported": True, "errors": [], "program_format": "fp16-hardware-v1",
            "shader_abi_version": 2, "address_units": "bytes", "stage": stage, "bases": dict(BASES),
            "inputs": inputs or [], "outputs": outputs or [], "uniforms": uniforms or []}


def scalar(offset, location=0, component=0, interpolation="smooth"):
    return {"word_offset": offset, "location": location, "component": component,
            "type": "float", "bit_size": 32, "interpolation": interpolation}


def synthetic_artifact():
    inputs = [scalar(component, component=component) for component in range(4)]
    vertex = "; mgc fp16-hardware-v1 backend, ABI=2, stage=MESA_SHADER_VERTEX\n"
    fragment = "; mgc fp16-hardware-v1 backend, ABI=2, stage=MESA_SHADER_FRAGMENT\n"
    for component in range(4):
        vertex += f"movi_i32 {BASES['input'] + component * 4}, r0\nload r0, r1\nwait_load\nmovi_i32 {BASES['vertex'] + component * 4}, r0\nstore r1, r0\n"
        fragment += f"movi_i32 {15360 if component in (0, 3) else 0}, r1\nmovi_i32 {BASES['fragment'] + component * 4}, r0\nstore r1, r0\n"
    positions = [(-1, -1, 0, 1), (1, -1, 0, 1), (1, 1, 0, 1),
                 (-1, -1, 0, 1), (1, 1, 0, 1), (-1, 1, 0, 1)]
    return {"schema": "mgc-host-draw-v1", "program_format": "fp16-hardware-v1", "shader_abi_version": 2,
            "draw": {"mode": "triangles", "count": 6, "start": 0, "instance_count": 1,
                     "start_instance": 0, "primitive_restart": False, "index_size": 0, "index_bias": 0},
            "shaders": {"vs": {"assembly": vertex, "layout": layout("vertex", inputs, inputs)},
                        "fs": {"assembly": fragment, "layout": layout("fragment", outputs=[scalar(component, 2, component) for component in range(4)])}},
            "vertex_buffers": [{"stride": 16, "data_hex": b"".join(struct.pack("<4f", *position) for position in positions).hex()}],
            "vertex_elements": [{"src_offset": 0, "vertex_buffer_index": 0, "instance_divisor": 0, "format": "R32G32B32A32_FLOAT"}],
            "constants": {"vs": [], "fs": []}, "indices": None}


def point(position, varying=0):
    return {"position": np.array(position, dtype=float), "varyings": {0: varying}}


def test_synthetic_verified_words_render_shared_edge_once():
    image, coverage, report = render_artifact(synthetic_artifact(), width=4, height=4)
    assert np.all(image == [255, 0, 0, 255])
    assert np.all(coverage == 1)
    assert report["covered_pixels"] == 16 and report["max_overdraw"] == 1
    assert report["fpga_execution"] is False


@pytest.mark.parametrize("index_size", [1, 2, 4])
def test_indexed_fetch_matches_nonindexed(index_size):
    artifact = synthetic_artifact()
    expected = fetch_vertices(artifact, ShaderLayout(artifact["shaders"]["vs"]["layout"]))
    artifact["draw"]["index_size"] = index_size
    artifact["indices"] = b"".join(index.to_bytes(index_size, "little") for index in range(6)).hex()
    assert fetch_vertices(artifact, ShaderLayout(artifact["shaders"]["vs"]["layout"])) == expected


@pytest.mark.parametrize("change", ["instances", "mode", "count", "bad-span", "type", "negative-index", "target", "layout"])
def test_host_draw_rejects_unsupported_or_malformed_capture(change):
    artifact = synthetic_artifact()
    if change == "instances":
        artifact["draw"]["instance_count"] = 2
    elif change == "mode":
        artifact["draw"]["mode"] = "lines"
    elif change == "count":
        artifact["draw"]["count"] = 4
    elif change == "bad-span":
        artifact["vertex_buffers"][0]["data_hex"] = "00"
    elif change == "type":
        artifact["vertex_elements"][0]["format"] = "R32G32B32A32_UINT"
    elif change == "negative-index":
        artifact["draw"].update(index_size=1, index_bias=-1)
        artifact["indices"] = bytes(range(6)).hex()
    elif change == "target":
        artifact["program_format"] = "legacy-mgcsim-v1"
    else:
        artifact["shaders"]["vs"]["layout"]["supported"] = False
    with pytest.raises(ValueError):
        render_artifact(artifact)


def test_layout_linkage_uses_semantics_not_equal_offsets():
    vertex = ShaderLayout(layout("vertex", outputs=[scalar(9, 32, 2)]))
    fragment = ShaderLayout(layout("fragment", inputs=[scalar(2, 32, 2)]))
    assert link_varyings(vertex, fragment) == {2: 9}
    mismatch = copy.deepcopy(fragment.source)
    mismatch["inputs"][0]["location"] = 33
    with pytest.raises(ValueError, match="unmatched"):
        link_varyings(vertex, ShaderLayout(mismatch))


@pytest.mark.parametrize("mutation", ["alignment", "overlap", "duplicate", "unknown-type", "bounds", "version", "units"])
def test_strict_layout_rejects_ambiguous_contract(mutation):
    source = layout("vertex", inputs=[scalar(0)])
    if mutation == "alignment":
        source["bases"]["input"] += 1
    elif mutation == "overlap":
        source["bases"]["uniform"] = source["bases"]["input"]
    elif mutation == "duplicate":
        source["inputs"].append(scalar(0))
    elif mutation == "unknown-type":
        source["inputs"][0]["type"] = "unknown"
    elif mutation == "bounds":
        source["inputs"][0]["word_offset"] = 65536
    elif mutation == "version":
        source["version"] = 2
    else:
        source["address_units"] = "words"
    with pytest.raises(ValueError):
        ShaderLayout(source)


def test_uniform_pack_is_typed_and_bounded():
    source = layout("fragment", uniforms=[{"word_offset": 0, "type": "float", "bit_size": 32},
                                         {"word_offset": 1, "type": "int", "bit_size": 32}])
    assert pack_constants(ShaderLayout(source), [struct.pack("<fi", -0.5, -7).hex()]) == {0: 0xB800, 1: 0xFFFFFFF9}
    with pytest.raises(ValueError, match="span"):
        pack_constants(ShaderLayout(source), [struct.pack("<f", 1).hex()])


def test_clipping_preserves_inside_vertices_and_rejects_outside():
    original = [point([-0.5, -0.5, 0, 1]), point([0.5, -0.5, 0, 1]), point([0, 0.5, 0, 1])]
    assert len(clip_triangle(original)) == 1
    outside = [point([2, -0.5, 0, 1]), point([3, -0.5, 0, 1]), point([2, 0.5, 0, 1])]
    assert clip_triangle(outside) == []
    crossed = [point([-2, -0.5, 0, 1], 0), point([0.5, -0.5, 0, 1], 1), point([0, 0.5, 0, 1], 2)]
    triangles = clip_triangle(crossed)
    assert len(triangles) == 2
    assert all(abs(vertex["position"][axis]) <= vertex["position"][3]
               for triangle in triangles for vertex in triangle for axis in range(3))


def test_perspective_weights_are_not_affine_for_nonunit_w():
    triangle = [point([-1, -1, 0, 1]), point([2, -2, 0, 2]), point([0, 4, 0, 4])]
    samples = list(raster_samples(triangle, 4, 4))
    assert samples
    assert any(not np.allclose(sample[3], sample[4]) for sample in samples)
    for sample in samples:
        assert np.isclose(sample[4].sum(), 1)


def test_instruction_budget_is_fail_closed():
    with pytest.raises(ValueError, match="budget"):
        render_artifact(synthetic_artifact(), max_steps=1)


@pytest.fixture(scope="module")
def fresh_draws(tmp_path_factory):
    directory = tmp_path_factory.mktemp("fresh-render")
    return {(name, target): capture_draw(ROOT / "shaders" / name, ROOT / "mesa/build", directory / f"{name}-{target.value}", target)
            for name, target in [("01_flat_color", ProgramFormat.LEGACY), ("01_flat_color", ProgramFormat.HARDWARE),
                                 ("02_varying", ProgramFormat.HARDWARE), ("03_uniform", ProgramFormat.LEGACY),
                                 ("12_runtime_arithmetic", ProgramFormat.LEGACY), ("07_lighting_phong", ProgramFormat.LEGACY)]}


@pytest.mark.parametrize("name,target,expected", [("01_flat_color", ProgramFormat.LEGACY, [255, 0, 0, 255]),
                                                ("01_flat_color", ProgramFormat.HARDWARE, [255, 0, 0, 255]),
                                                ("02_varying", ProgramFormat.HARDWARE, [255, 128, 64, 255]),
                                                ("03_uniform", ProgramFormat.LEGACY, [128, 32, 64, 128]),
                                                ("12_runtime_arithmetic", ProgramFormat.LEGACY, [128, 0, 64, 255])])
def test_fresh_actual_gallium_draw_to_managed_framebuffer(fresh_draws, name, target, expected):
    image, coverage, report = render_artifact(fresh_draws[name, target], width=8, height=8)
    covered = coverage > 0
    assert covered.sum() > 0
    assert np.all(image[covered] == expected)
    assert report["program_format"] == target.value
    assert report["linkage"] == "compiler semantic location/component reflection"
    assert report["fpga_execution"] is False


def test_fresh_phong_draw_executes_real_compiled_fragment_program(fresh_draws):
    image, coverage, report = render_artifact(fresh_draws["07_lighting_phong", ProgramFormat.LEGACY], width=8, height=8)
    covered = coverage > 0
    assert covered.sum() > 0
    assert np.all(image[covered, 3] == 255)
    assert np.all(image[covered, :3] > 0)
    assert report["host_steps"] > 1000


def test_fresh_gradient_draw_has_analytic_pixel_oracle(tmp_path):
    artifact = capture_draw(ROOT / "tests/fixtures/host-gradient", ROOT / "mesa/build", tmp_path)
    image, coverage, _ = render_artifact(artifact, width=8, height=8)
    for row, column in np.argwhere(coverage > 0):
        expected = np.rint(np.array([(column + 0.5) / 8, (7.5 - row) / 8, 0.5, 1]) * 255).astype(np.uint8)
        assert np.array_equal(image[row, column], expected)


@pytest.mark.parametrize("target", [ProgramFormat.HARDWARE, ProgramFormat.LEGACY])
def test_fresh_runtime_add_draw_matches_analytic_pixels_and_softpipe(tmp_path, target):
    shader = ROOT / "tests/fixtures/hardware-add"
    artifact = capture_draw(shader, ROOT / "mesa/build", tmp_path / "capture", target)
    image, coverage, report = render_artifact(artifact)
    assert np.count_nonzero(coverage) == 72
    for row, column in np.argwhere(coverage > 0):
        values = np.array([2 * (column + 0.5) / 16 - 0.25,
                           2 * (15.5 - row) / 16 - 0.25, 0.75, 1])
        expected = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
        assert np.array_equal(image[row, column], expected)
    reference = softpipe_reference(shader, ROOT / "mesa/build", tmp_path / "reference")
    comparison = compare_images(image, reference, 0)
    assert comparison["passed"]
    assert comparison["alpha_support_mismatch"] == 0
    assert report["program_format"] == target.value
    assert report["fpga_execution"] is False


def test_fresh_hardware_format_pixels_match_actual_softpipe(fresh_draws, tmp_path):
    image, _, _ = render_artifact(fresh_draws["01_flat_color", ProgramFormat.HARDWARE])
    reference = softpipe_reference(ROOT / "shaders/01_flat_color", ROOT / "mesa/build", tmp_path)
    assert compare_images(image, reference, 0)["passed"]


@pytest.mark.parametrize("invalid", [True, -1, 256, "8"])
def test_pixel_comparison_rejects_invalid_tolerance(invalid):
    with pytest.raises(ValueError):
        compare_images(np.zeros((1, 1, 4), dtype=np.uint8), np.zeros((1, 1, 4), dtype=np.uint8), invalid)


def test_pixel_comparison_does_not_hide_precision_or_support_differences():
    image = np.array([[[10, 20, 30, 255]]], dtype=np.uint8)
    reference = np.array([[[11, 20, 30, 255]]], dtype=np.uint8)
    assert not compare_images(image, reference, 0)["passed"]
    assert compare_images(image, reference, 1)["passed"]
    reference[0, 0, 3] = 0
    assert not compare_images(image, reference, 255)["passed"]


def test_fresh_canonical_teapot_is_compiled_and_front_lit(tmp_path):
    mesh = tmp_path / "teapot.host.bin"
    payload, _ = generate_mesh()
    mesh.write_bytes(payload)
    artifact = capture_draw(ROOT / "shaders/07_lighting_phong", ROOT / "mesa/build", tmp_path, mesh=mesh)
    image, coverage, report = render_artifact(artifact)
    assert report["vertices"] == 1728 and report["triangles"] == 576
    assert np.count_nonzero(coverage) > 0
    assert int(image[:, :, 2].max()) > 64
    reference = softpipe_reference(ROOT / "shaders/07_lighting_phong", ROOT / "mesa/build", tmp_path, mesh=mesh)
    assert compare_images(image, reference, 16)["passed"]
