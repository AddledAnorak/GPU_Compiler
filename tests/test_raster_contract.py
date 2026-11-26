"""Hardware-format shader execution around a CPU-only raster reference."""

import struct

import numpy as np
import pytest

from test_host_render import BASES, layout, point, scalar, synthetic_artifact
from tools.host_render import ShaderExecutor, clip_triangle, fetch_vertices, pack_constants, pack_float, render_artifact
from tools.mgcdriver import ExecutionTimeoutError
from tools.mgchw import assemble_hardware
from tools.shader_layout import ShaderLayout


QUAD = [(-1, -1, 0, 1), (1, -1, 0, 1), (1, 1, 0, 1),
        (-1, -1, 0, 1), (1, 1, 0, 1), (-1, 1, 0, 1)]


def copy_shader(stage, transfers):
    assembly = f"; mgc fp16-hardware-v1 backend, ABI=2, stage=MESA_SHADER_{stage.upper()}\n"
    for source, destination in transfers:
        assembly += (f"movi_i32 {BASES['input'] + source * 4}, r0\nload r0, r1\nwait_load\n"
                     f"copy r1, r2\nmovi_i32 {BASES[stage] + destination * 4}, r0\nstore r2, r0\n")
    return assembly


def varying_artifact(records, interpolation="smooth"):
    artifact = synthetic_artifact()
    positions = [scalar(component, component=component) for component in range(4)]
    attributes = [scalar(4 + component, location=1, component=component) for component in range(4)]
    varyings = [scalar(12 + component, location=32, component=component) for component in range(4)]
    fragment_inputs = [scalar(8 + component, location=32, component=component, interpolation=interpolation)
                       for component in range(4)]
    artifact["shaders"] = {
        "vs": {"assembly": copy_shader("vertex", [(component, component) for component in range(4)]
                                      + [(4 + component, 12 + component) for component in range(4)]),
               "layout": layout("vertex", positions + attributes, positions + varyings)},
        "fs": {"assembly": copy_shader("fragment", [(8 + component, component) for component in range(4)]),
               "layout": layout("fragment", fragment_inputs, [scalar(component, 2, component) for component in range(4)])},
    }
    artifact["draw"]["count"] = len(records)
    artifact["vertex_buffers"] = [{"stride": 32,
                                   "data_hex": b"".join(struct.pack("<8f", *position, *color)
                                                        for position, color in records).hex()}]
    artifact["vertex_elements"] = [{"src_offset": offset, "vertex_buffer_index": 0,
                                    "instance_divisor": 0, "format": "R32G32B32A32_FLOAT"}
                                   for offset in (0, 16)]
    return artifact


def quantized_rgba(values):
    return np.rint(np.clip(np.asarray(values, dtype=np.float16).astype(float), 0, 1) * 255).astype(np.uint8)


def word_counts(artifact):
    return tuple(len(assemble_hardware(artifact["shaders"][stage]["assembly"])) for stage in ("vs", "fs"))


@pytest.mark.parametrize("interpolation", ["smooth", "noperspective"])
def test_nonconstant_varyings_match_analytical_pixel_values(interpolation):
    artifact = varying_artifact([((-1, -1, 0, 1), (0, 0, 0, 1)),
                                ((2, -2, 0, 2), (1, 0, 0, 1)),
                                ((0, 4, 0, 4), (0, 1, 0, 1))], interpolation)
    image, coverage, report = render_artifact(artifact, width=4, height=4)
    expected_coverage = np.array([[0, 0, 0, 0], [0, 1, 1, 0],
                                  [0, 1, 1, 0], [1, 1, 1, 1]], dtype=np.uint32)
    expected = np.zeros_like(image)
    for pixel_row, column in np.argwhere(expected_coverage):
        bottom_row = 3 - pixel_row
        upper = (bottom_row + 0.5) / 4
        right = (column + 0.5 - 2 * upper) / 4
        weights = np.array([1 - right - upper, right, upper])
        if interpolation == "smooth":
            weights *= [1, 0.5, 0.25]
            weights /= weights.sum()
        expected[pixel_row, column] = quantized_rgba([weights[1], weights[2], 0, 1])
    np.testing.assert_array_equal(coverage, expected_coverage)
    np.testing.assert_array_equal(image, expected)
    np.testing.assert_array_equal(image[3, 1], [53, 11, 0, 255] if interpolation == "smooth" else [80, 32, 0, 255])
    assert len(np.unique(image[coverage > 0, 0])) > 1
    vertex_words, fragment_words = word_counts(artifact)
    assert report["host_steps"] == 3 * vertex_words + 8 * fragment_words
    assert report["rasterization"] == "bounded CPU only"
    assert report["program_format"] == "fp16-hardware-v1"
    assert report["fpga_execution"] is False and report["rtl_timing_validated"] is False
    assert "not physical GPU or OpenGL conformance" in report["limitations"]


@pytest.mark.parametrize("axis,sign", [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)],
                         ids=["left", "right", "bottom", "top", "near", "far"])
def test_clip_intersections_transport_varyings_and_preserve_orientation(axis, sign):
    positions = []
    for coordinate, tangent, other in [(-2 * sign, -0.5, -0.5), (0, -0.5, 0.5), (0, 0.5, 0)]:
        position = np.ones(4)
        position[axis] = coordinate
        position[(axis + 1) % 3] = tangent
        position[(axis + 2) % 3] = other
        positions.append(position)
    vertices = [point(position, value) for position, value in zip(positions, [0, 0.5, 1])]
    expected = {tuple(positions[1]): 0.5, tuple(positions[2]): 1,
                tuple((positions[0] + positions[1]) / 2): 0.25,
                tuple((positions[0] + positions[2]) / 2): 0.5}
    for ordered in (vertices, [vertices[0], vertices[2], vertices[1]]):
        clipped = clip_triangle(ordered)
        observed = {}
        normal = np.cross(ordered[1]["position"][:3] - ordered[0]["position"][:3],
                          ordered[2]["position"][:3] - ordered[0]["position"][:3])
        assert len(clipped) == 2
        for triangle in clipped:
            clipped_normal = np.cross(triangle[1]["position"][:3] - triangle[0]["position"][:3],
                                      triangle[2]["position"][:3] - triangle[0]["position"][:3])
            assert np.dot(normal, clipped_normal) > 0
            for vertex in triangle:
                key = tuple(vertex["position"])
                assert vertex["varyings"][0] == expected[key]
                observed[key] = vertex["varyings"][0]
        assert observed == expected


@pytest.mark.parametrize("interpolation", ["smooth", "flat"])
def test_clipped_fan_has_winding_invariant_pixels_and_original_provoking_vertex(interpolation):
    positions = [(-2, -1, 0, 1), (1, -1, 0, 1), (0, 1, 0, 1)]
    records = [(position, ((position[0] + 2) / 4, 0, 0, 1)) for position in positions]
    expected_coverage = np.array([[0, 1, 0, 0], [1, 1, 1, 0],
                                  [1, 1, 1, 0], [1, 1, 1, 1]], dtype=np.uint32)
    expected = np.zeros((4, 4, 4), dtype=np.uint8)
    for pixel_row, column in np.argwhere(expected_coverage):
        value = (column + 2.5) / 8 if interpolation == "smooth" else 0.5
        expected[pixel_row, column] = quantized_rgba([value, 0, 0, 1])
    for ordered in (records, [records[1], records[0], records[2]]):
        image, coverage, report = render_artifact(varying_artifact(ordered, interpolation), width=4, height=4)
        np.testing.assert_array_equal(coverage, expected_coverage)
        np.testing.assert_array_equal(image, expected)
        assert report["clipped_triangles"] == 2 and report["fragments"] == 11
        assert report["max_overdraw"] == 1


@pytest.mark.parametrize("positions", [
    [(-0.5, -0.5, 0, 1), (-0.5, -0.5, 0, 1), (0.5, 0.5, 0, 1)],
    [(-1, -1, 0, 1), (0, 0, 0, 1), (1, 1, 0, 1)],
    [(-2, 0, 0, 1), (-1, -0.5, 0, 1), (-1, 0.5, 0, 1)],
], ids=["repeated-vertex", "collinear", "collapsed-by-clipping"])
def test_degenerate_triangles_never_invoke_fragment_shader(positions):
    artifact = varying_artifact([(position, (1, 0, 0, 1)) for position in positions])
    image, coverage, report = render_artifact(artifact, width=4, height=4, max_fragments=1)
    assert not image.any() and not coverage.any()
    assert report["fragments"] == 0 and report["covered_pixels"] == 0
    assert report["host_steps"] == 3 * word_counts(artifact)[0]


@pytest.mark.parametrize("layers,fragments", [
    ([(-0.5, (1, 0, 0, 1)), (0.5, (0, 0, 1, 1))], 4),
    ([(0.5, (0, 0, 1, 1)), (-0.5, (1, 0, 0, 1))], 8),
    ([(0, (1, 0, 0, 1)), (0, (0, 0, 1, 1))], 4),
], ids=["near-before-far", "far-before-near", "equal-depth-first-wins"])
def test_explicit_less_depth_occludes_far_and_equal_fragments(layers, fragments):
    records = [((position[0], position[1], depth, position[3]), color)
               for depth, color in layers for position in QUAD]
    artifact = varying_artifact(records)
    image, coverage, report = render_artifact(artifact, width=2, height=2,
                                              depth_test=True, max_fragments=fragments)
    assert np.all(image == [255, 0, 0, 255])
    assert np.all(coverage == fragments // 4)
    assert report["depth"] == "less" and report["fragments"] == fragments
    disabled, disabled_coverage, disabled_report = render_artifact(artifact, width=2, height=2, depth_test=False)
    assert np.all(disabled == quantized_rgba(layers[-1][1]))
    assert np.all(disabled_coverage == 2)
    assert disabled_report["depth"] == "disabled" and disabled_report["fragments"] == 8


@pytest.mark.parametrize("domain", ["vertices", "constants"])
def test_aggregate_capture_resource_last_valid_first_invalid(domain):
    artifact = synthetic_artifact()
    resource_limit = 32 << 20
    first = artifact["vertex_buffers"][0]["data_hex"] if domain == "vertices" else struct.pack("<f", 0.5).hex()
    padding = "00" * ((resource_limit - len(first)) // 4)
    slots = [first, padding, padding]
    assert sum(map(len, slots)) == resource_limit
    assert all(len(slot) < resource_limit for slot in slots)
    if domain == "vertices":
        shader_layout = ShaderLayout(artifact["shaders"]["vs"]["layout"])
        expected = fetch_vertices(artifact, shader_layout)
        artifact["vertex_buffers"] += [{"stride": 0, "data_hex": slot} for slot in slots[1:]]
        assert fetch_vertices(artifact, shader_layout) == expected
        artifact["vertex_buffers"][-1]["data_hex"] += "00"
        with pytest.raises(ValueError, match="aggregate resource budget"):
            fetch_vertices(artifact, shader_layout)
    else:
        shader_layout = ShaderLayout(layout("fragment", uniforms=[{"word_offset": 0, "type": "float", "bit_size": 32}]))
        assert pack_constants(shader_layout, slots) == {0: 0x3800}
        slots[-1] += "00"
        with pytest.raises(ValueError, match="aggregate resource budget"):
            pack_constants(shader_layout, slots)


def test_aggregate_instruction_budget_counts_vertices_and_fragments_at_boundary():
    artifact = varying_artifact([(position, (0.25, 0.5, 0.75, 1)) for position in QUAD])
    vertex_words, fragment_words = word_counts(artifact)
    vertex_total = 6 * vertex_words
    draw_total = vertex_total + 4 * fragment_words
    image, coverage, report = render_artifact(artifact, width=2, height=2, max_steps=draw_total)
    assert np.all(image == quantized_rgba([0.25, 0.5, 0.75, 1])) and np.all(coverage == 1)
    assert report["host_steps"] == draw_total
    for budget in (vertex_total - 1, draw_total - 1):
        assert budget > max(vertex_words, fragment_words)
        with pytest.raises(ValueError, match="aggregate instruction budget"):
            render_artifact(artifact, width=2, height=2, max_steps=budget)


def test_real_shader_invocation_instruction_budget_last_valid_first_invalid():
    records = [(position, (0.25, 0.5, 0.75, 1)) for position in QUAD]
    artifact = varying_artifact(records)
    shader = artifact["shaders"]["vs"]
    shader_layout = ShaderLayout(shader["layout"])
    budget = word_counts(artifact)[0]
    inputs = {offset: pack_float(value) for offset, value in enumerate((*records[0][0], *records[0][1]))}
    executor = ShaderExecutor(shader["assembly"], shader_layout, max_steps=budget)
    try:
        outputs = executor.invoke(inputs)
        assert outputs == {**{offset: inputs[offset] for offset in range(4)},
                           **{12 + component: inputs[4 + component] for component in range(4)}}
        assert executor.steps == budget and executor.invocations == 1
    finally:
        executor.close()
    executor = ShaderExecutor(shader["assembly"], shader_layout, max_steps=budget - 1)
    try:
        with pytest.raises(ExecutionTimeoutError, match="max_steps"):
            executor.invoke(inputs)
        assert executor.steps == 0 and executor.invocations == 0
    finally:
        executor.close()


def test_fragment_invocation_cap_accepts_exact_boundary_and_rejects_one_less():
    artifact = varying_artifact([(position, (1, 0, 0, 1)) for position in QUAD])
    image, coverage, report = render_artifact(artifact, width=2, height=2, max_fragments=4)
    assert np.all(image == [255, 0, 0, 255]) and np.all(coverage == 1)
    assert report["fragments"] == 4
    with pytest.raises(ValueError, match="aggregate fragment budget"):
        render_artifact(artifact, width=2, height=2, max_fragments=3)


@pytest.mark.parametrize("budget", [True, 0, 65537, 1.5], ids=["boolean", "zero", "over-limit", "noninteger"])
def test_fragment_invocation_cap_rejects_invalid_metadata(budget):
    with pytest.raises(ValueError, match="fragment budget"):
        render_artifact(synthetic_artifact(), max_fragments=budget)


def test_declared_discard_requires_output_commit_contract():
    artifact = synthetic_artifact()
    artifact["shaders"]["fs"]["layout"]["uses_discard"] = True
    with pytest.raises(ValueError, match="discard.*output-commit contract"):
        render_artifact(artifact, width=1, height=1)


@pytest.mark.parametrize("path,value", [
    (("shaders",), None),
    (("shaders", "fs"), None),
    (("draw",), None),
    (("constants",), None),
    (("vertex_buffers", 0), []),
    (("vertex_elements", 0), None),
    (("vertex_elements", 0, "format"), 64),
    (("draw", "instance_count"), True),
    (("draw", "start_instance"), False),
    (("draw", "primitive_restart"), 0),
    (("vertex_buffers", 0, "data_hex"), "zz"),
    (("vertex_buffers", 0, "data_hex"), "0"),
], ids=["null-shaders", "null-fragment", "null-draw", "null-constants", "nonobject-buffer",
        "null-element", "nonstring-format", "boolean-instance-count", "boolean-start-instance",
        "nonboolean-restart", "nonhex-payload", "odd-hex-payload"])
def test_malformed_capture_metadata_is_rejected_as_validation_error(path, value):
    artifact = synthetic_artifact()
    container = artifact
    for key in path[:-1]:
        container = container[key]
    container[path[-1]] = value
    with pytest.raises(ValueError):
        render_artifact(artifact, width=1, height=1)
