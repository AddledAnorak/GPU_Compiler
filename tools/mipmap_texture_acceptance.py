"""Fresh explicit-LOD mipmap GL acceptance, with independent Fraction axis oracles."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np

from tools.host_executor import HostExecutorServer
from tools.host_render import render_artifact
from tools.host_texture import SAMPLER_STATE, TEXTURE_BASE, pack_texture
from tools.linear_texture_acceptance import MAX_CHANNEL_ERROR, TextureCase, case_mesh, image_comparison, shader_directory
from tools.mgcdriver import ProgramFormat
from tools.mirrored_texture_acceptance import MAX_DRAW_STEPS, mirrored_coordinates, resource_observations
from tools.mixed_texture_acceptance import WRAPS, rational_sample as axis_sample
from tools.run_render import capture_draw, native_host_readback, softpipe_reference
from tools.run_shader import ROOT
from tools.shader_layout import ShaderLayout
from tools.texture_flow_acceptance import STATUS


MIPMAP_MAGIC = b"MGCMIP1\0"
MIPMAP_HEADER = struct.Struct("<8sIII")
MAX_LEVELS = 5
MAX_MIPMAP_BYTES = 20 + 4 * (256 + 64 + 16 + 4 + 1)


@dataclass(frozen=True)
class MipmapCase(TextureCase):
    texture_filter: str = "nearest"
    texture_wrap_s: str = "clamp-to-edge"
    texture_wrap_t: str = "clamp-to-edge"
    lod: int = 0


_LOD_PROFILES = (
    ("identity", "clamp-to-edge", "clamp-to-edge"),
    ("negative-seams", "repeat", "clamp-to-edge"),
    ("even-odd-seams", "mirrored-repeat", "repeat"),
    ("clamp", "clamp-to-edge", "mirrored-repeat"),
    ("multiple-periods", "repeat", "mirrored-repeat"),
)
CASES = tuple(
    MipmapCase(f"{texture_filter}-lod{lod}-16x16", 16, 16, phase=phase,
               texture_filter=texture_filter, texture_wrap_s=wrap_s, texture_wrap_t=wrap_t, lod=lod)
    for texture_filter in ("nearest", "linear")
    for lod, (phase, wrap_s, wrap_t) in enumerate(_LOD_PROFILES)
) + tuple(
    MipmapCase(f"{texture_filter}-{name}", width, height, phase=phase, variant=variant,
               depth_test=depth, repeats=repeats, texture_filter=texture_filter,
               texture_wrap_s=wrap_s, texture_wrap_t=wrap_t, lod=lod)
    for texture_filter in ("nearest", "linear")
    for name, width, height, lod, phase, wrap_s, wrap_t, variant, depth, repeats in (
        ("lod1-asymmetric-3x5", 3, 5, 1, "negative-seams", "repeat", "clamp-to-edge", "sample", False, 1),
        ("lod1-asymmetric-5x3", 5, 3, 1, "even-odd-seams", "mirrored-repeat", "repeat", "sample", False, 1),
        ("lod4-last-level-3x5", 3, 5, 4, "clamp", "clamp-to-edge", "mirrored-repeat", "sample", False, 1),
        ("lod4-zero-alpha-1x1", 1, 1, 4, "multiple-periods", "mirrored-repeat", "clamp-to-edge", "sample", False, 1),
        ("lod0-seams-5x3", 5, 3, 0, "seams", "clamp-to-edge", "repeat", "sample", False, 1),
        ("lod1-late-kill-3x5", 3, 5, 1, "identity", "repeat", "mirrored-repeat", "late", False, 1),
        ("lod1-early-kill-5x3", 5, 3, 1, "identity", "mirrored-repeat", "clamp-to-edge", "early", False, 1),
        ("lod4-depth-loop-repeated-1x1", 1, 1, 4, "identity", "clamp-to-edge", "repeat", "runtime-loop", True, 2),
    )
)
DEFAULT_CASES = tuple(case for case in CASES if case.name in {
    f"{texture_filter}-{name}"
    for texture_filter, asymmetric in (("nearest", "3x5"), ("linear", "5x3"))
    for name in ("lod0-16x16", f"lod1-asymmetric-{asymmetric}", "lod2-16x16", "lod3-16x16",
                 "lod4-depth-loop-repeated-1x1")
})


def mipmap_dimensions(width, height):
    if any(type(extent) is not int or not 1 <= extent <= 16 for extent in (width, height)):
        raise ValueError("mipmap base dimensions must be exact integers in [1, 16]")
    dimensions = [(width, height)]
    while width != 1 or height != 1:
        width, height = max(1, width // 2), max(1, height // 2)
        dimensions.append((width, height))
    return tuple(dimensions)


def _validate_levels(levels):
    if type(levels) not in (tuple, list) or not 1 <= len(levels) <= MAX_LEVELS:
        raise ValueError("mipmaps require one complete bounded chain of 1..5 levels")
    for texels in levels:
        if (not isinstance(texels, np.ndarray) or texels.dtype != np.uint8 or texels.ndim != 3 or
                texels.shape[2] != 4 or not texels.flags.c_contiguous):
            raise ValueError("every mipmap level must be tightly packed RGBA8 input texels")
    dimensions = mipmap_dimensions(levels[0].shape[1], levels[0].shape[0])
    if len(levels) != len(dimensions) or any(texels.shape != (height, width, 4)
                                           for texels, (width, height) in zip(levels, dimensions)):
        raise ValueError("mipmaps require every consecutive floor-half level through 1x1")
    return dimensions


def input_levels(width, height):
    """Independent per-level patterns; no downsampling or generated lower levels."""
    levels = []
    for level, (level_width, level_height) in enumerate(mipmap_dimensions(width, height)):
        rows, columns = np.indices((level_height, level_width))
        texels = np.empty((level_height, level_width, 4), dtype=np.uint8)
        texels[:, :, 0] = (17 + 53 * level + 37 * columns + 19 * rows) % 256
        texels[:, :, 1] = (127 + 29 * level + 11 * columns + 47 * rows) % 256
        texels[:, :, 2] = (250 + 71 * level + 61 * columns + 7 * rows) % 256
        texels[:, :, 3] = (31 * level + 67 * columns + 43 * rows) % 256
        if width == height == 1:
            texels[0, 0] = [13, 127, 250, 0]
        levels.append(texels)
    return tuple(levels)


def encode_mipmaps(levels):
    dimensions = _validate_levels(levels)
    width, height = dimensions[0]
    return MIPMAP_HEADER.pack(MIPMAP_MAGIC, width, height, len(levels)) + b"".join(
        texels.tobytes() for texels in levels)


def decode_mipmaps(contents):
    if type(contents) is not bytes or not MIPMAP_HEADER.size < len(contents) <= MAX_MIPMAP_BYTES:
        raise ValueError("mipmap input must contain a bounded 20-byte header and exact RGBA8 levels")
    magic, width, height, count = MIPMAP_HEADER.unpack_from(contents)
    dimensions = mipmap_dimensions(width, height)
    if magic != MIPMAP_MAGIC or count != len(dimensions):
        raise ValueError("mipmap input requires MGCMIP1 and the complete bounded level count")
    if len(contents) != MIPMAP_HEADER.size + sum(width * height * 4 for width, height in dimensions):
        raise ValueError("mipmap input must contain all levels without truncation or trailing bytes")
    offset, levels = MIPMAP_HEADER.size, []
    for width, height in dimensions:
        size = width * height * 4
        levels.append(np.frombuffer(contents, dtype=np.uint8, count=size, offset=offset).reshape(height, width, 4).copy())
        offset += size
    return tuple(levels)


def selected_level(levels, lod):
    _validate_levels(levels)
    if type(lod) is not int or not 0 <= lod <= 4:
        raise ValueError("mipmap source LOD must be an exact explicit integer in [0, 4]")
    return min(lod, len(levels) - 1)


def rational_sample(levels, horizontal, vertical, texture_filter="nearest", *, lod=0,
                    texture_wrap_s="clamp-to-edge", texture_wrap_t="clamp-to-edge"):
    """Select uploaded input then use independent Fraction per-axis sampling, never packing."""
    texels = levels[selected_level(levels, lod)]
    return axis_sample(texels, Fraction(horizontal), Fraction(vertical), texture_filter,
                       texture_wrap_s=texture_wrap_s, texture_wrap_t=texture_wrap_t)


def rational_oracle(levels, *, lod=0, phase="identity", texture_filter="nearest", discard=False,
                    texture_wrap_s="clamp-to-edge", texture_wrap_t="clamp-to-edge"):
    selected = selected_level(levels, lod)
    if type(discard) is not bool:
        raise ValueError("mipmap oracle discard must be explicitly boolean")
    image = np.zeros((16, 16, 4), dtype=np.uint8)
    coverage = np.ones((16, 16), dtype=np.uint32)
    for row in range(16):
        for column in range(16):
            horizontal, vertical = mirrored_coordinates(row, column, phase)
            channels = axis_sample(levels[selected], horizontal, vertical, texture_filter,
                                   texture_wrap_s=texture_wrap_s, texture_wrap_t=texture_wrap_t)
            if discard and column < 8:
                coverage[row, column] = 0
            else:
                image[row, column] = [round(channel) for channel in channels]
    return image, coverage


def changed_levels(levels, level):
    _validate_levels(levels)
    if type(level) is not int or not 0 <= level < len(levels):
        raise ValueError("control must change one existing integer mipmap level")
    changed = tuple(texels.copy() for texels in levels)
    np.bitwise_xor(changed[level], np.uint8(255), out=changed[level])
    return changed


def texture_contract(case):
    if (type(case.lod) is not int or not 0 <= case.lod <= 4 or
            type(case.texture_filter) is not str or case.texture_filter not in ("nearest", "linear") or
            any(type(wrap) is not str or wrap not in WRAPS for wrap in (case.texture_wrap_s, case.texture_wrap_t))):
        raise ValueError("mipmap acceptance requires canonical filter, S/T wraps and explicit integer LOD")
    return {"version": 10 if case.texture_filter == "linear" else 9, "texture": 0, "sampler": 0,
            "storage_block": 15, "max_width": 16, "max_height": 16, "filter": case.texture_filter,
            "wrap_s": case.texture_wrap_s, "wrap_t": case.texture_wrap_t, "lod": case.lod,
            "mip_filter": "nearest", "max_levels": 5}


def captured_texture(levels, case):
    dimensions = _validate_levels(levels)
    texture_contract(case)
    sampler = {**SAMPLER_STATE, "min_filter": case.texture_filter, "mag_filter": case.texture_filter,
               "wrap_s": case.texture_wrap_s, "wrap_t": case.texture_wrap_t, "mip_filter": "nearest"}
    return {"width": dimensions[0][0], "height": dimensions[0][1], "format": "R8G8B8A8_UNORM",
            "data_hex": levels[0].tobytes().hex(), "sampler": sampler,
            "mipmaps": [{"level": level, "width": width, "height": height, "data_hex": texels.tobytes().hex()}
                        for level, (texels, (width, height)) in enumerate(zip(levels, dimensions)) if level]}


def _shader(case, directory):
    texture_contract(case)
    template = shader_directory(replace(case, phase="identity"))
    vertex_template = shader_directory(replace(case, phase="identity", variant="sample"))
    vertex = (vertex_template / "vert.glsl").read_text()
    fragment = (template / "frag.glsl").read_text()
    expressions = {"identity": None, "clamp": "v_uv * 4.0 - vec2(1.5)",
                   "seams": "v_uv * 2.0 - vec2(0.0625)",
                   "negative-seams": "v_uv * 2.0 - vec2(1.0625)",
                   "even-odd-seams": "v_uv * 4.0 - vec2(0.125)",
                   "multiple-periods": "v_uv * 12.0 - vec2(4.375)"}
    assignment = "v_uv = (a_position.xy + vec2(1.0)) * 0.5;"
    original = "texture2D(u_texture, v_uv)"
    if (case.phase not in expressions or vertex.count(assignment) != 1 or fragment.count(original) != 1 or
            not fragment.startswith("#version 100\n") or (case.variant != "sample" and case.phase != "identity")):
        raise ValueError("mipmap shader requires the retained varying/attribute fixture and identity kill coordinates")
    if expressions[case.phase] is not None:
        vertex = vertex.replace(assignment, assignment + f"\n    v_uv = {expressions[case.phase]};")
    vertex = vertex.replace("#version 100\n", "#version 120\n", 1).replace("precision mediump float;\n", "")
    fragment = fragment.replace("#version 100\n", "#version 120\n#extension GL_ARB_shader_texture_lod : require\n", 1)
    fragment = fragment.replace("precision mediump float;\n", "")
    fragment = fragment.replace(original, f"texture2DLod(u_texture, v_uv, {case.lod}.0)")
    shader = directory / "shader"
    shader.mkdir(parents=True, exist_ok=True)
    (shader / "vert.glsl").write_text(vertex)
    (shader / "frag.glsl").write_text(fragment)
    return shader


def check_contract(artifact, levels, case):
    selected = selected_level(levels, case.lod)
    vertex = ShaderLayout(artifact["shaders"]["vs"]["layout"])
    fragment = ShaderLayout(artifact["shaders"]["fs"]["layout"])
    contract = texture_contract(case)
    if (artifact.get("schema") != "mgc-host-draw-v2" or artifact.get("program_format") != ProgramFormat.LEGACY.value or
            type(artifact.get("shader_abi_version")) is not int or artifact["shader_abi_version"] != 1 or
            vertex.source["version"] != 1 or vertex.host_texture is not None or vertex.fragment_status is not None or
            fragment.source["version"] != 2 or fragment.host_texture != contract or fragment.fragment_status != STATUS or
            fragment.source.get("uses_discard", False) is not (case.variant != "sample")):
        raise ValueError(f"{case.name}: fresh mipmap reflection/layout2/ABI1/status differs")
    expected = [captured_texture(levels, case)]
    if json.dumps(artifact.get("textures"), sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError(f"{case.name}: capture changed uploaded level bytes/metadata or actual GL mip sampler")
    texels = levels[selected]
    scalar_values = np.array([texels.shape[1], texels.shape[0], 0, 0] +
                             [int(channel) / 255.0 for channel in texels.reshape(-1)], dtype=np.float16).view(np.uint16)
    expected_words = {TEXTURE_BASE + offset: int(value) for offset, value in enumerate(scalar_values)}
    if pack_texture(fragment, artifact["textures"]) != expected_words:
        raise ValueError(f"{case.name}: UBO15 must pack only selected uploaded dimensions/channels, not output pixels")
    if case.variant == "runtime-loop" and "mgc_block_" not in artifact["shaders"]["fs"]["assembly"]:
        raise ValueError(f"{case.name}: actual explicit-LOD runtime-loop lowering was not retained")


def _check_execution(coverage, report, expected_coverage, case, draw_index=0):
    if case.depth_test and draw_index:
        expected_coverage = np.zeros_like(expected_coverage)
    committed = int(expected_coverage.sum())
    discarded = (256 if case.depth_test else 128) if case.variant != "sample" else 0
    version = texture_contract(case)["version"]
    expected = {"vertices": 12 if case.depth_test else 6, "triangles": 4 if case.depth_test else 2,
                "fragments": committed + discarded, "committed_fragments": committed,
                "discarded_fragments": discarded, "covered_pixels": committed, "max_overdraw": int(committed > 0),
                "fpga_execution": False, "program_format": ProgramFormat.LEGACY.value, "shader_abi_version": 1,
                "fragment_completion": "explicit legacy host status v1",
                "texture_sampling": f"bounded legacy {case.texture_filter}-S:{case.texture_wrap_s}/T:{case.texture_wrap_t} "
                                    f"v{version} explicit integer LOD {case.lod} nearest-mip"}
    if (coverage.dtype != np.uint32 or not np.array_equal(coverage, expected_coverage) or
            any(type(report.get(name)) is not type(value) or report.get(name) != value for name, value in expected.items()) or
            type(report.get("host_steps")) is not int or not 256 < report["host_steps"] <= MAX_DRAW_STEPS):
        raise ValueError(f"{case.name}: mipmap coverage/status/ABI/budget differs on draw {draw_index + 1}")


def _native(shader, build, output, options, case, levels, expected_coverage):
    with tempfile.TemporaryDirectory(prefix="mgc-mip-") as temporary:
        with HostExecutorServer(Path(temporary) / "executor.sock", max_requests=case.repeats,
                                max_steps=MAX_DRAW_STEPS) as server:
            image = native_host_readback(shader, build, output, socket_path=server.path, repeats=case.repeats, **options)
        records = list(server.records)
        if server.errors or len(records) != case.repeats:
            raise ValueError(f"{case.name}: native mipmap requests failed: {server.errors}")
    report = json.loads((output / "native-report.json").read_text())
    expected_wrap = case.texture_wrap_s if case.texture_wrap_s == case.texture_wrap_t else "mixed"
    if (report.get("native_gl_readback") is not True or report.get("fpga_execution") is not False or
            report.get("mipmaps") is not True or
            report.get("texture_filter") != case.texture_filter or report.get("texture_wrap") != expected_wrap or
            report.get("texture_wrap_s") != case.texture_wrap_s or report.get("texture_wrap_t") != case.texture_wrap_t or
            len(report.get("capture_paths", [])) != case.repeats or not np.array_equal(image, records[-1]["rgba"]) or
            [commit["sequence"] for commit in report["commits"]] != [record["sequence"] for record in records]):
        raise ValueError(f"{case.name}: native glReadPixels differs from actual executor replies/GL state")
    expected_depth = np.full((16, 16), 65535, dtype=np.uint16)
    expected_depth[:, 8:] = 16384
    for draw_index, (record, capture_path) in enumerate(zip(records, report["capture_paths"])):
        check_contract(record["artifact"], levels, case)
        resource_observations(record["artifact"])
        if json.loads(Path(capture_path).read_text()) != record["artifact"]:
            raise ValueError(f"{case.name}: immutable native C capture differs from the executed artifact")
        _check_execution(record["coverage"], record["report"], expected_coverage, case, draw_index)
        if not np.array_equal(record["rgba"], image):
            raise ValueError(f"{case.name}: repeated draw changed retained color commits")
        if case.depth_test and not np.array_equal(record["depth"], expected_depth):
            raise ValueError(f"{case.name}: killed or equal-depth fragments changed Z16")
    if case.depth_test:
        (output / "depth.z16").write_bytes(records[-1]["depth"].astype("<u2").tobytes())
    return image, report, [record["report"] for record in records]


def _evaluate(directory, build, case, levels, shader, mesh):
    directory.mkdir(parents=True, exist_ok=True)
    texture = directory / "texture.bin"
    contents = encode_mipmaps(levels)
    texture.write_bytes(contents)
    options = {"mesh": mesh, "texture": texture, "texture_filter": case.texture_filter,
               "texture_wrap_s": case.texture_wrap_s, "texture_wrap_t": case.texture_wrap_t,
               "depth_test": case.depth_test, "mipmaps": True}
    oracle_options = {"lod": case.lod, "phase": case.phase, "texture_filter": case.texture_filter,
                      "texture_wrap_s": case.texture_wrap_s, "texture_wrap_t": case.texture_wrap_t,
                      "discard": case.variant != "sample"}
    expected, expected_coverage = rational_oracle(levels, **oracle_options)
    artifact = capture_draw(shader, build, directory / "capture", host_texture=True, **options)
    captured = json.dumps(artifact, sort_keys=True)
    check_contract(artifact, levels, case)
    resources = resource_observations(artifact)
    managed, coverage, report = render_artifact(artifact, depth_test=case.depth_test, max_steps=MAX_DRAW_STEPS)
    _check_execution(coverage, report, expected_coverage, case)
    if json.dumps(artifact, sort_keys=True) != captured:
        raise ValueError(f"{case.name}: managed execution mutated captured input or level metadata")
    native, native_report, draw_reports = _native(shader, build, directory / "native", options, case, levels, expected_coverage)
    reference = softpipe_reference(shader, build, directory / "reference", repeats=case.repeats, **options)
    tolerance = MAX_CHANNEL_ERROR if case.texture_filter == "linear" else 0
    images = {"managed": managed, "native": native, "softpipe": reference, "rational": expected}
    comparisons = {"native_vs_managed": image_comparison(native, managed, 0),
                   "managed_vs_rational": image_comparison(managed, expected, tolerance),
                   "native_vs_rational": image_comparison(native, expected, tolerance),
                   "softpipe_vs_rational": image_comparison(reference, expected, tolerance),
                   "native_vs_softpipe": image_comparison(native, reference, tolerance)}
    if not all(comparison["passed"] for comparison in comparisons.values()):
        raise ValueError(f"{case.name}: same-input explicit-LOD full-RGBA comparisons failed: {comparisons}")
    nearest, _ = rational_oracle(levels, **{**oracle_options, "texture_filter": "nearest"})
    interpolation_required = case.texture_filter == "linear" and image_comparison(expected, nearest, 0)["max_channel_error"] > tolerance
    nearest_controls = {name: image_comparison(images[name], nearest, 0) for name in ("native", "softpipe")}
    if interpolation_required and any(check["max_channel_error"] <= tolerance for check in nearest_controls.values()):
        raise ValueError(f"{case.name}: linear selected-level interpolation collapsed to nearest")
    if texture.read_bytes() != contents or encode_mipmaps(levels) != contents:
        raise ValueError(f"{case.name}: GL or managed execution mutated uploaded mipmap bytes")
    for name, image in images.items():
        (directory / f"{name}.rgba8").write_bytes(image.tobytes())
    (directory / "coverage.u32").write_bytes(coverage.astype("<u4").tobytes())
    return images, {"comparisons": comparisons, "managed_report": report, "native_draw_reports": draw_reports,
                    "resources": resources, "interpolation_required": interpolation_required, "nearest_controls": nearest_controls,
                    "maximum_channel_error": max(check["max_channel_error"] for check in comparisons.values()),
                    "tested_channel_tolerance": tolerance,
                    "evidence": {"draw": str(directory / "capture/draw.json"), "shader_directory": str(shader),
                                 "mesh": str(mesh), "texture": str(texture), "texture_sha256": hashlib.sha256(contents).hexdigest(),
                                 "native_report": str(directory / "native/native-report.json"), "native_captures": native_report["capture_paths"],
                                 "native_raw_bottom_left": str(directory / "native/native.rgba8"),
                                 "softpipe_raw_bottom_left": str(directory / "reference/softpipe.rgba8")}}


def _run_case(directory, build, case):
    directory.mkdir(parents=True, exist_ok=True)
    shader = _shader(case, directory)
    mesh = directory / "vertices.bin"
    mesh.write_bytes(case_mesh(case))
    levels = input_levels(case.width, case.height)
    selected = selected_level(levels, case.lod)
    images, result = _evaluate(directory, build, case, levels, shader, mesh)
    controls = {}
    for name, level in (("selected", selected), ("unselected", next((level for level in range(len(levels)) if level != selected), None))):
        if level is None:
            controls[name] = {"applicable": False, "reason": "complete 1x1 chain has no unselected level"}
            continue
        changed = changed_levels(levels, level)
        controlled, evidence = _evaluate(directory / f"{name}-control", build, case, changed, shader, mesh)
        checks = {channel: image_comparison(images[channel], controlled[channel], 0) for channel in images}
        if name == "selected" and any(check["max_channel_error"] <= result["tested_channel_tolerance"] for check in checks.values()):
            raise ValueError(f"{case.name}: changing only the selected uploaded level did not change expected/actual RGBA")
        if name == "unselected" and any(not check["passed"] for check in checks.values()):
            raise ValueError(f"{case.name}: changing only an unselected uploaded level changed expected/actual RGBA")
        controls[name] = {"applicable": True, "changed_level": level, "image_changes": checks, "result": evidence}
    maximum = max([result["maximum_channel_error"]] + [control["result"]["maximum_channel_error"]
                   for control in controls.values() if control["applicable"]])
    return {"case": case.name, "passed": True, "texture_width": case.width, "texture_height": case.height,
            "texture_filter": case.texture_filter, "texture_wrap_s": case.texture_wrap_s, "texture_wrap_t": case.texture_wrap_t,
            "source_lod": case.lod, "selected_level": selected, "level_dimensions": [list(pair) for pair in _validate_levels(levels)],
            "phase": case.phase, "variant": case.variant, "depth_test": case.depth_test, "repeats": case.repeats,
            **result, "maximum_channel_error": maximum, "input_controls": controls,
            "coverage_oracle": "explicit status commits, never an alpha/RGBA proxy",
            "depth_oracle": "right half 16384 Z16; killed left half 65535; second draw has zero commits" if case.depth_test else None}


def _extension_unavailable(directory, error):
    diagnostics = [str(error)]
    for path in directory.rglob("*.log"):
        if path.is_file() and path.stat().st_size <= 1 << 20:
            diagnostics.append(path.read_text(errors="replace"))
    return any(("GL_ARB_shader_texture_lod" in diagnostic and any(
        marker in diagnostic.lower() for marker in ("unavailable", "not advertised", "not supported", "unsupported", "require advertised"))) or
        "MGC_MIPMAP_EXTENSION_UNAVAILABLE" in diagnostic for diagnostic in diagnostics)


def validate(output, build, *, cases=DEFAULT_CASES):
    cases = tuple(cases)
    if not cases or any(case not in CASES for case in cases) or len({case.name for case in cases}) != len(cases):
        raise ValueError("mipmap acceptance requires distinct named bounded cases")
    output, build = Path(output), Path(build)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"passed": False, "status": "running", "cases": [], "tested_cases": [case.name for case in cases],
               "native_gl_readback": False, "fpga_execution": False, "exact_fp32_conformance": False,
               "texture_lod_extension": "GL_ARB_shader_texture_lod", "glsl_profile": "desktop-120", "extension_available": None,
               "linear_tested_channel_tolerance": MAX_CHANNEL_ERROR, "nearest_tested_channel_tolerance": 0,
               "oracle": "selected uploaded level min(explicit source LOD,last); independent Fraction per-axis taps and nearest-even RGBA8 rounding",
               "contract": "host v9 nearest/v10 linear explicit shader integer LOD0..4; layout2/ABI1, UBO15 selected-input RGBA8, nearest-mip, complete <=5 levels",
               "aggregate_draw_step_budget": MAX_DRAW_STEPS,
               "input_precision": "uploaded RGBA8/255 texels and vertex/varying values use binary16 in CPU-host execution",
               "scope": "fresh same-input managed/native CPU-host GL readback and independent CPU softpipe; not FPGA, implicit derivatives, trilinear or universal FP32 conformance"}
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    for case in cases:
        try:
            result = _run_case(output / case.name, build, case)
        except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
            unavailable = _extension_unavailable(output / case.name, error)
            summary.update(status="unavailable" if unavailable else "error", failed_case=case.name, message=str(error))
            if unavailable:
                summary["extension_available"] = False
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            raise
        (output / case.name / "render.json").write_text(json.dumps(result, indent=2) + "\n")
        summary["cases"].append(result)
    summary.update(passed=True, status="passed", native_gl_readback=True, extension_available=True,
                   maximum_channel_error=max(result["maximum_channel_error"] for result in summary["cases"]))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build")))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/mipmap-texture")
    parser.add_argument("--cases", nargs="+", choices=[case.name for case in CASES], help="explicit case names; defaults to DEFAULT_CASES")
    parser.add_argument("--extended", action="store_true", help="run all 26 cases when no explicit --cases are supplied")
    args = parser.parse_args(argv)
    try:
        selected = (CASES if args.extended else DEFAULT_CASES) if args.cases is None else tuple(
            next(case for case in CASES if case.name == name) for name in args.cases)
        summary = validate(args.output, args.build, cases=selected)
        print(json.dumps({"passed": True, "tested_cases": summary["tested_cases"],
                          "maximum_channel_error": summary["maximum_channel_error"], "native_gl_readback": True,
                          "fpga_execution": False, "exact_fp32_conformance": False, "summary": str(args.output / "summary.json")}, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
