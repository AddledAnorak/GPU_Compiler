"""Independent full-RGBA mirrored-repeat oracle and fresh managed/native GL evidence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

import numpy as np

from tools.host_executor import HostExecutorServer
from tools.host_render import render_artifact
from tools.host_texture import host_texture_contract, SAMPLER_STATE
from tools.linear_texture_acceptance import (MAX_CHANNEL_ERROR, TextureCase, case_mesh,
                                             image_comparison, input_texels, pixel_coordinates,
                                             rational_sample as clamp_sample, shader_directory)
from tools.mgcasm import assemble
from tools.mgcdriver import ProgramFormat
from tools.repeat_texture_acceptance import rational_sample as repeat_sample
from tools.run_render import capture_draw, native_host_readback, softpipe_reference
from tools.run_shader import ROOT
from tools.shader_layout import ShaderLayout
from tools.texture_flow_acceptance import STATUS


MAX_DRAW_STEPS = 2_000_000


@dataclass(frozen=True)
class MirroredCase(TextureCase):
    phase: str = "clamp"
    texture_wrap: str = "mirrored-repeat"


CASES = tuple(MirroredCase(f"{texture_filter}-mirrored-repeat-{name}", width, height,
                           phase=phase, variant=variant, depth_test=depth, repeats=repeats,
                           texture_filter=texture_filter)
              for texture_filter in ("nearest", "linear")
              for name, width, height, phase, variant, depth, repeats in (
                  ("2x2", 2, 2, "clamp", "sample", False, 1),
                  ("asymmetric-3x5", 3, 5, "clamp", "sample", False, 1),
                  ("asymmetric-5x3", 5, 3, "clamp", "sample", False, 1),
                  ("zero-alpha-1x1", 1, 1, "clamp", "sample", False, 1),
                  ("16x16", 16, 16, "clamp", "sample", False, 1),
                  ("seams-4x4", 4, 4, "seams", "sample", False, 1),
                  ("negative-seams-3x5", 3, 5, "negative-seams", "sample", False, 1),
                  ("even-odd-seams-5x3", 5, 3, "even-odd-seams", "sample", False, 1),
                  ("multiple-periods-3x5", 3, 5, "multiple-periods", "sample", False, 1),
                  ("late-kill-2x2", 2, 2, "clamp", "late", False, 1),
                  ("early-kill-4x4", 4, 4, "clamp", "early", False, 1),
                  ("loop-kill-3x5", 3, 5, "clamp", "runtime-loop", False, 1),
                  ("depth-loop-repeated-1x1", 1, 1, "clamp", "runtime-loop", True, 2),
                  ("depth-loop-repeated-2x2", 2, 2, "clamp", "runtime-loop", True, 2),
              ))

DEFAULT_CASES = tuple(case for case in CASES if case.name in {
    f"{texture_filter}-mirrored-repeat-{name}"
    for texture_filter in ("nearest", "linear")
    for name in ("2x2", "negative-seams-3x5", "even-odd-seams-5x3", "16x16",
                 "depth-loop-repeated-1x1")
})


def mirrored_coordinates(row, column, phase="clamp"):
    if phase in ("identity", "clamp"):
        return pixel_coordinates(row, column, phase)
    transforms = {"seams": (2, Fraction(1, 16)),
                  "negative-seams": (2, Fraction(17, 16)),
                  "even-odd-seams": (4, Fraction(1, 8)),
                  "multiple-periods": (12, Fraction(35, 8))}
    if phase not in transforms:
        raise ValueError("unsupported mirrored coordinate case")
    horizontal, vertical = pixel_coordinates(row, column)
    scale, offset = transforms[phase]
    return horizontal * scale - offset, vertical * scale - offset


def _validate_texels(texels):
    if (not isinstance(texels, np.ndarray) or texels.dtype != np.uint8 or
            texels.ndim != 3 or texels.shape[2] != 4 or
            not all(1 <= extent <= 16 for extent in texels.shape[:2])):
        raise ValueError("mirror oracle requires bounded 1..16 RGBA8 input texels")


def mirror_fold(coordinate):
    coordinate = Fraction(coordinate)
    period = coordinate - (coordinate / 2).__floor__() * 2
    return min(period, 2 - period)


def rational_sample(texels, horizontal, vertical, texture_filter="linear"):
    """Exact period-two fold, edge-clamped taps and Fraction byte-unit weights."""
    _validate_texels(texels)
    height, width = texels.shape[:2]
    horizontal, vertical = mirror_fold(horizontal), mirror_fold(vertical)
    if texture_filter == "nearest":
        column = min(max((horizontal * width).__floor__(), 0), width - 1)
        row = min(max((vertical * height).__floor__(), 0), height - 1)
        return tuple(Fraction(int(channel)) for channel in texels[row, column])
    if texture_filter != "linear":
        raise ValueError("unsupported rational mirrored filter")
    horizontal_position = horizontal * width - Fraction(1, 2)
    vertical_position = vertical * height - Fraction(1, 2)
    first_column, first_row = horizontal_position.__floor__(), vertical_position.__floor__()
    horizontal_weight = horizontal_position - first_column
    vertical_weight = vertical_position - first_row
    return tuple(sum(Fraction(int(texels[min(max(row, 0), height - 1),
                                         min(max(column, 0), width - 1), component])) *
                     row_weight * column_weight
                     for row, row_weight in ((first_row, 1 - vertical_weight),
                                             (first_row + 1, vertical_weight))
                     for column, column_weight in ((first_column, 1 - horizontal_weight),
                                                   (first_column + 1, horizontal_weight)))
                 for component in range(4))


def rational_oracle(texels, *, phase="clamp", texture_filter="linear", discard=False,
                    texture_wrap="mirrored-repeat"):
    _validate_texels(texels)
    samplers = {"mirrored-repeat": rational_sample, "repeat": repeat_sample,
                "clamp-to-edge": clamp_sample}
    if type(texture_wrap) is not str or texture_wrap not in samplers:
        raise ValueError("unsupported rational oracle wrap")
    if type(discard) is not bool:
        raise ValueError("mirror oracle discard must be explicitly boolean")
    if type(texture_filter) is not str or texture_filter not in ("nearest", "linear"):
        raise ValueError("unsupported rational mirrored filter")
    image = np.zeros((16, 16, 4), dtype=np.uint8)
    coverage = np.ones((16, 16), dtype=np.uint32)
    for row in range(16):
        for column in range(16):
            horizontal, vertical = mirrored_coordinates(row, column, phase)
            if discard and column < 8:
                coverage[row, column] = 0
            else:
                image[row, column] = [round(channel) for channel in
                                      samplers[texture_wrap](texels, horizontal, vertical, texture_filter)]
    return image, coverage


def _version(case):
    return (2 if case.texture_filter == "linear" else 1) + {
        "clamp-to-edge": 0, "repeat": 2, "mirrored-repeat": 4}[case.texture_wrap]


def _shader(case, directory):
    template = shader_directory(replace(case, phase="identity"))
    if case.phase == "identity":
        return template
    if case.variant == "sample" and case.phase == "clamp":
        return shader_directory(case)
    expressions = {"clamp": "v_uv * 4.0 - vec2(1.5)",
                   "seams": "v_uv * 2.0 - vec2(0.0625)",
                   "negative-seams": "v_uv * 2.0 - vec2(1.0625)",
                   "even-odd-seams": "v_uv * 4.0 - vec2(0.125)",
                   "multiple-periods": "v_uv * 12.0 - vec2(4.375)"}
    source = (template / "frag.glsl").read_text()
    original = "texture2D(u_texture, v_uv)"
    if source.count(original) != 1 or case.phase not in expressions:
        raise ValueError("existing full-RGBA fixture must expose exactly one sample expression")
    shader = directory / "shader"
    shader.mkdir(parents=True, exist_ok=True)
    (shader / "vert.glsl").write_text((template / "vert.glsl").read_text())
    (shader / "frag.glsl").write_text(source.replace(original,
                                                    f"texture2D(u_texture, {expressions[case.phase]})"))
    return shader


def check_contract(artifact, texels, case):
    vertex = ShaderLayout(artifact["shaders"]["vs"]["layout"])
    fragment = ShaderLayout(artifact["shaders"]["fs"]["layout"])
    if (artifact.get("schema") != "mgc-host-draw-v2" or
            artifact.get("program_format") != ProgramFormat.LEGACY.value or
            type(artifact.get("shader_abi_version")) is not int or artifact["shader_abi_version"] != 1 or
            vertex.source["version"] != 1 or vertex.host_texture is not None or
            vertex.fragment_status is not None or fragment.source["version"] != 2 or
            fragment.host_texture != host_texture_contract(_version(case)) or
            fragment.fragment_status != STATUS or
            bool(fragment.source.get("uses_discard")) != (case.variant != "sample")):
        raise ValueError(f"{case.name}: fresh mirrored reflection/ABI/status contract differs")
    sampler = {**SAMPLER_STATE, "min_filter": case.texture_filter, "mag_filter": case.texture_filter,
               "wrap_s": case.texture_wrap, "wrap_t": case.texture_wrap}
    expected = {"width": case.width, "height": case.height, "format": "R8G8B8A8_UNORM",
                "data_hex": texels.tobytes().hex(), "sampler": sampler}
    if artifact.get("textures") != [expected]:
        raise ValueError(f"{case.name}: fresh capture changed uploaded bytes or actual GL S/T sampler")
    if case.variant == "runtime-loop" and "mgc_block_" not in artifact["shaders"]["fs"]["assembly"]:
        raise ValueError(f"{case.name}: compiled runtime-loop control flow was not retained")


def resource_observations(artifact):
    stages = {}
    for stage in ("vs", "fs"):
        assembly = artifact["shaders"][stage]["assembly"]
        words = assemble(assembly)
        registers = sorted({int(index) for index in re.findall(r"\br([0-9]+)\b", assembly)})
        if not words or len(words) * 4 > 4 << 20 or not registers or registers[-1] > 31:
            raise ValueError("mirrored compiler exceeds bounded assembly or scalar register aperture")
        stages[stage] = {"instruction_words": len(words), "assembly_bytes": len(assembly.encode()),
                         "referenced_scalar_registers": registers, "maximum_scalar_register": registers[-1]}
    return {"stages": stages, "aggregate_draw_step_budget": MAX_DRAW_STEPS,
            "per_invocation_step_budget": 100_000, "maximum_scalar_register": 31,
            "scope": "emitted instruction words and referenced physical scalar registers, not liveness or GPU cycles"}


def _check_execution(coverage, report, expected_coverage, case, draw_index=0):
    if case.depth_test and draw_index:
        expected_coverage = np.zeros_like(expected_coverage)
    committed = int(expected_coverage.sum())
    discarded = (256 if case.depth_test else 128) if case.variant != "sample" else 0
    expected = {"vertices": 12 if case.depth_test else 6, "triangles": 4 if case.depth_test else 2,
                "fragments": committed + discarded, "committed_fragments": committed,
                "discarded_fragments": discarded, "covered_pixels": committed,
                "max_overdraw": int(committed > 0), "fpga_execution": False,
                "fragment_completion": "explicit legacy host status v1",
                "texture_sampling": f"bounded legacy {case.texture_filter}-"
                                    f"{'clamp' if case.texture_wrap == 'clamp-to-edge' else case.texture_wrap} "
                                    f"v{_version(case)}"}
    if (coverage.dtype != np.uint32 or not np.array_equal(coverage, expected_coverage) or
            any(report.get(name) != value for name, value in expected.items()) or
            type(report.get("host_steps")) is not int or not 256 < report["host_steps"] <= MAX_DRAW_STEPS):
        raise ValueError(f"{case.name}: managed coverage/status/budget differs on draw {draw_index + 1}")


def _native(shader, build, output, options, case, texels, expected_coverage):
    with tempfile.TemporaryDirectory(prefix="mgc-mirror-") as temporary:
        socket_path = Path(temporary) / "executor.sock"
        with HostExecutorServer(socket_path, max_requests=case.repeats, max_steps=MAX_DRAW_STEPS) as server:
            image = native_host_readback(shader, build, output, socket_path=socket_path,
                                         repeats=case.repeats, **options)
        records = list(server.records)
        if server.errors or len(records) != case.repeats:
            raise ValueError(f"{case.name}: native managed requests failed: {server.errors}")
    report = json.loads((output / "native-report.json").read_text())
    if (report.get("texture_filter") != case.texture_filter or report.get("texture_wrap") != case.texture_wrap or
            len(report.get("capture_paths", [])) != case.repeats or
            not np.array_equal(image, records[-1]["rgba"]) or
            [commit["sequence"] for commit in report["commits"]] != [record["sequence"] for record in records]):
        raise ValueError(f"{case.name}: native glReadPixels differs from accepted executor replies")
    expected_depth = np.full((16, 16), 65535, dtype=np.uint16)
    expected_depth[:, 8:] = 16384
    for draw_index, (record, capture_path) in enumerate(zip(records, report["capture_paths"])):
        check_contract(record["artifact"], texels, case)
        resource_observations(record["artifact"])
        if json.loads(Path(capture_path).read_text()) != record["artifact"]:
            raise ValueError(f"{case.name}: native C capture differs from submitted artifact")
        _check_execution(record["coverage"], record["report"], expected_coverage, case, draw_index)
        if not np.array_equal(record["rgba"], image):
            raise ValueError(f"{case.name}: repeated draw changed retained color commits")
        if case.depth_test and not np.array_equal(record["depth"], expected_depth):
            raise ValueError(f"{case.name}: killed/equal-depth fragments changed Z16 depth")
    if case.depth_test:
        (output / "depth.z16").write_bytes(records[-1]["depth"].astype("<u2").tobytes())
    return image, report, [record["report"] for record in records]


def _run_case(directory, build, case):
    directory.mkdir(parents=True, exist_ok=True)
    shader = _shader(case, directory)
    mesh, texture = directory / "vertices.bin", directory / "texture.bin"
    texels = input_texels(case.width, case.height)
    mesh.write_bytes(case_mesh(case))
    texture.write_bytes(struct.pack("<II", case.width, case.height) + texels.tobytes())
    options = {"mesh": mesh, "texture": texture, "texture_filter": case.texture_filter,
               "texture_wrap": case.texture_wrap, "depth_test": case.depth_test}
    oracle_options = {"phase": case.phase, "texture_filter": case.texture_filter,
                      "texture_wrap": case.texture_wrap, "discard": case.variant != "sample"}
    expected, expected_coverage = rational_oracle(texels, **oracle_options)
    artifact = capture_draw(shader, build, directory / "capture", host_texture=True, **options)
    check_contract(artifact, texels, case)
    resources = resource_observations(artifact)
    managed, coverage, report = render_artifact(artifact, depth_test=case.depth_test, max_steps=MAX_DRAW_STEPS)
    _check_execution(coverage, report, expected_coverage, case)
    native, native_report, draw_reports = _native(shader, build, directory / "native", options,
                                                 case, texels, expected_coverage)
    reference = softpipe_reference(shader, build, directory / "reference", repeats=case.repeats, **options)
    tolerance = MAX_CHANNEL_ERROR if case.texture_filter == "linear" else 0
    comparisons = {"native_vs_managed": image_comparison(native, managed, 0),
                   "managed_vs_rational": image_comparison(managed, expected, tolerance),
                   "native_vs_rational": image_comparison(native, expected, tolerance),
                   "softpipe_vs_rational": image_comparison(reference, expected, tolerance),
                   "native_vs_softpipe": image_comparison(native, reference, tolerance)}
    if not all(comparison["passed"] for comparison in comparisons.values()):
        raise ValueError(f"{case.name}: same-input mirrored RGBA checks failed: {comparisons}")
    nearest, _ = rational_oracle(texels, **{**oracle_options, "texture_filter": "nearest"})
    interpolation_required = (case.texture_filter == "linear" and
                              image_comparison(expected, nearest, 0)["max_channel_error"] > tolerance)
    native_vs_nearest = image_comparison(native, nearest, 0)
    softpipe_vs_nearest = image_comparison(reference, nearest, 0)
    if interpolation_required and min(native_vs_nearest["max_channel_error"],
                                      softpipe_vs_nearest["max_channel_error"]) <= tolerance:
        raise ValueError(f"{case.name}: linear-mirrored-repeat collapsed to nearest")
    controls = {}
    if case.name in ("nearest-mirrored-repeat-2x2", "linear-mirrored-repeat-2x2"):
        for wrap in ("clamp-to-edge", "repeat"):
            control_case = replace(case, texture_wrap=wrap)
            control_options = {**options, "texture_wrap": wrap}
            control_expected, control_coverage = rational_oracle(texels, **{**oracle_options, "texture_wrap": wrap})
            control_native, control_report, _ = _native(shader, build, directory / f"{wrap}-native",
                                                        control_options, control_case, texels, control_coverage)
            control_reference = softpipe_reference(shader, build, directory / f"{wrap}-reference",
                                                   **control_options)
            control = {"same_input": True, "native_gl_readback": True,
                       "native_vs_rational": image_comparison(control_native, control_expected, tolerance),
                       "softpipe_vs_rational": image_comparison(control_reference, control_expected, tolerance),
                       "native_vs_softpipe": image_comparison(control_native, control_reference, tolerance),
                       "mirror_vs_control_native": image_comparison(native, control_native, 0),
                       "mirror_vs_control_softpipe": image_comparison(reference, control_reference, 0),
                       "host_texture_version": _version(control_case),
                       "native_commits": len(control_report["commits"]),
                       "native_report": str(directory / f"{wrap}-native/native-report.json")}
            if (not all(control[name]["passed"] for name in ("native_vs_rational", "softpipe_vs_rational",
                                                             "native_vs_softpipe")) or
                    min(control["mirror_vs_control_native"]["max_channel_error"],
                        control["mirror_vs_control_softpipe"]["max_channel_error"]) <= tolerance):
                raise ValueError(f"{case.name}: actual same-input {wrap} control failed: {control}")
            controls[wrap] = control
    for name, image in (("managed", managed), ("rational", expected), ("nearest", nearest)):
        (directory / f"{name}.rgba8").write_bytes(image.tobytes())
    (directory / "coverage.u32").write_bytes(coverage.astype("<u4").tobytes())
    return {"case": case.name, "passed": True, "texture_dimensions": [case.width, case.height],
            "texture_filter": case.texture_filter, "texture_wrap": case.texture_wrap,
            "coordinate_case": case.phase, "fixture": case.variant, "layout_version": 2,
            "shader_abi_version": 1, "host_texture_version": _version(case),
            "native_gl_readback": True, "reply_matches_gl_readback": True, "fpga_execution": False,
            "native_commits": len(draw_reports), "draw_repeats": case.repeats,
            "managed_report": report, "native_draw_reports": draw_reports, "comparisons": comparisons,
            "maximum_channel_error": max(comparison["max_channel_error"] for comparison in comparisons.values()),
            "tested_channel_tolerance": tolerance, "interpolation_required": interpolation_required,
            "native_vs_nearest": native_vs_nearest, "softpipe_vs_nearest": softpipe_vs_nearest,
            "actual_wrap_controls": controls, "resources": resources,
            "coverage_oracle": "explicit commits, never an alpha/RGBA proxy",
            "depth_oracle": "right half 16384 Z16, killed left half 65535; repeat has zero commits" if case.depth_test else None,
            "evidence": {"draw": str(directory / "capture/draw.json"), "shader_directory": str(shader),
                         "mesh": str(mesh), "texture": str(texture), "managed_rgba8": str(directory / "managed.rgba8"),
                         "native_report": str(directory / "native/native-report.json"),
                         "native_captures": native_report["capture_paths"],
                         "native_raw_bottom_left": str(directory / "native/native.rgba8"),
                         "softpipe_raw_bottom_left": str(directory / "reference/softpipe.rgba8"),
                         "rational_rgba8": str(directory / "rational.rgba8"), "coverage": str(directory / "coverage.u32")}}


def validate(output, build, *, cases=DEFAULT_CASES):
    cases = tuple(cases)
    if not cases or any(case not in CASES for case in cases) or len({case.name for case in cases}) != len(cases):
        raise ValueError("mirrored acceptance requires distinct named bounded cases")
    output, build = Path(output), Path(build)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"passed": False, "status": "running", "cases": [], "tested_cases": [case.name for case in cases],
               "native_gl_readback": False, "fpga_execution": False, "exact_fp32_conformance": False,
               "linear_tested_channel_tolerance": MAX_CHANNEL_ERROR,
               "input_precision": "uploaded RGBA8/255 texels and vertex/varying values use binary16 in managed execution",
               "output_precision": "compiled binary16 intermediates/color converted to RGBA8 UNORM; native glReadPixels returns RGBA8",
               "oracle": "independent Fraction period-two fold, edge-clamped nearest/bilinear taps and nearest-even byte rounding",
               "contract": "host v5 nearest and v6 linear mirrored-repeat S/T; layout2/ABI1, binding-0 block-15 RGBA8 without mips; v1..4 unchanged",
               "aggregate_draw_step_budget": MAX_DRAW_STEPS,
               "scope": "only these fresh 16x16 same-GLSL/mesh/texel CPU softpipe and managed/native CPU-host cases; not native-device or universal FP32 conformance"}
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    for case in cases:
        try:
            result = _run_case(output / case.name, build, case)
        except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
            summary.update(status="error", failed_case=case.name, message=str(error))
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            raise
        (output / case.name / "render.json").write_text(json.dumps(result, indent=2) + "\n")
        summary["cases"].append(result)
    summary.update(passed=True, status="passed", native_gl_readback=True,
                   maximum_channel_error=max(result["maximum_channel_error"] for result in summary["cases"]))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/mirrored-texture")
    parser.add_argument("--build", type=Path, default=Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build")))
    parser.add_argument("--case", action="append", choices=[case.name for case in CASES])
    args = parser.parse_args(argv)
    try:
        selected = DEFAULT_CASES if args.case is None else tuple(next(case for case in CASES if case.name == name)
                                                                for name in args.case)
        summary = validate(args.output, args.build, cases=selected)
        print(json.dumps({"passed": True, "tested_cases": summary["tested_cases"],
                          "maximum_channel_error": summary["maximum_channel_error"],
                          "native_gl_readback": True, "fpga_execution": False, "exact_fp32_conformance": False,
                          "summary": str(args.output / "summary.json")}))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "error", "message": str(error),
                          "summary": str(args.output / "summary.json")}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
