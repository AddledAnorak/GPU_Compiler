"""Independent mixed-axis Fraction sampling and fresh full-RGBA actual GL evidence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
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
from tools.host_texture import SAMPLER_STATE, host_texture_contract
from tools.linear_texture_acceptance import (MAX_CHANNEL_ERROR, TextureCase, case_mesh,
                                             image_comparison, input_texels)
from tools.mgcdriver import ProgramFormat
from tools.mirrored_texture_acceptance import (MAX_DRAW_STEPS, _shader, _validate_texels,
                                               mirror_fold, mirrored_coordinates,
                                               resource_observations)
from tools.run_render import capture_draw, native_host_readback, softpipe_reference
from tools.run_shader import ROOT
from tools.shader_layout import ShaderLayout
from tools.texture_flow_acceptance import STATUS


WRAPS = ("clamp-to-edge", "repeat", "mirrored-repeat")
AXES = tuple((wrap_s, wrap_t) for wrap_s in WRAPS for wrap_t in WRAPS if wrap_s != wrap_t)


@dataclass(frozen=True)
class MixedCase(TextureCase):
    phase: str = "clamp"
    texture_wrap_s: str = "repeat"
    texture_wrap_t: str = "clamp-to-edge"


_PROFILES = (
    (3, 5, "negative-seams", "sample", False, 1),
    (5, 3, "even-odd-seams", "sample", False, 1),
    (2, 2, "clamp", "late", False, 1),
    (3, 5, "negative-seams", "runtime-loop", False, 1),
    (1, 1, "multiple-periods", "runtime-loop", True, 2),
    (5, 3, "even-odd-seams", "early", False, 1),
)
DEFAULT_CASES = tuple(
    MixedCase(f"{texture_filter}-s-{wrap_s}-t-{wrap_t}", width, height, phase, variant, depth, repeats,
              texture_filter, wrap_s, wrap_t)
    for texture_filter in ("nearest", "linear")
    for (wrap_s, wrap_t), (width, height, phase, variant, depth, repeats) in zip(AXES, _PROFILES)
)
CASES = DEFAULT_CASES + tuple(
    MixedCase(f"{texture_filter}-s-{wrap_s}-t-{wrap_t}-{name}", width, height, phase, variant,
              depth, repeats, texture_filter, wrap_s, wrap_t)
    for texture_filter in ("nearest", "linear")
    for wrap_s, wrap_t, name, width, height, phase, variant, depth, repeats in (
        ("repeat", "mirrored-repeat", "16x16", 16, 16, "clamp", "sample", False, 1),
        ("mirrored-repeat", "repeat", "seams-4x4", 4, 4, "seams", "sample", False, 1),
        ("clamp-to-edge", "repeat", "zero-alpha-1x1", 1, 1, "clamp", "sample", False, 1),
        ("repeat", "clamp-to-edge", "negative-5x3", 5, 3, "negative-seams", "sample", False, 1),
        ("mirrored-repeat", "clamp-to-edge", "periods-3x5", 3, 5, "multiple-periods", "sample", False, 1),
        ("clamp-to-edge", "mirrored-repeat", "overlap-2x2", 2, 2, "clamp", "sample", False, 2),
        ("repeat", "mirrored-repeat", "depth-kill-2x2", 2, 2, "clamp", "runtime-loop", True, 2),
        ("mirrored-repeat", "repeat", "late-kill-3x5", 3, 5, "clamp", "late", False, 1),
        ("clamp-to-edge", "repeat", "early-kill-4x4", 4, 4, "clamp", "early", False, 1),
    )
)


def _axis_taps(coordinate, extent, wrap, texture_filter):
    coordinate = mirror_fold(coordinate) if wrap == "mirrored-repeat" else Fraction(coordinate)
    position = coordinate * extent
    if texture_filter == "nearest":
        indices = ((position.__floor__(), Fraction(1)),)
    else:
        position -= Fraction(1, 2)
        first = position.__floor__()
        weight = position - first
        indices = ((first, 1 - weight), (first + 1, weight))
    return tuple((index % extent if wrap == "repeat" else min(max(index, 0), extent - 1), weight)
                 for index, weight in indices)


def rational_sample(texels, horizontal, vertical, texture_filter="linear", *,
                    texture_wrap_s="repeat", texture_wrap_t="clamp-to-edge"):
    """Independent axis taps, exact byte-unit weights, no managed packing or shader execution."""
    _validate_texels(texels)
    if type(texture_filter) is not str or texture_filter not in ("nearest", "linear"):
        raise ValueError("unsupported rational mixed filter")
    if any(type(wrap) is not str or wrap not in WRAPS for wrap in (texture_wrap_s, texture_wrap_t)):
        raise ValueError("unsupported rational S/T wrap")
    height, width = texels.shape[:2]
    columns = _axis_taps(horizontal, width, texture_wrap_s, texture_filter)
    rows = _axis_taps(vertical, height, texture_wrap_t, texture_filter)
    return tuple(sum(Fraction(int(texels[row, column, component])) * row_weight * column_weight
                     for row, row_weight in rows for column, column_weight in columns)
                 for component in range(4))


def rational_oracle(texels, *, phase="clamp", texture_filter="linear", discard=False,
                    texture_wrap_s="repeat", texture_wrap_t="clamp-to-edge"):
    if type(discard) is not bool:
        raise ValueError("mixed oracle discard must be explicitly boolean")
    image = np.zeros((16, 16, 4), dtype=np.uint8)
    coverage = np.ones((16, 16), dtype=np.uint32)
    for row in range(16):
        for column in range(16):
            horizontal, vertical = mirrored_coordinates(row, column, phase)
            channels = rational_sample(texels, horizontal, vertical, texture_filter,
                                       texture_wrap_s=texture_wrap_s, texture_wrap_t=texture_wrap_t)
            if discard and column < 8:
                coverage[row, column] = 0
            else:
                image[row, column] = [round(channel) for channel in channels]
    return image, coverage


def _version(case):
    return 8 if case.texture_filter == "linear" else 7


def check_contract(artifact, texels, case):
    vertex = ShaderLayout(artifact["shaders"]["vs"]["layout"])
    fragment = ShaderLayout(artifact["shaders"]["fs"]["layout"])
    contract = host_texture_contract(_version(case), wrap_s=case.texture_wrap_s, wrap_t=case.texture_wrap_t)
    if (artifact.get("schema") != "mgc-host-draw-v2" or
            artifact.get("program_format") != ProgramFormat.LEGACY.value or
            type(artifact.get("shader_abi_version")) is not int or artifact["shader_abi_version"] != 1 or
            vertex.source["version"] != 1 or vertex.host_texture is not None or
            vertex.fragment_status is not None or fragment.source["version"] != 2 or
            fragment.host_texture != contract or fragment.fragment_status != STATUS or
            bool(fragment.source.get("uses_discard")) != (case.variant != "sample")):
        raise ValueError(f"{case.name}: fresh mixed reflection/ABI/status contract differs")
    sampler = {**SAMPLER_STATE, "min_filter": case.texture_filter, "mag_filter": case.texture_filter,
               "wrap_s": case.texture_wrap_s, "wrap_t": case.texture_wrap_t}
    expected = {"width": case.width, "height": case.height, "format": "R8G8B8A8_UNORM",
                "data_hex": texels.tobytes().hex(), "sampler": sampler}
    if artifact.get("textures") != [expected]:
        raise ValueError(f"{case.name}: fresh capture changed uploaded bytes or actual GL S/T sampler")
    if case.variant == "runtime-loop" and "mgc_block_" not in artifact["shaders"]["fs"]["assembly"]:
        raise ValueError(f"{case.name}: compiled runtime-loop control flow was not retained")


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
                                    f"S:{case.texture_wrap_s}/T:{case.texture_wrap_t} v{_version(case)}"}
    if (coverage.dtype != np.uint32 or not np.array_equal(coverage, expected_coverage) or
            any(report.get(name) != value for name, value in expected.items()) or
            type(report.get("host_steps")) is not int or not 256 < report["host_steps"] <= MAX_DRAW_STEPS):
        raise ValueError(f"{case.name}: mixed coverage/status/budget differs on draw {draw_index + 1}")


def _native(shader, build, output, options, case, texels, expected_coverage):
    with tempfile.TemporaryDirectory(prefix="mgc-mixed-") as temporary:
        with HostExecutorServer(Path(temporary) / "executor.sock", max_requests=case.repeats,
                                max_steps=MAX_DRAW_STEPS) as server:
            image = native_host_readback(shader, build, output, socket_path=server.path,
                                         repeats=case.repeats, **options)
        records = list(server.records)
        if server.errors or len(records) != case.repeats:
            raise ValueError(f"{case.name}: native mixed requests failed: {server.errors}")
    report = json.loads((output / "native-report.json").read_text())
    if (report.get("texture_filter") != case.texture_filter or report.get("texture_wrap") != "mixed" or
            report.get("texture_wrap_s") != case.texture_wrap_s or report.get("texture_wrap_t") != case.texture_wrap_t or
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
               "texture_wrap_s": case.texture_wrap_s, "texture_wrap_t": case.texture_wrap_t,
               "depth_test": case.depth_test}
    oracle_options = {"phase": case.phase, "texture_filter": case.texture_filter,
                      "texture_wrap_s": case.texture_wrap_s, "texture_wrap_t": case.texture_wrap_t,
                      "discard": case.variant != "sample"}
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
        raise ValueError(f"{case.name}: same-input mixed RGBA checks failed: {comparisons}")
    nearest, _ = rational_oracle(texels, **{**oracle_options, "texture_filter": "nearest"})
    interpolation_required = case.texture_filter == "linear" and image_comparison(expected, nearest, 0)["max_channel_error"] > tolerance
    nearest_checks = {"native": image_comparison(native, nearest, 0),
                      "softpipe": image_comparison(reference, nearest, 0)}
    if interpolation_required and min(check["max_channel_error"] for check in nearest_checks.values()) <= tolerance:
        raise ValueError(f"{case.name}: mixed linear sampling collapsed to nearest")
    controls = {}
    for name, wrap_s, wrap_t in (("paired_s", case.texture_wrap_s, case.texture_wrap_s),
                                ("paired_t", case.texture_wrap_t, case.texture_wrap_t),
                                ("swapped", case.texture_wrap_t, case.texture_wrap_s)):
        wrong, _ = rational_oracle(texels, **{**oracle_options, "texture_wrap_s": wrap_s, "texture_wrap_t": wrap_t})
        distinguishable = image_comparison(expected, wrong, 0)["max_channel_error"] > tolerance
        control = {"oracle_only": True, "distinguishable": distinguishable,
                   "native": image_comparison(native, wrong, 0), "softpipe": image_comparison(reference, wrong, 0)}
        if distinguishable and min(control[source]["max_channel_error"] for source in ("native", "softpipe")) <= tolerance:
            raise ValueError(f"{case.name}: actual GL mixed axes collapsed to {name}")
        controls[name] = control
    for name, image in (("managed", managed), ("native", native), ("softpipe", reference), ("rational", expected)):
        (directory / f"{name}.rgba8").write_bytes(image.tobytes())
    (directory / "coverage.u32").write_bytes(coverage.astype("<u4").tobytes())
    return {"case": case.name, "passed": True, "texture_width": case.width, "texture_height": case.height,
            "texture_filter": case.texture_filter, "texture_wrap": "mixed",
            "texture_wrap_s": case.texture_wrap_s, "texture_wrap_t": case.texture_wrap_t,
            "phase": case.phase, "variant": case.variant, "depth_test": case.depth_test, "repeats": case.repeats,
            "comparisons": comparisons, "managed_report": report, "native_draw_reports": draw_reports,
            "maximum_channel_error": max(comparison["max_channel_error"] for comparison in comparisons.values()),
            "tested_channel_tolerance": tolerance, "interpolation_required": interpolation_required,
            "nearest_controls": nearest_checks, "axis_controls": controls, "resources": resources,
            "coverage_oracle": "explicit commits, never an alpha/RGBA proxy",
            "depth_oracle": "right half 16384 Z16, killed left half 65535; repeat has zero commits" if case.depth_test else None,
            "evidence": {"draw": str(directory / "capture/draw.json"), "shader_directory": str(shader),
                         "mesh": str(mesh), "texture": str(texture), "native_report": str(directory / "native/native-report.json"),
                         "native_captures": native_report["capture_paths"],
                         "native_raw_bottom_left": str(directory / "native/native.rgba8"),
                         "softpipe_raw_bottom_left": str(directory / "reference/softpipe.rgba8")}}


def validate(output, build, *, cases=DEFAULT_CASES):
    cases = tuple(cases)
    if not cases or any(case not in CASES for case in cases) or len({case.name for case in cases}) != len(cases):
        raise ValueError("mixed acceptance requires distinct named bounded cases")
    output, build = Path(output), Path(build)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"passed": False, "status": "running", "cases": [], "tested_cases": [case.name for case in cases],
               "native_gl_readback": False, "fpga_execution": False, "exact_fp32_conformance": False,
               "linear_tested_channel_tolerance": MAX_CHANNEL_ERROR, "nearest_tested_channel_tolerance": 0,
               "oracle": "independent per-axis Fraction taps: repeat modulo, mirrored period-two fold, clamp edges; nearest-even byte rounding",
               "contract": "host v7 nearest/v8 linear independent S/T; layout2/ABI1, binding-0 block-15 RGBA8 without mips; v1..6 unchanged",
               "aggregate_draw_step_budget": MAX_DRAW_STEPS,
               "input_precision": "uploaded RGBA8/255 texels and vertex/varying values use binary16 in managed execution",
               "output_precision": "compiled binary16 intermediates converted to RGBA8 UNORM; native glReadPixels returns RGBA8",
               "scope": "fresh 16x16 same-GLSL/mesh/texel CPU softpipe and managed/native CPU-host cases; not FPGA or universal FP32 conformance"}
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
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/mixed-texture")
    parser.add_argument("--build", type=Path, default=Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build")))
    parser.add_argument("--case", action="append", choices=[case.name for case in CASES])
    parser.add_argument("--extended", action="store_true", help="run all 30 cases instead of the bounded 12-case default")
    args = parser.parse_args(argv)
    try:
        selected = (CASES if args.extended else DEFAULT_CASES) if args.case is None else tuple(
            next(case for case in CASES if case.name == name) for name in args.case)
        summary = validate(args.output, args.build, cases=selected)
        print(json.dumps({"passed": True, "tested_cases": summary["tested_cases"],
                          "maximum_channel_error": summary["maximum_channel_error"],
                          "native_gl_readback": True, "fpga_execution": False, "exact_fp32_conformance": False,
                          "summary": str(args.output / "summary.json")}))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "error", "message": str(error), "summary": str(args.output / "summary.json")}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
