"""Fresh bounded linear-clamp capture/managed/native GL checks, not FP32 conformance."""

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
from tools.host_texture import (HOST_TEXTURE_LAYOUT, HOST_TEXTURE_LINEAR_LAYOUT,
                                LINEAR_SAMPLER_STATE, SAMPLER_STATE)
from tools.mgcdriver import ProgramFormat
from tools.run_render import capture_draw, native_host_readback, softpipe_reference
from tools.run_shader import ROOT
from tools.shader_layout import ShaderLayout
from tools.texture_acceptance import quad_mesh, texture_pattern
from tools.texture_flow_acceptance import STATUS


MAX_CHANNEL_ERROR = 2


@dataclass(frozen=True)
class TextureCase:
    name: str
    width: int
    height: int
    phase: str = "identity"
    variant: str = "sample"
    depth_test: bool = False
    repeats: int = 1
    texture_filter: str = "linear"


CASES = (
    TextureCase("linear-2x2", 2, 2),
    TextureCase("linear-checker-4x4", 4, 4),
    TextureCase("linear-asymmetric-3x5", 3, 5),
    TextureCase("linear-asymmetric-5x3", 5, 3),
    TextureCase("linear-zero-alpha-1x1", 1, 1),
    TextureCase("linear-centers-16x16", 16, 16),
    TextureCase("linear-clamp-2x2", 2, 2, phase="clamp"),
    TextureCase("linear-clamp-3x5", 3, 5, phase="clamp"),
    TextureCase("linear-clamp-1x1", 1, 1, phase="clamp"),
    TextureCase("linear-phase-4x4", 4, 4, phase="phase"),
    TextureCase("linear-late-kill-2x2", 2, 2, variant="late"),
    TextureCase("linear-early-kill-4x4", 4, 4, variant="early"),
    TextureCase("linear-loop-kill-3x5", 3, 5, variant="runtime-loop"),
    TextureCase("linear-depth-loop-repeated-2x2", 2, 2, variant="runtime-loop",
                depth_test=True, repeats=2),
    TextureCase("nearest-v1-regression-2x2", 2, 2, texture_filter="nearest"),
)


def input_texels(width, height):
    if any(type(extent) is not int or not 1 <= extent <= 16 for extent in (width, height)):
        raise ValueError("acceptance texture dimensions must be integers in [1, 16]")
    if width == height and width in (1, 2, 4, 16):
        return texture_pattern(width)
    rows, columns = np.indices((height, width))
    return np.stack(((columns * 67 + rows * 13 + 17) % 256,
                     (rows * 47 + columns * 29 + 31) % 256,
                     (columns * 113 + rows * 71 + 7) % 256,
                     np.where((rows + columns) % 3, 255, 0)), axis=2).astype(np.uint8)


def pixel_coordinates(row, column, phase="identity"):
    horizontal = Fraction(2 * column + 1, 32)
    vertical = Fraction(31 - 2 * row, 32)
    if phase == "clamp":
        return horizontal * 4 - Fraction(3, 2), vertical * 4 - Fraction(3, 2)
    if phase == "phase":
        return 1 - vertical, horizontal
    if phase != "identity":
        raise ValueError("unsupported analytic texture coordinate case")
    return horizontal, vertical


def rational_sample(texels, horizontal, vertical, texture_filter="linear"):
    """Return exact byte-unit channels; no compiler, packer, renderer or FP16 helpers."""
    height, width = texels.shape[:2]
    horizontal, vertical = Fraction(horizontal), Fraction(vertical)
    if texture_filter == "nearest":
        column = min(max((horizontal * width).__floor__(), 0), width - 1)
        row = min(max((vertical * height).__floor__(), 0), height - 1)
        return tuple(Fraction(int(channel)) for channel in texels[row, column])
    if texture_filter != "linear":
        raise ValueError("unsupported analytic texture filter")
    horizontal_position = horizontal * width - Fraction(1, 2)
    vertical_position = vertical * height - Fraction(1, 2)
    first_column = horizontal_position.__floor__()
    first_row = vertical_position.__floor__()
    horizontal_weight = horizontal_position - first_column
    vertical_weight = vertical_position - first_row
    column_weights = ((first_column, 1 - horizontal_weight),
                      (first_column + 1, horizontal_weight))
    row_weights = ((first_row, 1 - vertical_weight), (first_row + 1, vertical_weight))
    return tuple(sum(Fraction(int(texels[min(max(row, 0), height - 1),
                                         min(max(column, 0), width - 1), component])) *
                     row_weight * column_weight
                     for row, row_weight in row_weights
                     for column, column_weight in column_weights)
                 for component in range(4))


def rational_oracle(texels, *, phase="identity", texture_filter="linear", discard=False):
    if (not isinstance(texels, np.ndarray) or texels.dtype != np.uint8 or
            texels.ndim != 3 or texels.shape[2] != 4 or
            not all(1 <= extent <= 16 for extent in texels.shape[:2])):
        raise ValueError("analytic input must be bounded tightly packed RGBA8 texels")
    if type(discard) is not bool:
        raise ValueError("analytic discard must be explicitly boolean")
    image = np.zeros((16, 16, 4), dtype=np.uint8)
    coverage = np.ones((16, 16), dtype=np.uint32)
    for row in range(16):
        for column in range(16):
            if discard and column < 8:
                coverage[row, column] = 0
                continue
            horizontal, vertical = pixel_coordinates(row, column, phase)
            image[row, column] = [round(channel) for channel in
                                  rational_sample(texels, horizontal, vertical, texture_filter)]
    return image, coverage


def image_comparison(image, reference, tolerance):
    if (image.shape != (16, 16, 4) or reference.shape != image.shape or
            image.dtype != np.uint8 or reference.dtype != np.uint8):
        raise ValueError("acceptance images must be same-geometry 16x16 RGBA8")
    differences = np.abs(image.astype(np.int16) - reference.astype(np.int16))
    maximum = int(differences.max())
    return {"max_channel_error": maximum,
            "mismatch_pixels": int(np.count_nonzero(np.any(differences, axis=2))),
            "channel_tolerance": tolerance, "passed": maximum <= tolerance}


def shader_directory(case):
    if case.variant == "sample":
        fixture = ROOT / "tests/fixtures/host-texture"
        return fixture if case.phase == "identity" else fixture / case.phase
    fixture = ROOT / "tests/fixtures/host-texture-flow"
    return fixture if case.variant == "late" else fixture / case.variant


def case_mesh(case):
    if not case.depth_test:
        return quad_mesh()
    positions = [(-1, -1), (1, -1), (1, 1), (-1, -1), (1, 1), (-1, 1)]
    return b"".join(struct.pack("<7f", horizontal, vertical, depth, 1, 0, 0, 1)
                    for depth in (-0.5, 0.5) for horizontal, vertical in positions)


def check_contract(artifact, texels, case):
    vertex = ShaderLayout(artifact["shaders"]["vs"]["layout"])
    fragment = ShaderLayout(artifact["shaders"]["fs"]["layout"])
    metadata = HOST_TEXTURE_LINEAR_LAYOUT if case.texture_filter == "linear" else HOST_TEXTURE_LAYOUT
    sampler = LINEAR_SAMPLER_STATE if case.texture_filter == "linear" else SAMPLER_STATE
    if (artifact.get("schema") != "mgc-host-draw-v2" or
            artifact.get("program_format") != ProgramFormat.LEGACY.value or
            type(artifact.get("shader_abi_version")) is not int or artifact["shader_abi_version"] != 1 or
            vertex.source["version"] != 1 or vertex.host_texture is not None or
            vertex.fragment_status is not None or fragment.source["version"] != 2 or
            fragment.host_texture != metadata or fragment.fragment_status != STATUS or
            bool(fragment.source.get("uses_discard")) != (case.variant != "sample")):
        raise ValueError(f"{case.name}: fresh capture lost the exact legacy texture/status reflection")
    expected = {"width": case.width, "height": case.height, "format": "R8G8B8A8_UNORM",
                "data_hex": texels.tobytes().hex(), "sampler": dict(sampler)}
    if artifact.get("textures") != [expected]:
        raise ValueError(f"{case.name}: capture differs from original uploaded texels or actual GL sampler")
    if case.variant == "runtime-loop" and "mgc_block_" not in artifact["shaders"]["fs"]["assembly"]:
        raise ValueError(f"{case.name}: actual compiled runtime control flow was not retained")


def check_execution(coverage, report, expected_coverage, case, draw_index=0):
    discarded = case.variant != "sample"
    counts = {"vertices": 12 if case.depth_test else 6, "triangles": 4 if case.depth_test else 2,
              "fragments": 384 if case.depth_test else 256,
              "committed_fragments": 128 if discarded else 256,
              "discarded_fragments": (256 if case.depth_test else 128) if discarded else 0,
              "covered_pixels": 128 if discarded else 256, "max_overdraw": 1}
    if case.depth_test and draw_index:
        counts.update(fragments=256, committed_fragments=0, discarded_fragments=256,
                      covered_pixels=0, max_overdraw=0)
        expected_coverage = np.zeros_like(expected_coverage)
    if not np.array_equal(coverage, expected_coverage) or coverage.dtype != np.uint32:
        raise ValueError(f"{case.name}: explicit coverage differs on draw {draw_index + 1}")
    if (any(report.get(key) != value for key, value in counts.items()) or
            report.get("host_steps", 0) <= 256 or report.get("fpga_execution") is not False or
            report.get("fragment_completion") != "explicit legacy host status v1" or
            report.get("texture_sampling") !=
            f"bounded legacy {case.texture_filter}-clamp v{2 if case.texture_filter == 'linear' else 1}"):
        raise ValueError(f"{case.name}: actual managed invocation/status counts differ on draw {draw_index + 1}")


def validate(output, build, *, cases=CASES):
    output, build = Path(output), Path(build)
    cases = tuple(cases)
    if not cases or len({case.name for case in cases}) != len(cases):
        raise ValueError("acceptance requires distinct explicit cases")
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").unlink(missing_ok=True)
    results = []
    for case in cases:
        directory = output / case.name
        directory.mkdir(parents=True, exist_ok=True)
        shader = shader_directory(case)
        mesh = directory / "vertices.bin"
        mesh.write_bytes(case_mesh(case))
        texels = input_texels(case.width, case.height)
        texture = directory / "texture.bin"
        texture.write_bytes(struct.pack("<II", case.width, case.height) + texels.tobytes())
        options = {"mesh": mesh, "texture": texture, "texture_filter": case.texture_filter,
                   "depth_test": case.depth_test}
        artifact = capture_draw(shader, build, directory / "capture", host_texture=True,
                                fragment_status=True, **options)
        check_contract(artifact, texels, case)
        expected, expected_coverage = rational_oracle(texels, phase=case.phase,
                                                     texture_filter=case.texture_filter,
                                                     discard=case.variant != "sample")
        image, coverage, report = render_artifact(artifact, depth_test=case.depth_test)
        check_execution(coverage, report, expected_coverage, case)
        with tempfile.TemporaryDirectory(prefix="mgc-linear-") as temporary:
            socket_path = Path(temporary) / "executor.sock"
            with HostExecutorServer(socket_path, max_requests=case.repeats) as server:
                native = native_host_readback(shader, build, directory / "native",
                                              socket_path=socket_path, repeats=case.repeats, **options)
            records = list(server.records)
            if server.errors or len(records) != case.repeats:
                raise ValueError(f"{case.name}: native managed requests failed: {server.errors}")
        native_report = json.loads((directory / "native/native-report.json").read_text())
        if (not np.array_equal(native, records[-1]["rgba"]) or not np.array_equal(native, image) or
                [commit["sequence"] for commit in native_report["commits"]] !=
                [record["sequence"] for record in records]):
            raise ValueError(f"{case.name}: glReadPixels, accepted replies and offline managed capture differ")
        expected_depth = np.full((16, 16), 65535, dtype=np.uint16)
        expected_depth[:, 8:] = 16384
        for draw_index, (record, capture_path) in enumerate(zip(records, native_report["capture_paths"])):
            check_contract(record["artifact"], texels, case)
            if json.loads(Path(capture_path).read_text()) != record["artifact"]:
                raise ValueError(f"{case.name}: native captured JSON differs from submitted artifact")
            check_execution(record["coverage"], record["report"], expected_coverage, case, draw_index)
            if not np.array_equal(record["rgba"], image):
                raise ValueError(f"{case.name}: native repeated color did not retain the expected commits")
            if case.depth_test and not np.array_equal(record["depth"], expected_depth):
                raise ValueError(f"{case.name}: killed fragments modified depth or repeated LESS rewrote equal depth")
        reference = softpipe_reference(shader, build, directory / "reference", repeats=case.repeats, **options)
        tolerance = MAX_CHANNEL_ERROR if case.texture_filter == "linear" else 0
        comparisons = {"managed_vs_rational": image_comparison(image, expected, tolerance),
                       "softpipe_vs_rational": image_comparison(reference, expected, tolerance),
                       "native_vs_rational": image_comparison(native, expected, tolerance),
                       "managed_vs_softpipe": image_comparison(image, reference, tolerance),
                       "native_vs_softpipe": image_comparison(native, reference, tolerance)}
        if not all(comparison["passed"] for comparison in comparisons.values()):
            raise ValueError(f"{case.name}: bounded same-input image or rational oracle failed: {comparisons}")
        nearest, _ = rational_oracle(texels, phase=case.phase, texture_filter="nearest",
                                     discard=case.variant != "sample")
        nearest_difference = image_comparison(expected, nearest, 0)
        interpolation_required = case.texture_filter == "linear" and nearest_difference["max_channel_error"] > tolerance
        native_nearest_difference = image_comparison(native, nearest, 0)
        softpipe_nearest_difference = image_comparison(reference, nearest, 0)
        if interpolation_required and min(native_nearest_difference["max_channel_error"],
                                           softpipe_nearest_difference["max_channel_error"]) <= tolerance:
            raise ValueError(f"{case.name}: actual interpolation was indistinguishable from nearest sampling")
        nearest_control = None
        if case.name == "linear-2x2":
            nearest_options = {**options, "texture_filter": "nearest"}
            control = softpipe_reference(shader, build, directory / "nearest-control", **nearest_options)
            nearest_control = {"nearest_oracle": image_comparison(control, nearest, 0),
                               "linear_vs_actual_nearest": image_comparison(reference, control, 0)}
            if (not nearest_control["nearest_oracle"]["passed"] or
                    nearest_control["linear_vs_actual_nearest"]["max_channel_error"] <= tolerance):
                raise ValueError("linear-2x2: actual GL nearest control failed to distinguish interpolation")
        (directory / "managed.rgba8").write_bytes(image.tobytes())
        (directory / "rational.rgba8").write_bytes(expected.tobytes())
        (directory / "nearest.rgba8").write_bytes(nearest.tobytes())
        (directory / "coverage.u32").write_bytes(coverage.astype("<u4").tobytes())
        if case.depth_test:
            (directory / "native/depth.z16").write_bytes(records[-1]["depth"].astype("<u2").tobytes())
        result = {"case": case.name, "passed": True, "texture_dimensions": [case.width, case.height],
                  "texture_filter": case.texture_filter, "coordinate_case": case.phase,
                  "fixture": case.variant, "layout_version": 2,
                  "host_texture_version": 2 if case.texture_filter == "linear" else 1,
                  "native_gl_readback": True, "reply_matches_gl_readback": True,
                  "native_commits": len(records), "draw_repeats": case.repeats,
                  "managed_report": report, "native_draw_reports": [record["report"] for record in records],
                  "comparisons": comparisons, "maximum_channel_error":
                  max(comparison["max_channel_error"] for comparison in comparisons.values()),
                  "tested_channel_tolerance": tolerance, "interpolation_required": interpolation_required,
                  "rational_vs_nearest": nearest_difference, "native_vs_nearest": native_nearest_difference,
                  "softpipe_vs_nearest": softpipe_nearest_difference, "actual_nearest_control": nearest_control,
                  "coverage_oracle": "explicit full quad commits, or right-half commits/left-half discard; no RGBA/alpha proxy",
                  "depth_oracle": "right half 16384 Z16, killed left half 65535; second draw has zero commits" if case.depth_test else None,
                  "oracle": "independent Fraction pixel-center coordinates, four clamped texel weights and nearest-even RGBA8 rounding",
                  "numerical_scope": "ideal rational filtering versus ordinary FP16 compiled execution on these exact inputs only; not exact FP32 or universal error conformance",
                  "fpga_execution": False,
                  "evidence": {"draw": str(directory / "capture/draw.json"),
                               "shader_directory": str(shader), "mesh": str(mesh), "texture": str(texture),
                               "managed_rgba8": str(directory / "managed.rgba8"),
                               "native_report": str(directory / "native/native-report.json"),
                               "native_captures": native_report["capture_paths"],
                               "native_raw_bottom_left": str(directory / "native/native.rgba8"),
                               "softpipe_raw_bottom_left": str(directory / "reference/softpipe.rgba8"),
                               "rational_rgba8": str(directory / "rational.rgba8"),
                               "coverage": str(directory / "coverage.u32")}}
        (directory / "render.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
    summary = {"passed": True, "cases": results, "tested_cases": [case.name for case in cases],
               "native_gl_readback": True, "fpga_execution": False, "exact_fp32_conformance": False,
               "maximum_channel_error": max(result["maximum_channel_error"] for result in results),
               "linear_tested_channel_tolerance": MAX_CHANNEL_ERROR,
               "contract": "legacy host texture v2 linear-clamp, same block 15, one binding-0 RGBA8 texture without mips; compiler layout remains v2; nearest host v1 unchanged",
               "scope": "only the named fresh 16x16 cases, same GLSL/mesh/uploaded bytes on CPU softpipe and managed/native host execution"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/linear-texture")
    parser.add_argument("--build", type=Path, default=Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build")))
    parser.add_argument("--case", action="append", choices=[case.name for case in CASES],
                        help="run only explicitly named fresh cases; may be repeated")
    args = parser.parse_args(argv)
    try:
        selected = CASES if args.case is None else tuple(next(case for case in CASES if case.name == name)
                                                        for name in args.case)
        summary = validate(args.output, args.build, cases=selected)
        print(json.dumps({"passed": True, "tested_cases": summary["tested_cases"],
                          "maximum_channel_error": summary["maximum_channel_error"],
                          "native_gl_readback": True, "exact_fp32_conformance": False,
                          "fpga_execution": False, "summary": str(args.output / "summary.json")}))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"linear texture acceptance failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
