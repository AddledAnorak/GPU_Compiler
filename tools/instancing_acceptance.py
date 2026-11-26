"""Fresh attribute-divisor GL draws with independent rational rectangle/commit evidence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
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
from tools.host_instancing import HOST_INSTANCING_CONTRACT, validate_instancing
from tools.host_render import render_artifact
from tools.linear_texture_acceptance import image_comparison
from tools.mgcdriver import ProgramFormat
from tools.mirrored_texture_acceptance import MAX_DRAW_STEPS, resource_observations
from tools.run_render import (INSTANCE_HEADER, INSTANCE_MAGIC, capture_draw,
                              native_host_readback, softpipe_reference)
from tools.run_shader import ROOT
from tools.shader_layout import ShaderLayout
from tools.texture_flow_acceptance import STATUS


INSTANCE_ID_WORD = 65535
INSTANCING_SHADER_ABI = 3
INSTANCE_ID_COLOR = (Fraction(1, 8), Fraction(3, 8), Fraction(3, 4), Fraction(1))


@dataclass(frozen=True)
class InstanceCase:
    name: str
    count: int = 5
    offset_divisor: int = 1
    color_divisor: int = 1
    offset_components: int = 3
    color_components: int = 4
    kill: bool = False
    zero_alpha: bool = False
    depth_test: bool = False
    blend: str = "disabled"
    repeats: int = 1
    instance_id_compare: int | None = None


CASES = (
    InstanceCase("single", count=1),
    InstanceCase("divisor1-count4", count=4),
    InstanceCase("color-divisor2-count5", color_divisor=2),
    InstanceCase("color-divisor3-count5", color_divisor=3),
    InstanceCase("offset-divisor2-count5", offset_divisor=2),
    InstanceCase("offset-divisor3-count4", count=4, offset_divisor=3, color_divisor=2),
    InstanceCase("vec4-offset-count4", count=4, offset_components=4),
    InstanceCase("vec3-color-count4", count=4, color_components=3, color_divisor=2),
    InstanceCase("max16", count=16),
    InstanceCase("kill-count5", kill=True, color_divisor=2),
    InstanceCase("zero-alpha-count4", count=4, zero_alpha=True),
    InstanceCase("overlap-source-over", offset_divisor=2, blend="source-over", repeats=2),
    InstanceCase("overlap-additive", count=4, offset_divisor=3, blend="additive"),
    InstanceCase("depth-kill-repeat", offset_divisor=2, kill=True, depth_test=True, repeats=2),
    InstanceCase("depth-zero-alpha-repeat", count=4, zero_alpha=True, depth_test=True, repeats=2),
    InstanceCase("instance-id1-count4", count=4, instance_id_compare=1),
    InstanceCase("instance-id1-divisors2-3-count5", offset_divisor=2, color_divisor=3, instance_id_compare=1),
    InstanceCase("instance-id0-max16", count=16, instance_id_compare=0),
    InstanceCase("instance-id15-max16", count=16, instance_id_compare=15),
)
DEFAULT_CASES = tuple(case for case in CASES if case.name not in (
    "vec4-offset-count4", "vec3-color-count4", "overlap-additive",
    "instance-id1-divisors2-3-count5", "instance-id0-max16"))


def shader_instancing_contract(case, stage):
    return {"version": 1, "max_instances": 16, "instance_id_word": INSTANCE_ID_WORD,
            "instance_id": "local-zero-based", "uses_instance_id": stage == "vertex" and case.instance_id_compare is not None}


def instance_records(case):
    offsets = []
    for index in range((case.count - 1) // case.offset_divisor + 1):
        if case.depth_test:
            position = (0, 0, (Fraction(1, 2), Fraction(-1, 2), 0)[index % 3])
        else:
            position = (Fraction(2 * (index % 4) - 3, 4), Fraction(2 * (index // 4) - 3, 4), 0)
        offsets.append(position + (0,) if case.offset_components == 4 else position)
    palette = ((1, Fraction(1, 4), 0), (0, 1, Fraction(1, 2)), (Fraction(1, 4), 0, 1),
               (Fraction(1, 2), 1, Fraction(1, 4)))
    colors = []
    for index in range((case.count - 1) // case.color_divisor + 1):
        color = palette[index % len(palette)]
        alpha = 0 if case.zero_alpha else Fraction(1, 2)
        colors.append(color + (alpha,) if case.color_components == 4 else color)
    return tuple(offsets), tuple(colors)


def instance_bytes(case):
    offsets, colors = instance_records(case)
    return INSTANCE_HEADER.pack(INSTANCE_MAGIC, case.count, case.offset_divisor, case.color_divisor,
                                case.offset_components, case.color_components) + b"".join(
                                    struct.pack("<" + "f" * len(record), *record) for record in offsets + colors)


def case_mesh():
    positions = ((-1, -1), (1, -1), (1, 1), (-1, -1), (1, 1), (-1, 1))
    return b"".join(struct.pack("<7f", horizontal / 4, vertical / 4, 0, 1, 0, 0, 1)
                    for horizontal, vertical in positions)


def _shader(case, directory):
    shader = directory / "shader"
    shader.mkdir(parents=True, exist_ok=True)
    color = "a_instance_color" if case.color_components == 4 else "vec4(a_instance_color, 1.0)"
    extension = "#extension GL_EXT_draw_instanced : require\n" if case.instance_id_compare is not None else ""
    instance_id = (f"if (gl_InstanceIDEXT == {case.instance_id_compare}) v_color = vec4(0.125, 0.375, 0.75, 1.0);\n"
                   if case.instance_id_compare is not None else "")
    (shader / "vert.glsl").write_text(
        f"#version 100\n{extension}attribute vec4 a_position;\nattribute vec{case.offset_components} a_instance_offset;\n"
        f"attribute vec{case.color_components} a_instance_color;\n"
        "varying vec4 v_color;\nvarying float v_local_x;\nvoid main() {\n"
        "gl_Position = a_position + vec4(a_instance_offset.xyz, 0.0);\n"
        f"v_color = {color};\n{instance_id}v_local_x = a_position.x;\n}}\n")
    kill = "if (v_local_x < 0.0) discard;\n" if case.kill else ""
    (shader / "frag.glsl").write_text(
        "#version 100\nprecision mediump float;\nvarying vec4 v_color;\nvarying float v_local_x;\n"
        f"void main() {{\ngl_FragColor = v_color;\n{kill}}}\n")
    return shader


def rational_oracle(case, *, force_instance_id=None):
    """Pixel-center rectangles, explicit kill/depth, Fraction blend and per-commit byte rounding."""
    if force_instance_id is not None and (type(force_instance_id) is not int or not 0 <= force_instance_id < 16):
        raise ValueError("oracle instance ID control must be an explicit local integer 0..15")
    offsets, colors = instance_records(case)
    image = np.zeros((16, 16, 4), np.uint8)
    depth = np.full((16, 16), 65535, np.uint16)
    coverages, depths, images, discarded_counts = [], [], [], []
    for draw_index in range(case.repeats):
        coverage = np.zeros((16, 16), np.uint32)
        discarded = 0
        for local_instance in range(case.count):
            horizontal, vertical, offset_depth = offsets[local_instance // case.offset_divisor][:3]
            source = tuple(Fraction(channel) for channel in colors[local_instance // case.color_divisor])
            if len(source) == 3:
                source += (Fraction(1),)
            instance_id = local_instance if force_instance_id is None else force_instance_id
            if case.instance_id_compare is not None and instance_id == case.instance_id_compare:
                source = INSTANCE_ID_COLOR
            incoming_depth = round((Fraction(offset_depth) + 1) * Fraction(65535, 2))
            for row in range(16):
                vertical_pixel = Fraction(15 - 2 * row, 16)
                for column in range(16):
                    horizontal_pixel = Fraction(2 * column - 15, 16)
                    if not (horizontal - Fraction(1, 4) <= horizontal_pixel < horizontal + Fraction(1, 4) and
                            vertical - Fraction(1, 4) <= vertical_pixel < vertical + Fraction(1, 4)):
                        continue
                    if case.kill and horizontal_pixel < horizontal:
                        discarded += 1
                        continue
                    if case.depth_test and incoming_depth >= int(depth[row, column]):
                        continue
                    destination = tuple(Fraction(int(channel), 255) for channel in image[row, column])
                    if case.blend == "source-over":
                        alpha = source[3]
                        blended = tuple(source[index] * alpha + destination[index] * (1 - alpha) for index in range(3))
                        blended += (alpha + destination[3] * (1 - alpha),)
                    elif case.blend == "additive":
                        blended = tuple(min(first + second, 1) for first, second in zip(source, destination))
                    else:
                        blended = source
                    image[row, column] = [round(channel * 255) for channel in blended]
                    coverage[row, column] += 1
                    if case.depth_test:
                        depth[row, column] = incoming_depth
        coverages.append(coverage)
        depths.append(depth.copy())
        images.append(image.copy())
        discarded_counts.append(discarded)
    return images, coverages, depths, discarded_counts


def check_contract(artifact, case):
    expected_capture = {"version": 1, "max_instances": 16, "max_vertices": 4096,
                        "indexing": "base-instance-plus-floor-local-instance-over-divisor",
                        "shader_abi_version": INSTANCING_SHADER_ABI, "instance_id_word": INSTANCE_ID_WORD}
    if (not validate_instancing(artifact) or artifact.get("host_instancing") != expected_capture or
            artifact.get("program_format") != ProgramFormat.LEGACY.value or
            type(artifact.get("shader_abi_version")) is not int or artifact["shader_abi_version"] != INSTANCING_SHADER_ABI or
            artifact["draw"].get("instance_count") != case.count or artifact["draw"].get("start_instance", 0) != 0 or
            artifact["draw"].get("count") != 6 or artifact["draw"].get("index_size", 0) != 0 or
            artifact.get("indices") not in (None, "")):
        raise ValueError(f"{case.name}: actual GL schema3/count/base/direct-draw contract differs")
    vertex = ShaderLayout(artifact["shaders"]["vs"]["layout"])
    fragment = ShaderLayout(artifact["shaders"]["fs"]["layout"])
    if (vertex.source["version"] != 3 or vertex.source["shader_abi_version"] != INSTANCING_SHADER_ABI or
            vertex.host_texture is not None or vertex.fragment_status is not None or
            fragment.source["version"] != 3 or fragment.source["shader_abi_version"] != INSTANCING_SHADER_ABI or
            fragment.host_texture is not None or fragment.fragment_status != STATUS or
            bool(fragment.source.get("uses_discard")) != case.kill):
        raise ValueError(f"{case.name}: actual VS/FS reflection or explicit discard contract differs")
    for layout in (vertex, fragment):
        metadata = layout.source.get("host_instancing")
        expected = shader_instancing_contract(case, layout.source["stage"])
        if (type(metadata) is not dict or set(metadata) != set(expected) or
                any(type(metadata[key]) is not type(value) or metadata[key] != value for key, value in expected.items())):
            raise ValueError(f"{case.name}: ABI3 host instance-ID metadata differs from the actual VS/FS source")
    inputs = vertex.entries("inputs")
    position_locations = {entry["location"] for entry in inputs if entry["word_offset"] == 0}
    if len(position_locations) != 1:
        raise ValueError(f"{case.name}: nonconstant position/offset/color VS inputs were not retained")
    position_location = next(iter(position_locations))
    slots = {}
    for binding in (0, 3, 4):
        matching = {entry["word_offset"] // 4 for entry in inputs if entry["location"] == position_location + binding}
        if len(matching) != 1:
            raise ValueError(f"{case.name}: actual semantic binding {binding} lacks a unique reflected driver slot")
        slots[binding] = next(iter(matching))
    offsets, colors = instance_records(case)
    for binding, divisor, components, records in ((3, case.offset_divisor, case.offset_components, offsets),
                                                  (4, case.color_divisor, case.color_components, colors)):
        element = artifact["vertex_elements"][slots[binding]]
        buffer = artifact["vertex_buffers"][element["vertex_buffer_index"]]
        expected = b"".join(struct.pack("<" + "f" * len(record), *record) for record in records)
        if (element.get("instance_divisor") != divisor or buffer.get("stride") != components * 4 or
                element.get("src_offset") != 0 or bytes.fromhex(buffer["data_hex"]) != expected):
            raise ValueError(f"{case.name}: captured divisor/stride/finite FP32 records differ from uploaded input")


def _check_execution(coverage, report, expected, discarded, case):
    committed = int(expected.sum())
    if (coverage.dtype != np.uint32 or not np.array_equal(coverage, expected) or
            report.get("vertices") != case.count * 6 or report.get("triangles") != case.count * 2 or
            report.get("instances") != case.count or report.get("instancing") != "explicit host attribute-divisor v1" or
            report.get("committed_fragments") != committed or report.get("discarded_fragments") != discarded or
            report.get("covered_pixels") != int(np.count_nonzero(expected)) or
            report.get("max_overdraw") != int(expected.max()) or report.get("fpga_execution") is not False or
            type(report.get("host_steps")) is not int or not 0 < report["host_steps"] <= MAX_DRAW_STEPS):
        raise ValueError(f"{case.name}: instanced coverage/kill/status/budget differs")


def _run_case(directory, build, case):
    directory.mkdir(parents=True, exist_ok=True)
    shader = _shader(case, directory)
    mesh, instances = directory / "vertices.bin", directory / "instances.bin"
    mesh.write_bytes(case_mesh())
    instances.write_bytes(instance_bytes(case))
    options = {"mesh": mesh, "instance_data": instances, "depth_test": case.depth_test, "blend": case.blend}
    expected_images, expected_coverages, expected_depths, discarded = rational_oracle(case)
    instance_id_controls = {}
    if case.instance_id_compare is not None:
        controls = {"always_zero": rational_oracle(case, force_instance_id=0)[0][-1],
                    "no_builtin_branch": rational_oracle(replace(case, instance_id_compare=None))[0][-1]}
        for name, control in controls.items():
            comparison = image_comparison(expected_images[-1], control, 0)
            if comparison["passed"]:
                raise ValueError(f"{case.name}: instance-ID {name} control does not distinguish local IDs")
            instance_id_controls[name] = comparison
    artifact = capture_draw(shader, build, directory / "capture", **options)
    check_contract(artifact, case)
    resources = resource_observations(artifact)
    managed_images, managed_reports = [], []
    image = None
    depth = np.full((16, 16), 65535, np.uint16) if case.depth_test else None
    for draw_index in range(case.repeats):
        output_depth = np.empty((16, 16), np.uint16) if case.depth_test else None
        image, coverage, report = render_artifact(artifact, depth_test=case.depth_test, blend=case.blend,
                                                 initial_image=image, initial_depth=depth, depth_output=output_depth,
                                                 max_steps=MAX_DRAW_STEPS)
        _check_execution(coverage, report, expected_coverages[draw_index], discarded[draw_index], case)
        if case.depth_test and not np.array_equal(output_depth, expected_depths[draw_index]):
            raise ValueError(f"{case.name}: managed Z16 commit oracle differs")
        depth = output_depth
        managed_images.append(image.copy())
        managed_reports.append(report)
    with tempfile.TemporaryDirectory(prefix="mgc-instances-") as temporary:
        with HostExecutorServer(Path(temporary) / "executor.sock", max_requests=case.repeats,
                                max_steps=MAX_DRAW_STEPS) as server:
            native = native_host_readback(shader, build, directory / "native", socket_path=server.path,
                                          repeats=case.repeats, **options)
        if server.errors or len(server.records) != case.repeats:
            raise ValueError(f"{case.name}: native instancing requests failed: {server.errors}")
        records = list(server.records)
    native_report = json.loads((directory / "native/native-report.json").read_text())
    for draw_index, record in enumerate(records):
        check_contract(record["artifact"], case)
        _check_execution(record["coverage"], record["report"], expected_coverages[draw_index], discarded[draw_index], case)
        if (not np.array_equal(record["rgba"], managed_images[draw_index]) or
                json.loads(Path(native_report["capture_paths"][draw_index]).read_text()) != record["artifact"] or
                native_report["commits"][draw_index]["sequence"] != record["sequence"]):
            raise ValueError(f"{case.name}: actual native captures/accepted replies differ from exact managed draws")
        if case.depth_test and not np.array_equal(record["depth"], expected_depths[draw_index]):
            raise ValueError(f"{case.name}: native Z16 commits differ")
    if not np.array_equal(native, records[-1]["rgba"]):
        raise ValueError(f"{case.name}: actual glReadPixels differs from accepted final executor reply")
    reference = softpipe_reference(shader, build, directory / "reference", repeats=case.repeats, **options)
    tolerance = 1 if case.blend != "disabled" else 0
    comparisons = {"native_vs_managed": image_comparison(native, image, 0),
                   "managed_vs_rational": image_comparison(image, expected_images[-1], 0),
                   "native_vs_rational": image_comparison(native, expected_images[-1], 0),
                   "softpipe_vs_rational": image_comparison(reference, expected_images[-1], tolerance),
                   "native_vs_softpipe": image_comparison(native, reference, tolerance)}
    if not all(check["passed"] for check in comparisons.values()):
        raise ValueError(f"{case.name}: full-image instancing checks differ: {comparisons}")
    for name, pixels in (("managed", image), ("native", native), ("softpipe", reference), ("rational", expected_images[-1])):
        (directory / f"{name}.rgba8").write_bytes(pixels.tobytes())
    for draw_index, coverage in enumerate(expected_coverages):
        (directory / f"coverage-{draw_index + 1}.u32").write_bytes(coverage.astype("<u4").tobytes())
    if case.depth_test:
        (directory / "depth.z16").write_bytes(expected_depths[-1].astype("<u2").tobytes())
    return {"case": case.name, "passed": True, "instance_count": case.count, "offset_divisor": case.offset_divisor,
            "color_divisor": case.color_divisor, "kill": case.kill, "zero_alpha": case.zero_alpha,
            "depth_test": case.depth_test, "blend": case.blend, "repeats": case.repeats,
            "shader_abi_version": INSTANCING_SHADER_ABI, "instance_id_word": INSTANCE_ID_WORD,
            "instance_id_compare": case.instance_id_compare, "instance_id_controls": instance_id_controls,
            "comparisons": comparisons, "managed_draw_reports": managed_reports, "resources": resources,
            "maximum_channel_error": max(check["max_channel_error"] for check in comparisons.values()),
            "coverage_oracle": "explicit per-instance commits, never RGBA/alpha", "start_instance": 0,
            "evidence": {"draw": str(directory / "capture/draw.json"), "shader_directory": str(shader),
                         "mesh": str(mesh), "instance_data": str(instances),
                         "native_report": str(directory / "native/native-report.json"),
                         "native_captures": native_report["capture_paths"]}}


def validate(output, build, *, cases=DEFAULT_CASES):
    cases = tuple(cases)
    if not cases or any(case not in CASES for case in cases) or len({case.name for case in cases}) != len(cases):
        raise ValueError("instancing acceptance requires distinct named bounded cases")
    output, build = Path(output), Path(build)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"passed": False, "status": "running", "cases": [], "tested_cases": [case.name for case in cases],
               "native_gl_readback": False, "fpga_execution": False, "exact_fp32_conformance": False,
               "contract": HOST_INSTANCING_CONTRACT, "oracle": "independent Fraction pixel-center rectangles, kill/Z16 commits and per-commit RGBA8 blend rounding",
               "scope": "advertised ANGLE/EXT glDrawArraysInstanced and attribute divisors, legacy host ABI3 local integer instance-ID branches; base instance zero, no GL indexed/base-instance claim",
               "additional_indexed_base_evidence": "separate direct C/Python instancing capture/fetch guard tests",
               "softpipe_blend_channel_tolerance": 1, "unblended_channel_tolerance": 0}
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
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/instancing")
    parser.add_argument("--build", type=Path, default=Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build")))
    parser.add_argument("--case", action="append", choices=[case.name for case in CASES])
    parser.add_argument("--extended", action="store_true", help="run all 19 cases, including four real instance-ID branches")
    args = parser.parse_args(argv)
    try:
        cases = (CASES if args.extended else DEFAULT_CASES) if args.case is None else tuple(
            next(case for case in CASES if case.name == name) for name in args.case)
        summary = validate(args.output, args.build, cases=cases)
        print(json.dumps({"passed": True, "tested_cases": summary["tested_cases"], "native_gl_readback": True,
                          "maximum_channel_error": summary["maximum_channel_error"], "summary": str(args.output / "summary.json")}))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "error", "message": str(error), "summary": str(args.output / "summary.json")}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
