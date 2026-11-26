"""Fresh bounded blend equations through reflected shaders and actual host GL."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from tools.blend_acceptance import mesh_words
from tools.host_blend import BLEND_MODES
from tools.host_executor import HostExecutorServer
from tools.host_render import render_artifact
from tools.linear_texture_acceptance import image_comparison
from tools.mgcreference.artifacts import write_png
from tools.run_render import capture_draw, native_host_readback, softpipe_reference
from tools.run_shader import ROOT
from tools.shader_layout import ShaderLayout
from tools.texture_flow_acceptance import STATUS


CLEAR_COLOR = (0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class BlendCase:
    name: str
    blend: str
    variant: str
    depths: tuple
    depth_test: bool = False
    repeats: int = 1


CASES = tuple(BlendCase(f"{mode}-{name}", mode, variant, depths, depth_test, repeats)
              for mode in BLEND_MODES[2:]
              for name, variant, depths, depth_test, repeats in (
                  ("single-background", "frag.glsl", (-0.5,), False, 1),
                  ("near-first", "frag.glsl", (-0.5, 0.5), False, 1),
                  ("far-first", "frag.glsl", (0.5, -0.5), False, 1),
                  ("three-layers", "frag.glsl", (-0.5, 0, 0.5), False, 1),
                  ("clamped-rgba", "clamped.frag.glsl", (-0.5, 0.5), False, 1),
                  ("zero-alpha-depth", "zero-alpha.frag.glsl", (-0.5, 0.5), True, 1),
                  ("kill-depth-repeated", "discard.frag.glsl", (0.5, -0.5), True, 2),
                  ("kill-zero-alpha-depth", "discard-zero-alpha.frag.glsl", (-0.5, 0.5), True, 1),
              ))
DEFAULT_CASES = tuple(case for case in CASES if case.name in {
    f"{mode}-{name}" for mode in BLEND_MODES[2:]
    for name in ("single-background", "zero-alpha-depth", "kill-depth-repeated")})


def rational_commit(source, destination, blend):
    if type(blend) is not str or blend not in BLEND_MODES:
        raise ValueError("unsupported rational blend mode")
    if (len(source) != 4 or len(destination) != 4 or
            any(type(value) is not int or not 0 <= value <= 255 for value in destination)):
        raise ValueError("rational blending needs four source channels and RGBA8 destination")
    source = tuple(min(max(Fraction(channel), 0), 1) for channel in source)
    normalized = tuple(Fraction(channel, 255) for channel in destination)
    if blend == "disabled":
        result = source
    elif blend == "source-over":
        result = tuple(source[component] * source[3] + normalized[component] * (1 - source[3])
                       for component in range(3)) + (source[3] + normalized[3] * (1 - source[3]),)
    elif blend == "additive":
        result = tuple(incoming + previous for incoming, previous in zip(source, normalized))
    elif blend == "subtract":
        result = tuple(incoming - previous for incoming, previous in zip(source, normalized))
    elif blend == "reverse-subtract":
        result = tuple(previous - incoming for incoming, previous in zip(source, normalized))
    elif blend == "minimum":
        result = tuple(min(incoming, previous) for incoming, previous in zip(source, normalized))
    else:
        result = tuple(max(incoming, previous) for incoming, previous in zip(source, normalized))
    return tuple(round(min(max(channel, 0), 1) * 255) for channel in result)


def rational_oracle(case, clear_color=CLEAR_COLOR):
    if (type(clear_color) is not tuple or len(clear_color) != 4 or
            any(type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1
                for value in clear_color)):
        raise ValueError("clear color must be four explicit finite unit channels")
    if type(case.repeats) is not int or not 1 <= case.repeats <= 4:
        raise ValueError("rational draw repeats must be bounded")
    initial = tuple(round(Fraction(value) * 255) for value in clear_color)
    image = np.tile(np.asarray(initial, dtype=np.uint8), (16, 16, 1))
    coverage = np.zeros((16, 16), dtype=np.uint32)
    counts = {"fragments": 0, "discarded_fragments": 0, "committed_fragments": 0}
    for column in range(16):
        destination = initial
        stored_depth = Fraction(1)
        for iteration in range(case.repeats):
            for value in case.depths:
                depth = Fraction(value)
                window_depth = (depth + 1) / 2
                if case.depth_test and window_depth >= stored_depth:
                    continue
                counts["fragments"] += 16
                discarded = case.variant in ("discard.frag.glsl", "discard-zero-alpha.frag.glsl")
                if discarded and column < 8 and depth < 0:
                    counts["discarded_fragments"] += 16
                    continue
                if case.variant == "clamped.frag.glsl":
                    source = (Fraction(3, 2), Fraction(-1, 2), Fraction(1, 4), Fraction(1, 2))
                else:
                    alpha = Fraction(1, 2)
                    if case.variant in ("zero-alpha.frag.glsl", "discard-zero-alpha.frag.glsl"):
                        alpha += depth
                    elif case.variant not in ("frag.glsl", "discard.frag.glsl"):
                        raise ValueError("unsupported rational shader fixture")
                    source = (Fraction(1, 2) - depth, Fraction(0), Fraction(1, 2) + depth, alpha)
                destination = rational_commit(source, destination, case.blend)
                counts["committed_fragments"] += 16
                coverage[:, column] += 1
                stored_depth = window_depth
        image[:, column] = destination
    return image, coverage, counts


def validate(output, build, cases=DEFAULT_CASES):
    output = Path(output)
    build = Path(build)
    output.mkdir(parents=True, exist_ok=True)
    fixture = ROOT / "tests/fixtures/host-blend"
    results = []
    for case in cases:
        directory = output / case.name
        shader = directory / "shader"
        shader.mkdir(parents=True, exist_ok=True)
        (shader / "vert.glsl").write_text((fixture / "vert.glsl").read_text())
        (shader / "frag.glsl").write_text((fixture / case.variant).read_text())
        mesh = directory / "vertices.bin"
        mesh.write_bytes(mesh_words(case.depths))
        artifact = capture_draw(shader, build, directory, mesh=mesh, depth_test=case.depth_test,
                                blend=case.blend, clear_color=CLEAR_COLOR)
        vertex_layout = ShaderLayout(artifact["shaders"]["vs"]["layout"])
        fragment_layout = ShaderLayout(artifact["shaders"]["fs"]["layout"])
        if (artifact["program_format"] != "legacy-mgcsim-v1" or
                vertex_layout.source["shader_abi_version"] != 1 or
                fragment_layout.fragment_status != STATUS):
            raise ValueError(f"{case.name}: actual legacy reflection/status missing")
        initial = np.tile(np.rint(np.asarray(CLEAR_COLOR) * 255).astype(np.uint8), (16, 16, 1))
        image = initial
        depth = np.full((16, 16), 65535, dtype=np.uint16)
        coverage = np.zeros((16, 16), dtype=np.uint32)
        totals = {"fragments": 0, "discarded_fragments": 0, "committed_fragments": 0}
        reports = []
        for iteration in range(case.repeats):
            image, draw_coverage, report = render_artifact(
                artifact, depth_test=case.depth_test, blend=case.blend, initial_image=image,
                initial_depth=depth.copy() if case.depth_test else None,
                depth_output=depth if case.depth_test else None)
            coverage += draw_coverage
            for name in totals:
                totals[name] += report[name]
            reports.append(report)
        expected, expected_coverage, counts = rational_oracle(case)
        reference = softpipe_reference(shader, build, directory, mesh=mesh, depth_test=case.depth_test,
                                       repeats=case.repeats, blend=case.blend, clear_color=CLEAR_COLOR)
        with tempfile.TemporaryDirectory(prefix="btp-equations-") as temporary:
            socket_path = Path(temporary) / "executor.sock"
            with HostExecutorServer(socket_path) as server:
                native = native_host_readback(shader, build, directory / "native", socket_path=socket_path,
                                             mesh=mesh, depth_test=case.depth_test, repeats=case.repeats,
                                             blend=case.blend, clear_color=CLEAR_COLOR)
            if server.errors or len(server.records) != case.repeats:
                raise ValueError(f"{case.name}: native request/commit failed: {server.errors}")
            native_counts = {name: sum(record["report"][name] for record in server.records) for name in totals}
        comparison = image_comparison(image, reference, 1)
        native_comparison = image_comparison(native, expected, 0)
        if (not np.array_equal(image, expected) or not np.array_equal(coverage, expected_coverage) or
                totals != counts or native_counts != counts or not comparison["passed"] or
                not native_comparison["passed"]):
            raise ValueError(f"{case.name}: rational/commit/native/softpipe mismatch: "
                             f"{comparison}, {native_comparison}, {totals}, {native_counts}, {counts}")
        if any(report["blend"] != case.blend or report["host_steps"] <= 0 for report in reports):
            raise ValueError(f"{case.name}: actual managed execution evidence missing")
        (directory / "host.rgba8").write_bytes(image.tobytes())
        (directory / "expected.rgba8").write_bytes(expected.tobytes())
        (directory / "coverage.u32").write_bytes(coverage.astype("<u4").tobytes())
        write_png(directory / "host.png", image[:, :, :3])
        report = {"case": case.name, "blend": case.blend, "clear_color": list(CLEAR_COLOR),
                  "depth": "Z16 LESS" if case.depth_test else "disabled", "repeats": case.repeats,
                  "counts": counts, "softpipe_comparison": comparison,
                  "native_rational_comparison": native_comparison, "native_gl_readback": True,
                  "managed_draws": reports, "fpga_execution": False,
                  "evidence": {"capture": str(directory / "draw.json"),
                               "host": str(directory / "host.rgba8"),
                               "expected": str(directory / "expected.rgba8"),
                               "softpipe": str(directory / "softpipe.rgba8"),
                               "native": str(directory / "native/native.rgba8"),
                               "native_report": str(directory / "native/native-report.json")}}
        (directory / "render.json").write_text(json.dumps(report, indent=2) + "\n")
        results.append(report)
    summary = {"passed": True, "cases": results, "fpga_execution": False,
               "contract": "bounded unit-factor legacy host equations, RGBA8 per-commit rounding",
               "oracle": "independent rational source/destination, explicit kill/depth counts",
               "native_channel_tolerance": 0, "softpipe_channel_tolerance": 1}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/host-blend-equations")
    parser.add_argument("--build", type=Path, default=Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build")))
    parser.add_argument("--case", action="append", choices=[case.name for case in CASES])
    args = parser.parse_args(argv)
    cases = tuple(case for case in CASES if case.name in args.case) if args.case else DEFAULT_CASES
    try:
        summary = validate(args.output, args.build, cases)
        print(json.dumps({"passed": summary["passed"], "cases": len(summary["cases"]),
                          "fpga_execution": False, "summary": str(args.output / "summary.json")}))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"blend-equation acceptance failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
