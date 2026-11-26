"""Fresh source-over blend acceptance with exact per-commit and softpipe oracles."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

from tools.host_render import render_artifact
from tools.mgcreference.artifacts import write_png
from tools.run_render import capture_draw, compare_images, softpipe_reference
from tools.run_shader import ROOT


CASES = [
    ("disabled-near-first", "frag.glsl", (-0.5, 0.5), False, "disabled"),
    ("source-over-near-first", "frag.glsl", (-0.5, 0.5), False, "source-over"),
    ("source-over-far-first", "frag.glsl", (0.5, -0.5), False, "source-over"),
    ("three-layer-rounding", "frag.glsl", (-0.5, 0, 0.5), False, "source-over"),
    ("depth-near-first", "frag.glsl", (-0.5, 0.5), True, "source-over"),
    ("depth-far-first", "frag.glsl", (0.5, -0.5), True, "source-over"),
    ("zero-alpha-near-first", "zero-alpha.frag.glsl", (-0.5, 0.5), False, "source-over"),
    ("zero-alpha-depth-near-first", "zero-alpha.frag.glsl", (-0.5, 0.5), True, "source-over"),
    ("zero-alpha-depth-far-first", "zero-alpha.frag.glsl", (0.5, -0.5, 0), True, "source-over"),
    ("clamped-rgba", "clamped.frag.glsl", (-0.5, 0.5), False, "source-over"),
    ("clamped-alpha", "clamped-alpha.frag.glsl", (-0.5, 0.5), False, "source-over"),
    ("discard-near-first", "discard.frag.glsl", (-0.5, 0.5), True, "source-over"),
    ("discard-far-first", "discard.frag.glsl", (0.5, -0.5), True, "source-over"),
    ("discard-no-depth", "discard.frag.glsl", (-0.5, 0.5), False, "source-over"),
    ("discard-zero-alpha-near-first", "discard-zero-alpha.frag.glsl", (-0.5, 0.5), True, "source-over"),
    ("discard-zero-alpha-far-first", "discard-zero-alpha.frag.glsl", (0.5, -0.5), True, "source-over"),
]


def mesh_words(depths):
    positions = [(-1, -1), (1, -1), (1, 1), (-1, -1), (1, 1), (-1, 1)]
    return b"".join(struct.pack("<7f", horizontal, vertical, depth, 1, 0, 0, 1)
                    for depth in depths for horizontal, vertical in positions)


def _oracle_source(variant, depth):
    half = Fraction(1, 2)
    if variant == "clamped.frag.glsl":
        return (Fraction(3, 2), Fraction(-1, 2), Fraction(1, 4), half)
    if variant in ("zero-alpha.frag.glsl", "discard-zero-alpha.frag.glsl"):
        alpha = half + depth
    elif variant == "clamped-alpha.frag.glsl":
        alpha = half + depth * 2
    elif variant in ("frag.glsl", "discard.frag.glsl"):
        alpha = half
    else:
        raise ValueError("unsupported analytic blend fixture")
    return (half - depth, Fraction(0), half + depth, alpha)


def analytic_oracle(depths, variant="frag.glsl", *, depth_test=False, blend="source-over"):
    if type(blend) is not str or blend not in ("disabled", "source-over"):
        raise ValueError("unsupported analytic blend mode")
    image = np.zeros((16, 16, 4), dtype=np.uint8)
    coverage = np.zeros((16, 16), dtype=np.uint32)
    fragments = 0
    discarded = 0
    for column in range(16):
        destination = (0, 0, 0, 0)
        stored_depth = Fraction(1)
        for value in depths:
            depth = Fraction(value)
            window_depth = (depth + 1) / 2
            if depth_test and window_depth >= stored_depth:
                continue
            fragments += 16
            if variant in ("discard.frag.glsl", "discard-zero-alpha.frag.glsl") and column < 8 and depth < 0:
                discarded += 16
                continue
            source = tuple(min(max(channel, Fraction(0)), Fraction(1))
                           for channel in _oracle_source(variant, depth))
            if blend == "source-over":
                alpha = source[3]
                result = tuple(source[component] * alpha + Fraction(destination[component], 255) * (1 - alpha)
                               for component in range(3)) + (alpha + Fraction(destination[3], 255) * (1 - alpha),)
            else:
                result = source
            destination = tuple(round(channel * 255) for channel in result)
            coverage[:, column] += 1
            stored_depth = window_depth
        image[:, column] = destination
    return image, coverage, {"fragments": fragments, "discarded_fragments": discarded,
                             "committed_fragments": fragments - discarded}


def validate(output, build):
    fixture = ROOT / "tests/fixtures/host-blend"
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for name, variant, depths, depth_test, blend in CASES:
        directory = output / name
        shader = directory / "shader"
        shader.mkdir(parents=True, exist_ok=True)
        (shader / "vert.glsl").write_text((fixture / "vert.glsl").read_text())
        (shader / "frag.glsl").write_text((fixture / variant).read_text())
        mesh = directory / "vertices.bin"
        mesh.write_bytes(mesh_words(depths))
        artifact = capture_draw(shader, build, directory, mesh=mesh, depth_test=depth_test, blend=blend)
        image, coverage, report = render_artifact(artifact, depth_test=depth_test, blend=blend)
        reference = softpipe_reference(shader, build, directory, mesh=mesh, depth_test=depth_test, blend=blend)
        expected, expected_coverage, counts = analytic_oracle(depths, variant, depth_test=depth_test, blend=blend)
        comparison = compare_images(image, reference, 1)
        reference_oracle = compare_images(expected, reference, 1)
        if (not np.array_equal(image, expected) or not np.array_equal(coverage, expected_coverage) or
                not comparison["passed"] or not reference_oracle["passed"]):
            raise ValueError(f"{name}: source-over/commit/softpipe oracle failed: {comparison}")
        if (any(report[key] != value for key, value in counts.items()) or
                report["blend"] != blend or report["vertices"] != len(depths) * 6 or
                report["covered_pixels"] != 256 or report["host_steps"] <= counts["fragments"]):
            raise ValueError(f"{name}: explicit blending or managed invocation counts differ")
        (directory / "host.rgba8").write_bytes(image.tobytes())
        (directory / "expected.rgba8").write_bytes(expected.tobytes())
        (directory / "coverage.u32").write_bytes(coverage.astype("<u4").tobytes())
        write_png(directory / "host.png", image[:, :, :3])
        write_png(directory / "softpipe.png", reference[:, :, :3])
        report["softpipe_comparison"] = comparison
        report["softpipe_oracle_comparison"] = reference_oracle
        report["blend_oracle"] = "exact rational source-over, source RGBA clamping, RGBA8 ties-to-even after each accepted commit"
        report["coverage_oracle"] = "depth and explicit discard/commit state, never inferred from RGBA or alpha"
        report["depth_order"] = list(depths)
        report["fixture"] = variant
        report["evidence"] = {"draw": str(directory / "draw.json"), "mesh": str(mesh),
                              "shader_directory": str(shader), "host_rgba8": str(directory / "host.rgba8"),
                              "softpipe_rgba8": str(directory / "softpipe.rgba8"),
                              "expected_rgba8": str(directory / "expected.rgba8"),
                              "coverage": str(directory / "coverage.u32")}
        (directory / "render.json").write_text(json.dumps(report, indent=2) + "\n")
        results.append({"case": name, **report})
    summary = {"passed": True, "cases": results, "fpga_execution": False,
               "contract": "explicit bounded CPU source-over RGBA8, separate alpha factors, per-commit rounding",
               "softpipe_channel_tolerance": 1}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/host-blend")
    parser.add_argument("--build", type=Path, default=Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build")))
    args = parser.parse_args(argv)
    try:
        summary = validate(args.output, args.build)
        print(json.dumps({"passed": summary["passed"], "cases": len(summary["cases"]),
                          "fpga_execution": False, "summary": str(args.output / "summary.json"),
                          "evidence": [case["evidence"] for case in summary["cases"]]}))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"blend acceptance failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
