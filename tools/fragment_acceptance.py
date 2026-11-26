"""Fresh legacy fragment commit acceptance against independent CPU softpipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys

import numpy as np

from tools.host_render import render_artifact
from tools.mgcreference.artifacts import write_png
from tools.run_render import capture_draw, compare_images, softpipe_reference
from tools.run_shader import ROOT


def mesh_words(depths, left=-1):
    positions = [(left, -1), (1, -1), (1, 1), (left, -1), (1, 1), (left, 1)]
    return b"".join(struct.pack("<7f", horizontal, vertical, depth, 1, 0, 0, 1)
                    for depth in depths for horizontal, vertical in positions)


def validate(output, build):
    fixture = ROOT / "tests/fixtures/fragment-discard"
    cases = [("kill", fixture, (0,), -1, False), ("no-kill", fixture, (0,), 0, False),
             ("zero", fixture / "zero-rgba", (0,), -1, False),
             ("near-first", fixture / "overlap", (-0.5, 0.5), -1, True),
             ("far-first", fixture / "overlap", (0.5, -0.5), -1, True),
             ("branch-near-first", fixture / "branch-color", (-0.5, 0.5), -1, True),
             ("branch-far-first", fixture / "branch-color", (0.5, -0.5), -1, True),
             ("loop-near-first", fixture / "runtime-loop", (-0.5, 0.5), -1, True),
             ("loop-far-first", fixture / "runtime-loop", (0.5, -0.5), -1, True)]
    results = []
    for name, shader, depths, left, depth_test in cases:
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        mesh = directory / "vertices.bin"
        mesh.write_bytes(mesh_words(depths, left))
        artifact = capture_draw(shader, build, directory, mesh=mesh, depth_test=depth_test)
        if name.startswith("loop-") and "mgc_block_" not in artifact["shaders"]["fs"]["assembly"]:
            raise ValueError("runtime loop acceptance must retain actual compiled control flow")
        image, coverage, report = render_artifact(artifact, depth_test=depth_test)
        reference = softpipe_reference(shader, build, directory, mesh=mesh, depth_test=depth_test)
        comparison = compare_images(image, reference, 0)
        if not comparison["passed"]:
            raise ValueError(f"{name}: fresh softpipe pixels differ: {comparison}")
        if name in ("kill", "no-kill"):
            expected = np.zeros((16, 16, 4), dtype=np.uint8)
            expected[:, 8:] = [64, 128, 191, 0]
            expected_coverage = np.zeros((16, 16), dtype=np.uint32)
            expected_coverage[:, 8:] = 1
            discarded = 128 if name == "kill" else 0
        elif name == "zero":
            expected = np.zeros((16, 16, 4), dtype=np.uint8)
            expected_coverage = np.ones((16, 16), dtype=np.uint32)
            discarded = 0
        else:
            expected = np.zeros((16, 16, 4), dtype=np.uint8)
            expected[:, :8] = [0, 0, 255, 255]
            expected[:, 8:] = [255, 0, 0, 0]
            expected_coverage = np.ones((16, 16), dtype=np.uint32)
            if name.endswith("far-first"):
                expected_coverage[:, 8:] = 2
            discarded = 128
        if not np.array_equal(image, expected) or not np.array_equal(coverage, expected_coverage):
            raise ValueError(f"{name}: independent color/commit-count oracle failed")
        if (report["discarded_fragments"] != discarded or
                report["committed_fragments"] != int(expected_coverage.sum()) or
                report["fragments"] != report["committed_fragments"] + discarded):
            raise ValueError(f"{name}: explicit invocation completion counts differ")
        image.tofile(directory / "host.rgba8")
        write_png(directory / "host.png", image[:, :, :3])
        write_png(directory / "softpipe.png", reference[:, :, :3])
        report["softpipe_comparison"] = comparison
        report["commit_oracle"] = "explicit status and analytic coverage; never RGBA/alpha sentinel inference"
        (directory / "render.json").write_text(json.dumps(report, indent=2) + "\n")
        results.append({"case": name, **report})
    summary = {"passed": True, "cases": results, "fpga_execution": False,
               "contract": "legacy host fragment completion v1; bounded structured CFG discard"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/fragment-commit")
    parser.add_argument("--build", type=Path, default=ROOT / "mesa/build")
    args = parser.parse_args(argv)
    try:
        summary = validate(args.output, args.build)
        print(json.dumps({"passed": summary["passed"], "cases": len(summary["cases"]), "fpga_execution": False}))
        return 0
    except (ValueError, OSError) as error:
        print(f"fragment acceptance failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
