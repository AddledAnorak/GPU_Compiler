"""Fresh legacy texture capture, managed execution and independent texel oracles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

from tools.host_texture import HOST_TEXTURE_LAYOUT, SAMPLER_STATE
from tools.host_render import render_artifact
from tools.mgcreference.artifacts import write_png
from tools.run_render import capture_draw, compare_images, softpipe_reference
from tools.run_shader import ROOT


def quad_mesh():
    positions = [(-1, -1), (1, -1), (1, 1), (-1, -1), (1, 1), (-1, 1)]
    return b"".join(struct.pack("<7f", horizontal, vertical, 0, 1, 0, 0, 1)
                    for horizontal, vertical in positions)


def texture_pattern(size):
    if size == 1:
        return np.array([[[13, 127, 250, 0]]], dtype=np.uint8)
    if size == 2:
        return np.array([[[255, 0, 0, 255], [0, 255, 0, 0]],
                         [[0, 0, 255, 128], [0, 0, 0, 0]]], dtype=np.uint8)
    rows, columns = np.indices((size, size))
    if size == 4:
        palette = np.array([[17, 63, 191, 0], [229, 127, 3, 255]], dtype=np.uint8)
        return palette[(rows + columns) % 2]
    if size == 16:
        return np.stack((columns * 17, rows * 17, (rows * 16 + columns),
                         np.where((rows + columns) % 3, 255, 0)), axis=2).astype(np.uint8)
    raise ValueError("acceptance pattern size must be 1, 2, 4 or 16")


def texel_oracle(texels, phase="identity"):
    rows, columns = np.indices((16, 16))
    horizontal = (columns + 0.5) / 16
    vertical = (15.5 - rows) / 16
    if phase == "clamp":
        horizontal, vertical = horizontal * 4 - 1.5, vertical * 4 - 1.5
    elif phase == "phase":
        horizontal, vertical = 1 - vertical, horizontal
    elif phase != "identity":
        raise ValueError("unsupported analytic texture coordinate case")
    height, width = texels.shape[:2]
    columns = np.clip(np.floor(horizontal * width), 0, width - 1).astype(int)
    rows = np.clip(np.floor(vertical * height), 0, height - 1).astype(int)
    return texels[rows, columns].copy()


def validate(output, build):
    fixture = ROOT / "tests/fixtures/host-texture"
    cases = [("checker-4x4", 4, "identity"), ("pattern-2x2", 2, "identity"),
             ("zero-alpha-1x1", 1, "identity"), ("centers-16x16", 16, "identity"),
             ("clamp-2x2", 2, "clamp"), ("phase-2x2", 2, "phase")]
    results = []
    output.mkdir(parents=True, exist_ok=True)
    for name, size, phase in cases:
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        shader = fixture if phase == "identity" else fixture / phase
        mesh = directory / "vertices.bin"
        mesh.write_bytes(quad_mesh())
        texels = texture_pattern(size)
        texture = directory / "texture.bin"
        texture.write_bytes(struct.pack("<II", size, size) + texels.tobytes())
        artifact = capture_draw(shader, build, directory, mesh=mesh,
                                host_texture=True, texture=texture)
        if (artifact.get("schema") != "mgc-host-draw-v2" or
                artifact["shaders"]["fs"]["layout"].get("host_texture") != HOST_TEXTURE_LAYOUT or
                artifact["shaders"]["fs"]["layout"]["version"] != 2 or
                artifact["shaders"]["vs"]["layout"]["version"] != 1 or
                artifact.get("shader_abi_version") != 1):
            raise ValueError(f"{name}: fresh capture did not preserve the actual host texture contract")
        expected_texture = {"width": size, "height": size, "format": "R8G8B8A8_UNORM",
                            "data_hex": texels.tobytes().hex(), "sampler": dict(SAMPLER_STATE)}
        if artifact.get("textures") != [expected_texture]:
            raise ValueError(f"{name}: fresh capture differs from raw uploaded RGBA8 bytes/state")
        image, coverage, report = render_artifact(artifact)
        reference = softpipe_reference(shader, build, directory, mesh=mesh, texture=texture)
        comparison = compare_images(image, reference, 0)
        expected = texel_oracle(texels, phase)
        if (not comparison["passed"] or not np.array_equal(image, expected) or
                not np.array_equal(reference, expected)):
            raise ValueError(f"{name}: independent raw-texel or softpipe oracle failed: {comparison}")
        if (not np.all(coverage == 1) or report["vertices"] != 6 or report["fragments"] != 256 or
                report["covered_pixels"] != 256 or report["host_steps"] <= 256 or
                report["discarded_fragments"] != 0 or report["committed_fragments"] != 256):
            raise ValueError(f"{name}: actual managed invocation or explicit coverage counts differ")
        (directory / "host.rgba8").write_bytes(image.tobytes())
        (directory / "expected.rgba8").write_bytes(expected.tobytes())
        (directory / "coverage.u32").write_bytes(coverage.astype("<u4").tobytes())
        write_png(directory / "host.png", image[:, :, :3])
        write_png(directory / "softpipe.png", reference[:, :, :3])
        report["softpipe_comparison"] = comparison
        report["texel_oracle"] = "analytic pixel-center coordinates, floor and clamp indexing of original RGBA8 input"
        report["coverage_oracle"] = "256 explicitly committed quad pixels, including zero RGBA and alpha-zero texels"
        report["coordinate_case"] = phase
        report["texture_width"] = size
        report["texture_height"] = size
        (directory / "render.json").write_text(json.dumps(report, indent=2) + "\n")
        results.append({"case": name, **report})
    summary = {"passed": True, "cases": results, "fpga_execution": False,
               "contract": "one legacy normalized nearest clamp-to-edge RGBA8 texture, host texture v1"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/host-texture")
    parser.add_argument("--build", type=Path, default=Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build")))
    args = parser.parse_args(argv)
    try:
        summary = validate(args.output, args.build)
        print(json.dumps({"passed": summary["passed"], "cases": len(summary["cases"]), "fpga_execution": False}))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"texture acceptance failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
