"""Fresh bounded legacy texture/discard CFG acceptance against CPU softpipe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

from tools.host_render import render_artifact
from tools.host_texture import HOST_TEXTURE_LAYOUT, SAMPLER_STATE
from tools.mgcdriver import ProgramFormat
from tools.mgcreference.artifacts import write_png
from tools.run_render import capture_draw, compare_images, softpipe_reference
from tools.run_shader import ROOT
from tools.shader_layout import ShaderLayout
from tools.texture_acceptance import quad_mesh, texel_oracle, texture_pattern


FIXTURE = ROOT / "tests/fixtures/host-texture-flow"
CASES = tuple((f"{variant}-{size}x{size}", variant, size)
              for variant in ("late", "early", "runtime-loop") for size in (4, 2))
STATUS = {"version": 1, "word_offset": 65535, "pending": 0, "discarded": 1, "committed": 2}


def flow_oracle(texels):
    expected = texel_oracle(texels, "identity")
    expected[:, :8] = 0
    coverage = np.zeros((16, 16), dtype=np.uint32)
    coverage[:, 8:] = 1
    return expected, coverage


def validate(output, build):
    output = Path(output)
    build = Path(build)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for name, variant, size in CASES:
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        shader = FIXTURE if variant == "late" else FIXTURE / variant
        mesh = directory / "vertices.bin"
        mesh.write_bytes(quad_mesh())
        texels = texture_pattern(size)
        texture = directory / "texture.bin"
        texture.write_bytes(struct.pack("<II", size, size) + texels.tobytes())
        artifact = capture_draw(shader, build, directory, mesh=mesh, host_texture=True,
                                texture=texture, fragment_status=True)
        vertex_layout = ShaderLayout(artifact["shaders"]["vs"]["layout"])
        fragment_layout = ShaderLayout(artifact["shaders"]["fs"]["layout"])
        if (artifact.get("schema") != "mgc-host-draw-v2" or
                artifact.get("program_format") != ProgramFormat.LEGACY.value or
                type(artifact.get("shader_abi_version")) is not int or
                artifact["shader_abi_version"] != 1 or
                vertex_layout.source["version"] != 1 or vertex_layout.host_texture is not None or
                vertex_layout.fragment_status is not None or fragment_layout.source["version"] != 2 or
                fragment_layout.host_texture != HOST_TEXTURE_LAYOUT or
                fragment_layout.fragment_status != STATUS or
                fragment_layout.source.get("uses_discard") is not True):
            raise ValueError(f"{name}: fresh capture lost the legacy texture/status contract")
        if variant == "runtime-loop" and "mgc_block_" not in artifact["shaders"]["fs"]["assembly"]:
            raise ValueError(f"{name}: runtime loop must retain actual compiled control flow")
        expected_texture = {"width": size, "height": size, "format": "R8G8B8A8_UNORM",
                            "data_hex": texels.tobytes().hex(), "sampler": dict(SAMPLER_STATE)}
        if artifact.get("textures") != [expected_texture]:
            raise ValueError(f"{name}: captured resources differ from uploaded RGBA8 bytes/state")
        image, coverage, report = render_artifact(artifact)
        reference = softpipe_reference(shader, build, directory, mesh=mesh, texture=texture)
        expected, expected_coverage = flow_oracle(texels)
        comparison = compare_images(image, reference, 0)
        if (not comparison["passed"] or not np.array_equal(image, expected) or
                not np.array_equal(reference, expected) or
                not np.array_equal(coverage, expected_coverage)):
            raise ValueError(f"{name}: raw-texel, explicit coverage or softpipe oracle failed: {comparison}")
        counts = {"vertices": 6, "triangles": 2, "fragments": 256, "committed_fragments": 128,
                  "discarded_fragments": 128, "covered_pixels": 128, "max_overdraw": 1}
        if (any(report[key] != value for key, value in counts.items()) or
                report["host_steps"] <= 256 or report["fpga_execution"] is not False or
                report["fragment_completion"] != "explicit legacy host status v1"):
            raise ValueError(f"{name}: actual managed invocation/completion counts differ")
        (directory / "host.rgba8").write_bytes(image.tobytes())
        (directory / "expected.rgba8").write_bytes(expected.tobytes())
        (directory / "coverage.u32").write_bytes(coverage.astype("<u4").tobytes())
        write_png(directory / "host.png", image[:, :, :3])
        write_png(directory / "softpipe.png", reference[:, :, :3])
        report["softpipe_comparison"] = comparison
        report["texel_oracle"] = "uploaded RGBA8, normalized nearest-clamp identity coordinates"
        report["coverage_oracle"] = "right eight columns commit; left eight discard; never RGBA/alpha sentinel inference"
        report["fixture"] = variant
        report["texture_dimensions"] = [size, size]
        report["runtime_cfg_retained"] = variant == "runtime-loop"
        report["softpipe_raw_pixel_origin"] = "bottom-left; compared after conversion to top-left"
        report["evidence"] = {"draw": str(directory / "draw.json"), "mesh": str(mesh),
                              "texture": str(texture), "shader_directory": str(shader),
                              "host_rgba8": str(directory / "host.rgba8"),
                              "softpipe_rgba8": str(directory / "softpipe.rgba8"),
                              "expected_rgba8": str(directory / "expected.rgba8"),
                              "coverage": str(directory / "coverage.u32")}
        (directory / "render.json").write_text(json.dumps(report, indent=2) + "\n")
        results.append({"case": name, **report})
    summary = {"passed": True, "cases": results, "fpga_execution": False,
               "contract": "bounded legacy texture layout v2, resource wire v2, explicit fragment status v1 and structured CFG",
               "softpipe_channel_tolerance": 0}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/host-texture-flow")
    parser.add_argument("--build", type=Path,
                        default=Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build")))
    args = parser.parse_args(argv)
    try:
        summary = validate(args.output, args.build)
        print(json.dumps({"passed": summary["passed"], "cases": len(summary["cases"]),
                          "fpga_execution": False, "summary": str(args.output / "summary.json"),
                          "evidence": [case["evidence"] for case in summary["cases"]]}))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"texture flow acceptance failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
