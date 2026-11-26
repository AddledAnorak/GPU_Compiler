"""Fresh Gallium managed-host feedback and actual GL readback against softpipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

import numpy as np

from tools.blend_acceptance import analytic_oracle, mesh_words
from tools.host_executor import HostExecutorServer
from tools.mgcreference.artifacts import write_png
from tools.run_render import compare_images, native_host_readback, softpipe_reference
from tools.run_shader import ROOT
from tools.teapot_mesh import generate_mesh
from tools.texture_acceptance import quad_mesh, texture_pattern


CASES = (
    ("flat", "shaders/01_flat_color", None, None, "disabled", 1, 0),
    ("gradient", "shaders/02_varying", None, None, "disabled", 1, 0),
    ("uniform", "shaders/03_uniform", None, None, "disabled", 1, 1),
    ("phong", "shaders/07_lighting_phong", None, None, "disabled", 1, 8),
    ("teapot", "shaders/07_lighting_phong", "teapot", None, "disabled", 1, 16),
    ("zero-rgba", "tests/fixtures/fragment-discard/zero-rgba", "quad", None, "disabled", 1, 0),
    ("discard", "tests/fixtures/fragment-discard", "quad", None, "disabled", 1, 0),
    ("texture", "tests/fixtures/host-texture", "quad", 4, "disabled", 1, 0),
    ("texture-late-discard", "tests/fixtures/host-texture-flow", "quad", 4, "disabled", 1, 0),
    ("texture-loop-discard", "tests/fixtures/host-texture-flow/runtime-loop", "quad", 2, "disabled", 1, 0),
    ("blend-repeated", "tests/fixtures/host-blend", "overlap", None, "source-over", 3, 1),
)
CASES = tuple((*case, False) for case in CASES) + (
    ("depth-discard-alpha-zero", "tests/fixtures/fragment-discard/overlap", "overlap", None, "disabled", 2, 0, True),
    ("depth-discard-far-first", "tests/fixtures/fragment-discard/overlap", "far-overlap", None, "disabled", 2, 0, True),
    ("depth-loop-discard", "tests/fixtures/fragment-discard/runtime-loop", "overlap", None, "disabled", 2, 0, True),
    ("depth-texture-loop-discard", "tests/fixtures/host-texture-flow/runtime-loop", "overlap", 2, "disabled", 2, 0, True),
    ("depth-blend-repeated", "tests/fixtures/host-blend", "overlap", None, "source-over", 2, 1, True),
    ("teapot-depth", "shaders/07_lighting_phong", "teapot", None, "disabled", 1, 16, True),
)


def validate(output, build):
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for name, shader_name, geometry, texture_size, blend, repeats, tolerance, depth_test in CASES:
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        shader = ROOT / shader_name
        if not shader.is_dir():
            raise ValueError(f"missing native acceptance shader: {shader}")
        mesh = None
        if geometry:
            mesh = directory / "vertices.bin"
            if geometry == "teapot":
                payload, metadata = generate_mesh(3)
                (directory / "mesh.json").write_text(json.dumps(metadata, indent=2) + "\n")
            elif geometry == "quad":
                payload = quad_mesh()
            else:
                payload = mesh_words((0.5, -0.5) if geometry == "far-overlap" else (-0.5, 0.5))
            mesh.write_bytes(payload)
        texture = None
        if texture_size:
            texture = directory / "texture.bin"
            texture.write_bytes(struct.pack("<II", texture_size, texture_size) + texture_pattern(texture_size).tobytes())
        with tempfile.TemporaryDirectory(prefix="mgc-native-") as temporary:
            path = Path(temporary) / "executor.sock"
            with HostExecutorServer(path, max_requests=repeats) as server:
                image = native_host_readback(shader, build, directory, socket_path=path,
                                             mesh=mesh, texture=texture, blend=blend, repeats=repeats,
                                             depth_test=depth_test)
            records = list(server.records)
            if server.errors or len(records) != repeats:
                raise ValueError(f"{name}: managed host requests failed: {server.errors}")
        if not np.array_equal(image, records[-1]["rgba"]):
            raise ValueError(f"{name}: actual GL readback differs from accepted host reply")
        commits = re.findall(r"MGC_HOST_EXECUTOR_COMMIT sequence=(\d+) width=16 height=16", (directory / "native.log").read_text())
        if [int(sequence) for sequence in commits] != [record["sequence"] for record in records]:
            raise ValueError(f"{name}: explicit native commits and managed draw sequences disagree")
        reference = softpipe_reference(shader, build, directory, mesh=mesh, texture=texture,
                                       blend=blend, repeats=repeats, depth_test=depth_test)
        comparison = compare_images(image, reference, tolerance)
        if not comparison["passed"]:
            raise ValueError(f"{name}: native readback softpipe comparison failed: {comparison}")
        if name == "zero-rgba" and (np.any(image) or records[-1]["report"]["committed_fragments"] != 256):
            raise ValueError("zero RGBA must have 256 explicit committed fragments")
        if "discard" in name and not depth_test and (records[-1]["report"]["discarded_fragments"] != 128 or
                                  records[-1]["report"]["committed_fragments"] != 128):
            raise ValueError(f"{name}: explicit kill/commit counts disagree with the half-quad oracle")
        if name == "blend-repeated":
            oracle = analytic_oracle((-0.5, 0.5) * repeats, blend=blend)[0]
            if not np.array_equal(image, oracle):
                raise ValueError("repeated native source-over disagrees with the rational per-commit oracle")
        if depth_test:
            if any(record["depth"] is None or record["report"]["depth_storage"] != "Z16_UNORM" for record in records):
                raise ValueError(f"{name}: native depth must use explicit Z16 feedback")
            if repeats > 1 and records[-1]["report"]["committed_fragments"] != 0:
                raise ValueError(f"{name}: repeated equal-depth fragments must not commit")
            if name in ("depth-discard-alpha-zero", "depth-discard-far-first", "depth-loop-discard"):
                expected = np.empty((16, 16, 4), dtype=np.uint8)
                expected[:, :8] = (0, 0, 255, 255)
                expected[:, 8:] = (255, 0, 0, 0)
                expected_depth = np.empty((16, 16), dtype=np.uint16)
                expected_depth[:, :8] = 49151
                expected_depth[:, 8:] = 16384
                if not np.array_equal(image, expected) or not np.array_equal(records[-1]["depth"], expected_depth):
                    raise ValueError(f"{name}: kill/transparent-depth independent oracle failed")
            if name == "depth-texture-loop-discard":
                expected_depth = np.empty((16, 16), dtype=np.uint16)
                expected_depth[:, :8] = 65535
                expected_depth[:, 8:] = 16384
                if np.any(image[:, :8]) or not np.array_equal(records[-1]["depth"], expected_depth):
                    raise ValueError("texture-loop discard must preserve initial color and depth")
        for record in records:
            (directory / f"draw-{record['sequence']}.json").write_text(json.dumps(record["artifact"], indent=2) + "\n")
        (directory / "coverage.u32").write_bytes(records[-1]["coverage"].astype("<u4").tobytes())
        (directory / "host-reply.rgba8").write_bytes(records[-1]["rgba"].tobytes())
        if depth_test:
            (directory / "host-reply.z16").write_bytes(records[-1]["depth"].astype("<u2").tobytes())
        write_png(directory / "native.png", image[:, :, :3])
        write_png(directory / "softpipe.png", reference[:, :, :3])
        report = {**records[-1]["report"], "case": name, "native_gl_readback": True,
                  "fpga_execution": False, "native_commits": len(commits), "draw_repeats": repeats,
                  "rgba_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                  "reply_matches_gl_readback": True, "softpipe_comparison": comparison,
                  "native_contract": "explicit same-user legacy host executor v2 RGBA8/Z16 LESS" if depth_test else
                                     "explicit same-user legacy host executor v1 RGBA8; depth disabled"}
        (directory / "render.json").write_text(json.dumps(report, indent=2) + "\n")
        results.append(report)
    failure_dir = output / "execution-failure"
    failure_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mgc-native-failure-") as temporary:
        with HostExecutorServer(Path(temporary) / "executor.sock", max_requests=1, max_steps=1) as server:
            try:
                native_host_readback(ROOT / "shaders/01_flat_color", build, failure_dir, socket_path=server.path)
            except ValueError:
                pass
            else:
                raise ValueError("native shader budget failure must not report a successful draw")
        if server.records or not server.errors or "MGC_HOST_EXECUTOR_COMMIT" in (failure_dir / "native.log").read_text():
            raise ValueError("execution failure must have no accepted reply or native framebuffer commit")
    summary = {"passed": True, "cases": results, "expected_failures": [{"case": "instruction-budget",
               "native_commits": 0, "errors": server.errors}], "native_gl_readback": True,
               "fpga_execution": False, "physical_or_rtl_validation": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/native-host")
    parser.add_argument("--build", type=Path, default=Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build")))
    args = parser.parse_args(argv)
    try:
        summary = validate(args.output, args.build)
        print(json.dumps({"passed": summary["passed"], "cases": len(summary["cases"]),
                          "expected_failures": len(summary["expected_failures"]),
                          "native_gl_readback": True, "fpga_execution": False}))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"native host acceptance failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
