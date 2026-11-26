"""Fresh GLSL compilation for the separate c-interpreter-v1 target."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from tools.mgcc import assemble_c, disassemble_c
from tools.mgcc.runtime import execute_c_pair
from tools.mgcdriver import DriverError, Program, ProgramFormat
from tools.run_shader import compile_glsl
from tools.shader_layout import ShaderLayout


def compile_c_glsl(shader_directory, build, output):
    shader_directory, build, output = map(Path, (shader_directory, build, output))
    if output.exists() and any(output.iterdir()):
        raise ValueError("C compilation requires a new/empty output folder; previous evidence is not overwritten")
    libraries = list((build / "src/gallium/targets/dri").glob("libgallium-*.so"))
    if len(libraries) != 1:
        raise ValueError("build Mesa first with bash tools/build_project.sh")
    files = {name: hashlib.sha256((shader_directory / name).read_bytes()).hexdigest()
             for name in ("vert.glsl", "frag.glsl")}
    library_hash = hashlib.sha256(libraries[0].read_bytes()).hexdigest()
    assembly = compile_glsl(shader_directory, build, output, ProgramFormat.C_INTERPRETER)
    layouts, programs = {}, {}
    for stage in ("vs", "fs"):
        layouts[stage] = json.loads((output / f"{stage}.layout.json").read_text())
        checked = ShaderLayout(layouts[stage])
        if checked.target != ProgramFormat.C_INTERPRETER or checked.source.get("uses_discard"):
            raise ValueError("C artifact reflection is unsupported")
        words = assemble_c(assembly[stage])
        decoded = disassemble_c(words)
        if assemble_c("\n".join(decoded)) != words:
            raise ValueError("C decoder roundtrip failed")
        binary = Program.from_words(words, format=ProgramFormat.C_INTERPRETER).to_bytes()
        (output / f"{stage}.bin").write_bytes(binary)
        (output / f"{stage}.mem").write_text("".join(f"{word:08x}\n" for word in words))
        (output / f"{stage}.disasm").write_text("\n".join(decoded) + "\n")
        programs[stage] = {"words": len(words), "binary_sha256": hashlib.sha256(binary).hexdigest(),
                           "instructions": dict(sorted(Counter(line.split()[0] for line in decoded).items()))}
    if files != {name: hashlib.sha256((shader_directory / name).read_bytes()).hexdigest() for name in files} or library_hash != hashlib.sha256(libraries[0].read_bytes()).hexdigest():
        raise ValueError("GLSL source or Gallium build changed during compilation")
    manifest = {"compiler_target": "c-interpreter-v1", "program_format": "c-interpreter-v1",
                "shader_abi_version": 5, "address_units": "bytes", "source_sha256": files,
                "gallium_sha256": library_hash, "programs": programs,
                "fpga_execution": False, "rtl_timing_validated": False,
                "wait_policy": "no waits: supplied C has no hardware latency model",
                "numerics": "supplied-C FP16 truncation; negative-half signed-shift issue remains unresolved"}
    (output / "compile.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"assembly": assembly, "layouts": layouts, "manifest": manifest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shader", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", type=Path, default=Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build")))
    parser.add_argument("--payload", type=Path, help="execute VS->FS with explicit reflected JSON values")
    parser.add_argument("--phong", action="store_true", help="execute the repository Phong shader and compare a float64 formula")
    parser.add_argument("--max-steps", type=int, default=1000000)
    arguments = parser.parse_args(argv)
    try:
        compiled = compile_c_glsl(arguments.shader, arguments.build, arguments.output)
        report = {"compiled": True, **compiled["manifest"]}
        if arguments.payload and arguments.phong:
            raise ValueError("choose explicit payload or the Phong fixture, not both")
        if arguments.payload or arguments.phong:
            if arguments.phong:
                from tools.run_vector_shader import phong_payload, reference_phong
                payload = phong_payload(compiled)
            else:
                source = json.loads(arguments.payload.read_text())
                if set(source) != {"vertex_inputs", "vertex_uniforms", "fragment_uniforms"}:
                    raise ValueError("payload requires vertex_inputs, vertex_uniforms and fragment_uniforms")
                payload = {name: {int(offset): value for offset, value in values.items()}
                           for name, values in source.items()}
            report["execution"] = execute_c_pair(compiled, **payload, max_steps=arguments.max_steps)
            if arguments.phong:
                expected = reference_phong((0.25, -0.125, 0.0), (0.0, 0.0, 1.0),
                                           (1.0, 0.5, 3.0), (0.0, 0.0, 3.0), (0.7, 0.4, 0.2))
                actual = [report["execution"]["fragment"]["outputs"][offset] for offset in range(4)]
                error = max(abs(lhs - rhs) for lhs, rhs in zip(actual, expected))
                report["phong_reference"] = {"actual": actual, "expected": expected, "max_error": error, "tolerance": 0.05}
                report["passed"] = error <= 0.05
        (arguments.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0 if report.get("passed", True) else 1
    except (ValueError, OSError, DriverError, subprocess.TimeoutExpired) as error:
        print(f"C target failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
