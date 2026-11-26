"""Separate experimental GLSL -> recovered vector words -> managed reference path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from tools.mgcdriver import ProgramFormat
from tools.mgchw import assemble_hardware, decode_word, disassemble_hardware
from tools.run_shader import ROOT, mesa_environment
from tools.shader_layout import ShaderLayout, link_varyings


VECTOR_TARGET = "fp16-hardware-vector-experimental-v1"


def compile_vector_glsl(shader_directory: Path, output: Path, build: Path | None = None) -> dict:
    shader_directory, output = Path(shader_directory).resolve(), Path(output).resolve()
    build = Path(build or ROOT / "mesa/build").resolve()
    harness = ROOT / "tools/mgc_harness"
    if not harness.is_file():
        raise ValueError("build Mesa and its GLES harness with bash tools/build_project.sh")
    if any((output / name).exists() for name in ("compile.json", "vs.asm", "fs.asm", "result.json")):
        raise ValueError("output already contains vector artifacts; choose a fresh directory")
    library = build / "src/gallium/targets/dri/libgallium-25.2.0.so"
    source_hashes = {name: hashlib.sha256((shader_directory / name).read_bytes()).hexdigest()
                     for name in ("vert.glsl", "frag.glsl")}
    library_hash = hashlib.sha256(library.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="btp-vector-glsl-") as temporary:
        dump = Path(temporary)
        environment = mesa_environment(build, dump, ProgramFormat.HARDWARE)
        for name in ("MGC_HOST_CAPTURE_DIR", "MGC_HOST_INSTANCING", "MGC_NIR_DUMP"):
            environment.pop(name, None)
        environment["MGC_COMPILE_TARGET"] = VECTOR_TARGET
        completed = subprocess.run(
            [str(harness), str(shader_directory / "vert.glsl"), str(shader_directory / "frag.glsl")],
            env=environment, capture_output=True, text=True, timeout=60,
        )
        (output / "compile.log").write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise ValueError(f"vector GLSL compilation failed; see {output / 'compile.log'}")
        assembly, layouts, stages = {}, {}, {}
        for stage, stage_name in (("vs", "vertex"), ("fs", "fragment")):
            source, layout_path = dump / f"{stage}.asm", dump / f"{stage}.layout.json"
            if not source.is_file() or not layout_path.is_file():
                raise ValueError(f"vector compiler produced no complete {stage} artifact")
            text = source.read_text()
            layout = json.loads(layout_path.read_text())
            if not text.startswith(f"; mgc {VECTOR_TARGET} backend, ABI=2, stage={stage_name}"):
                raise ValueError("compiler artifact is not from the separate vector target")
            if layout.get("compiler_target") != VECTOR_TARGET:
                raise ValueError("vector reflection provenance mismatch")
            ShaderLayout(layout)
            words = assemble_hardware(text)
            decoded = "\n".join(disassemble_hardware(words)) + "\n"
            if assemble_hardware(decoded) != words:
                raise ValueError("vector encoding does not round-trip every machine word")
            binary = struct.pack(f"<{len(words)}I", *words)
            (output / f"{stage}.asm").write_text(text)
            (output / f"{stage}.layout.json").write_text(json.dumps(layout, indent=2) + "\n")
            (output / f"{stage}.mem").write_text("".join(f"{word:08x}\n" for word in words))
            (output / f"{stage}.bin").write_bytes(binary)
            (output / f"{stage}.disasm").write_text(decoded)
            assembly[stage], layouts[stage] = text, layout
            stages[stage] = {
                "words": len(words), "bytes": len(binary),
                "binary_sha256": hashlib.sha256(binary).hexdigest(),
                "assembly_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "instructions": dict(sorted(Counter(decode_word(word).mnemonic for word in words).items())),
            }
    if source_hashes != {name: hashlib.sha256((shader_directory / name).read_bytes()).hexdigest()
                          for name in source_hashes} or library_hash != hashlib.sha256(library.read_bytes()).hexdigest():
        raise ValueError("GLSL source or Gallium library changed during compilation; retry with a frozen build")
    manifest = {
        "compiler_target": VECTOR_TARGET, "program_format": ProgramFormat.HARDWARE.value,
        "shader_abi_version": 2, "experimental": True, "lane_mode": "broadcast-scalar",
        "source_sha256": source_hashes, "gallium_sha256": library_hash,
        "stages": stages, "machine_code_emitted": True, "board_upload_ready": False,
        "fpga_execution": False, "rtl_timing_validated": False, "rasterization": False,
        "numerics": "approximate FP16 functional reference; no RTL numerical certification",
    }
    (output / "compile.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"assembly": assembly, "layouts": layouts, "manifest": manifest}


def execute_vector_pair(compiled: dict, vertex_inputs: dict, vertex_uniforms: dict,
                        fragment_uniforms: dict, *, allow_approximate_fp16: bool = False,
                        max_steps: int = 1000000) -> dict:
    from tools.mgcvector import VectorShaderExecutor

    layouts = {stage: ShaderLayout(source) for stage, source in compiled["layouts"].items()}
    mapping = link_varyings(layouts["vs"], layouts["fs"])
    with VectorShaderExecutor(compiled["assembly"]["vs"], layouts["vs"],
                              allow_approximate_fp16=allow_approximate_fp16, max_steps=max_steps) as executor:
        vertex = executor.execute(vertex_inputs, vertex_uniforms)
    fragment_inputs = {destination: vertex["outputs"][source] for destination, source in mapping.items()}
    with VectorShaderExecutor(compiled["assembly"]["fs"], layouts["fs"],
                              allow_approximate_fp16=allow_approximate_fp16, max_steps=max_steps) as executor:
        fragment = executor.execute(fragment_inputs, fragment_uniforms)
    return {
        "compiler_target": VECTOR_TARGET, "program_format": ProgramFormat.HARDWARE.value,
        "vertex": vertex, "fragment": fragment, "varying_map": mapping,
        "rasterization": False, "interpolation": False, "fpga_execution": False,
        "rtl_timing_validated": False, "approximate_fp16": True,
    }


def phong_payload(compiled: dict, *, position=(0.25, -0.125, 0.0), normal=(0.0, 0.0, 1.0),
                  light=(1.0, 0.5, 3.0), view=(0.0, 0.0, 3.0), color=(0.7, 0.4, 0.2)) -> dict:
    values = {}
    normal_locations = {entry["location"] for entry in compiled["layouts"]["vs"]["inputs"]} - {15}
    if len(normal_locations) != 1:
        raise ValueError("Phong demonstration requires one reflected position and one normal attribute")
    normal_location = normal_locations.pop()
    for entry in compiled["layouts"]["vs"]["inputs"]:
        source = (*position, 1.0) if entry["location"] == 15 else normal if entry["location"] == normal_location else None
        if source is None or entry["component"] >= len(source):
            raise ValueError("Phong demonstration expects fresh position/normal vertex semantics")
        values[entry["word_offset"]] = float(source[entry["component"]])
    vertex_uniforms = {entry["word_offset"]: float(entry["word_offset"] % 16 in (0, 5, 10, 15))
                       for entry in compiled["layouts"]["vs"]["uniforms"]}
    fragment_uniforms = {}
    for entry in compiled["layouts"]["fs"]["uniforms"]:
        offset = entry["word_offset"]
        block, component = divmod(offset, 4)
        if block not in (0, 1, 2) or component >= 3:
            raise ValueError("Phong demonstration per-stage uniform layout changed; supply explicit payload")
        fragment_uniforms[offset] = float((light, view, color)[block][component])
    return {"vertex_inputs": values, "vertex_uniforms": vertex_uniforms, "fragment_uniforms": fragment_uniforms}


def reference_phong(position, normal, light, view, color) -> list[float]:
    def normalize(values):
        length = math.sqrt(sum(value * value for value in values))
        if length == 0:
            raise ValueError("Phong reference requires nonzero normalization vectors")
        return [value / length for value in values]

    unit_normal = normalize(normal)
    light_direction = normalize([source - origin for source, origin in zip(light, position)])
    view_direction = normalize([source - origin for source, origin in zip(view, position)])
    diffuse = max(sum(left * right for left, right in zip(unit_normal, light_direction)), 0.0)
    projection = sum(left * right for left, right in zip(unit_normal, light_direction))
    reflected = [-direction + 2 * projection * component
                 for direction, component in zip(light_direction, unit_normal)]
    specular = max(sum(left * right for left, right in zip(view_direction, reflected)), 0.0) ** 32
    return [(0.1 + diffuse) * component + 0.5 * specular for component in color] + [1.0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shader", nargs="?", type=Path, default=ROOT / "shaders/07_lighting_phong")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", type=Path, default=ROOT / "mesa/build")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--allow-approximate-fp16", action="store_true")
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--max-steps", type=int, default=1000000)
    args = parser.parse_args(argv)
    try:
        if not args.compile_only and not args.allow_approximate_fp16:
            raise ValueError("vector reference execution requires --allow-approximate-fp16; use --compile-only otherwise")
        compiled = compile_vector_glsl(args.shader, args.output, args.build)
        if args.compile_only:
            print(json.dumps(compiled["manifest"], indent=2))
            return 0
        if args.payload:
            payload = json.loads(args.payload.read_text())
            payload = {category: {int(offset): value for offset, value in payload[category].items()}
                       for category in ("vertex_inputs", "vertex_uniforms", "fragment_uniforms")}
        elif args.shader.resolve() == (ROOT / "shaders/07_lighting_phong").resolve():
            payload = phong_payload(compiled)
        else:
            raise ValueError("provide an explicit --payload for shaders other than the original Phong demonstration")
        result = execute_vector_pair(compiled, **payload, allow_approximate_fp16=True, max_steps=args.max_steps)
        if not args.payload:
            expected = reference_phong((0.25, -0.125, 0), (0, 0, 1), (1, 0.5, 3), (0, 0, 3), (0.7, 0.4, 0.2))
            actual = [result["fragment"]["outputs"][component] for component in range(4)]
            error = max(abs(observed - reference) for observed, reference in zip(actual, expected))
            result["independent_reference"] = {"rgba": expected, "max_absolute_error": error, "tolerance": 0.05}
            if error > 0.05:
                raise ValueError(f"Phong output differs from independent reference by {error}")
        result["manifest"] = compiled["manifest"]
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "failure.json").write_text(json.dumps({"passed": False, "error": str(error),
                                                              "fpga_execution": False}, indent=2) + "\n")
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
