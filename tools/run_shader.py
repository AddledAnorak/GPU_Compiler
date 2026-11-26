"""Fresh GLSL compilation followed by managed legacy host-driver execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.mgcasm import assemble
from tools.mgcasm.fp16 import float_to_fp16_bits, fp16_bits_to_float
from tools.mgcdriver import Driver, HardwareSimulationBackend, Program, ProgramFormat, SimulationBackend
from tools.mgchw.encoding import assemble_hardware, disassemble_hardware

ROOT = Path(__file__).resolve().parents[1]
ABI_BASES = {"input": 0x10000, "uniform": 0x20000, "vertex": 0x30000, "fragment": 0x100000}
ABI_REGION_WORDS = 0x10000


def mesa_environment(build: Path, dump: Path, target: ProgramFormat = ProgramFormat.LEGACY) -> dict[str, str]:
    target = ProgramFormat(target)
    environment = dict(os.environ)
    for name in ("MGC_HOST_TEXTURE", "MGC_HOST_FEATURES", "MGC_FRAGMENT_STATUS", "MGC_HOST_EXECUTOR_SOCKET"):
        environment.pop(name, None)
    if target == ProgramFormat.C_INTERPRETER:
        for name in ("MGC_HOST_INSTANCING", "MGC_HOST_CAPTURE_DIR", "MGC_HOST_TEXTURE_WRAP_S", "MGC_HOST_TEXTURE_WRAP_T"):
            environment.pop(name, None)
    library_paths = [build / "src/egl", build / "src/mapi/es2api", build / "src/gallium/targets/dri"]
    environment.update({
        "GALLIUM_DRIVER": "mgc", "EGL_PLATFORM": "surfaceless", "LIBGL_ALWAYS_SOFTWARE": "1",
        "MESA_SHADER_CACHE_DISABLE": "1", "MGC_ASM_DUMP": str(dump),
        "MESA_LOADER_DRIVER_OVERRIDE": "swrast", "LIBGL_DRIVERS_PATH": str(build / "src/gallium/targets/dri"),
        "LD_LIBRARY_PATH": os.pathsep.join(map(str, library_paths)) + os.pathsep + environment.get("LD_LIBRARY_PATH", ""),
        "MGC_COMPILE_TARGET": target.value,
    })
    return environment


def compile_glsl(shader_directory: Path, build: Path, output: Path,
                 target: ProgramFormat = ProgramFormat.LEGACY) -> dict[str, str]:
    target = ProgramFormat(target)
    harness = ROOT / "tools/mgc_harness"
    if not harness.is_file():
        raise ValueError("build the fresh Mesa harness with bash tools/build_project.sh")
    with tempfile.TemporaryDirectory(prefix="btp-shader-") as temporary:
        dump = Path(temporary)
        completed = subprocess.run(
            [str(harness), str(shader_directory / "vert.glsl"), str(shader_directory / "frag.glsl")],
            env=mesa_environment(build, dump, target), capture_output=True, text=True, timeout=60,
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "compile.log").write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise ValueError(f"GLSL/MGC compilation failed; see {output / 'compile.log'}")
        assembly = {}
        for stage in ("vs", "fs"):
            path = dump / f"{stage}.asm"
            if not path.is_file():
                raise ValueError(f"MGC produced no {stage} assembly; see {output / 'compile.log'}")
            assembly[stage] = path.read_text()
            profile = "legacy-simulator" if target == ProgramFormat.LEGACY else target.value
            version = 1 if target == ProgramFormat.LEGACY else 5 if target == ProgramFormat.C_INTERPRETER else 2
            if not assembly[stage].startswith(f"; mgc {profile} backend, ABI={version},"):
                raise ValueError("compiler artifact does not match the explicitly selected target and ABI")
            (output / f"{stage}.asm").write_text(assembly[stage])
            layout_path = dump / f"{stage}.layout.json"
            if layout_path.is_file():
                (output / layout_path.name).write_text(layout_path.read_text())
        return assembly


def execute_assembly(assembly: str, values: dict[int, float] | None = None,
                     raw_values: dict[int, int] | None = None, *,
                     target: ProgramFormat = ProgramFormat.LEGACY) -> dict:
    if assembly.startswith("; mgc fp16-hardware-vector-experimental-v1"):
        raise ValueError("experimental vector compiler output requires tools.run_vector_shader; never a scalar executor")
    target = ProgramFormat(target)
    if target == ProgramFormat.C_INTERPRETER or assembly.startswith("; mgc c-interpreter-v1"):
        raise ValueError("current C compiler output requires tools.run_c_shader; never a legacy/recovered executor")
    hardware = target == ProgramFormat.HARDWARE
    if hardware and not assembly.startswith("; mgc fp16-hardware-v1 backend, ABI=2,"):
        raise ValueError("hardware shader execution requires an explicit compiler target/ABI artifact")
    if not hardware and assembly.startswith("; mgc fp16-hardware-v1"):
        raise ValueError("hardware compiler output must never execute on the legacy backend")
    values = values or {}
    raw_values = raw_values or {}
    if values.keys() & raw_values.keys():
        raise ValueError("floating-point and raw payloads must not overlap")
    payloads = {}
    for address, value in raw_values.items():
        if isinstance(value, bool) or not isinstance(value, int) or not -(1 << 31) <= value < (1 << 32):
            raise ValueError("raw ABI values must be signed int32 or unsigned uint32 integers")
        payloads[address] = value & 0xFFFFFFFF
    for address, value in values.items():
        payloads[address] = float_to_fp16_bits(value)
    words = assemble_hardware(assembly) if hardware else assemble(assembly)
    if hardware and assemble_hardware("\n".join(disassemble_hardware(words))) != words:
        raise ValueError("hardware encoding failed complete-word decoder roundtrip")
    program = Program.from_words(words, format=target)
    bases = {name: base * (4 if hardware else 1) for name, base in ABI_BASES.items()}
    stride = 4 if hardware else 1
    backend = HardwareSimulationBackend(memory_size=5 << 20) if hardware else SimulationBackend(memory_size=5 << 20)
    with Driver(backend) as driver:
        buffers = {name: driver.allocate(ABI_REGION_WORDS * 4, alignment=4, address=base * 4)
                   for name, base in ABI_BASES.items()}
        for address, value in payloads.items():
            if isinstance(address, bool) or not isinstance(address, int):
                raise ValueError("ABI addresses must be integer indices in the selected target's address units")
            if hardware and address % 4:
                raise ValueError("hardware host ABI payload addresses must be four-byte aligned")
            matches = [(name, base) for name, base in bases.items()
                       if base <= address < base + ABI_REGION_WORDS * stride]
            if len(matches) != 1:
                raise ValueError(f"address 0x{address:x} outside documented host-test ABI")
            name, base = matches[0]
            buffers[name].upload(struct.pack("<I", value), offset=(address - base) * (1 if hardware else 4))
        fence = driver.submit(program, buffers=tuple(buffers.values()), max_steps=1000000,
                              execution_timeout=10.0)
        result = fence.wait(15.0)
        outputs = {}
        for stage in ("vertex", "fragment"):
            contents = buffers[stage].download(offset=0, size=64)
            words = struct.unpack("<16I", contents)
            outputs[stage] = {"raw_words": list(words),
                              "fp16": [fp16_bits_to_float(value) for value in words]}
        return {
            "pipeline": f"GLSL -> Mesa NIR -> MGC {target.value} assembly -> managed host driver -> functional CPU",
            "fpga_execution": False, "rasterization": False, "rtl_timing_validated": False,
            "shader_abi_version": 2 if hardware else 1,
            "address_units": "provisional host buffer byte addresses" if hardware else "compiler legacy word indices; driver buffers use bytes",
            "abi_bases": bases,
            "program_format": program.format.value, "program_words": len(program.words),
            "program_sha256": hashlib.sha256(program.to_bytes()).hexdigest(),
            "fence": fence.status.value, "host_steps": result.steps, "outputs": outputs,
            "input_payload_words": {"fp16": len(values), "raw_i32_u32": len(raw_values)},
        }


def execute_shader_pair(assembly: dict[str, str], *, varying_map: dict[int, int],
                        vertex_values: dict[int, float] | None = None,
                        vertex_raw_values: dict[int, int] | None = None,
                        fragment_values: dict[int, float] | None = None,
                        fragment_raw_values: dict[int, int] | None = None,
                        target: ProgramFormat = ProgramFormat.LEGACY) -> dict:
    target = ProgramFormat(target)
    stride = 4 if target == ProgramFormat.HARDWARE else 1
    input_base = ABI_BASES["input"] * stride
    if any(isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset < 16
           for offset in varying_map):
        raise ValueError("vertex varying offsets must index the 16 captured output words")
    if any(isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset < ABI_REGION_WORDS
           for offset in varying_map.values()):
        raise ValueError("fragment varying offsets must stay inside the host input region")
    if len(set(varying_map.values())) != len(varying_map):
        raise ValueError("varying destinations must be unique")
    fragment_values = fragment_values or {}
    fragment_raw_values = dict(fragment_raw_values or {})
    destinations = {input_base + offset * stride for offset in varying_map.values()}
    if destinations & (fragment_values.keys() | fragment_raw_values.keys()):
        raise ValueError("bridged varyings must not overlap fragment payloads")
    vertex = execute_assembly(assembly["vs"], vertex_values, vertex_raw_values, target=target)
    words = vertex["outputs"]["vertex"]["raw_words"]
    fragment_raw_values.update({input_base + destination * stride: words[source]
                                for source, destination in varying_map.items()})
    fragment = execute_assembly(assembly["fs"], fragment_values, fragment_raw_values, target=target)
    return {
        "pipeline": f"linked GLSL -> Mesa NIR -> {target.value} VS CPU invocation -> explicit raw varying copy -> FS CPU invocation",
        "fpga_execution": False, "rasterization": False, "interpolation": False,
        "invocation_contract": "one vertex and one fragment, not a triangle or framebuffer renderer",
        "shader_abi_version": vertex["shader_abi_version"], "program_format": target.value,
        "varying_map": [{"vertex_output_word_offset": source, "fragment_input_word_offset": target}
                        for source, target in varying_map.items()],
        "host_steps": vertex["host_steps"] + fragment["host_steps"],
        "fence": fragment["fence"], "stages": {"vs": vertex, "fs": fragment},
        "outputs": {"vertex": vertex["outputs"]["vertex"], "fragment": fragment["outputs"]["fragment"]},
    }


def load_values(path: Path | None, *, raw: bool = False) -> dict:
    if path is None:
        return {}
    source = json.loads(path.read_text())
    if not isinstance(source, dict):
        raise ValueError("values JSON must be an address/value object")
    return {int(address, 0): (int(value, 0) if isinstance(value, str) else value) if raw else float(value)
            for address, value in source.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shader_directory", type=Path)
    parser.add_argument("--build", type=Path, default=Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build")))
    parser.add_argument("--stage", choices=("vs", "fs"), default="fs")
    parser.add_argument("--target", choices=[profile.value for profile in ProgramFormat],
                        default=ProgramFormat.LEGACY.value)
    parser.add_argument("--values", type=Path, help="JSON object: legacy ABI word address -> floating-point value")
    parser.add_argument("--raw-values", type=Path,
                        help="JSON object: legacy ABI word address -> int32/uint32 integer or hexadecimal string")
    parser.add_argument("--varying-map", type=Path,
                        help="enable VS->FS invocation bridge; JSON vertex output word offset -> fragment input word offset")
    parser.add_argument("--auto-link", action="store_true", help="derive VS->FS linkage from compiler reflection")
    parser.add_argument("--vertex-values", type=Path, help="vertex-stage FP16 address/value JSON for explicit bridge")
    parser.add_argument("--vertex-raw-values", type=Path, help="vertex-stage integer address/value JSON for explicit bridge")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/shader")
    args = parser.parse_args(argv)
    try:
        if (args.vertex_values or args.vertex_raw_values) and not (args.varying_map or args.auto_link):
            raise ValueError("vertex payload options require --varying-map or --auto-link")
        if args.varying_map and args.auto_link:
            raise ValueError("explicit and reflected linkage are mutually exclusive")
        target = ProgramFormat(args.target)
        assembly = compile_glsl(args.shader_directory.resolve(), args.build.resolve(), args.output, target)
        values = load_values(args.values)
        raw_values = load_values(args.raw_values, raw=True)
        if args.varying_map or args.auto_link:
            if args.auto_link:
                from tools.shader_layout import ShaderLayout, link_varyings
                vertex_layout = ShaderLayout(json.loads((args.output / "vs.layout.json").read_text()))
                fragment_layout = ShaderLayout(json.loads((args.output / "fs.layout.json").read_text()))
                linkage = {source: destination for destination, source in link_varyings(vertex_layout, fragment_layout).items()}
                if len(linkage) != len(fragment_layout.entries("inputs")):
                    raise ValueError("single-invocation bridge does not support duplicated source varyings")
            else:
                linkage = load_values(args.varying_map, raw=True)
            result = execute_shader_pair(assembly, varying_map=linkage,
                                         vertex_values=load_values(args.vertex_values),
                                         vertex_raw_values=load_values(args.vertex_raw_values, raw=True),
                                         fragment_values=values, fragment_raw_values=raw_values, target=target)
            result["stage"] = "vs->fs"
            result["linkage"] = "compiler-reflected" if args.auto_link else "explicit manual map"
        else:
            result = execute_assembly(assembly[args.stage], values, raw_values, target=target)
            result["stage"] = args.stage
        result["shader_directory"] = str(args.shader_directory)
        (args.output / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "error", "error": type(error).__name__, "message": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
