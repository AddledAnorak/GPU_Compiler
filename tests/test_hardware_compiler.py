import json
import os
import subprocess
import sys

import pytest

from test_backend_runtime import backend_binary
from tools.mgcdriver import Driver, HardwareSimulationBackend, Program, ProgramFormat, ProgramFormatError, SimulationBackend
from tools.mgchw import ByteMemory, HardwareCPU, assemble_hardware, decode_word, disassemble_hardware
from tools.run_shader import ABI_BASES, ROOT, compile_glsl, execute_assembly, execute_shader_pair, mesa_environment

HARDWARE = ProgramFormat.HARDWARE
BASES = {name: address * 4 for name, address in ABI_BASES.items()}


def compile_hardware(binary, mode, option="hardware", *, success=True, count=65):
    result = subprocess.run([str(binary), mode, str(count), option], capture_output=True, text=True, timeout=30)
    assert result.returncode == (0 if success else 1), result.stderr
    if not success:
        assert not result.stdout
        assert "error exposed partial assembly" not in result.stderr
        return result.stderr
    assert "abi=2 " in result.stderr
    assert result.stdout.startswith("; mgc fp16-hardware-v1 backend, ABI=2,")
    words = assemble_hardware(result.stdout)
    assert assemble_hardware("\n".join(disassemble_hardware(words))) == words
    assert all(decode_word(word).mnemonic in {"preload_low", "load_high", "copy", "load", "store", "nop", "fadd16"}
               for word in words)
    assert "stop_core" not in result.stdout and "wait_short" not in result.stdout
    return result.stdout, words


@pytest.fixture(scope="module")
def hardware_glsl(tmp_path_factory):
    if not (ROOT / "mesa/build/build.ninja").is_file() or not (ROOT / "tools/mgc_harness").is_file():
        pytest.skip("configured Mesa and freshly built GLES harness required")
    directory = tmp_path_factory.mktemp("hardware-glsl")
    return {name: compile_glsl(ROOT / "shaders" / name, ROOT / "mesa/build", directory / name, HARDWARE)
            for name in ("01_flat_color", "02_varying")}


def test_flat_nir_is_real_verified_hardware_words(backend_binary):
    assembly, words = compile_hardware(backend_binary, "flat")
    assert len(words) == 16
    assert words[:2] == [0x080F0000, 0x08000020]
    assert decode_word(words[4]).mnemonic == "store"
    result = execute_assembly(assembly, target=HARDWARE)
    assert result["outputs"]["fragment"]["raw_words"][:4] == [0x3C00, 0, 0, 0x3C00]
    assert result["fence"] == "completed"
    assert result["host_steps"] == len(words)


@pytest.mark.parametrize("payload", [0, 0x3C00, 0xB800, 0x0001, 0x8000])
def test_hardwarecpu_direct_byte_input_output(backend_binary, payload):
    _, words = compile_hardware(backend_binary, "copy")
    memory = ByteMemory()
    memory.map_region(BASES["input"], 4)
    memory.map_region(BASES["fragment"], 4)
    memory.write32(BASES["input"], payload)
    cpu = HardwareCPU(memory=memory)
    cpu.run(words)
    assert memory.read32(BASES["fragment"]) == payload
    assert cpu.pc == len(words)


@pytest.mark.parametrize("mode,offset", [("uniform", 5), ("ubo", 4099)])
def test_hardware_direct_uniform_offsets_are_byte_scaled(backend_binary, mode, offset):
    assembly, _ = compile_hardware(backend_binary, mode)
    result = execute_assembly(assembly, {BASES["uniform"] + offset * 4: -0.5}, target=HARDWARE)
    assert result["outputs"]["fragment"]["fp16"][0] == -0.5


def test_hardware_custom_abi_is_explicit_owned_memory(backend_binary):
    _, words = compile_hardware(backend_binary, "copy", "hardware-custom")
    with Driver(HardwareSimulationBackend(memory_size=4 << 20)) as driver:
        source = driver.allocate(4, alignment=4, address=0x1000)
        output = driver.allocate(4, alignment=4, address=0x200000)
        source.upload((0x3800).to_bytes(4, "little"))
        program = Program.from_words(words, format=HARDWARE)
        fence = driver.submit(program, buffers=[source, output], max_steps=100)
        fence.wait(5)
        assert output.download() == (0x3800).to_bytes(4, "little")


@pytest.mark.parametrize("mode", ["io", "phong", "trig", "runtime-loop", "integer-runtime-loop", "branch-store",
                                   "discard", "texture", "unaligned", "compute", "nonfinite", "finite-overflow",
                                   "copy-register-pressure", "integer-output", "empty", "hardware-multiply",
                                   "hardware-subtract", "hardware-add-overflow-constant"])
def test_unsupported_hardware_nir_fails_without_partial_assembly(backend_binary, mode):
    assert compile_hardware(backend_binary, mode, success=False)


@pytest.mark.parametrize("mode,limit", [("input-boundary", 16383), ("uniform-boundary", 65535),
                                      ("ubo-block-boundary", 15), ("ubo-word-boundary", 4095)])
@pytest.mark.parametrize("valid", [True, False])
def test_hardware_io_last_valid_first_invalid_boundary(backend_binary, mode, limit, valid):
    if not valid:
        assert compile_hardware(backend_binary, mode, success=False, count=limit + 1)
        return
    assembly, _ = compile_hardware(backend_binary, mode, count=limit)
    domain = "input" if mode == "input-boundary" else "uniform"
    result = execute_assembly(assembly, {BASES[domain] + 65535 * 4: -0.5}, target=HARDWARE)
    assert result["outputs"]["fragment"]["fp16"][0] == -0.5


@pytest.mark.parametrize("mode,payload,expected", [
    ("hardware-add", {BASES["input"]: -0.25}, 0.25),
    ("hardware-add-registers", {BASES["input"]: -0.5, BASES["input"] + 4: 0.25}, -0.25),
    ("hardware-add-uniform", {BASES["input"]: 0.25, BASES["uniform"]: -0.75}, -0.5),
])
def test_shared_hardware_add_lowering_uses_registers_and_observed_wait(backend_binary, mode, payload, expected):
    assembly, words = compile_hardware(backend_binary, mode)
    additions = [index for index, word in enumerate(words) if decode_word(word).mnemonic == "fadd16"]
    assert additions
    for index in additions:
        assert decode_word(words[index]).source_a_register
        assert words[index + 1:index + 7] == [0] * 6
    result = execute_assembly(assembly, payload, target=HARDWARE)
    assert result["outputs"]["fragment"]["fp16"][0] == expected


def test_hardware_dependent_add_chain_reuses_bounded_allocator(backend_binary):
    assembly, words = compile_hardware(backend_binary, "chain", count=240)
    assert sum(decode_word(word).mnemonic == "fadd16" for word in words) == 240
    result = execute_assembly(assembly, {BASES["input"]: -0.5}, target=HARDWARE)
    assert result["outputs"]["fragment"]["fp16"][0] == 7.0


@pytest.mark.parametrize("option", ["hardware-unaligned", "hardware-overlap", "hardware-overflow",
                                     "hardware-scheduled", "unknown-target"])
def test_hardware_abi_or_scheduler_guesses_are_rejected(backend_binary, option):
    assert compile_hardware(backend_binary, "flat", option, success=False)


def test_fresh_glsl_constant_to_managed_hardware_backend(hardware_glsl):
    result = execute_assembly(hardware_glsl["01_flat_color"]["fs"], target=HARDWARE)
    assert result["program_format"] == HARDWARE.value
    assert result["shader_abi_version"] == 2
    assert result["outputs"]["fragment"]["fp16"][:4] == [1, 0, 0, 1]
    assert not result["fpga_execution"] and not result["rtl_timing_validated"]


def test_fresh_glsl_hardware_vertex_to_fragment_copy(hardware_glsl):
    values = {BASES["input"] + component * 4: value for component, value in enumerate([0.125, 0.25, 0.5, 1])}
    values.update({BASES["input"] + 16 + component * 4: value
                   for component, value in enumerate([0.25, -0.5, 0.75])})
    result = execute_shader_pair(hardware_glsl["02_varying"], varying_map={4: 0, 5: 1, 6: 2},
                                 vertex_values=values, target=HARDWARE)
    assert result["outputs"]["vertex"]["fp16"][:4] == [0.125, 0.25, 0.5, 1]
    assert result["outputs"]["fragment"]["fp16"][:4] == [0.25, -0.5, 0.75, 1]
    assert not result["interpolation"] and not result["rasterization"]


@pytest.mark.parametrize("shader", ["03_uniform", "09_runtime_loop", "08_texture"])
def test_fresh_unsupported_glsl_hardware_target_rejects(tmp_path, shader):
    with pytest.raises(ValueError, match="compilation failed"):
        compile_glsl(ROOT / "shaders" / shader, ROOT / "mesa/build", tmp_path, HARDWARE)


def test_hardware_program_never_routes_to_legacy(hardware_glsl):
    assembly = hardware_glsl["01_flat_color"]["fs"]
    with pytest.raises(ValueError, match="never execute"):
        execute_assembly(assembly)
    program = Program.from_words(assemble_hardware(assembly), format=HARDWARE)
    with Driver(SimulationBackend(memory_size=5 << 20)) as driver:
        with pytest.raises(ProgramFormatError, match="legacy"):
            driver.submit(program)


def test_hardware_payload_requires_byte_alignment(hardware_glsl):
    with pytest.raises(ValueError, match="aligned"):
        execute_assembly(hardware_glsl["01_flat_color"]["vs"], {BASES["input"] + 1: 0.5}, target=HARDWARE)


def test_shader_cli_explicit_hardware_format_without_tools_path(tmp_path):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-m", "tools.run_shader", "shaders/01_flat_color",
                             "--target", HARDWARE.value, "--output", str(tmp_path)],
                            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["program_format"] == HARDWARE.value
    assert report["outputs"]["fragment"]["fp16"][:4] == [1, 0, 0, 1]


def test_ambient_target_cannot_change_explicit_default(hardware_glsl, monkeypatch):
    monkeypatch.setenv("MGC_COMPILE_TARGET", HARDWARE.value)
    assert mesa_environment(ROOT / "mesa/build", ROOT)["MGC_COMPILE_TARGET"] == ProgramFormat.LEGACY.value


def test_unknown_gallium_compiler_target_has_no_legacy_fallback(tmp_path):
    environment = mesa_environment(ROOT / "mesa/build", tmp_path)
    environment["MGC_COMPILE_TARGET"] = "unverified-hardware-v2"
    result = subprocess.run([str(ROOT / "tools/mgc_harness"), str(ROOT / "shaders/01_flat_color/vert.glsl"),
                             str(ROOT / "shaders/01_flat_color/frag.glsl")], env=environment,
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "unknown MGC_COMPILE_TARGET" in result.stderr
    assert not list(tmp_path.glob("*.asm"))
