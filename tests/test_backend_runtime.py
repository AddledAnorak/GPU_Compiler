"""Fresh Mesa NIR -> legacy assembly -> assembler -> strict simulator checks.

The helper is rebuilt from the current compiler and test C source, never a
checked-in assembly fixture. These tests do not validate FPGA machine code.
"""

import math
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = pathlib.Path(os.environ.get("MGC_BUILD_DIR", ROOT / "mesa/build"))
sys.path.insert(0, str(ROOT / "tools"))

from mgcasm import ISA, assemble, disassemble
from mgcasm.fp16 import float_to_fp16_bits, fp16_bits_to_float
from mgcsim import CPU

INPUT = 0x10000
UNIFORM = 0x20000
VERTEX = 0x30000
FRAMEBUFFER = 0x100000
ISA_SPEC = ISA()


@pytest.fixture(scope="module")
def backend_binary(tmp_path_factory):
    if not (BUILD / "build.ninja").is_file():
        pytest.skip("a configured Mesa build is required for fresh compiler tests")
    ninja = shutil.which("ninja") or str(ROOT / ".venv/bin/ninja")
    environment = dict(os.environ)
    environment["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + environment.get("PATH", "")
    target = "src/gallium/drivers/mgc/mgc_test_trig"
    compiler_object = "src/gallium/drivers/mgc/libmgc.a.p/mgc_compiler.c.o"
    subprocess.run([ninja, "-C", str(BUILD), compiler_object], check=True, env=environment,
                   capture_output=True, text=True)
    directory = tmp_path_factory.mktemp("fresh_backend")
    binary = directory / "mgc_test_backend"
    original_object = target + ".p/mgc_test_trig.c.o"
    new_object = directory / "mgc_test_backend.c.o"
    source = ROOT / "mesa/src/gallium/drivers/mgc/mgc_test_backend.c"
    compile_text = subprocess.check_output(
        [ninja, "-C", str(BUILD), "-t", "commands", original_object], text=True,
        env=environment,
    ).splitlines()[-1]
    compile_command = [
        str(new_object) + token[len(original_object):]
        if token.startswith(original_object) else
        str(source) if token.endswith("/mgc_test_trig.c") else token
        for token in shlex.split(compile_text)
    ]
    subprocess.run(compile_command, cwd=BUILD, check=True, env=environment,
                   capture_output=True, text=True)
    link_text = subprocess.check_output(
        [ninja, "-C", str(BUILD), "-t", "commands", target], text=True,
        env=environment,
    ).splitlines()[-1]
    link_command = [str(binary) if token == target else
                    str(new_object) if token == original_object else token
                    for token in shlex.split(link_text)]
    link_command.insert(link_command.index(str(new_object)) + 1, str(BUILD / compiler_object))
    subprocess.run(link_command, cwd=BUILD, check=True, env=environment,
                   capture_output=True, text=True)
    return binary


def compile_shader(binary, mode, count=4, option=None, success=True):
    command = [str(binary), mode, str(count)]
    if option:
        command.append(option)
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if success:
        assert result.returncode == 0, result.stderr
        assert "legacy-simulator" in result.stdout
        statistics = dict((name, int(value)) for name, value in
                          re.findall(r"(\w+)=(\d+)", result.stderr))
        assert statistics["abi"] == 1
        return result.stdout, statistics
    assert result.returncode == 1, result.stderr
    assert not result.stdout
    return result.stderr


def execute(assembly, memory=None, raw_words=None):
    words = assemble(assembly, ISA_SPEC)
    assert assemble("\n".join(disassemble(words, ISA_SPEC)), ISA_SPEC) == words
    cpu = CPU(isa=ISA_SPEC, max_cycles=100000)
    for address, value in (memory or {}).items():
        cpu.mem.store(address, float_to_fp16_bits(value))
    for address, value in (raw_words or {}).items():
        cpu.mem.store(address, value)
    cpu.run(words)
    assert cpu.halted
    return cpu


def scalar(cpu, address=FRAMEBUFFER):
    return fp16_bits_to_float(cpu.mem.load(address) & 0xFFFF)


@pytest.mark.parametrize("attribute,uniform", [(2, 3), (-1.25, 0.75), (0, -4), (10, 0.125)])
def test_io_real_values_and_nonzero_offsets(backend_binary, attribute, uniform):
    assembly, _ = compile_shader(backend_binary, "io")
    cpu = execute(assembly, {INPUT + 14: attribute, UNIFORM + 2 * 4096 + 11: uniform})
    expected = np.float16(np.float16(np.float16(attribute) * np.float16(uniform)) + np.float16(0.5))
    expected = np.float16(expected / np.float16(2))
    assert scalar(cpu, FRAMEBUFFER + 9) == float(expected)
    assert cpu.mem.load(FRAMEBUFFER) == 0


def test_custom_abi_bases(backend_binary):
    assembly, _ = compile_shader(backend_binary, "io", option="custom")
    cpu = execute(assembly, {0x40000 + 14: 2, 0x50000 + 2 * 4096 + 11: 3})
    assert scalar(cpu, 0x70000 + 9) == 3.25
    assert cpu.mem.load(FRAMEBUFFER + 9) == 0


@pytest.mark.parametrize("mode,address", [("ubo", UNIFORM + 4096 + 3), ("uniform", UNIFORM + 5)])
def test_uniform_address_units(backend_binary, mode, address):
    assembly, _ = compile_shader(backend_binary, mode)
    cpu = execute(assembly, {address: -1.5})
    assert scalar(cpu) == -1.5


def test_vertex_outputs_are_not_silently_dropped(backend_binary):
    assembly, _ = compile_shader(backend_binary, "vertex")
    cpu = execute(assembly, {INPUT + 9: 0.625})
    assert scalar(cpu, VERTEX + 14) == 0.625
    assert not cpu.mem.dump_framebuffer()


@pytest.mark.parametrize("value", [-3, -0.125, 0, 0.5, 4])
def test_pure_if_phi_is_flattened(backend_binary, value):
    assembly, _ = compile_shader(backend_binary, "branch")
    cpu = execute(assembly, {INPUT: value})
    expected = -value if value < 0 else value + 1
    assert scalar(cpu) == float(np.float16(expected))
    assert "copy.cr" in assembly
    assert "goto" not in assembly


@pytest.mark.parametrize("count", [1, 4, 8, 16, 32])
def test_constant_bound_loops_are_unrolled(backend_binary, count):
    assembly, _ = compile_shader(backend_binary, "loop", count)
    cpu = execute(assembly, {INPUT: 0.125})
    assert scalar(cpu) == count * 0.125
    assert "goto" not in assembly


@pytest.mark.parametrize("count", [0, 1, 4, 20, 65])
@pytest.mark.parametrize("mode", ["runtime-loop", "integer-runtime-loop"])
def test_live_counted_loops_and_phi_edges(backend_binary, mode, count):
    assembly, _ = compile_shader(backend_binary, mode)
    memory = {INPUT: 0.125, INPUT + 1: count}
    cpu = execute(assembly, memory, {UNIFORM: count})
    assert scalar(cpu) == count * 0.125
    assert "goto.cr" in assembly
    assert "mgc_edge" in assembly


def test_large_static_loop_uses_cfg_instead_of_false_success(backend_binary):
    assembly, _ = compile_shader(backend_binary, "loop", 65)
    assert scalar(execute(assembly, {INPUT: 0.125})) == 8.125


@pytest.mark.parametrize("count", [0, 1, 2, 7, 12])
def test_loop_phi_copies_are_parallel_not_sequential(backend_binary, count):
    assembly, _ = compile_shader(backend_binary, "swap-loop")
    cpu = execute(assembly, {INPUT: 0.25, INPUT + 1: 0.75, INPUT + 2: count})
    assert scalar(cpu) == (0.75 if count % 2 else 0.25)
    assert scalar(cpu, FRAMEBUFFER + 1) == (0.25 if count % 2 else 0.75)


@pytest.mark.parametrize("value", [-2, 0, 0.75])
def test_live_if_else_output_side_effects(backend_binary, value):
    assembly, _ = compile_shader(backend_binary, "branch-store")
    assert scalar(execute(assembly, {INPUT: value})) == (-value if value < 0 else value + 1)
    assert "goto.cr" in assembly


def test_host_preinterpolated_values(backend_binary):
    assembly, _ = compile_shader(backend_binary, "interpolated")
    assert scalar(execute(assembly, {INPUT + 6: 0.625})) == 0.625


@pytest.mark.parametrize("value,expected", [(0.25, 2), (0.75, 7)])
def test_integer_select_preserves_raw_bits(backend_binary, value, expected):
    assembly, _ = compile_shader(backend_binary, "integer-select")
    assert scalar(execute(assembly, {INPUT: value})) == expected


@pytest.mark.parametrize("value,killed", [(0.25, False), (0.5, True), (0.75, True)])
def test_early_discard_preserves_framebuffer_when_killed(backend_binary, value, killed):
    assembly, _ = compile_shader(backend_binary, "discard")
    cpu = execute(assembly, {INPUT: value, FRAMEBUFFER: 0.875})
    assert scalar(cpu) == (0.875 if killed else value)


def test_predicates_recycle_after_last_use(backend_binary):
    assembly, statistics = compile_shader(backend_binary, "predicates", 40)
    cpu = execute(assembly, {INPUT: 0.75})
    expected = np.float16(0)
    for threshold in np.arange(40) * 0.125:
        expected = np.float16(expected + (0.75 if 0.75 < threshold else -0.75))
    assert scalar(cpu) == float(expected)
    assert statistics["predicates"] <= 2


def test_long_chain_reuses_scalar_registers(backend_binary):
    assembly, statistics = compile_shader(backend_binary, "chain", 240)
    cpu = execute(assembly, {INPUT: 0.125})
    assert scalar(cpu) == 7.625
    assert statistics["registers"] < 8
    assert statistics["instructions"] > 200


def test_dependency_waits_preserve_results_and_reduce_wait_count(backend_binary):
    scheduled, scheduled_stats = compile_shader(backend_binary, "schedule")
    immediate, immediate_stats = compile_shader(backend_binary, "schedule", option="unscheduled")
    memory = {INPUT: 2, INPUT + 1: 3, INPUT + 2: 0.5, INPUT + 3: -1.25}
    scheduled_cpu = execute(scheduled, memory)
    immediate_cpu = execute(immediate, memory)
    assert scalar(scheduled_cpu) == scalar(immediate_cpu) == 5.375
    assert scheduled_stats["waits"] < immediate_stats["waits"]
    assert scheduled.count("wait_load") < immediate.count("wait_load")
    assert scheduled_cpu.cycles < immediate_cpu.cycles


@pytest.mark.parametrize("value", np.geomspace(2 ** -24, 65504, 75))
def test_inverse_sqrt_across_fp16_finite_range(backend_binary, value):
    assembly, _ = compile_shader(backend_binary, "rsqrt")
    cpu = execute(assembly, {INPUT: value})
    quantized = float(np.float16(value))
    assert math.isclose(scalar(cpu), 1 / math.sqrt(quantized), rel_tol=0.004)


@pytest.mark.parametrize("value", [0, 2 ** -24, 0.125, 0.5, 1, 2, 9, 65504, float("inf")])
def test_sqrt(backend_binary, value):
    assembly, _ = compile_shader(backend_binary, "sqrt")
    cpu = execute(assembly, {INPUT: value})
    assert math.isclose(scalar(cpu), math.sqrt(float(np.float16(value))), rel_tol=0.004)


@pytest.mark.parametrize("value", [-1, float("nan"), -float("inf")])
def test_inverse_sqrt_invalid_inputs_produce_nan(backend_binary, value):
    assembly, _ = compile_shader(backend_binary, "rsqrt")
    assert math.isnan(scalar(execute(assembly, {INPUT: value})))


@pytest.mark.parametrize("value", [0, 0.125, 0.5, 0.9, 1, 1.125])
def test_constant_integer_power(backend_binary, value):
    assembly, _ = compile_shader(backend_binary, "pow32")
    assert "flog" not in assembly and "fexp" not in assembly
    cpu = execute(assembly, {INPUT: value})
    expected = np.float16(value)
    for _ in range(5):
        expected = np.float16(expected * expected)
    assert scalar(cpu) == float(expected)


@pytest.mark.parametrize("exponent", [0, 1, 2, 4, 16, 64])
def test_supported_power_exponent_bounds(backend_binary, exponent):
    assembly, _ = compile_shader(backend_binary, "power", exponent)
    assert scalar(execute(assembly, {INPUT: 0.5})) == float(np.float16(0.5 ** exponent))


@pytest.mark.parametrize("value", [-2, -0.5, 0, 0.5, 2])
def test_transcendentals_with_runtime_input(backend_binary, value):
    assembly, _ = compile_shader(backend_binary, "trig")
    cpu = execute(assembly, {INPUT: value})
    for component, function in enumerate((math.sin, math.cos, math.atan)):
        assert abs(scalar(cpu, FRAMEBUFFER + component) - function(value)) < 0.01


@pytest.mark.parametrize("normal,light,view,color", [
    ([0, 0, 1], [0, 0, 1], [0, 0, 1], [0.2, 0.4, 0.6]),
    ([1, 2, 3], [-1, 0.5, 2], [0, 1, 2], [0.8, 0.2, 0.1]),
    ([0, 0, 1], [1, 0, -1], [0, 0, 1], [0.3, 0.5, 0.7]),
])
def test_phong_normalize_dot_reflect_specular(backend_binary, normal, light, view, color):
    assembly, statistics = compile_shader(backend_binary, "phong")
    memory = {}
    quantized = []
    for slot, vector in enumerate((normal, light, view)):
        vector = np.asarray(vector, dtype=np.float16).astype(float)
        quantized.append(vector / np.linalg.norm(vector))
        memory.update((INPUT + slot * 4 + component, value)
                      for component, value in enumerate(vector))
    memory.update((UNIFORM + component, value) for component, value in enumerate(color))
    cpu = execute(assembly, memory)
    normal, light, view = quantized
    dot = float(np.dot(normal, light))
    reflection = 2 * dot * normal - light
    specular = max(float(np.dot(view, reflection)), 0) ** 32
    expected = (0.1 + max(dot, 0)) * np.asarray(color, dtype=np.float16).astype(float) + 0.5 * specular
    for component in range(3):
        assert abs(scalar(cpu, FRAMEBUFFER + component) - expected[component]) < 0.025
    assert scalar(cpu, FRAMEBUFFER + 3) == 1
    assert statistics["registers"] <= 32
    assert statistics["predicates"] <= 8


@pytest.mark.parametrize("mode,count,message", [
    ("predicate-pressure", 9, "predicate register pressure"),
    ("unaligned", 4, "aligned to four bytes"),
    ("integer-input", 4, "integer and boolean"),
    ("overflow", 4, "address exceeds"),
    ("fractional-power", 4, "unsupported NIR ALU"),
    ("indirect", 4, "indirect input"),
    ("scalar-pressure", 4, "scalar register pressure"),
    ("late-discard", 4, "discard is supported only before"),
    ("texture", 4, "texture sampling is not supported"),
    ("double", 4, "floating operands require"),
    ("compute", 4, "only fragment and vertex"),
])
def test_unsupported_forms_fail_without_partial_assembly(backend_binary, mode, count, message):
    error = compile_shader(backend_binary, mode, count, success=False)
    assert message in error


@pytest.mark.parametrize("option,message", [("overlap", "regions overlap"),
                                            ("overflow", "32-bit address space")])
def test_invalid_abi_options_reject_without_assembly(backend_binary, option, message):
    assert message in compile_shader(backend_binary, "io", option=option, success=False)


@pytest.mark.parametrize("value", [-2.5, -0.25, 0, 0.25, 2.5])
def test_floor_ceil_trunc_fraction(backend_binary, value):
    assembly, _ = compile_shader(backend_binary, "rounding")
    cpu = execute(assembly, {INPUT: value})
    expected = (math.floor(value), math.ceil(value), math.trunc(value), value - math.floor(value))
    for component, reference in enumerate(expected):
        assert scalar(cpu, FRAMEBUFFER + component) == reference
