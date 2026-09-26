"""Week 5-6 NIR -> MGC ISA -> assembler -> emulator numerical checks.

Build the standalone Mesa helper first:
  PATH="$PWD/.venv/bin:$PATH" .venv/bin/ninja -C mesa/build \
    src/gallium/drivers/mgc/mgc_test_trig
The helper uses Mesa's own NIR builder and atan expansion. It isolates the
backend because shader uniform/attribute input is not implemented yet.
"""

import math
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from mgcasm import ISA, assemble, disassemble
from mgcasm.fp16 import fp16_bits_to_float
from mgcsim import CPU

BIN = pathlib.Path(os.environ.get(
    "MGC_TRIG_TEST_BIN",
    ROOT / "mesa/build/src/gallium/drivers/mgc/mgc_test_trig",
))
ISA_SPEC = ISA()
OPS = ("sin", "cos", "atan")


@pytest.fixture(scope="module", autouse=True)
def require_mesa_helper():
    if not BIN.is_file():
        if "MGC_TRIG_TEST_BIN" in os.environ:
            pytest.fail(f"MGC_TRIG_TEST_BIN does not exist: {BIN}")
        pytest.skip("build Mesa's mgc_test_trig helper to run Week 5-6 tests")


def compile_asm(op, x):
    result = subprocess.run(
        [str(BIN), op, repr(float(x))], text=True, capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def run_asm(src):
    words = assemble(src, ISA_SPEC)
    # Every emitted instruction, including conditional copies and waits,
    # must be representable by the assembler's 32-bit encoding.
    assert assemble("\n".join(disassemble(words, ISA_SPEC)), ISA_SPEC) == words
    cpu = CPU(isa=ISA_SPEC, max_cycles=30000)
    cpu.run(words)  # raises LatencyViolation if backend omitted a required wait
    assert cpu.halted
    return [fp16_bits_to_float(cpu.mem.load(0x100000 + i) & 0xFFFF)
            for i in range(3)], cpu


def test_emitted_instruction_shapes():
    sin_asm = compile_asm("sin", 0.5)
    cos_asm = compile_asm("cos", 0.5)
    atan_asm = compile_asm("atan", 0.5)
    assert "floor16" in sin_asm and "floor16" in cos_asm
    assert "fcomp_ge" in atan_asm and "fdiv16" in atan_asm
    assert "copy.cr" in atan_asm and "wait_fcomp" in atan_asm
    assert "and " in atan_asm and "add " in atan_asm
    assert all("stop_core" in asm for asm in (sin_asm, cos_asm, atan_asm))


@pytest.mark.parametrize("x", np.linspace(-2 * math.pi, 2 * math.pi, 201))
def test_accuracy_and_roundtrip_on_one_period(x):
    values, _ = run_asm(compile_asm("all", x))
    q = float(np.float16(x))  # GPU arithmetic and input are FP16
    expected = (math.sin(q), math.cos(q), math.atan(q))
    assert abs(values[0] - expected[0]) < 0.01
    assert abs(values[1] - expected[1]) < 0.01
    assert abs(values[2] - expected[2]) < 0.003


@pytest.mark.parametrize("x", np.linspace(-8 * math.pi, 8 * math.pi, 65))
def test_accuracy_across_four_periods(x):
    values, _ = run_asm(compile_asm("all", x))
    q = float(np.float16(x))
    assert abs(values[0] - math.sin(q)) < 0.025
    assert abs(values[1] - math.cos(q)) < 0.025
    assert abs(values[2] - math.atan(q)) < 0.003


@pytest.mark.parametrize("x", [-4.0, -1.0, -0.5, 0.0, 0.5, 1.0, 4.0])
def test_combined_shader_matches_individual_functions(x):
    together, _ = run_asm(compile_asm("all", x))
    for i, op in enumerate(OPS):
        alone, _ = run_asm(compile_asm(op, x))
        assert together[i] == alone[i]


def test_atan_preserves_negative_zero_sign():
    _, cpu = run_asm(compile_asm("atan", -0.0))
    assert cpu.mem.load(0x100002) & 0xFFFF == 0x8000


@pytest.mark.parametrize("x", [float("inf"), -float("inf"), float("nan")])
def test_nonfinite_inputs(x):
    values, _ = run_asm(compile_asm("all", x))
    assert math.isnan(values[0]) and math.isnan(values[1])
    if math.isnan(x):
        assert math.isnan(values[2])
    else:
        assert abs(values[2] - math.copysign(math.pi / 2, x)) < 0.003


def test_unknown_operation_is_rejected():
    result = subprocess.run(
        [str(BIN), "unsupported", "0.5"], text=True, capture_output=True,
        check=False,
    )
    assert result.returncode == 2
