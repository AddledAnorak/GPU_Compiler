from pathlib import Path

import pytest

from tools.mgcc import CEncodingError, assemble_c, disassemble_c, fp16_bits
from tools.mgcc.backend import CInterpreterBackend
from tools.mgcc.encoding import OPERATIONS
from tools.mgcdriver import (Driver, ExecutionTimeoutError, HardwareSimulationBackend, Program,
                             ProgramFormat, ProgramFormatError, ResourceError, SimulationBackend,
                             SubmissionError)


@pytest.fixture(scope="module")
def native_backend():
    with CInterpreterBackend(quantum=16) as backend:
        yield backend


def program(source):
    return Program.from_words(assemble_c(source), format=ProgramFormat.C_INTERPRETER)


def execute(backend, source, *, size=16, max_steps=1000):
    with Driver(backend) as driver:
        output = driver.allocate(size, address=0x40000)
        fence = driver.submit(program(source), buffers=[output], max_steps=max_steps)
        result = fence.wait(5)
        return result, output.download()


@pytest.mark.parametrize("name", sorted(OPERATIONS))
def test_every_named_native_form_roundtrips(name):
    operands = {"a": "r1", "b": "r2", "d": "r3", "c": "cr4", "p": "-3",
                "va": "v1", "vb": "v2", "vd": "v3"}
    source = name + " " + ", ".join(operands[role] for role in OPERATIONS[name][2])
    words = assemble_c(source)
    assert assemble_c("\n".join(disassemble_c(words))) == words


@pytest.mark.parametrize("source", ["nop", "regbank 1,0,1", "preload_low 0xcafe",
                                    "load_high 0x1234,r1", "jump -3",
                                    "movi_i32 -1,r63", "movi_u16 123,r32", "movi_f16 -0.25,r1"])
def test_special_forms_roundtrip(source):
    words = assemble_c(source)
    assert assemble_c("\n".join(disassemble_c(words))) == words


@pytest.mark.parametrize("source", ["vfadd16 v1,v2,v3", "vget_lane0_i32 v0,r1", "wait_fadd",
                                    "call somewhere", "push r1", "i32tof16 #3,r1", "load #3,r1",
                                    "copy #8,r1", "fadd16 r64,r1,r2", "jump 131072",
                                    "jump -131073", "movi_u16 65536,r1", "movi_i32 4294967296,r1",
                                    "regbank.cr0 0,0,1", "label: nop\nlabel: stop_core", ""])
def test_unsupported_and_invalid_forms_fail_closed(source):
    with pytest.raises(CEncodingError):
        assemble_c(source)


@pytest.mark.parametrize("word", [(16 << 27) | (7 << 10), 27 << 27, 17 << 27,
                                  (7 << 27) | (4 << 10), (4 << 27) | (2 << 10),
                                  2 << 10, (21 << 27) | (7 << 10), 1 << 25])
def test_silent_c_noops_and_aliases_are_rejected(word):
    with pytest.raises(CEncodingError):
        disassemble_c([word])


def test_branch_offsets_include_macro_and_prefix_words():
    source = "start: movi_i32 1,r63\njump.cr0 end\ncopy r63,r62\njump start\nend: stop_core"
    words = assemble_c(source)
    text = disassemble_c(words)
    assert text[3] == "jump.cr0 4"
    assert text[6] == "jump -6"
    assert assemble_c("\n".join(text)) == words


@pytest.mark.parametrize("value,expected", [(0.5, 0x3800), (-0.5, 0xb800), (0.0, 0), (-0.0, 0x8000),
                                         (0.500732421875, 0x3801), (2 ** -24, 1), (65504.0, 0x7bff)])
def test_c_quantization_truncates_like_supplied_code(value, expected):
    assert fp16_bits(value) == expected


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 1e50, 70000.0])
def test_input_quantization_rejects_undefined_nonfinite_or_overflow(value):
    with pytest.raises(CEncodingError):
        fp16_bits(value)


@pytest.mark.parametrize("operation,source,expected", [
    ("fadd16", "movi_f16 0.5,r2\nfadd16 #3,r2,r3", 0x3e00),
    ("fsub16", "movi_f16 2,r2\nfsub16 #3,r2,r3", 0xbc00),
    ("fmul16", "movi_f16 2,r2\nfmul16 #4,r2,r3", 0x3c00),
    ("fdiv16", "movi_f16 2,r2\nfdiv16 #3,r2,r3", 0x3800),
    ("fsqrt16", "movi_f16 4,r2\nfsqrt16 r2,r3", 0x4000),
    ("frsqrt16", "movi_f16 4,r2\nfrsqrt16 r2,r3", 0x3800),
    ("fmulp2_16", "movi_f16 0.5,r2\nfmulp2_16 r2,2,r3", 0x4000),
    ("i32tof16", "movi_i32 3,r2\ni32tof16 r2,r3", 0x4200),
    ("u8tof16_lh", "movi_i32 0x12345678,r2\nu8tof16_lh r2,r3", fp16_bits(86)),
    ("f16tou8_lh", "movi_f16 2,r2\nmovi_i32 0x12345678,r4\nf16tou8_lh r2,r4,r3", 0x12340278),
])
def test_native_c_executes_canonical_operands(native_backend, operation, source, expected):
    result, contents = execute(native_backend, "movi_i32 0x40000,r1\n" + source + "\nstore r3,r1\nstop_core")
    assert int.from_bytes(contents[:4], "little") == expected
    assert result.backend == "supplied-c-interpreter"


@pytest.mark.parametrize("predicate,expected", [("cr0", 0x3c00), ("!cr0", 0x3800)])
def test_conditional_current_pc_branch(native_backend, predicate, expected):
    source = ("movi_i32 0x40000,r1\nmovi_f16 0,r2\nfcomp_e #3,r2,cr0\n"
              f"jump.{predicate} taken\ncopy #3,r3\njump finish\ntaken: copy #4,r3\n"
              "finish: store r3,r1\nstop_core")
    result, contents = execute(native_backend, source)
    assert int.from_bytes(contents[:4], "little") == expected


def test_upper_register_prefix_and_original_managed_buffer(native_backend):
    result, contents = execute(native_backend,
        "movi_i32 0x40000,r33\nmovi_i32 0x12345678,r34\ncopy r34,r63\nstore r63,r33\nstop_core")
    assert contents[:4] == bytes.fromhex("78563412")
    assert result.registers[63] == 0x12345678


def test_exact_budget_stop_succeeds(native_backend):
    result, contents = execute(native_backend, "stop_core", max_steps=1)
    assert result.steps == 1


@pytest.mark.parametrize("source,error", [("jump 0", ExecutionTimeoutError), ("nop", SubmissionError),
    ("movi_i32 0x40004,r1\nload r1,r2\nstop_core", SubmissionError),
    ("movi_i32 0x40001,r1\nload r1,r2\nstop_core", SubmissionError),
    ("movi_i32 0,r1\nstore r1,r1\nstop_core", SubmissionError),
    ("movi_i32 0x7c00,r1\nf16toi32 r1,r2\nstop_core", SubmissionError)])
def test_managed_execution_failure_guards(native_backend, source, error):
    with pytest.raises(error):
        execute(native_backend, source, size=4, max_steps=12)
    result, contents = execute(native_backend, "stop_core", size=4, max_steps=1)
    assert result.steps == 1


@pytest.mark.parametrize("backend", [SimulationBackend, HardwareSimulationBackend])
def test_old_backends_never_dispatch_new_format(backend):
    with pytest.raises(ProgramFormatError):
        backend().validate(program("stop_core"))


@pytest.mark.parametrize("format", [ProgramFormat.LEGACY, ProgramFormat.HARDWARE])
def test_c_backend_never_dispatches_old_format(native_backend, format):
    with pytest.raises(ProgramFormatError):
        native_backend.validate(Program.from_words([24 << 27], format=format))


def test_native_build_uses_preserved_supplied_sources(native_backend):
    assert len(native_backend.build_evidence["source_sha256"]) == 4
    assert native_backend.build_evidence["negative_fp16_signed_shift_issue_unresolved"] is True
    assert native_backend.build_evidence["original_sources_modified"] is False
