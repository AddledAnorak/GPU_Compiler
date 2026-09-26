"""mgcsim correctness tests: FP16 bit-exactness, latency-hazard detection,
and one hand-written assembly kernel producing an expected framebuffer
pixel -- the three checks the plan's verification section calls out for
Student B's Week 3-4 deliverable.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

import numpy as np
import pytest

from mgcasm import ISA, assemble
from mgcasm.fp16 import float_to_fp16_bits, fp16_bits_to_float
from mgcsim import CPU, LatencyViolation
from mgcsim.memory import FRAMEBUFFER_BASE

isa = ISA()


def run(src: str, max_cycles=1000) -> CPU:
    words = assemble(src, isa)
    cpu = CPU(isa=isa, max_cycles=max_cycles)
    cpu.run(words)
    return cpu


# ---------------------------------------------------------------------
# FP16 bit-exactness against numpy.float16, including the constant bank
# ---------------------------------------------------------------------
@pytest.mark.parametrize("idx,bits", [
    (0, 0xFC00), (1, 0xBC00), (2, 0x0000), (3, 0x3C00),
    (4, 0x3800), (5, 0x4248), (6, 0x7C00), (7, 0x7E00),
])
def test_constant_bank_matches_spec_table4(idx, bits):
    entry = isa.constants_by_index[idx]
    assert entry["bits"] == bits


def test_fp16_bits_roundtrip_matches_numpy():
    for v in [0.0, 1.0, -1.0, 0.5, 3.14, -3.14, 65504.0, 1e-5]:
        bits = float_to_fp16_bits(v)
        expected = int(np.float16(v).view(np.uint16))
        assert bits == expected
        assert fp16_bits_to_float(bits) == float(np.float16(v))


def test_fp16_add_matches_numpy_including_rounding():
    cpu = run("""
        movi_f16 0.1, r1
        movi_f16 0.2, r2
        fadd16 r1, r2, r3
        wait_fadd
        stop_core
    """)
    expected = float(np.float16(np.float16(0.1) + np.float16(0.2)))
    assert fp16_bits_to_float(cpu.regs[3] & 0xFFFF) == expected


def test_fp16_inf_and_nan_from_constant_bank():
    # srcB has no constant-bank select bit in the instruction format (only
    # srcA does, per Table 2) -- +inf must be copied into a register first.
    cpu = run("""
        copy #6, r2
        fadd16 #6, r2, r1
        wait_fadd
        stop_core
    """)
    # +inf + +inf = +inf
    assert fp16_bits_to_float(cpu.regs[1] & 0xFFFF) == float("inf")


def test_fp16_divide_by_zero_is_inf_not_a_crash():
    cpu = run("""
        movi_f16 1.0, r1
        movi_f16 0.0, r2
        fdiv16 r1, r2, r3
        wait_fdiv
        stop_core
    """)
    assert fp16_bits_to_float(cpu.regs[3] & 0xFFFF) == float("inf")


@pytest.mark.parametrize("bits", [0x7C00, 0xFC00, 0x7E00])
def test_rounding_preserves_nonfinite_fp16(bits):
    cpu = run(f"movi_i32 {bits}, r1\nfloor16 r1, r2\nwait_short\nstop_core")
    assert cpu.regs[2] & 0xFFFF == bits


# ---------------------------------------------------------------------
# Latency-hazard detection (Table 10) -- the emulator must catch a
# missing/insufficient wait, not silently produce a wrong answer.
# ---------------------------------------------------------------------
def test_missing_wait_after_fmul_is_detected():
    with pytest.raises(LatencyViolation):
        run("""
            movi_f16 2.0, r1
            movi_f16 3.0, r2
            fmul16 r1, r2, r3
            fadd16 r3, r3, r4
            stop_core
        """)


def test_missing_wait_after_fadd_is_detected():
    with pytest.raises(LatencyViolation):
        run("""
            movi_f16 1.0, r1
            movi_f16 2.0, r2
            fadd16 r1, r2, r3
            copy r3, r4
            stop_core
        """)


def test_correct_wait_sequence_does_not_raise():
    cpu = run("""
        movi_f16 2.0, r1
        movi_f16 3.0, r2
        fmul16 r1, r2, r3
        wait_fmul
        fadd16 r3, r3, r4
        wait_fadd
        stop_core
    """)
    assert cpu.halted


def test_multiple_outstanding_ops_of_one_class_are_allowed():
    # matches FP16_ISA_V1.pdf sec. 11.3's own pattern (icmp_ge then
    # icmp_lt, no wait between them): issuing a second op of the same
    # class before waiting is fine; a single wait_<class> clears both.
    cpu = run("""
        movi_f16 1.0, r1
        movi_f16 2.0, r2
        fmul16 r1, r2, r3
        fmul16 r2, r2, r4
        wait_fmul
        fadd16 r3, r4, r5
        wait_fadd
        stop_core
    """)
    assert cpu.halted


def test_overwriting_an_outstanding_register_clears_its_hazard():
    # fmul16 leaves r3 outstanding under fp16_multiply. An unrelated write
    # to r3 by an instruction with no latency class (copy) means r3 no
    # longer holds the fmul16 result -- reading r3 afterward must not raise
    # a stale LatencyViolation for a value that's no longer there.
    cpu = run("""
        movi_f16 1.0, r1
        movi_f16 2.0, r2
        fmul16 r1, r2, r3
        movi_f16 3.0, r5
        copy r5, r3
        icmp_eq r3, r3, cr0
        stop_core
    """)
    assert cpu.halted


def test_reading_cr_before_wait_fcomp_is_detected():
    with pytest.raises(LatencyViolation):
        run("""
            movi_f16 1.0, r1
            movi_f16 2.0, r2
            fcomp_l r1, r2, cr0
            fadd16.cr0 r1, r2, r3
            stop_core
        """)


def test_sec_11_2_example_from_spec_is_hazard_free():
    # FP16 Arithmetic With Explicit Wait, sec. 11.2 of FP16_ISA_V1.pdf
    cpu = run("""
        movi_f16 1.0, r1
        movi_f16 0.5, r3
        fmul16 r1, r3, r4
        wait_fmul
        fadd16 r4, r3, r5
        wait_fadd
        stop_core
    """)
    assert cpu.halted


# ---------------------------------------------------------------------
# Integer compare (sec. 11.3) and conditional execution
# ---------------------------------------------------------------------
def test_sec_11_3_integer_compare_example():
    cpu = run("""
        movi_i32 5, r10
        movi_i32 0, r0
        movi_i32 5, r11
        movi_i32 5, r12
        icmp_ge r10, r0, cr0
        icmp_lt r11, r12, cr1
        wait_short
        stop_core
    """)
    assert cpu.cr[0] is True   # 5 >= 0
    assert cpu.cr[1] is False  # 5 < 5 is false


def test_conditional_execution_gates_on_cr():
    cpu = run("""
        movi_i32 1, r0
        movi_i32 1, r1
        icmp_eq r0, r1, cr0
        wait_short
        movi_i32 0, r2
        movi_i32 99, r3
        copy.cr0 r3, r2
        stop_core
    """)
    assert cpu.regs[2] == 99  # predicate true -> copy executed

    cpu2 = run("""
        movi_i32 1, r0
        movi_i32 2, r1
        icmp_eq r0, r1, cr0
        wait_short
        movi_i32 0, r2
        movi_i32 99, r3
        copy.cr0 r3, r2
        stop_core
    """)
    assert cpu2.regs[2] == 0  # predicate false -> copy skipped


# ---------------------------------------------------------------------
# A hand-written kernel producing an expected framebuffer pixel --
# the plan's integration-gate check.
# ---------------------------------------------------------------------
def test_hand_written_kernel_writes_expected_pixel():
    cpu = run("""
        movi_i32 7, r1              ; framebuffer address
        movi_f16 1.0, r2            ; color value (fp16 bit pattern in low 16 bits)
        store r2, r1
        stop_core
    """)
    from mgcsim.memory import Memory

    assert cpu.mem.load(7) & 0xFFFF == float_to_fp16_bits(1.0)


def test_vector_helper_example_sec_11_4():
    # sec 11.4: vset4 rEdgeBase, rDeltaX, v0 ; vcmp_ge_zero_i32 v0 ;
    # vstore_mask vColor, rFbAddr
    cpu = run("""
        movi_i32 0, r1        ; edge base
        movi_i32 1, r2        ; delta
        vset4 r1, r2, v0
        vcmp_ge_zero_i32 v0
        movi_i32 3, r3        ; color pattern
        movi_i32 100, r4      ; framebuffer address
        vbcast r3, v1
        vstore_mask v1, r4
        stop_core
    """)
    assert cpu.vregs[0] == [0, 1, 2, 3]
    assert cpu.vmask == [True, True, True, True]
    for lane in range(4):
        assert cpu.mem.load(100 + lane) == 3
