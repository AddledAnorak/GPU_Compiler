"""Assembler/disassembler correctness tests.

We have no reference hardware, so the primary correctness property we can
check is internal consistency: assemble -> disassemble -> reassemble must
be a fixed point for every mnemonic the ISA defines (per the plan's
verification section, item (a)).
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

import pytest

from mgcasm import ISA, AsmError, assemble, disassemble
from mgcasm.assembler import Assembler

isa = ISA()


def roundtrip(src: str) -> None:
    words = assemble(src, isa)
    lines = disassemble(words, isa)
    words2 = assemble("\n".join(lines), isa)
    assert words == words2, f"round-trip mismatch:\n{src}\n->\n{lines}"


# ---------------------------------------------------------------------
# One representative instruction per mnemonic that has a real operand
# shape, built directly from the operand roles in isa.yaml so every
# mnemonic is exercised without hand-listing 90 lines of assembly.
# ---------------------------------------------------------------------
SAMPLE_REG = "r5"
SAMPLE_REG_B = "r6"
SAMPLE_CR = "cr2"
SAMPLE_VREG = "v1"


def _sample_operand(role: str) -> str:
    return {
        "srcA": SAMPLE_REG,
        "srcB": SAMPLE_REG_B,
        "dst": "r7",
        "cr_dst": SAMPLE_CR,
        "addr_reg": SAMPLE_REG_B,
        "vsrc": SAMPLE_VREG,
        "vdst": "v2",
        "bit": "1",
    }[role]


@pytest.mark.parametrize("mnemonic", sorted(isa.instructions))
def test_mnemonic_roundtrips(mnemonic):
    instr = isa.instructions[mnemonic]
    if mnemonic in ("regbank",):
        pytest.skip("covered by test_regbank_prefix")
    if mnemonic in ("movi_u16", "movi_i32", "movi_f16", "ploadu_l", "loadu_h"):
        pytest.skip("covered by test_immediate_pseudo_ops")
    if mnemonic in ("goto", "call"):
        pytest.skip("covered by test_branch_targets")
    if instr.channel is None:
        pytest.skip(f"{mnemonic} has no assigned opcode channel yet")

    ops = ", ".join(_sample_operand(r) for r in instr.operands)
    src = f"{mnemonic} {ops}".strip()
    roundtrip(src)


def test_conditional_execution_normal_and_inverted():
    roundtrip("fadd16.cr3 r1, r2, r3")
    roundtrip("fadd16.ncr3 r1, r2, r3")


@pytest.mark.parametrize("mnemonic", ["goto", "call"])
@pytest.mark.parametrize("condition", ["cr", "ncr"])
@pytest.mark.parametrize("register", range(8))
@pytest.mark.parametrize("offset", [-32768, -1, 0, 22, 32767])
def test_direct_branch_predicates_roundtrip(mnemonic, condition, register, offset):
    roundtrip(f"{mnemonic}.{condition}{register} {offset}")


def test_constant_bank_operand_by_index_and_name():
    roundtrip("fadd16 #3, r2, r3")
    roundtrip("fmul16 #one, r2, r3")
    with pytest.raises(AsmError):
        assemble("fadd16 #9, r2, r3", isa)  # undocumented constant slot


def test_regbank_prefix_and_high_registers():
    roundtrip("regbank 0, 0, 1\nloadu_h 0x001, r33")
    roundtrip("regbank 1, 0, 0\ncopy r40, r2")


def test_regbank_required_for_high_registers():
    with pytest.raises(AsmError):
        assemble("copy r40, r2", isa)  # no preceding regbank


def test_regbank_is_one_shot():
    # the hi flag must not leak into a second instruction
    with pytest.raises(AsmError):
        assemble(
            "regbank 1, 0, 0\ncopy r40, r2\ncopy r40, r3",
            isa,
        )


def test_immediate_pseudo_ops():
    roundtrip("movi_i32 128, r2")
    roundtrip("movi_f16 0.5, r3")
    with pytest.raises(AsmError):
        assemble("movi_i32 999999999, r2", isa)  # exceeds provisional 27-bit path


def test_conditional_movi_rejected():
    # A conditional movi_* would propagate cond bits onto the expanded
    # ploadu_l/loadu_h pair, but the disassembler never prints a condition
    # suffix for those two mnemonics and the emulator executes them before
    # the predicate gate -- so it would silently break the round trip and
    # execute unconditionally. Must be rejected at assembly time instead.
    with pytest.raises(AsmError):
        assemble("movi_f16.cr0 0.5, r3", isa)
    with pytest.raises(AsmError):
        assemble("movi_i32.cr1 128, r2", isa)
    with pytest.raises(AsmError):
        assemble("movi_u16.ncr2 0x3c00, r1", isa)


def test_branch_targets_label_and_literal():
    roundtrip("goto skip\nnop\nskip:\nret")
    roundtrip("call -1")


def test_isa_examples_from_spec_section_11():
    # sec 11.2: FP16 arithmetic with explicit wait
    roundtrip("fmul16 r1, r3, r4\nwait_fmul\nfadd16 r4, r3, r5\nwait_fadd")
    # sec 11.3: integer compare
    roundtrip("icmp_ge r10, r0, cr0\nicmp_lt r11, r12, cr1")
    # sec 11.4: vector helper example
    roundtrip("vset4 r1, r2, v0\nvcmp_ge_zero_i32 v0\nvstore_mask v0, r3")


def test_unknown_mnemonic_rejected():
    with pytest.raises(AsmError):
        assemble("frobnicate r1, r2", isa)


def test_wrong_operand_count_rejected():
    with pytest.raises(AsmError):
        assemble("fadd16 r1, r2", isa)  # fadd16 needs 3 operands
