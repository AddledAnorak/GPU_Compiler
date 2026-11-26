"""Canonical word forms from the C implementation and the supplied samples."""

from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType

WORD_MASK = 0xFFFFFFFF
CONSTANTS = (0xFC00, 0xBC00, 0, 0x3C00, 0x3800, 0x4248, 0x7C00, 0x7E00)
ASSUMPTIONS = MappingProxyType({
    "sample-next-pc": "Selector 7 jumps relative to the next word, as required by sample labels.",
    "sample-vector": "Sample vector masks, raw lane extraction and masked writes have the recovered effects.",
    "negative-fp16-sign": "Decode negative halves by moving the sign bit with unsigned arithmetic.",
    "arithmetic-nan": "Use positive quiet NaN for host-dependent FP16 arithmetic NaN results.",
    "sample-vector-fp16": "Use sequential lane setup, nearest-even vector FP16, and clamped RGB565 packing.",
    "sample-fadd-rne": "Use the sample replay's nearest-even scalar add instead of C's truncation.",
    "sample-rgb565-fp16-scale": "Round the RGB565 channel products to FP16 before truncating to integers.",
})

PDF_WAIT_SLOTS = MappingProxyType({
    "wait_fadd": 6, "wait_fmul": 8, "wait_fdiv": 8, "wait_fcomp": 3,
    "wait_conv": 3, "wait_load": 4, "wait_short": 3,
})
UNRESOLVED_PDF_FORMS = frozenset({
    "load_addr", "store_addr", "i_load", "i_store", "push", "pop", "call", "rcall",
    "v4cr_store", "v4cr_store_step", "vstore_mask", "v4cmp_ge_zero", "v4cmp_ge_zero_and",
    "v4cmp_ge_zero_step", "v4cmp_ge_zero_step_and_p1", "v4mask_to_cr", "vcmp_ge_zero_i32",
})

_FIELD_MASKS = {
    "a": 63 << 16, "b": 31 << 5, "d": 31, "p": 31 << 5,
    "va": 7 << 16, "vb": 7 << 5, "vd": 7, "cr": 7,
    "bank": (1 << 20) | (1 << 9) | (1 << 4),
    "imm": 0xFFFF << 6,
    "offset": (0xFF << 14) | 0x3FF,
}


class DecodeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Form:
    name: str
    opcode: int
    selector: int
    operands: tuple[str, ...]
    evidence: str
    assumption: str | None
    modeled: bool
    operand_mask: int
    pc_bias: int
    result_wait: str | None = None

    @property
    def encoding(self) -> int:
        shift, _ = _selector_bits(self.opcode)
        return (self.opcode << 27) | (self.selector << shift)


def _selector_bits(opcode):
    if opcode == 1:
        return 5, 1
    return (10, 15) if opcode == 21 else (10, 63)


def _forms():
    forms = {}

    def family(opcode, first, names, operands, *, evidence="C", assumption=None, modeled=True, pc_bias=0,
               result_wait=None):
        roles = tuple(operands.split())
        operand_mask = 0
        for role in roles:
            operand_mask |= _FIELD_MASKS[role]
        for selector, name in enumerate(names.split(), first):
            forms[opcode, selector] = Form(name, opcode, selector, roles, evidence,
                                           assumption, modeled, operand_mask, pc_bias, result_wait)

    family(0, 0, "nop", "")
    family(0, 1, "regbank", "bank")
    family(1, 0, "preload_low", "imm")
    family(1, 1, "load_high", "imm d")
    family(2, 0, "copy fabs16 fneg16 fnabs16", "a d")
    family(3, 0, "fmulp2_16", "a p d")
    family(4, 0, "fadd16 fsub16", "a b d", result_wait="wait_fadd")
    family(5, 0, "fmul16", "a b d", result_wait="wait_fmul")
    family(6, 0, "fdiv16", "a b d", result_wait="wait_fdiv")
    family(7, 0, "fmin16 fmax16", "a b d")
    family(7, 2, "fsqrt16 frsqrt16", "a d")
    family(8, 0, "trunc16 round16 ceil16 floor16", "a d")
    family(9, 0, "f16tou8_ll f16tou8_lh f16tou8_hl f16tou8_hh", "a b d", result_wait="wait_conv")
    family(10, 0, "f16tou16_l f16tou16_h", "a b d", result_wait="wait_conv")
    family(11, 0, "f16toi32", "a d", result_wait="wait_conv")
    family(12, 0, "u8tof16_ll u8tof16_lh u8tof16_hl u8tof16_hh", "b d", result_wait="wait_conv")
    family(13, 0, "u16tof16_l u16tof16_h", "b d", result_wait="wait_conv")
    family(14, 0, "i32tof16", "b d", result_wait="wait_conv")
    family(15, 0, "fcomp_l fcomp_le fcomp_e fcomp_ge fcomp_g fcomp_ne fcomp_o fcomp_u", "a b cr",
           result_wait="wait_fcomp")
    family(16, 0, "add and istep4", "a b d")
    family(16, 3, "vset4", "a b vd")
    family(16, 4, "vbcast", "a vd")
    family(16, 5, "vadd_i32", "va vb vd")
    family(19, 0, "load", "b d", result_wait="wait_load")
    family(20, 0, "store", "a b")
    family(21, 0, "jump", "offset")
    family(22, 0, "goto", "b")
    family(23, 0, "ret", "b")
    family(24, 0, "stop_core", "")
    family(26, 0, "icmp_lt icmp_le icmp_eq icmp_ge icmp_gt icmp_ne", "a b cr")

    family(21, 7, "jump_next", "offset", evidence="sample", assumption="sample-next-pc", pc_bias=1)
    family(25, 3, "vstore_mask_r", "a vd b", evidence="sample", assumption="sample-vector")
    family(25, 5, "vstore_mask_vr", "va vd b", evidence="sample", assumption="sample-vector")
    family(27, 1, "vcmp_ge_zero_i32_mask", "va d", evidence="sample", assumption="sample-vector")
    family(27, 2, "vget_lane0_i32", "va d", evidence="sample", assumption="sample-vector")
    family(28, 1, "vcmp_ge_zero_i32_and", "va b d", evidence="sample", assumption="sample-vector")

    family(16, 6, "vfset4", "a b vd", evidence="sample", assumption="sample-vector-fp16")
    family(16, 7, "vfadd16 vfsub16 vfmul16 vfmax16", "va vb vd", evidence="sample",
           assumption="sample-vector-fp16")
    family(16, 11, "vrgb565_ceramic", "va vd", evidence="sample", assumption="sample-vector-fp16")
    return MappingProxyType(forms)


FORMS = _forms()


@dataclass(frozen=True, slots=True)
class Instruction:
    form: Form
    source_a: int
    source_b: int
    destination: int
    constant_a: bool
    condition: int | None
    invert: bool
    immediate: int
    offset: int


@lru_cache(maxsize=8192, typed=True)
def decode_word(word: int) -> Instruction:
    """Decode words in the documented canonical forms, rejecting reserved bits."""
    if type(word) is not int or not 0 <= word <= WORD_MASK:
        raise DecodeError("instruction is not an unsigned 32-bit word")
    opcode = word >> 27
    shift, mask = _selector_bits(opcode)
    selector = (word >> shift) & mask
    form = FORMS.get((opcode, selector))
    if form is None:
        raise DecodeError(f"unknown opcode {opcode}, selector {selector}")
    conditional = bool(word & (1 << 26))
    allowed = (31 << 27) | (mask << shift) | form.operand_mask
    if conditional:
        allowed |= 31 << 22
    if word & ~allowed:
        raise DecodeError(f"unexpected operand or reserved bits for {form.name}")
    source_a = (word >> 16) & 31
    constant_a = not bool(word & (1 << 21))
    if "a" in form.operands and constant_a and source_a >= len(CONSTANTS):
        raise DecodeError("undocumented constant-bank index")
    offset = (((word >> 14) & 0xFF) << 10) | (word & 0x3FF)
    return Instruction(form, source_a, (word >> 5) & 31, word & 31, constant_a,
                       (word >> 22) & 7 if conditional else None, bool(word & (1 << 25)),
                       (word >> 6) & 0xFFFF, offset - (1 << 18) if offset & (1 << 17) else offset)
