"""Assemble canonical Emu instructions and a small set of explicit macros."""

from dataclasses import dataclass
import math
import re

from .fp16 import float_to_half
from .isa import FORMS, PDF_WAIT_SLOTS, UNRESOLVED_PDF_FORMS, WORD_MASK, Form

_FORMS_BY_NAME = {form.name: form for form in FORMS.values()}
_NAME = r"[a-zA-Z_][a-zA-Z_0-9]*"
_LABEL = re.compile(rf"^({_NAME}):\s*")
_PREDICATE = re.compile(r"^\[(!?)cr([0-7])\]\s*")
_REG_FIELDS = {"a": (16, 20), "b": (5, 9), "d": (0, 4)}
_IMMEDIATE_MACROS = {"movi_u16", "movi_i32", "movi_f16"}
_ALIASES = {"ploadu_l": "preload_low", "loadu_h": "load_high", "u32tof16": "i32tof16"}
_SAMPLE_INCLUDE = '.include "../../rtl_fp16/tests/assembly/asm_lib/fp16_graphics_macros.inc"'


class AssemblyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _Branch:
    line: int
    pc: int
    pc_bias: int
    label: str


def _number(token, minimum, maximum, kind):
    try:
        base = 10 if re.fullmatch(r"[+-]?[0-9]+", token) else 0
        value = int(token, base)
    except ValueError as error:
        raise AssemblyError(f"invalid {kind}: {token!r}") from error
    if not minimum <= value <= maximum:
        raise AssemblyError(f"{kind} must be in [{minimum}, {maximum}], got {value}")
    return value


def _register(token, prefix, maximum):
    matched = re.fullmatch(rf"{prefix}([0-9]+)", token)
    if matched is None:
        raise AssemblyError(f"expected {prefix}0–{prefix}{maximum}, got {token!r}")
    return _number(matched[1], 0, maximum, f"{prefix} register")


def _offset_bits(offset):
    if not -(1 << 17) <= offset < (1 << 17):
        raise AssemblyError("branch displacement must fit a signed 18-bit word offset")
    offset &= (1 << 18) - 1
    return ((offset >> 10) << 14) | (offset & 0x3FF)


def _check_count(name, operands, expected):
    if len(operands) != expected:
        raise AssemblyError(f"{name} expects {expected} operand(s), got {len(operands)}")


def _encode(form: Form, operands: tuple[str, ...], predicate: int) -> list[int]:
    word = form.encoding | predicate
    if form.operands == ("bank",):
        _check_count(form.name, operands, 3)
        for operand, (_, flag_bit) in zip(operands, _REG_FIELDS.values()):
            word |= _number(operand, 0, 1, "bank flag") << flag_bit
        return [word]
    _check_count(form.name, operands, len(form.operands))
    bank = 0
    for role, operand in zip(form.operands, operands):
        if role == "imm":
            word |= _number(operand, 0, 0xFFFF, "immediate") << 6
        elif role == "offset":
            word |= _offset_bits(_number(operand, -(1 << 17), (1 << 17) - 1, "branch displacement"))
        elif role == "p":
            word |= (_number(operand, -16, 15, "power") & 31) << 5
        elif role == "cr":
            word |= _register(operand, "cr", 7)
        else:
            shift, flag_bit = _REG_FIELDS[role[-1]]
            if role.startswith("v"):
                word |= _register(operand, "v", 7) << shift
            elif role == "a" and operand.startswith("#"):
                word |= _number(operand[1:], 0, 7, "constant-bank index") << shift
            else:
                register = _register(operand, "r", 63)
                word |= (register & 31) << shift
                if role == "a":
                    word |= 1 << 21
                if register >= 32:
                    bank |= 1 << flag_bit
    return [_FORMS_BY_NAME["regbank"].encoding | bank, word] if bank else [word]


def _reject_nonfinite(assumption):
    raise AssemblyError(f"FP16 literal requires unsupported numeric assumption {assumption!r}")


def _literal(name, operand):
    if name == "movi_u16":
        return _number(operand, 0, 0xFFFF, "unsigned 16-bit literal")
    if name == "movi_i32":
        return _number(operand, -(1 << 31), WORD_MASK, "32-bit literal") & WORD_MASK
    try:
        value = float(operand)
        if math.isfinite(value):
            bits = float_to_half(value, _reject_nonfinite)
            if bits & 0x7C00 != 0x7C00:
                return bits
    except (ValueError, OverflowError) as error:
        raise AssemblyError(f"invalid finite FP16 literal: {operand!r}") from error
    raise AssemblyError(f"FP16 literal does not fit a finite half: {operand!r}")


def _emit(name, operands, predicate, pc, line, branches):
    if name in _IMMEDIATE_MACROS:
        _check_count(name, operands, 2)
        value = _literal(name, operands[0])
        return (_encode(_FORMS_BY_NAME["preload_low"], (str(value & 0xFFFF),), predicate)
                + _encode(_FORMS_BY_NAME["load_high"], (str(value >> 16), operands[1]), predicate))
    if name in PDF_WAIT_SLOTS:
        _check_count(name, operands, 0)
        return _encode(_FORMS_BY_NAME["nop"], (), predicate) * PDF_WAIT_SLOTS[name]
    if name == "wait":
        _check_count(name, operands, 1)
        return _encode(_FORMS_BY_NAME["nop"], (), predicate) * _number(operands[0], 1, 65535, "wait slots")
    if name in UNRESOLVED_PDF_FORMS:
        raise AssemblyError(f"{name}: named in the ISA PDF but its encoding and effect are not established")
    form = _FORMS_BY_NAME.get(_ALIASES.get(name, name))
    if form is None:
        raise AssemblyError(f"unknown instruction {name!r}")
    if not form.modeled:
        raise AssemblyError(f"{name}: encoding is known but Emu does not model its effect")
    if form.operands == ("offset",) and len(operands) == 1 and re.fullmatch(_NAME, operands[0]):
        branches.append(_Branch(line, pc, form.pc_bias, operands[0]))
        operands = ("0",)
    return _encode(form, operands, predicate)


def assemble(source: str, *, sample_source: bool = False) -> list[int]:
    """Translate Emu assembly to words, resolving labels after macro expansion."""
    words: list[int] = []
    labels: dict[str, int] = {}
    branches: list[_Branch] = []
    for line_number, raw in enumerate(source.splitlines(), 1):
        line = re.split(r";|//", raw, maxsplit=1)[0].strip()
        try:
            while label := _LABEL.match(line):
                name = label[1]
                if name in labels:
                    raise AssemblyError(f"duplicate label {name!r}")
                labels[name] = len(words)
                line = line[label.end():]
            if not line:
                continue
            if line.startswith(".include"):
                if sample_source and line == _SAMPLE_INCLUDE:
                    continue
                raise AssemblyError("includes are not read; --sample-source allows the recovered graphics macros")
            predicate = 0
            if line.startswith("["):
                condition = _PREDICATE.match(line)
                if condition is None:
                    raise AssemblyError("expected a predicate such as [cr0] or [!cr0]")
                predicate = (1 << 26) | (bool(condition[1]) << 25) | (int(condition[2]) << 22)
                line = line[condition.end():]
            parts = line.split(None, 1)
            if not parts:
                raise AssemblyError("missing instruction after predicate")
            operands = tuple(piece.strip() for piece in parts[1].split(",")) if len(parts) == 2 else ()
            name = "jump_next" if sample_source and parts[0] == "jump" else parts[0]
            words.extend(_emit(name, operands, predicate, len(words), line_number, branches))
        except AssemblyError as error:
            raise AssemblyError(f"line {line_number}: {error}") from error
    if not words:
        raise AssemblyError("program has no instructions")
    for branch in branches:
        try:
            if branch.label not in labels:
                raise AssemblyError(f"undefined label {branch.label!r}")
            target = labels[branch.label]
            if target == len(words):
                raise AssemblyError(f"label {branch.label!r} has no instruction at its address")
            words[branch.pc] |= _offset_bits(target - branch.pc - branch.pc_bias)
        except AssemblyError as error:
            raise AssemblyError(f"line {branch.line}: {error}") from error
    return words
