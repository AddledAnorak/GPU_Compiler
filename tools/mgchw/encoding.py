from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tools.mgcasm.fp16 import float_to_fp16_bits

HARDWARE_FORMAT = "fp16-hardware-v1"


class HardwareEncodingError(ValueError):
    pass


OPERATIONS = {
    "copy": (2, 0, (("a", "r"), ("d", "r"))),
    "fadd16": (4, 0, (("a", "r"), ("b", "r"), ("d", "r"))),
    "add": (16, 0, (("a", "r"), ("b", "r"), ("d", "r"))),
    "istep4": (16, 2, (("a", "r"), ("b", "r"), ("d", "r"))),
    "vset4": (16, 3, (("a", "r"), ("b", "r"), ("d", "v"))),
    "vbcast": (16, 4, (("a", "r"), ("d", "v"))),
    "vadd_i32": (16, 5, (("a", "v"), ("b", "v"), ("d", "v"))),
    "vfset4": (16, 6, (("a", "r"), ("b", "r"), ("d", "v"))),
    "vfadd16": (16, 7, (("a", "v"), ("b", "v"), ("d", "v"))),
    "vfsub16": (16, 8, (("a", "v"), ("b", "v"), ("d", "v"))),
    "vfmul16": (16, 9, (("a", "v"), ("b", "v"), ("d", "v"))),
    "vfmax16": (16, 10, (("a", "v"), ("b", "v"), ("d", "v"))),
    "vrgb565_ceramic": (16, 11, (("a", "v"), ("d", "v"))),
    "load": (19, 0, (("b", "r"), ("d", "r"))),
    "store": (20, 0, (("a", "r"), ("b", "r"))),
    "vstore_mask_r": (25, 3, (("a", "r"), ("d", "v"), ("b", "r"))),
    "vstore_mask_vr": (25, 5, (("a", "v"), ("d", "v"), ("b", "r"))),
    "icmp_lt": (26, 0, (("a", "r"), ("b", "r"), ("d", "cr"))),
    "icmp_ge": (26, 3, (("a", "r"), ("b", "r"), ("d", "cr"))),
    "icmp_gt": (26, 4, (("a", "r"), ("b", "r"), ("d", "cr"))),
    "vcmp_ge_zero_i32_mask": (27, 1, (("a", "v"), ("d", "r"))),
    "vget_lane0_i32": (27, 2, (("a", "v"), ("d", "r"))),
    "vcmp_ge_zero_i32_and": (28, 1, (("a", "v"), ("b", "r"), ("d", "r"))),
}
WAIT_NOPS = {"wait_load": 4, "wait_fadd": 6}
REVERSE_OPERATIONS = {(spec[0], spec[1]): name for name, spec in OPERATIONS.items()}


@dataclass(frozen=True)
class Instruction:
    word: int
    mnemonic: str
    opcode: int
    subop: int
    source_a: int
    source_b: int
    destination: int
    source_a_register: bool
    condition: int | None
    invert: bool
    immediate: int = 0
    branch_offset: int = 0


@dataclass(frozen=True)
class SourceInstruction:
    line_number: int
    pc: int
    mnemonic: str
    operands: tuple[str, ...]
    condition: int | None
    invert: bool


def load_mem(path: str | Path) -> list[int]:
    words = []
    for line_number, text in enumerate(Path(path).read_text().splitlines(), 1):
        text = text.split("//", 1)[0].strip()
        if not text:
            continue
        if not re.fullmatch(r"[0-9a-fA-F]{8}", text):
            raise HardwareEncodingError(f"{path}:{line_number}: expected one 32-bit hex word")
        words.append(int(text, 16))
    if not words:
        raise HardwareEncodingError(f"{path}: empty memory image")
    return words


def decode_word(word: int) -> Instruction:
    if not isinstance(word, int) or isinstance(word, bool) or not 0 <= word <= 0xFFFFFFFF:
        raise HardwareEncodingError("instruction must be an unsigned 32-bit integer")
    opcode = word >> 27
    subop = (word >> 10) & 63
    immediate = 0
    offset = 0
    if opcode == 0 and word & 0x3FFFFF == 0:
        mnemonic = "nop"
    elif opcode == 1:
        mnemonic = "load_high" if word & 32 else "preload_low"
        immediate = (word >> 6) & 0xFFFF
    elif opcode == 21 and (word >> 10) & 15 == 7:
        mnemonic = "jump"
        offset = ((word >> 14) & 255) << 10 | (word & 1023)
        if offset & (1 << 17):
            offset -= 1 << 18
    else:
        mnemonic = REVERSE_OPERATIONS.get((opcode, subop))
        if mnemonic is None:
            raise HardwareEncodingError(f"unobserved instruction encoding 0x{word:08x}")
    return Instruction(
        word, mnemonic, opcode, subop, (word >> 16) & 31,
        (word >> 5) & 31, word & 31, bool(word & (1 << 21)),
        (word >> 22) & 7 if word & (1 << 26) else None,
        bool(word & (1 << 25)), immediate, offset,
    )


def parse_source(source: str) -> tuple[list[SourceInstruction], dict[str, int]]:
    instructions = []
    labels = {}
    pc = 0
    for line_number, text in enumerate(source.splitlines(), 1):
        text = text.split("//", 1)[0].split(";", 1)[0].strip()
        if not text:
            continue
        if text.startswith(".include"):
            if not re.fullmatch(r"\.include\s+\"[^\"]*fp16_graphics_macros\.inc\"", text):
                raise HardwareEncodingError(f"line {line_number}: unsupported include")
            continue
        label_match = re.match(r"^([A-Za-z_][\w]*):", text)
        if label_match:
            label = label_match.group(1)
            if label in labels:
                raise HardwareEncodingError(f"line {line_number}: duplicate label {label}")
            labels[label] = pc
            text = text[label_match.end():].strip()
            if not text:
                continue
        condition = None
        invert = False
        predicate_match = re.match(r"^\[(!?)cr([0-7])\]\s*", text, re.IGNORECASE)
        if predicate_match:
            invert = bool(predicate_match.group(1))
            condition = int(predicate_match.group(2))
            text = text[predicate_match.end():]
        parts = text.split(None, 1)
        mnemonic = parts[0].lower()
        operands = tuple(piece.strip() for piece in parts[1].split(",")) if len(parts) == 2 else ()
        instructions.append(SourceInstruction(line_number, pc, mnemonic, operands, condition, invert))
        pc += 2 if mnemonic in {"movi_u16", "movi_i32", "movi_f16"} else WAIT_NOPS.get(mnemonic, 1)
    return instructions, labels


def _number(text: str) -> int:
    try:
        return int(text, 0) if not text.lstrip("+-").isdigit() else int(text, 10)
    except ValueError as error:
        raise HardwareEncodingError(f"invalid integer {text!r}") from error


def _register(text: str, kind: str) -> int:
    match = re.fullmatch(re.escape(kind) + r"([0-9]+)", text, re.IGNORECASE)
    maximum = 7 if kind in {"v", "cr"} else 31
    if match is None or int(match.group(1)) > maximum:
        raise HardwareEncodingError(f"expected {kind}0..{kind}{maximum}, got {text!r}")
    return int(match.group(1))


def _encode(parsed: SourceInstruction, labels: dict[str, int]) -> list[int]:
    mnemonic, operands = parsed.mnemonic, parsed.operands
    predicate = 0
    if parsed.condition is not None:
        predicate = 1 << 26 | parsed.condition << 22 | int(parsed.invert) << 25
    if mnemonic in {"nop", *WAIT_NOPS}:
        if operands:
            raise HardwareEncodingError(f"{mnemonic} takes no operands")
        return [predicate] * WAIT_NOPS.get(mnemonic, 1)
    if mnemonic in {"movi_u16", "movi_i32", "movi_f16"}:
        if len(operands) != 2:
            raise HardwareEncodingError(f"{mnemonic} requires immediate, register")
        destination = _register(operands[1], "r")
        if mnemonic == "movi_f16":
            value = float_to_fp16_bits(float(operands[0]))
        else:
            value = _number(operands[0])
        if mnemonic == "movi_u16" and not 0 <= value <= 65535:
            raise HardwareEncodingError("movi_u16 immediate out of range")
        if not -(1 << 31) <= value <= 0xFFFFFFFF:
            raise HardwareEncodingError("32-bit immediate out of range")
        value &= 0xFFFFFFFF
        return [predicate | 1 << 27 | (value & 65535) << 6,
                predicate | 1 << 27 | (value >> 16) << 6 | 32 | destination]
    if mnemonic == "preload_low":
        if len(operands) != 1 or not 0 <= _number(operands[0]) <= 65535:
            raise HardwareEncodingError("preload_low requires one unsigned 16-bit immediate")
        return [predicate | 1 << 27 | _number(operands[0]) << 6]
    if mnemonic == "load_high":
        if len(operands) != 2 or not 0 <= _number(operands[0]) <= 65535:
            raise HardwareEncodingError("load_high requires high immediate, register")
        return [predicate | 1 << 27 | _number(operands[0]) << 6 | 32 | _register(operands[1], "r")]
    if mnemonic == "jump":
        if len(operands) != 1:
            raise HardwareEncodingError("jump requires a label or relative displacement")
        offset = labels[operands[0]] - parsed.pc - 1 if operands[0] in labels else _number(operands[0])
        if not -(1 << 17) <= offset < (1 << 17):
            raise HardwareEncodingError("branch displacement exceeds observed signed 18-bit field")
        offset &= (1 << 18) - 1
        return [predicate | 21 << 27 | 7 << 10 | (offset >> 10) << 14 | (offset & 1023)]
    if mnemonic not in OPERATIONS:
        raise HardwareEncodingError(f"unobserved mnemonic {mnemonic!r}; no guessed hardware encoding")
    opcode, subop, roles = OPERATIONS[mnemonic]
    if len(operands) != len(roles):
        raise HardwareEncodingError(f"{mnemonic} requires {len(roles)} operands")
    word = predicate | opcode << 27 | subop << 10
    for operand, (slot, kind) in zip(operands, roles):
        if slot == "a" and operand.startswith("#") and mnemonic == "fadd16":
            value = _number(operand[1:])
            if not 0 <= value <= 7:
                raise HardwareEncodingError("only eight documented FP16 constants are allowed")
        else:
            value = _register(operand, kind)
            if slot == "a" and kind == "r":
                word |= 1 << 21
        word |= value << {"a": 16, "b": 5, "d": 0}[slot]
    return [word]


def assemble_hardware(source: str) -> list[int]:
    instructions, labels = parse_source(source)
    words = []
    for parsed in instructions:
        try:
            words.extend(_encode(parsed, labels))
        except (HardwareEncodingError, ValueError) as error:
            raise HardwareEncodingError(f"line {parsed.line_number}: {error}") from error
    if not words:
        raise HardwareEncodingError("empty program")
    return words


def disassemble_hardware(words: list[int]) -> list[str]:
    lines = []
    for word in words:
        decoded = decode_word(word)
        prefix = ""
        if decoded.condition is not None:
            inversion = "!" if decoded.invert else ""
            prefix = f"[{inversion}cr{decoded.condition}] "
        if decoded.mnemonic == "jump":
            operands = [str(decoded.branch_offset)]
        elif decoded.mnemonic == "preload_low":
            operands = [str(decoded.immediate)]
        elif decoded.mnemonic == "load_high":
            operands = [str(decoded.immediate), f"r{decoded.destination}"]
        elif decoded.mnemonic == "nop":
            operands = []
        else:
            operands = []
            for slot, kind in OPERATIONS[decoded.mnemonic][2]:
                value = {"a": decoded.source_a, "b": decoded.source_b, "d": decoded.destination}[slot]
                operands.append(f"#{value}" if slot == "a" and kind == "r" and not decoded.source_a_register
                                else f"{kind}{value}")
        lines.append(prefix + decoded.mnemonic + (" " + ", ".join(operands) if operands else ""))
    return lines
