"""Canonical encoding for the supplied C interpreter, not the older ISA models."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import struct

FORMAT = "c-interpreter-v1"
MAX_WORDS = 65536


class CEncodingError(ValueError):
    pass


OPERATIONS = {}


def _family(channel, names, roles):
    for selector, name in enumerate(names.split()):
        OPERATIONS[name] = (channel, selector, tuple(roles.split()))


_family(2, "copy fabs16 fneg16 fnabs16", "a d")
_family(3, "fmulp2_16", "a p d")
_family(4, "fadd16 fsub16", "a b d")
_family(5, "fmul16", "a b d")
_family(6, "fdiv16", "a b d")
_family(7, "fmin16 fmax16", "a b d")
OPERATIONS.update({"fsqrt16": (7, 2, ("a", "d")), "frsqrt16": (7, 3, ("a", "d"))})
_family(8, "trunc16 round16 ceil16 floor16", "a d")
_family(9, "f16tou8_ll f16tou8_lh f16tou8_hl f16tou8_hh", "a b d")
_family(10, "f16tou16_l f16tou16_h", "a b d")
_family(11, "f16toi32", "a d")
_family(12, "u8tof16_ll u8tof16_lh u8tof16_hl u8tof16_hh", "b d")
_family(13, "u16tof16_l u16tof16_h", "b d")
_family(14, "i32tof16", "b d")
_family(15, "fcomp_l fcomp_le fcomp_e fcomp_ge fcomp_g fcomp_ne fcomp_o fcomp_u", "a b c")
_family(16, "add and istep4", "a b d")
OPERATIONS.update({"vset4": (16, 3, ("a", "b", "vd")), "vbcast": (16, 4, ("a", "vd")),
                   "vadd_i32": (16, 5, ("va", "vb", "vd"))})
_family(19, "load", "b d")
_family(20, "store", "a b")
_family(22, "goto", "b")
_family(23, "ret", "b")
_family(24, "stop_core", "")
_family(26, "icmp_lt icmp_le icmp_eq icmp_ge icmp_gt icmp_ne", "a b c")
REVERSE = {(channel, selector): name for name, (channel, selector, roles) in OPERATIONS.items()}


def integer(value, minimum, maximum, name="integer"):
    if type(value) is not int or not minimum <= value <= maximum:
        raise CEncodingError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def fp16_bits(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CEncodingError("FP16 payload must be a finite number")
    try:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise CEncodingError("payload does not fit FP32") from error
    sign = (bits >> 16) & 0x8000
    exponent = ((bits >> 23) & 255) - 127 + 15
    mantissa = bits & 0x7fffff
    if exponent <= 0:
        return sign if exponent < -10 else sign | (((mantissa | 0x800000) >> (1 - exponent)) >> 13)
    if exponent >= 31:
        raise CEncodingError("payload must fit finite C-contract FP16")
    return sign | (exponent << 10) | (mantissa >> 13)


def _number(text):
    try:
        return int(text, 0) if not text.lstrip("+-").isdigit() else int(text, 10)
    except ValueError as error:
        raise CEncodingError(f"invalid integer {text!r}") from error


def _register(text, kind="r"):
    match = re.fullmatch(re.escape(kind) + r"([0-9]+)", text)
    if match is None:
        raise CEncodingError(f"expected {kind} register, got {text!r}")
    return integer(int(match[1]), 0, 63 if kind == "r" else 7, "register")


@dataclass(frozen=True)
class Parsed:
    line: int
    name: str
    operands: tuple[str, ...]
    predicate: int = 0


def _parse(source):
    if not isinstance(source, str) or len(source) > 8 << 20:
        raise CEncodingError("assembly must be bounded text")
    instructions, label_indices = [], {}
    for line_number, text in enumerate(source.splitlines(), 1):
        text = text.split(";", 1)[0].split("//", 1)[0].strip()
        if not text:
            continue
        label = re.match(r"^([A-Za-z_][\w]*):", text)
        if label:
            if label[1] in label_indices:
                raise CEncodingError(f"duplicate label {label[1]!r}")
            label_indices[label[1]] = len(instructions)
            text = text[label.end():].strip()
            if not text:
                continue
        pieces = text.split(None, 1)
        name, predicate = pieces[0], 0
        matched = re.fullmatch(r"([a-z0-9_]+)(?:\.(!?)cr([0-7]))?", name)
        if matched is None:
            raise CEncodingError(f"line {line_number}: invalid instruction")
        name = matched[1]
        if matched[3] is not None:
            predicate = (1 << 26) | (bool(matched[2]) << 25) | (int(matched[3]) << 22)
        operands = tuple(part.strip() for part in pieces[1].split(",")) if len(pieces) == 2 else ()
        instructions.append(Parsed(line_number, name, operands, predicate))
        if len(instructions) > MAX_WORDS:
            raise CEncodingError("program exceeds bounded instruction count")
    return instructions, label_indices


def _encode(parsed, pc, labels):
    name, operands = parsed.name, parsed.operands
    prefix = parsed.predicate
    if name == "nop":
        if operands:
            raise CEncodingError("nop takes no operands")
        return [prefix]
    if name == "regbank":
        if len(operands) != 3 or prefix:
            raise CEncodingError("regbank requires three 0/1 bits and no predicate")
        high = [integer(_number(operand), 0, 1) for operand in operands]
        return [(1 << 10) | (high[0] << 20) | (high[1] << 9) | (high[2] << 4)]
    if name in {"movi_i32", "movi_u16", "movi_f16", "preload_low", "load_high"}:
        if len(operands) != (1 if name == "preload_low" else 2):
            raise CEncodingError(f"{name}: invalid immediate operands")
        value = fp16_bits(float(operands[0])) if name == "movi_f16" else _number(operands[0])
        if name != "movi_i32":
            integer(value, 0, 65535, "immediate")
        else:
            integer(value, -(1 << 31), (1 << 32) - 1, "immediate")
        value &= 0xffffffff
        if name == "preload_low":
            return [prefix | (1 << 27) | (value << 6)]
        destination = _register(operands[1])
        bank = [(1 << 10) | (1 << 4)] if destination >= 32 else []
        if name == "load_high":
            return bank + [prefix | (1 << 27) | (value << 6) | 32 | (destination & 31)]
        return [prefix | (1 << 27) | ((value & 65535) << 6)] + bank + [prefix | (1 << 27) | ((value >> 16) << 6) | 32 | (destination & 31)]
    if name == "jump":
        if len(operands) != 1:
            raise CEncodingError("jump requires a label or signed current-PC displacement")
        offset = labels[operands[0]] - pc if operands[0] in labels else _number(operands[0])
        integer(offset, -(1 << 17), (1 << 17) - 1, "branch displacement")
        offset &= (1 << 18) - 1
        return [prefix | (21 << 27) | (((offset >> 10) & 255) << 14) | (offset & 1023)]
    if name not in OPERATIONS:
        raise CEncodingError(f"unsupported current C instruction {name!r}")
    channel, selector, roles = OPERATIONS[name]
    if len(roles) != len(operands):
        raise CEncodingError(f"{name} requires {len(roles)} operands")
    word, bank = prefix | (channel << 27) | (selector << 10), 0
    for role, operand in zip(roles, operands):
        if role == "p":
            word |= (integer(_number(operand), -16, 15, "power") & 31) << 5
            continue
        if role == "a" and operand.startswith("#"):
            word |= integer(_number(operand[1:]), 0, 7, "constant bank index") << 16
            continue
        kind = "v" if role.startswith("v") else "cr" if role == "c" else "r"
        register = _register(operand, kind)
        slot = role[-1]
        if kind == "r" and register >= 32:
            bank |= 1 << {"a": 20, "b": 9, "d": 4}[slot]
        word |= (register & 31) << {"a": 16, "b": 5, "d": 0, "c": 0}[slot]
        if role == "a":
            word |= 1 << 21
    return ([(1 << 10) | bank] if bank else []) + [word]


def assemble_c(source):
    parsed, label_indices = _parse(source)
    positions, pc = [], 0
    for instruction in parsed:
        positions.append(pc)
        if instruction.name == "jump":
            pc += 1
        else:
            pc += len(_encode(instruction, pc, {}))
        if pc > MAX_WORDS:
            raise CEncodingError("program exceeds 65536 machine words")
    labels = {name: positions[index] if index < len(positions) else pc for name, index in label_indices.items()}
    words = []
    for instruction, position in zip(parsed, positions):
        try:
            words.extend(_encode(instruction, position, labels))
        except (ValueError, OverflowError) as error:
            raise CEncodingError(f"line {instruction.line}: {error}") from error
    if not words:
        raise CEncodingError("empty program")
    return words


def disassemble_c(words):
    if not 1 <= len(words) <= MAX_WORDS:
        raise CEncodingError("invalid program size")
    lines = []
    for word in words:
        integer(word, 0, (1 << 32) - 1, "machine word")
        channel, selector = word >> 27, (word >> 10) & 63
        predicate = f".{ '!' if word & (1 << 25) else ''}cr{(word >> 22) & 7}" if word & (1 << 26) else ""
        if not word & (1 << 26) and word & (31 << 22):
            raise CEncodingError("noncanonical unused predicate bits")
        if channel == 0:
            if selector == 1:
                line = f"regbank {(word >> 20) & 1}, {(word >> 9) & 1}, {(word >> 4) & 1}"
            else:
                line = "nop" + predicate
        elif channel == 1:
            line = ("load_high" if word & 32 else "preload_low") + predicate + f" {(word >> 6) & 65535}"
            if word & 32:
                line += f", r{word & 31}"
        elif channel == 21:
            offset = (((word >> 14) & 255) << 10) | (word & 1023)
            if offset & (1 << 17):
                offset -= 1 << 18
            line = f"jump{predicate} {offset}"
        else:
            name = REVERSE.get((channel, selector))
            if name is None:
                raise CEncodingError(f"unsupported/noncanonical word 0x{word:08x}")
            operands = []
            for role in OPERATIONS[name][2]:
                if role == "p":
                    power = (word >> 5) & 31
                    operands.append(str(power - 32 if power & 16 else power))
                elif role == "a" and not word & (1 << 21):
                    operands.append(f"#{(word >> 16) & 31}")
                else:
                    kind = "v" if role.startswith("v") else "cr" if role == "c" else "r"
                    register = (word >> {"a": 16, "b": 5, "d": 0, "c": 0}[role[-1]]) & (7 if kind != "r" else 31)
                    operands.append(f"{kind}{register}")
            line = name + predicate + (" " + ", ".join(operands) if operands else "")
        if _encode(_parse(line)[0][0], 0, {}) != [word]:
            raise CEncodingError(f"noncanonical word 0x{word:08x}")
        lines.append(line)
    return lines
