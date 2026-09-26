"""Text assembly -> list of 32-bit instruction words.

Syntax (all provisional -- this is our own tool's syntax, not vendor-defined):

    label:
    fadd16 r1, r2, r3          ; unconditional
    fadd16.cr0 r1, r2, r3      ; conditional on CR0 (normal)
    fadd16.ncr0 r1, r2, r3     ; conditional on CR0 (inverted)
    fadd16 #one, r2, r3        ; srcA from constant bank, by name
    fadd16 #3, r2, r3          ; srcA from constant bank, by index
    regbank 0, 0, 1            ; one-shot hi-register prefix
    icmp_ge r10, r0, cr0       ; CR destination
    goto some_label            ; direct control transfer (label or literal offset)
    movi_i32 128, r2           ; pseudo-op, expands to ploadu_l/loadu_h

See docs/isa/ENCODING_GUESSES.md for what every one of these choices stands
in for, and OPEN_QUESTIONS.md for what to ask the hardware author.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .fp16 import float_to_fp16_bits
from .isa import ISA

REGISTER_LIKE_ROLES = {"srcA", "srcB", "dst", "addr_reg", "vsrc", "vdst"}


class AsmError(Exception):
    pass


@dataclass
class Operand:
    kind: str  # 'reg' | 'const' | 'cr' | 'vreg' | 'imm' | 'label' | 'bit'
    value: object  # int, float, or str (label name)


@dataclass
class ParsedLine:
    lineno: int
    label: str | None
    mnemonic: str | None
    cond_cr: int | None
    cond_invert: bool
    operands: list[Operand]
    raw: str


def _strip_comment(line: str) -> str:
    for sep in (";", "#"):
        # '#' also introduces constant-bank operands like #one -- only strip
        # a leading-token '#' comment, not one embedded after an opcode.
        pass
    idx = line.find(";")
    return line if idx == -1 else line[:idx]


def _parse_operand(tok: str) -> Operand:
    tok = tok.strip()
    if tok.startswith("#"):
        body = tok[1:]
        if body.lstrip("-").isdigit():
            return Operand("const", int(body))
        return Operand("const", body)  # resolved by name later
    if tok.lower().startswith("cr") and tok[2:].isdigit():
        return Operand("cr", int(tok[2:]))
    if tok.lower().startswith("v") and tok[1:].isdigit():
        return Operand("vreg", int(tok[1:]))
    if tok.lower().startswith("r") and tok[1:].isdigit():
        return Operand("reg", int(tok[1:]))
    if tok.lower().startswith("0x"):
        return Operand("imm", int(tok, 16))
    try:
        if any(c in tok for c in ".eE") and not tok.lstrip("-").isdigit():
            return Operand("imm", float(tok))
        return Operand("imm", int(tok))
    except ValueError:
        return Operand("label", tok)


def parse_line(lineno: int, raw: str) -> ParsedLine | None:
    line = _strip_comment(raw).strip()
    if not line:
        return None

    label = None
    if ":" in line.split()[0] if line.split() else False:
        pass
    # label form: "name:" possibly followed by an instruction on the same line
    if line.endswith(":") and " " not in line and "," not in line:
        return ParsedLine(lineno, line[:-1], None, None, False, [], raw)
    if ":" in line:
        maybe_label, rest = line.split(":", 1)
        if maybe_label.strip() and " " not in maybe_label.strip():
            label = maybe_label.strip()
            line = rest.strip()
            if not line:
                return ParsedLine(lineno, label, None, None, False, [], raw)

    parts = line.split(None, 1)
    mnem_full = parts[0]
    operand_str = parts[1] if len(parts) > 1 else ""

    cond_cr = None
    cond_invert = False
    mnemonic = mnem_full
    if "." in mnem_full:
        mnemonic, suffix = mnem_full.split(".", 1)
        suffix_l = suffix.lower()
        if suffix_l.startswith("ncr") and suffix_l[3:].isdigit():
            cond_invert = True
            cond_cr = int(suffix_l[3:])
        elif suffix_l.startswith("cr") and suffix_l[2:].isdigit():
            cond_cr = int(suffix_l[2:])
        else:
            raise AsmError(f"line {lineno}: bad condition suffix '.{suffix}'")

    operands = [_parse_operand(t) for t in operand_str.split(",") if t.strip()] if operand_str else []
    return ParsedLine(lineno, label, mnemonic, cond_cr, cond_invert, operands, raw)


class Assembler:
    def __init__(self, isa: ISA | None = None):
        self.isa = isa or ISA()

    # ------------------------------------------------------------------
    # Pass 1: macro expansion (movi_* -> ploadu_l/loadu_h), label collection
    # ------------------------------------------------------------------
    def _expand_macros(self, parsed_lines: list[ParsedLine]) -> list[ParsedLine]:
        out: list[ParsedLine] = []
        for pl in parsed_lines:
            if pl.mnemonic is None:
                out.append(pl)
                continue
            if pl.cond_cr is not None and pl.mnemonic in (
                "movi_f16",
                "movi_u16",
                "movi_i32",
            ):
                # A conditional movi_* would propagate cond bits onto the
                # expanded ploadu_l/loadu_h pair, but the disassembler never
                # prints a condition suffix for those two mnemonics and the
                # emulator executes them before the predicate gate -- so the
                # result would assemble, silently fail to round-trip, and
                # execute unconditionally. Reject rather than emit something
                # broken; conditional immediate load needs its own encoding
                # decision, not a guess (see docs/isa/OPEN_QUESTIONS.md item 2).
                raise AsmError(
                    f"line {pl.lineno}: conditional '{pl.mnemonic}' is not "
                    "supported -- it would break the assemble/disassemble "
                    "round trip and execute unconditionally regardless of "
                    "the condition suffix (see docs/isa/OPEN_QUESTIONS.md item 2)"
                )
            if pl.mnemonic == "movi_f16":
                imm_op, dst_op = pl.operands
                bits = float_to_fp16_bits(_as_number(imm_op))
                out.extend(self._expand_movi_u16(pl, bits, dst_op))
            elif pl.mnemonic == "movi_u16":
                imm_op, dst_op = pl.operands
                out.extend(self._expand_movi_u16(pl, _as_int(imm_op), dst_op))
            elif pl.mnemonic == "movi_i32":
                imm_op, dst_op = pl.operands
                value = _as_int(imm_op)
                if not (0 <= value < (1 << 27)):
                    raise AsmError(
                        f"line {pl.lineno}: movi_i32 value {value} exceeds the "
                        "provisional 27-bit immediate path (real hardware "
                        "immediate encoding is unknown -- see "
                        "docs/isa/OPEN_QUESTIONS.md item 2)"
                    )
                low16 = value & 0xFFFF
                hi11 = (value >> 16) & 0x7FF
                out.append(_mk(pl, "ploadu_l", [Operand("imm", low16)]))
                out.append(_mk(pl, "loadu_h", [Operand("imm", hi11), dst_op]))
            else:
                out.append(pl)
        return out

    def _expand_movi_u16(self, pl: ParsedLine, bits: int, dst_op: Operand):
        return [
            _mk(pl, "ploadu_l", [Operand("imm", bits & 0xFFFF)]),
            _mk(pl, "loadu_h", [Operand("imm", 0), dst_op]),
        ]

    # ------------------------------------------------------------------
    # Pass 2: label resolution + encoding
    # ------------------------------------------------------------------
    def assemble(self, text: str) -> list[int]:
        raw_lines = text.splitlines()
        parsed = [parse_line(i + 1, l) for i, l in enumerate(raw_lines)]
        parsed = [p for p in parsed if p is not None]
        expanded = self._expand_macros(parsed)

        labels: dict[str, int] = {}
        pc = 0
        instr_lines: list[ParsedLine] = []
        for pl in expanded:
            if pl.label:
                labels[pl.label] = pc
            if pl.mnemonic:
                instr_lines.append(pl)
                pc += 1

        words: list[int] = []
        pending_hi = {"srcA": False, "srcB": False, "dst": False}
        for idx, pl in enumerate(instr_lines):
            if pl.mnemonic == "regbank":
                if len(pl.operands) != 3:
                    raise AsmError(f"line {pl.lineno}: regbank takes 3 bit operands")
                bits = [bool(_as_int(o)) for o in pl.operands]
                word = self._encode_regbank(pl, bits)
                words.append(word)
                pending_hi = {"srcA": bits[0], "srcB": bits[1], "dst": bits[2]}
                continue

            word = self._encode(pl, idx, labels, pending_hi)
            words.append(word)
            pending_hi = {"srcA": False, "srcB": False, "dst": False}
        return words

    def _encode_regbank(self, pl: ParsedLine, bits: list[bool]) -> int:
        f = self.isa.fields
        word = 0
        word = f["opcode"].insert(word, self.isa.opcode_by_channel[0])
        word = f["subop"].insert(word, self.isa.instructions["regbank"].subop)
        word = f["srcA"].insert(word, int(bits[0]))
        word = f["srcB"].insert(word, int(bits[1]))
        word = f["dst"].insert(word, int(bits[2]))
        return word

    def _encode(self, pl: ParsedLine, idx: int, labels: dict[str, int], pending_hi: dict) -> int:
        isa = self.isa
        f = isa.fields
        instr = isa.instructions.get(pl.mnemonic)
        if instr is None:
            raise AsmError(f"line {pl.lineno}: unknown mnemonic '{pl.mnemonic}'")
        if instr.channel is None:
            raise AsmError(
                f"line {pl.lineno}: '{pl.mnemonic}' has no assigned opcode channel "
                "(see docs/isa/OPEN_QUESTIONS.md)"
            )

        word = 0
        word = f["opcode"].insert(word, instr.opcode)
        word = f["subop"].insert(word, instr.subop)

        if pl.cond_cr is not None:
            word = f["cond_enable"].insert(word, 1)
            word = f["cond_invert"].insert(word, 1 if pl.cond_invert else 0)
            word = f["cond_cr"].insert(word, pl.cond_cr)

        roles = list(instr.operands)
        operands = list(pl.operands)
        if len(roles) != len(operands):
            raise AsmError(
                f"line {pl.lineno}: '{pl.mnemonic}' expects {len(roles)} operand(s) "
                f"({roles}), got {len(operands)}"
            )

        if roles == ["target"]:
            (op,) = operands
            target_val = labels[op.value] - (idx + 1) if op.kind == "label" else _as_int(op)
            width = isa.control_flow_target_width
            span = 1 << width
            if not (-(span // 2) <= target_val < span // 2):
                raise AsmError(f"line {pl.lineno}: branch target {target_val} out of range for {width}-bit field")
            raw = target_val & (span - 1)
            # subop[15:10] must stay intact (already inserted above) to
            # distinguish goto/call within their shared channel, so the
            # 16-bit target is split around it: high 6 bits into
            # srcA_select+srcA, low 10 bits into srcB+dst. See
            # control_flow_target in isa.yaml.
            hi6 = (raw >> 10) & 0x3F
            lo10 = raw & 0x3FF
            word = f["srcA_select"].insert(word, (hi6 >> 5) & 1)
            word = f["srcA"].insert(word, hi6 & 0x1F)
            word = f["srcB"].insert(word, (lo10 >> 5) & 0x1F)
            word = f["dst"].insert(word, lo10 & 0x1F)
            return word

        if roles and roles[0] in ("imm16",) and instr.mnemonic == "ploadu_l":
            (op,) = operands
            value = _as_int(op)
            word = f["srcA_select"].insert(word, (value >> 15) & 1)
            word = f["srcA"].insert(word, (value >> 10) & 0x1F)
            word = f["srcB"].insert(word, (value >> 5) & 0x1F)
            word = f["dst"].insert(word, value & 0x1F)
            return word

        if instr.mnemonic == "loadu_h":
            imm_op, dst_op = operands
            imm_hi = _as_int(imm_op)
            if not (0 <= imm_hi < (1 << 11)):
                raise AsmError(f"line {pl.lineno}: loadu_h immediate {imm_hi} exceeds 11-bit provisional field")
            word = f["srcA_select"].insert(word, (imm_hi >> 10) & 1)
            word = f["srcA"].insert(word, (imm_hi >> 5) & 0x1F)
            word = f["srcB"].insert(word, imm_hi & 0x1F)
            dst_idx, dst_hi = self._reg_index(dst_op, "dst", pending_hi, pl)
            word = f["dst"].insert(word, dst_idx)
            return word

        # Generic encoding: any of {dst, cr_dst, vdst} always occupies the
        # physical 'dst' field (there is at most one per instruction);
        # everything else ({srcA, srcB, addr_reg, vsrc}) is a source-like
        # operand consumed in order from the physical srcA/srcB fields.
        # This is independent of where the role sits in the operand list,
        # so single-operand cr_dst instructions (e.g. v4mask_to_cr) land
        # on 'dst', not 'srcA'.
        source_slots = iter(("srcA", "srcB"))
        for role, op in zip(roles, operands):
            if role in ("dst", "cr_dst", "vdst"):
                if role == "cr_dst":
                    if op.kind != "cr":
                        raise AsmError(f"line {pl.lineno}: expected a CR operand for '{role}', got {op}")
                    word = f["dst"].insert(word, op.value)
                else:
                    idx_val, _ = self._reg_index(op, "dst", pending_hi, pl)
                    word = f["dst"].insert(word, idx_val)
                continue
            slot = next(source_slots)
            if slot == "srcA" and op.kind == "const":
                const_idx = op.value if isinstance(op.value, int) else isa.constants_by_name.get(op.value)
                if const_idx is None:
                    raise AsmError(f"line {pl.lineno}: unknown constant name '{op.value}'")
                if not (0 <= const_idx <= 7):
                    raise AsmError(
                        f"line {pl.lineno}: constant bank index {const_idx} is undocumented "
                        "(only 0-7 are specified, see docs/isa/OPEN_QUESTIONS.md item 6)"
                    )
                word = f["srcA_select"].insert(word, 0)
                word = f["srcA"].insert(word, const_idx)
                continue
            if op.kind not in ("reg", "vreg"):
                raise AsmError(f"line {pl.lineno}: expected a register operand for '{role}', got {op}")
            idx_val, hi = self._reg_index(op, slot, pending_hi, pl)
            if slot == "srcA":
                word = f["srcA_select"].insert(word, 1)
            word = f[slot].insert(word, idx_val)
        return word

    def _reg_index(self, op: Operand, slot: str, pending_hi: dict, pl: ParsedLine) -> tuple[int, bool]:
        if op.kind not in ("reg", "vreg"):
            raise AsmError(f"line {pl.lineno}: expected register, got {op}")
        full = op.value
        if op.kind == "reg" and full >= 32:
            if not pending_hi.get(slot):
                raise AsmError(
                    f"line {pl.lineno}: register r{full} needs 'regbank' hi flag "
                    f"set for the '{slot}' slot on the immediately preceding line"
                )
            return full - 32, True
        if op.kind == "reg" and full < 32 and pending_hi.get(slot):
            raise AsmError(
                f"line {pl.lineno}: regbank hi flag was set for '{slot}' but "
                f"r{full} is a low register -- this would address r{full + 32} "
                "on real hardware under our provisional model"
            )
        if full > (63 if op.kind == "reg" else self._isa_vreg_max()):
            raise AsmError(f"line {pl.lineno}: register index {full} out of range")
        return full % 32, False

    def _isa_vreg_max(self) -> int:
        return self.isa.vector_register_count - 1


def _mk(pl: ParsedLine, mnemonic: str, operands: list[Operand]) -> ParsedLine:
    return ParsedLine(pl.lineno, None, mnemonic, pl.cond_cr, pl.cond_invert, operands, pl.raw)


def _as_int(op: Operand) -> int:
    if op.kind == "imm":
        return int(op.value)
    if op.kind == "const":
        return int(op.value)
    raise AsmError(f"expected immediate, got {op}")


def _as_number(op: Operand):
    if op.kind == "imm":
        return op.value
    raise AsmError(f"expected numeric immediate, got {op}")


def assemble(text: str, isa: ISA | None = None) -> list[int]:
    return Assembler(isa).assemble(text)
