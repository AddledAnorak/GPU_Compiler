"""32-bit instruction words -> text assembly, inverse of assembler.py.

Round-tripping (assemble -> disassemble -> reassemble must be a fixed
point) is the primary correctness property tested in tests/, since we have
no reference hardware to compare against.

Disassembly is stateful across a stream of words: a `regbank` instruction's
one-shot hi-register flags are tracked and folded into the *next*
instruction's printed register numbers (e.g. dst field bits=1 with a
pending dst-hi flag prints as `r33`, not `r1`), mirroring what the
assembler consumes. Without this, disassembling and reassembling a program
that uses r32-r63 would silently drop the hi flag and change the encoded
bits -- exactly the fixed-point property this module exists to guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

from .isa import ISA

_NO_HI = {"srcA": False, "srcB": False, "dst": False}


@dataclass
class DecodedInstr:
    mnemonic: str
    text: str
    channel: int
    subop: int
    cond_enable: bool
    cond_invert: bool
    cond_cr: int


class DisasmError(Exception):
    pass


class Disassembler:
    def __init__(self, isa: ISA | None = None):
        self.isa = isa or ISA()

    def decode_one(self, word: int, pending_hi: dict | None = None) -> tuple[DecodedInstr, dict]:
        """Decode one word. `pending_hi` is the regbank state left by the
        previous instruction; returns (decoded, pending_hi_for_next)."""
        pending_hi = pending_hi or dict(_NO_HI)
        isa = self.isa
        f = isa.fields
        opcode = f["opcode"].extract(word)
        channel = None
        for ch, op in isa.opcode_by_channel.items():
            if op == opcode:
                channel = ch
                break
        if channel is None:
            raise DisasmError(f"unknown opcode {opcode:#07b}")

        if channel == 22:
            # direct control-transfer (goto/call): subop[15:10] still
            # distinguishes the two mnemonics; the 16-bit signed target is
            # split around it into a high 6-bit chunk (srcA_select+srcA)
            # and a low 10-bit chunk (srcB+dst) -- see control_flow_target
            # in isa.yaml. Channel 21 (jump/rcall) is register-indirect and
            # decodes normally below.
            subop = f["subop"].extract(word)
            mnem = isa.mnemonic_for(channel, subop)
            if mnem not in ("goto", "call"):
                raise DisasmError(f"channel {channel} has no direct branch at subop {subop}")
            hi6 = (f["srcA_select"].extract(word) << 5) | f["srcA"].extract(word)
            lo10 = (f["srcB"].extract(word) << 5) | f["dst"].extract(word)
            raw = (hi6 << 10) | lo10
            width = isa.control_flow_target_width
            if raw & (1 << (width - 1)):
                raw -= 1 << width
            cond_enable = bool(f["cond_enable"].extract(word))
            cond_invert = bool(f["cond_invert"].extract(word))
            cond_cr = f["cond_cr"].extract(word)
            suffix = f".{'ncr' if cond_invert else 'cr'}{cond_cr}" if cond_enable else ""
            decoded = DecodedInstr(mnem, f"{mnem}{suffix} {raw}", channel, subop,
                                   cond_enable, cond_invert, cond_cr)
            return decoded, dict(_NO_HI)

        subop = f["subop"].extract(word)
        mnem = isa.mnemonic_for(channel, subop)
        if mnem is None:
            raise DisasmError(f"channel {channel} has no mnemonic at subop {subop}")

        if mnem == "regbank":
            a = f["srcA"].extract(word) & 1
            b = f["srcB"].extract(word) & 1
            d = f["dst"].extract(word) & 1
            decoded = DecodedInstr(mnem, f"regbank {a}, {b}, {d}", channel, subop, False, False, 0)
            return decoded, {"srcA": bool(a), "srcB": bool(b), "dst": bool(d)}

        if mnem == "ploadu_l":
            value = (
                (f["srcA_select"].extract(word) << 15)
                | (f["srcA"].extract(word) << 10)
                | (f["srcB"].extract(word) << 5)
                | f["dst"].extract(word)
            )
            return DecodedInstr(mnem, f"ploadu_l {value:#06x}", channel, subop, False, False, 0), dict(_NO_HI)

        if mnem == "loadu_h":
            imm_hi = (
                (f["srcA_select"].extract(word) << 10)
                | (f["srcA"].extract(word) << 5)
                | f["srcB"].extract(word)
            )
            dst = self._reg_text(f["dst"].extract(word), pending_hi["dst"])
            text = f"loadu_h {imm_hi:#05x}, {dst}"
            return DecodedInstr(mnem, text, channel, subop, False, False, 0), dict(_NO_HI)

        instr = isa.instructions[mnem]
        cond_enable = bool(f["cond_enable"].extract(word))
        cond_invert = bool(f["cond_invert"].extract(word))
        cond_cr = f["cond_cr"].extract(word)

        parts = []
        source_slots = iter(("srcA", "srcB"))
        for role in instr.operands:
            if role in ("dst", "cr_dst", "vdst"):
                idx = f["dst"].extract(word)
                if role == "cr_dst":
                    parts.append(f"cr{idx}")
                elif role == "vdst":
                    parts.append(f"v{idx}")
                else:
                    parts.append(self._reg_text(idx, pending_hi["dst"]))
                continue
            slot = next(source_slots)
            if slot == "srcA" and f["srcA_select"].extract(word) == 0:
                parts.append(f"#{f['srcA'].extract(word)}")
                continue
            idx = f[slot].extract(word)
            if role == "vsrc":
                parts.append(f"v{idx}")
            else:
                parts.append(self._reg_text(idx, pending_hi[slot]))

        mnem_text = mnem
        if cond_enable:
            mnem_text += f".{'ncr' if cond_invert else 'cr'}{cond_cr}"
        text = mnem_text + (" " + ", ".join(parts) if parts else "")
        return DecodedInstr(mnem, text, channel, subop, cond_enable, cond_invert, cond_cr), dict(_NO_HI)

    @staticmethod
    def _reg_text(field_value: int, hi: bool) -> str:
        return f"r{field_value + 32 if hi else field_value}"

    def disassemble(self, words: list[int]) -> list[str]:
        lines = []
        pending_hi = dict(_NO_HI)
        for w in words:
            decoded, pending_hi = self.decode_one(w, pending_hi)
            lines.append(decoded.text)
        return lines


def disassemble(words: list[int], isa: ISA | None = None) -> list[str]:
    return Disassembler(isa).disassemble(words)
