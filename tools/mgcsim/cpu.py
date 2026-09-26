"""Fetch/decode/execute loop for MGC-FP16-v1.

This is the load-bearing artifact of Weeks 1-4: with no board and no RTL
available, it is the only way to check that assembled programs -- and
later, backend-generated code -- do what they're supposed to. Two things
it must get right to be useful at all:

  1. FP16 arithmetic must be bit-exact against IEEE-754 binary16
     (delegated to mgcasm.fp16, which uses numpy.float16 as the
     reference).
  2. It must actively DETECT violations of the Table 10 latency contract
     (a missing/short wait_*) rather than silently tolerating them, so a
     backend scheduler bug shows up as a test failure instead of
     mystifying hardware misbehavior six weeks from now.

Every instruction's semantics implemented here beyond the literal spec
text (secs. 5, 8, 9, 11) is a guess -- see docs/isa/ENCODING_GUESSES.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mgcasm.fp16 import (
    fp16_add,
    fp16_bits_to_float,
    fp16_div,
    fp16_mul,
    fp16_sub,
    float_to_fp16_bits,
)
from mgcasm.isa import ISA

from .memory import Memory

_NO_HI = {"srcA": False, "srcB": False, "dst": False}


class MgcSimError(Exception):
    pass


class LatencyViolation(MgcSimError):
    pass


@dataclass
class Track:
    """One outstanding-result identity: a register or a condition register."""

    kind: str  # 'r' or 'cr'
    index: int

    def __hash__(self):
        return hash((self.kind, self.index))


@dataclass
class CPU:
    isa: ISA = field(default_factory=ISA)
    max_cycles: int = 1_000_000

    def __post_init__(self):
        self.regs = [0] * self.isa.register_count
        self.cr = [False] * len(self.isa.condition_registers)
        self.vregs = [[0, 0, 0, 0] for _ in range(self.isa.vector_register_count)]
        self.vmask = [False, False, False, False]
        self.mem = Memory()
        self.pc = 0
        self.halted = False
        self.stage_low = 0  # hidden latch written by ploadu_l
        self.load_addr_latch = 0
        self.store_addr_latch = 0
        self.call_stack: list[int] = []
        self.pending_hi = dict(_NO_HI)
        # outstanding[latency_class] = set of Track currently unsafe to read
        self.outstanding: dict[str, set[Track]] = {lc: set() for lc in self.isa.latency_classes}
        self.cycles = 0

    # ------------------------------------------------------------------
    # Register file helpers
    # ------------------------------------------------------------------
    def _get_reg(self, idx: int) -> int:
        self._check_read(Track("r", idx))
        return self.regs[idx] & 0xFFFFFFFF

    def _set_reg(self, idx: int, value: int) -> None:
        self.regs[idx] = value & 0xFFFFFFFF
        # A fresh write retires any latency hazard the *previous* occupant
        # of this register was tracked under -- e.g. an unrelated `add`
        # overwriting r2 while an earlier fmul16's result in r2 is still
        # outstanding should not leave a stale hazard that a later,
        # unrelated read of r2 (now the add's result) trips over. The
        # instruction doing this write re-marks the track itself right
        # after calling _set_reg if its own result has a latency class.
        track = Track("r", idx)
        for pending in self.outstanding.values():
            pending.discard(track)

    def _get_cr(self, idx: int) -> bool:
        self._check_read(Track("cr", idx))
        return self.cr[idx]

    def _check_read(self, track: Track) -> None:
        for lc, pending in self.outstanding.items():
            if track in pending:
                wait_mnem = self.isa.latency_classes[lc]["wait_mnemonic"]
                raise LatencyViolation(
                    f"cycle {self.cycles}: read of {track.kind}{track.index} before "
                    f"'{wait_mnem}' following a '{lc}' operation -- see Table 10 in "
                    "FP16_ISA_V1.pdf"
                )

    def _mark_outstanding(self, latency_class: str, track: Track) -> None:
        if not latency_class:
            return
        # Multiple ops of the same class may be in flight at once with no
        # wait between issuing them -- FP16_ISA_V1.pdf sec. 11.3 gives
        # exactly this pattern (icmp_ge then icmp_lt, no wait). A single
        # wait_<class> is modeled as a conservative barrier that clears
        # every outstanding op of that class at once, not a per-op stall,
        # matching the "N safe cycles" wording in Table 10. Reading a
        # result before the matching wait fires is still a violation --
        # see _check_read.
        self.outstanding[latency_class].add(track)

    def _do_wait(self, latency_class: str | None) -> None:
        if latency_class is None:
            for pending in self.outstanding.values():
                pending.clear()
        else:
            self.outstanding[latency_class].clear()

    # ------------------------------------------------------------------
    # Fetch / decode
    # ------------------------------------------------------------------
    def _channel_of(self, opcode: int) -> int:
        for ch, op in self.isa.opcode_by_channel.items():
            if op == opcode:
                return ch
        raise MgcSimError(f"unknown opcode {opcode:#07b}")

    def step(self, word: int) -> None:
        isa = self.isa
        f = isa.fields
        opcode = f["opcode"].extract(word)
        channel = self._channel_of(opcode)

        if channel == 22:
            self._exec_direct_branch(word, channel)
            return

        subop = f["subop"].extract(word)
        mnem = isa.mnemonic_for(channel, subop)
        if mnem is None:
            raise MgcSimError(f"channel {channel} has no mnemonic at subop {subop}")

        if mnem == "regbank":
            a = f["srcA"].extract(word) & 1
            b = f["srcB"].extract(word) & 1
            d = f["dst"].extract(word) & 1
            self.pending_hi = {"srcA": bool(a), "srcB": bool(b), "dst": bool(d)}
            self.pc += 1
            return

        if mnem == "ploadu_l":
            value = (
                (f["srcA_select"].extract(word) << 15)
                | (f["srcA"].extract(word) << 10)
                | (f["srcB"].extract(word) << 5)
                | f["dst"].extract(word)
            )
            self.stage_low = value
            self.pending_hi = dict(_NO_HI)
            self.pc += 1
            return

        if mnem == "loadu_h":
            imm_hi = (
                (f["srcA_select"].extract(word) << 10)
                | (f["srcA"].extract(word) << 5)
                | f["srcB"].extract(word)
            )
            dst = self._resolve_reg_field(f["dst"].extract(word), "dst")
            self._set_reg(dst, (imm_hi << 16) | self.stage_low)
            self.pending_hi = dict(_NO_HI)
            self.pc += 1
            return

        # Conditional execution gate. Reading the CR itself is checked for
        # hazards; a false predicate skips all effects but still retires
        # the one-shot regbank state and advances PC.
        cond_enable = f["cond_enable"].extract(word)
        if cond_enable:
            cond_invert = f["cond_invert"].extract(word)
            cond_cr = f["cond_cr"].extract(word)
            predicate = self._get_cr(cond_cr)
            if cond_invert:
                predicate = not predicate
            if not predicate:
                self.pending_hi = dict(_NO_HI)
                self.pc += 1
                return

        self._exec(mnem, word)
        # every non-branch path above either returns early or falls
        # through to here; branch mnemonics set self.pc themselves.
        if mnem not in ("jump", "rcall", "ret", "stop_core"):
            self.pc += 1
        self.pending_hi = dict(_NO_HI)

    def _resolve_reg_field(self, field_value: int, slot: str) -> int:
        hi = self.pending_hi.get(slot, False)
        return field_value + 32 if hi else field_value

    def _read_operand(self, isa_field_word: int, role: str, slot: str):
        f = self.isa.fields
        if role in ("srcA", "srcB", "addr_reg") and slot == "srcA" and f["srcA_select"].extract(isa_field_word) == 0:
            const_idx = f["srcA"].extract(isa_field_word)
            entry = self.isa.constants_by_index.get(const_idx)
            if entry is None or entry.get("bits") is None:
                raise MgcSimError(f"constant bank index {const_idx} is undocumented (see OPEN_QUESTIONS.md item 6)")
            return entry["bits"]
        idx = self._resolve_reg_field(f[slot].extract(isa_field_word), slot)
        return self._get_reg(idx)

    def _exec(self, mnem: str, word: int) -> None:
        isa = self.isa
        f = isa.fields
        instr = isa.instructions[mnem]
        roles = instr.operands

        # ---- FP16 arithmetic ----
        if mnem in ("fadd16", "fsub16", "fmul16", "fdiv16", "fmulp2_16", "fmin16", "fmax16"):
            a = self._read_operand(word, "srcA", "srcA") & 0xFFFF
            b = self._get_reg(self._resolve_reg_field(f["srcB"].extract(word), "srcB")) & 0xFFFF
            dst = self._resolve_reg_field(f["dst"].extract(word), "dst")
            if mnem == "fadd16":
                r = fp16_add(a, b)
            elif mnem == "fsub16":
                r = fp16_sub(a, b)
            elif mnem in ("fmul16", "fmulp2_16"):
                # fmulp2_16's distinct meaning ("multiply by power of 2"?)
                # is not documented beyond its name; treated as a plain
                # multiply pending clarification.
                r = fp16_mul(a, b)
            elif mnem == "fdiv16":
                r = fp16_div(a, b)
            elif mnem == "fmin16":
                r = a if fp16_bits_to_float(a) <= fp16_bits_to_float(b) else b
            else:  # fmax16
                r = a if fp16_bits_to_float(a) >= fp16_bits_to_float(b) else b
            self._set_reg(dst, r)  # FP16 write zero-extends [31:16] per sec. 5
            self._mark_outstanding(instr.latency_class, Track("r", dst))
            return

        # ---- Move and sign ----
        if mnem in ("copy", "fabs16", "fneg16", "fnabs16"):
            a = self._read_operand(word, "srcA", "srcA")
            dst = self._resolve_reg_field(f["dst"].extract(word), "dst")
            if mnem == "copy":
                self._set_reg(dst, a)
            else:
                bits = a & 0xFFFF
                if mnem == "fabs16":
                    bits &= 0x7FFF
                elif mnem == "fneg16":
                    bits ^= 0x8000
                elif mnem == "fnabs16":
                    bits |= 0x8000
                self._set_reg(dst, bits)
            return

        # ---- Rounding ----
        if mnem in ("trunc16", "round16", "ceil16", "floor16"):
            import math

            a = self._read_operand(word, "srcA", "srcA") & 0xFFFF
            dst = self._resolve_reg_field(f["dst"].extract(word), "dst")
            v = fp16_bits_to_float(a)
            if not math.isfinite(v):
                # Rounding NaN or infinity leaves the special value in
                # place; Python's math.floor/ceil instead raise errors.
                self._set_reg(dst, a)
                self._mark_outstanding(instr.latency_class, Track("r", dst))
                return
            fn = {"trunc16": math.trunc, "round16": round, "ceil16": math.ceil, "floor16": math.floor}[mnem]
            self._set_reg(dst, float_to_fp16_bits(float(fn(v))))
            self._mark_outstanding(instr.latency_class, Track("r", dst))
            return

        # ---- Conversions ----
        if mnem in (
            "f16tou8_ll", "f16tou8_lh", "f16tou8_hl", "f16tou8_hh",
            "f16tou16_l", "f16tou16_h", "f16toi32",
            "u8tof16_ll", "u8tof16_lh", "u8tof16_hl", "u8tof16_hh",
            "u16tof16_l", "u16tof16_h", "i32tof16", "u32tof16",
        ):
            a = self._read_operand(word, "srcA", "srcA")
            dst = self._resolve_reg_field(f["dst"].extract(word), "dst")
            self._set_reg(dst, self._convert(mnem, a))
            self._mark_outstanding(instr.latency_class, Track("r", dst))
            return

        # ---- FP compare-to-CR ----
        if mnem.startswith("fcomp_"):
            a = self._read_operand(word, "srcA", "srcA") & 0xFFFF
            b = self._get_reg(self._resolve_reg_field(f["srcB"].extract(word), "srcB")) & 0xFFFF
            cr = f["dst"].extract(word)
            self.cr[cr] = self._fcompare(mnem, a, b)
            self._mark_outstanding(instr.latency_class, Track("cr", cr))
            return

        # ---- Integer ----
        if mnem in ("add", "and"):
            a = self._read_operand(word, "srcA", "srcA")
            b = self._get_reg(self._resolve_reg_field(f["srcB"].extract(word), "srcB"))
            dst = self._resolve_reg_field(f["dst"].extract(word), "dst")
            r = (a + b) if mnem == "add" else (a & b)
            self._set_reg(dst, r)
            self._mark_outstanding(instr.latency_class, Track("r", dst))
            return
        if mnem == "istep4":
            a = self._read_operand(word, "srcA", "srcA")
            dst = self._resolve_reg_field(f["dst"].extract(word), "dst")
            self._set_reg(dst, a + 4)
            self._mark_outstanding(instr.latency_class, Track("r", dst))
            return
        if mnem == "vset4":
            base = self._get_reg(self._resolve_reg_field(f["srcA"].extract(word), "srcA"))
            delta = self._get_reg(self._resolve_reg_field(f["srcB"].extract(word), "srcB"))
            vdst = f["dst"].extract(word)
            base_s = _to_signed32(base)
            delta_s = _to_signed32(delta)
            self.vregs[vdst] = [_to_unsigned32(base_s + i * delta_s) for i in range(4)]
            return

        # ---- Signed integer compare-to-CR ----
        if mnem.startswith("icmp_"):
            a = _to_signed32(self._read_operand(word, "srcA", "srcA"))
            b = _to_signed32(self._get_reg(self._resolve_reg_field(f["srcB"].extract(word), "srcB")))
            cr = f["dst"].extract(word)
            ops = {
                "icmp_lt": a < b, "icmp_le": a <= b, "icmp_eq": a == b,
                "icmp_ge": a >= b, "icmp_gt": a > b, "icmp_ne": a != b,
            }
            self.cr[cr] = ops[mnem]
            self._mark_outstanding(instr.latency_class, Track("cr", cr))
            return

        # ---- Memory ----
        if mnem == "load_addr":
            self.load_addr_latch = self._read_operand(word, "srcA", "srcA")
            return
        if mnem == "store_addr":
            self.store_addr_latch = self._read_operand(word, "srcA", "srcA")
            return
        if mnem == "load":
            addr = self._read_operand(word, "srcA", "srcA")
            dst = self._resolve_reg_field(f["dst"].extract(word), "dst")
            self._set_reg(dst, self.mem.load(addr))
            self._mark_outstanding(instr.latency_class, Track("r", dst))
            return
        if mnem == "i_load":
            dst = self._resolve_reg_field(f["dst"].extract(word), "dst")
            self._set_reg(dst, self.mem.load(self.load_addr_latch))
            self._mark_outstanding(instr.latency_class, Track("r", dst))
            return
        if mnem == "pop":
            raise MgcSimError("pop: stack data model unspecified, see OPEN_QUESTIONS.md item 8")
        if mnem == "store":
            src = self._read_operand(word, "srcA", "srcA")
            addr = self._get_reg(self._resolve_reg_field(f["srcB"].extract(word), "srcB"))
            self.mem.store(addr, src)
            return
        if mnem == "i_store":
            src = self._read_operand(word, "srcA", "srcA")
            self.mem.store(self.store_addr_latch, src)
            return
        if mnem == "push":
            raise MgcSimError("push: stack data model unspecified, see OPEN_QUESTIONS.md item 8")

        # ---- Vector helpers ----
        if mnem in ("v4cmp_ge_zero", "v4cmp_ge_zero_and", "v4cmp_ge_zero_step", "v4cmp_ge_zero_step_and_p1", "vcmp_ge_zero_i32"):
            vsrc = f["srcA"].extract(word)
            lanes = [_to_signed32(x) >= 0 for x in self.vregs[vsrc]]
            if mnem == "v4cmp_ge_zero_and":
                self.vmask = [a and b for a, b in zip(self.vmask, lanes)]
            else:
                self.vmask = lanes
            return
        if mnem == "v4mask_to_cr":
            cr = f["dst"].extract(word)
            self.cr[cr] = any(self.vmask)
            return
        if mnem in ("vstore_mask", "v4cr_store", "v4cr_store_step"):
            vsrc = f["srcA"].extract(word)
            addr = self._get_reg(self._resolve_reg_field(f["srcB"].extract(word), "srcB"))
            for lane, (val, keep) in enumerate(zip(self.vregs[vsrc], self.vmask)):
                if keep:
                    self.mem.store(addr + lane, val)
            return
        if mnem == "vbcast":
            a = self._read_operand(word, "srcA", "srcA")
            vdst = f["dst"].extract(word)
            self.vregs[vdst] = [a, a, a, a]
            return
        if mnem == "vadd_i32":
            raise MgcSimError("vadd_i32: operand encoding unspecified, see docs/isa/OPEN_QUESTIONS.md item 5")

        # ---- Control flow (indirect) ----
        if mnem == "jump":
            target = self._get_reg(self._resolve_reg_field(f["srcA"].extract(word), "srcA"))
            self.pc = target
            return
        if mnem == "rcall":
            target = self._get_reg(self._resolve_reg_field(f["srcA"].extract(word), "srcA"))
            self.call_stack.append(self.pc + 1)
            self.pc = target
            return
        if mnem == "ret":
            if not self.call_stack:
                raise MgcSimError("ret with empty call stack")
            self.pc = self.call_stack.pop()
            return
        if mnem == "stop_core":
            self.halted = True
            return
        if mnem in ("wait", "wait_short", "wait_load", "wait_fcomp", "wait_fadd", "wait_fmul", "wait_fdiv", "wait_conv"):
            lc = None if mnem == "wait" else {
                "wait_short": "short", "wait_load": "load", "wait_fcomp": "fp_compare",
                "wait_fadd": "fp16_addsub", "wait_fmul": "fp16_multiply",
                "wait_fdiv": "fp16_divide", "wait_conv": "conversion",
            }[mnem]
            self._do_wait(lc)
            return
        if mnem == "nop":
            return

        raise MgcSimError(f"'{mnem}' has no emulator implementation yet")

    def _exec_direct_branch(self, word: int, channel: int) -> None:
        f = self.isa.fields
        subop = f["subop"].extract(word)
        mnem = self.isa.mnemonic_for(channel, subop)
        hi6 = (f["srcA_select"].extract(word) << 5) | f["srcA"].extract(word)
        lo10 = (f["srcB"].extract(word) << 5) | f["dst"].extract(word)
        raw = (hi6 << 10) | lo10
        width = self.isa.control_flow_target_width
        if raw & (1 << (width - 1)):
            raw -= 1 << width
        target = self.pc + 1 + raw
        if mnem == "call":
            self.call_stack.append(self.pc + 1)
        self.pc = target

    def _convert(self, mnem: str, a: int) -> int:
        if mnem == "f16toi32":
            return _to_unsigned32(int(fp16_bits_to_float(a & 0xFFFF)))
        if mnem in ("i32tof16",):
            return float_to_fp16_bits(float(_to_signed32(a)))
        if mnem == "u32tof16":
            return float_to_fp16_bits(float(a & 0xFFFFFFFF))
        if mnem.startswith("f16tou8_"):
            byte_sel = mnem[-2:]
            v = int(max(0.0, min(255.0, fp16_bits_to_float(a & 0xFFFF))))
            shift = {"ll": 0, "lh": 8, "hl": 16, "hh": 24}[byte_sel]
            return v << shift
        if mnem.startswith("f16tou16_"):
            half = mnem[-1]
            v = int(max(0.0, min(65535.0, fp16_bits_to_float(a & 0xFFFF))))
            return v << (16 if half == "h" else 0)
        if mnem.startswith("u8tof16_"):
            byte_sel = mnem[-2:]
            shift = {"ll": 0, "lh": 8, "hl": 16, "hh": 24}[byte_sel]
            byte_val = (a >> shift) & 0xFF
            return float_to_fp16_bits(float(byte_val))
        if mnem.startswith("u16tof16_"):
            half = mnem[-1]
            half_val = (a >> 16) & 0xFFFF if half == "h" else a & 0xFFFF
            return float_to_fp16_bits(float(half_val))
        raise MgcSimError(f"unhandled conversion '{mnem}'")

    @staticmethod
    def _fcompare(mnem: str, a: int, b: int) -> bool:
        fa, fb = fp16_bits_to_float(a), fp16_bits_to_float(b)
        import math

        nan = math.isnan(fa) or math.isnan(fb)
        if mnem == "fcomp_o":
            return not nan
        if mnem == "fcomp_u":
            return nan
        if nan:
            return False
        return {
            "fcomp_l": fa < fb, "fcomp_le": fa <= fb, "fcomp_e": fa == fb,
            "fcomp_ge": fa >= fb, "fcomp_g": fa > fb, "fcomp_ne": fa != fb,
        }[mnem]

    # ------------------------------------------------------------------
    def run(self, program: list[int], entry: int = 0) -> None:
        self.pc = entry
        self.halted = False
        for self.cycles in range(self.max_cycles):
            if self.halted or self.pc >= len(program):
                return
            self.step(program[self.pc])
        raise MgcSimError(f"exceeded max_cycles={self.max_cycles} without stop_core")


def _to_signed32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x8000_0000 else v


def _to_unsigned32(v: int) -> int:
    return v & 0xFFFFFFFF
