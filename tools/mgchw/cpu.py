from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from tools.mgcasm.fp16 import (
    fp16_add, fp16_bits_to_float, fp16_mul, fp16_sub, float_to_fp16_bits,
)
from .encoding import HardwareEncodingError, Instruction, decode_word


class MemoryAccessError(ValueError):
    pass


class StepLimitExceeded(RuntimeError):
    pass


class UnsupportedHardwareSemantics(RuntimeError):
    pass


@dataclass(frozen=True)
class MemoryRegion:
    base: int
    size: int
    writable: bool
    name: str


class ByteMemory:
    def __init__(self):
        self.regions: list[MemoryRegion] = []
        self.words: dict[int, int] = {}

    def map_region(self, base: int, size: int, *, writable: bool = True, name: str = "buffer") -> None:
        if base < 0 or size <= 0 or base % 4 or size % 4 or base + size > 1 << 32:
            raise MemoryAccessError("regions must fit the 32-bit address space and be word-aligned")
        if any(base < region.base + region.size and region.base < base + size for region in self.regions):
            raise MemoryAccessError("overlapping memory regions")
        self.regions.append(MemoryRegion(base, size, writable, name))

    def _check(self, address: int, write: bool = False) -> None:
        if address % 4:
            raise MemoryAccessError(f"unaligned byte address 0x{address:x}")
        for region in self.regions:
            if region.base <= address and address + 4 <= region.base + region.size:
                if write and not region.writable:
                    raise MemoryAccessError(f"write to read-only {region.name}")
                return
        raise MemoryAccessError(f"unmapped byte address 0x{address:x}")

    def read32(self, address: int) -> int:
        self._check(address)
        return self.words.get(address, 0)

    def write32(self, address: int, value: int) -> None:
        self._check(address, True)
        if not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
            raise MemoryAccessError("memory values must be unsigned 32-bit integers")
        self.words[address] = value

    def load_words(self, words: list[int], base: int = 0x100000, *, name: str = "sample data") -> None:
        self.map_region(base, len(words) * 4, name=name)
        for index, value in enumerate(words):
            self.write32(base + index * 4, value)


def signed32(value: int) -> int:
    return value - (1 << 32) if value & (1 << 31) else value


class HardwareCPU:
    """Functional model of observed instructions, not an RTL timing model."""

    def __init__(self, *, memory: ByteMemory | None = None,
                 allow_approximate_fp16: bool = False):
        self.memory = memory if memory is not None else ByteMemory()
        self.registers = [0] * 32
        self.vectors = [[0] * 4 for _ in range(8)]
        self.conditions = [False] * 8
        self.allow_approximate_fp16 = allow_approximate_fp16
        self.pc = 0
        self.steps = 0
        self.halted = False
        self.preload = 0
        self.program: list[Instruction] = []
        self.instruction_counts: Counter[str] = Counter()

    def load_program(self, words: list[int], *, reset: bool = True) -> None:
        if not words:
            raise HardwareEncodingError("cannot execute an empty hardware program")
        self.program = [decode_word(word) for word in words]
        if reset:
            self.pc = 0
            self.steps = 0
            self.halted = False
            self.preload = 0
            self.registers = [0] * 32
            self.vectors = [[0] * 4 for _ in range(8)]
            self.conditions = [False] * 8
            self.instruction_counts.clear()

    def _fp16_enabled(self) -> None:
        if not self.allow_approximate_fp16:
            raise UnsupportedHardwareSemantics(
                "vector FP16/RGB565 requires allow_approximate_fp16=True; RTL rounding is unverified"
            )

    def step(self) -> None:
        if not self.program:
            raise HardwareEncodingError("no hardware program loaded")
        if self.halted:
            return
        if not 0 <= self.pc < len(self.program):
            self.halted = True
            return
        instruction = self.program[self.pc]
        self.pc += 1
        self.steps += 1
        if instruction.condition is not None:
            enabled = self.conditions[instruction.condition] != instruction.invert
            if not enabled:
                self.instruction_counts["predicated-off"] += 1
                return
        mnemonic = instruction.mnemonic
        self.instruction_counts[mnemonic] += 1
        source_a, source_b, destination = instruction.source_a, instruction.source_b, instruction.destination
        registers, vectors = self.registers, self.vectors
        if mnemonic == "nop":
            return
        if mnemonic == "preload_low":
            self.preload = instruction.immediate
        elif mnemonic == "load_high":
            registers[destination] = instruction.immediate << 16 | self.preload
        elif mnemonic == "jump":
            self.pc += instruction.branch_offset
            if not 0 <= self.pc <= len(self.program):
                raise HardwareEncodingError(f"branch target {self.pc} outside instruction memory")
        elif mnemonic == "copy":
            registers[destination] = registers[source_a]
        elif mnemonic == "add":
            registers[destination] = (registers[source_a] + registers[source_b]) & 0xFFFFFFFF
        elif mnemonic == "istep4":
            registers[destination] = (registers[source_a] + registers[source_b] * 4) & 0xFFFFFFFF
        elif mnemonic.startswith("icmp_"):
            left, right = signed32(registers[source_a]), signed32(registers[source_b])
            self.conditions[destination] = {
                "icmp_lt": left < right, "icmp_ge": left >= right, "icmp_gt": left > right,
            }[mnemonic]
        elif mnemonic == "load":
            registers[destination] = self.memory.read32(registers[source_b])
        elif mnemonic == "store":
            self.memory.write32(registers[source_b], registers[source_a])
        elif mnemonic == "vset4":
            base, delta = registers[source_a], registers[source_b]
            vectors[destination] = [(base + lane * delta) & 0xFFFFFFFF for lane in range(4)]
        elif mnemonic == "vbcast":
            vectors[destination] = [registers[source_a]] * 4
        elif mnemonic == "vadd_i32":
            vectors[destination] = [(left + right) & 0xFFFFFFFF
                                    for left, right in zip(vectors[source_a], vectors[source_b])]
        elif mnemonic == "vget_lane0_i32":
            registers[destination] = vectors[source_a][0]
        elif mnemonic in {"vcmp_ge_zero_i32_mask", "vcmp_ge_zero_i32_and"}:
            mask = sum(1 << lane for lane, value in enumerate(vectors[source_a]) if signed32(value) >= 0)
            registers[destination] = mask if mnemonic.endswith("_mask") else mask & registers[source_b]
        elif mnemonic in {"vstore_mask_r", "vstore_mask_vr"}:
            values = [registers[source_a]] * 4 if mnemonic.endswith("_r") else vectors[source_a]
            writes = [(address, value) for lane, (address, value) in enumerate(zip(vectors[destination], values))
                      if registers[source_b] & (1 << lane)]
            for address, _ in writes:
                self.memory._check(address, True)
            for address, value in writes:
                self.memory.write32(address, value)
        elif mnemonic == "fadd16":
            constants = [0xFC00, 0xBC00, 0, 0x3C00, 0x3800, 0x4248, 0x7C00, 0x7E00]
            if not instruction.source_a_register and source_a >= len(constants):
                raise HardwareEncodingError("undocumented FP16 constant index")
            left = registers[source_a] if instruction.source_a_register else constants[source_a]
            registers[destination] = fp16_add(left, registers[source_b])
        elif mnemonic == "vfset4":
            self._fp16_enabled()
            value = registers[source_a] & 65535
            values = []
            for _ in range(4):
                values.append(value)
                value = fp16_add(value, registers[source_b])
            vectors[destination] = values
        elif mnemonic in {"vfadd16", "vfsub16", "vfmul16", "vfmax16"}:
            self._fp16_enabled()
            operation = {"vfadd16": fp16_add, "vfsub16": fp16_sub, "vfmul16": fp16_mul}.get(mnemonic)
            if operation:
                vectors[destination] = [operation(left, right)
                                        for left, right in zip(vectors[source_a], vectors[source_b])]
            else:
                vectors[destination] = [float_to_fp16_bits(max(fp16_bits_to_float(left), fp16_bits_to_float(right)))
                                        for left, right in zip(vectors[source_a], vectors[source_b])]
        elif mnemonic == "vrgb565_ceramic":
            self._fp16_enabled()
            colors = []
            for value in vectors[source_a]:
                intensity = fp16_bits_to_float(value)
                if math.isnan(intensity):
                    raise UnsupportedHardwareSemantics("NaN RGB565 conversion is not specified")
                intensity = max(0.0, min(1.0, intensity))
                colors.append(int(intensity * 31) << 11 | int(intensity * 63) << 5 | int(intensity * 31))
            vectors[destination] = colors
        else:
            raise UnsupportedHardwareSemantics(f"execution of {mnemonic} is not implemented")

    def run(self, words: list[int] | None = None, *, max_steps: int = 1000000,
            stop_pc: int | None = None, skip_pcs: dict[int, int] | None = None) -> int:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if words is not None:
            self.load_program(words)
        if stop_pc is not None and not 0 <= stop_pc <= len(self.program):
            raise ValueError("stop_pc outside program")
        for origin, target in (skip_pcs or {}).items():
            if not 0 <= origin < len(self.program) or not 0 <= target < len(self.program):
                raise ValueError("invalid explicit host-only skip")
        initial_steps = self.steps
        host_skips = 0
        while not self.halted and self.pc != stop_pc:
            if self.steps - initial_steps + host_skips >= max_steps:
                raise StepLimitExceeded(f"host instruction budget {max_steps} exhausted at pc={self.pc}")
            if skip_pcs and self.pc in skip_pcs:
                self.pc = skip_pcs[self.pc]
                self.instruction_counts["host-only-skip"] += 1
                host_skips += 1
            else:
                self.step()
        return self.steps - initial_steps
