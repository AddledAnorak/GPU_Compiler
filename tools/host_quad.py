"""Explicit cooperative legacy-host quads, never a device instruction or ABI."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tools.mgcasm.fp16 import float_to_fp16_bits, fp16_bits_to_float
from tools.mgcdriver.backends import ExecutionResult, SimulationBackend
from tools.mgcdriver.errors import ExecutionTimeoutError, ResourceError, SubmissionError, ValidationError
from tools.mgcdriver.program import ADDRESS_LIMIT, integer


REQUEST_WORD = 65496
OPERAND_WORD = 65497
RESPONSE_WORD = 65498


@dataclass(frozen=True)
class LaneRegion:
    address: int
    buffer: object
    writable: bool = False


class _QuadMemory:
    def __init__(self, backend, lane, regions):
        self.backend = backend
        self.lane = lane
        self.regions = regions

    def locate(self, word_address, write=False):
        integer(word_address, "quad legacy word address", 0, (ADDRESS_LIMIT - 4) // 4)
        address = word_address * 4
        for region in self.regions:
            if region.address <= address and address + 4 <= region.address + region.buffer.size:
                if write and not region.writable:
                    raise ResourceError("quad shader attempted to write a read-only submitted region")
                return region.buffer, address - region.address
        raise ResourceError(f"quad shader address {address:#x} is outside submitted lane resources")

    def load(self, address):
        buffer, offset = self.locate(address)
        return int.from_bytes(buffer._data[offset:offset + 4], "little")

    def store(self, address, value):
        buffer, offset = self.locate(address, True)
        buffer._data[offset:offset + 4] = (value & 0xFFFFFFFF).to_bytes(4, "little")
        if address == self.backend.output_base + REQUEST_WORD:
            if value not in (1, 2):
                raise SubmissionError("unsupported explicit host derivative mailbox operation")
            self.backend._waiting[self.lane] = (self.backend._cpus[self.lane].pc, value)


class QuadSimulationBackend(SimulationBackend):
    name = "legacy-host-quad"

    def __init__(self, *, input_base, output_base, memory_size=32 << 20, quantum=64):
        super().__init__(memory_size=memory_size, quantum=quantum)
        integer(input_base, "quad input word base", 0, (ADDRESS_LIMIT - 65536 * 4) // 4)
        integer(output_base, "quad output word base", 0, (ADDRESS_LIMIT - 65536 * 4) // 4)
        if max(input_base, output_base) < min(input_base + 65536, output_base + 65536):
            raise ValidationError("quad input and output regions overlap")
        self.input_base = input_base
        self.output_base = output_base
        self._regions = None
        self._cpus = []
        self._waiting = [None] * 4

    def configure(self, lanes):
        if not isinstance(lanes, (tuple, list)) or len(lanes) != 4:
            raise ValidationError("host derivatives require exactly four explicit lane resource maps")
        checked = []
        for regions in lanes:
            if not isinstance(regions, (tuple, list)) or not regions:
                raise ValidationError("each quad lane needs explicit submitted regions")
            spans = []
            for region in regions:
                if not isinstance(region, LaneRegion) or type(region.writable) is not bool:
                    raise ValidationError("invalid quad lane region")
                integer(region.address, "quad virtual byte address", 0, ADDRESS_LIMIT - 4)
                if region.address % 4 or region.address + region.buffer.size > ADDRESS_LIMIT:
                    raise ValidationError("quad region must be aligned and bounded")
                spans.append((region.address, region.address + region.buffer.size))
            if any(max(left[0], right[0]) < min(left[1], right[1])
                   for index, left in enumerate(spans) for right in spans[index + 1:]):
                raise ValidationError("quad virtual lane resources overlap")
            checked.append(tuple(regions))
        self._regions = tuple(checked)

    def start(self, program, buffers, entry, max_steps):
        from tools.mgcasm.isa import ISA
        from tools.mgcsim import CPU

        if self._regions is None:
            raise ValidationError("quad resource maps must be explicitly configured")
        submitted = {id(buffer) for buffer in buffers}
        if any(id(region.buffer) not in submitted for regions in self._regions for region in regions):
            raise ResourceError("quad lane refers to an unsubmitted buffer")
        self._isa = self._isa or ISA()
        self._program = program
        self._steps = 0
        self._max_steps = max_steps
        self._waiting = [None] * 4
        self._lane_steps = [0] * 4
        self._cpus = [CPU(isa=self._isa, max_cycles=max_steps) for _ in range(4)]
        for lane, cpu in enumerate(self._cpus):
            cpu.mem = _QuadMemory(self, lane, self._regions[lane])
            cpu.pc = entry
            cpu.mem.locate(self.input_base + RESPONSE_WORD)
            cpu.mem.locate(self.output_base + REQUEST_WORD, True)
            cpu.mem.locate(self.output_base + OPERAND_WORD, True)

    def _resolve(self):
        if not all(self._waiting):
            if any(self._waiting) and all(cpu.halted or self._waiting[lane]
                                          for lane, cpu in enumerate(self._cpus)):
                raise SubmissionError("divergent quad derivative or exited helper invocation")
            return
        if len(set(self._waiting)) != 1:
            raise SubmissionError("quad lanes reached different derivative barriers")
        operation = self._waiting[0][1]
        values = [fp16_bits_to_float(cpu.mem.load(self.output_base + OPERAND_WORD) & 0xFFFF)
                  for cpu in self._cpus]
        if not all(math.isfinite(value) for value in values):
            raise SubmissionError("host derivative operands must be finite binary16")
        differences = ([values[1] - values[0]] * 2 + [values[3] - values[2]] * 2 if operation == 1
                       else [values[2] - values[0], values[3] - values[1]] * 2)
        for cpu, value in zip(self._cpus, differences):
            bits = float_to_fp16_bits(value)
            if bits & 0x7C00 == 0x7C00:
                raise SubmissionError("host derivative result overflows finite binary16")
            buffer, offset = cpu.mem.locate(self.input_base + RESPONSE_WORD)
            buffer._data[offset:offset + 4] = bits.to_bytes(4, "little")
        self._waiting = [None] * 4

    def poll(self):
        for _ in range(self.quantum):
            if all(cpu.halted for cpu in self._cpus):
                return ExecutionResult(self.name, self._steps, tuple(self._cpus[0].regs))
            for lane, cpu in enumerate(self._cpus):
                if cpu.halted or self._waiting[lane]:
                    continue
                if self._steps >= self._max_steps:
                    raise ExecutionTimeoutError(f"host quad exceeded max_steps={self._max_steps}")
                if not 0 <= cpu.pc < len(self._program.words):
                    raise SubmissionError("host quad program counter escaped the loaded program")
                cpu.cycles = self._lane_steps[lane]
                cpu.step(self._program.words[cpu.pc])
                self._lane_steps[lane] += 1
                self._steps += 1
            self._resolve()
        return None

    def quiesce(self):
        self._cpus = []
        self._waiting = [None] * 4
        self._program = None
        return True

    def finish(self, buffers):
        self.quiesce()
