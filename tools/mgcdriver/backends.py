"""Separate cooperative simulators and explicitly configured MMIO submission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .errors import (
    ExecutionTimeoutError, HardwareSafetyError, ProgramFormatError,
    ResourceError, SubmissionError, ValidationError,
)
from .program import ADDRESS_LIMIT, Program, ProgramFormat, integer


@dataclass(frozen=True)
class ExecutionResult:
    backend: str
    steps: int | None = None
    registers: tuple[int, ...] | None = None


class _LegacyMemory:
    def __init__(self, buffers):
        self.buffers = buffers

    def _locate(self, word_address):
        integer(word_address, "legacy word address", 0, (ADDRESS_LIMIT - 4) // 4)
        address = word_address * 4
        for buffer in self.buffers:
            if buffer.address <= address and address + 4 <= buffer.address + buffer.size:
                return buffer, address - buffer.address
        raise ResourceError(f"shader access at byte address {address:#x} is outside submitted buffers")

    def load(self, word_address):
        buffer, offset = self._locate(word_address)
        return int.from_bytes(buffer._data[offset:offset + 4], "little")

    def store(self, word_address, value):
        buffer, offset = self._locate(word_address)
        buffer._data[offset:offset + 4] = (value & 0xFFFFFFFF).to_bytes(4, "little")


class SimulationBackend:
    """Legacy CPU only; word-addressed instructions are bridged to byte buffers."""

    format = ProgramFormat.LEGACY
    name = "legacy-simulation"

    def __init__(self, *, memory_base: int = 0, memory_size: int = 1 << 20, quantum: int = 256):
        integer(memory_base, "memory_base", 0, ADDRESS_LIMIT - 1)
        integer(memory_size, "memory_size", 1, ADDRESS_LIMIT)
        integer(quantum, "quantum", 1, 4096)
        if memory_base % 4 or memory_base + memory_size > ADDRESS_LIMIT:
            raise ValidationError("simulation memory must have an aligned 32-bit byte address range")
        self.memory_base = memory_base
        self.memory_size = memory_size
        self.quantum = quantum
        self._cpu = None
        self._isa = None

    def validate(self, program: Program) -> None:
        if program.format != self.format:
            raise ProgramFormatError(f"{self.name} requires {self.format.value}, not {program.format.value}")

    def write(self, buffer, offset: int, data: bytes) -> None:
        buffer._data[offset:offset + len(data)] = data

    def read(self, buffer, offset: int, size: int) -> bytes:
        return bytes(buffer._data[offset:offset + size])

    def start(self, program, buffers, entry, max_steps):
        from tools.mgcasm.isa import ISA
        from tools.mgcsim import CPU

        if self._isa is None:
            self._isa = ISA()
        self._cpu = CPU(isa=self._isa, max_cycles=max_steps)
        self._cpu.mem = _LegacyMemory(buffers)
        self._cpu.pc = entry
        self._program = program
        self._steps = 0
        self._max_steps = max_steps

    def poll(self) -> ExecutionResult | None:
        for _ in range(self.quantum):
            if self._steps >= self._max_steps:
                raise ExecutionTimeoutError(f"simulation exceeded max_steps={self._max_steps}")
            if not 0 <= self._cpu.pc < len(self._program.words):
                raise SubmissionError(f"program counter {self._cpu.pc} escaped the loaded program")
            self._cpu.cycles = self._steps
            self._cpu.step(self._program.words[self._cpu.pc])
            self._steps += 1
            if self._cpu.halted:
                return ExecutionResult(self.name, self._steps, tuple(self._cpu.regs))
        return None

    def quiesce(self) -> bool:
        self._cpu = None
        self._program = None
        return True

    def idle(self) -> bool:
        return True

    def finish(self, buffers) -> None:
        self._cpu = None
        self._program = None


def _hardware_memory(buffers):
    from tools.mgchw import ByteMemory

    class ManagedByteMemory(ByteMemory):
        def __init__(self):
            super().__init__()
            for buffer in buffers:
                complete_word_size = buffer.size // 4 * 4
                if complete_word_size:
                    self.map_region(buffer.address, complete_word_size, name=f"driver buffer {buffer.id}")

        def _locate(self, address, write=False):
            self._check(address, write)
            for buffer in buffers:
                if buffer.address <= address and address + 4 <= buffer.address + buffer.size:
                    return buffer, address - buffer.address
            raise ResourceError(f"byte address {address:#x} is outside submitted buffers")

        def read32(self, address):
            buffer, offset = self._locate(address)
            return int.from_bytes(buffer._data[offset:offset + 4], "little")

        def write32(self, address, value):
            buffer, offset = self._locate(address, True)
            integer(value, "memory word", 0, ADDRESS_LIMIT - 1)
            buffer._data[offset:offset + 4] = value.to_bytes(4, "little")

    return ManagedByteMemory()


class HardwareSimulationBackend(SimulationBackend):
    """Evidence-supported mgchw CPU, explicitly distinct from legacy simulation."""

    format = ProgramFormat.HARDWARE
    name = "hardware-reference-simulation"

    def __init__(self, *, memory_base=0, memory_size=1 << 20, quantum=256,
                 allow_approximate_fp16=False):
        super().__init__(memory_base=memory_base, memory_size=memory_size, quantum=quantum)
        if not isinstance(allow_approximate_fp16, bool):
            raise ValidationError("allow_approximate_fp16 must be an explicit boolean")
        self.allow_approximate_fp16 = allow_approximate_fp16

    def start(self, program, buffers, entry, max_steps):
        from tools.mgchw import HardwareCPU

        self._cpu = HardwareCPU(memory=_hardware_memory(buffers),
                                allow_approximate_fp16=self.allow_approximate_fp16)
        self._cpu.load_program(list(program.words))
        self._cpu.pc = entry
        self._program = program
        self._max_steps = max_steps

    def poll(self):
        for _ in range(self.quantum):
            if self._cpu.pc == len(self._program.words):
                self._cpu.step()
                return ExecutionResult(self.name, self._cpu.steps, tuple(self._cpu.registers))
            if self._cpu.steps >= self._max_steps:
                raise ExecutionTimeoutError(f"hardware reference exceeded max_steps={self._max_steps}")
            if not 0 <= self._cpu.pc < len(self._program.words):
                raise SubmissionError(f"hardware program counter {self._cpu.pc} escaped the loaded program")
            self._cpu.step()
            if self._cpu.halted or self._cpu.pc == len(self._program.words):
                if not self._cpu.halted:
                    self._cpu.step()
                return ExecutionResult(self.name, self._cpu.steps, tuple(self._cpu.registers))
        return None


class MMIOTransport(Protocol):
    """Board-owned bounded I/O, cache maintenance and device-ordering contract."""

    def read32(self, offset: int) -> int: ...
    def write32(self, offset: int, value: int) -> None: ...
    def read(self, byte_address: int, size: int) -> bytes: ...
    def write(self, byte_address: int, data: bytes) -> None: ...
    def sync_for_device(self, byte_address: int, size: int) -> None: ...
    def sync_for_cpu(self, byte_address: int, size: int) -> None: ...
    def barrier(self) -> None: ...


@dataclass(frozen=True)
class MMIOConfig:
    memory_base: int
    memory_size: int
    program_base: int
    program_capacity: int
    control_offset: int
    status_offset: int
    entry_offset: int
    word_count_offset: int
    start_value: int
    busy_mask: int
    done_mask: int
    error_mask: int
    acknowledge_offset: int
    acknowledge_value: int
    abort_offset: int | None = None
    abort_value: int | None = None
    byteorder: str = "little"

    def __post_init__(self):
        for name in ("memory_base", "program_base", "control_offset", "status_offset",
                     "entry_offset", "word_count_offset", "acknowledge_offset"):
            value = integer(getattr(self, name), name, 0, ADDRESS_LIMIT - 1)
            if value % 4:
                raise ValidationError(f"{name} must be 4-byte aligned")
        for name in ("memory_size", "program_capacity"):
            value = integer(getattr(self, name), name, 4, ADDRESS_LIMIT)
            if value % 4:
                raise ValidationError(f"{name} must be a multiple of 4")
        for name in ("start_value", "acknowledge_value", "busy_mask", "done_mask", "error_mask"):
            integer(getattr(self, name), name, 0, ADDRESS_LIMIT - 1)
        if not self.busy_mask or not self.done_mask or not self.start_value:
            raise ValidationError("busy_mask, done_mask and start_value must be nonzero")
        if (self.busy_mask & self.done_mask or self.busy_mask & self.error_mask
                or self.done_mask & self.error_mask):
            raise ValidationError("busy/done/error masks must not overlap")
        if self.memory_base + self.memory_size > ADDRESS_LIMIT:
            raise ValidationError("data window exceeds 32-bit byte address space")
        if self.program_base + self.program_capacity > ADDRESS_LIMIT:
            raise ValidationError("program window exceeds 32-bit byte address space")
        if (self.memory_base < self.program_base + self.program_capacity
                and self.program_base < self.memory_base + self.memory_size):
            raise ValidationError("data and program windows must not overlap")
        if (self.abort_offset is None) != (self.abort_value is None):
            raise ValidationError("abort_offset and abort_value must be configured together")
        if self.abort_offset is not None:
            integer(self.abort_offset, "abort_offset", 0, ADDRESS_LIMIT - 1)
            integer(self.abort_value, "abort_value", 0, ADDRESS_LIMIT - 1)
            if self.abort_offset % 4:
                raise ValidationError("abort_offset must be 4-byte aligned")
        if self.byteorder not in ("little", "big"):
            raise ValidationError("byteorder must be 'little' or 'big'")
        if len({self.control_offset, self.status_offset, self.entry_offset,
                self.word_count_offset}) != 4:
            raise ValidationError("control/status/entry/word-count registers must be distinct")


class MMIOBackend:
    """No device-opening code: a caller must provide and explicitly authorize I/O."""

    format = ProgramFormat.HARDWARE
    name = "configured-mmio"

    def __init__(self, config: MMIOConfig, transport: MMIOTransport, *, allow_hardware: bool = False):
        if allow_hardware is not True:
            raise HardwareSafetyError("MMIO requires explicit allow_hardware=True and board configuration")
        if not isinstance(config, MMIOConfig):
            raise ValidationError("config must be MMIOConfig")
        for method in ("read32", "write32", "read", "write", "sync_for_device", "sync_for_cpu", "barrier"):
            if not callable(getattr(transport, method, None)):
                raise ValidationError(f"transport must implement {method}")
        self.config = config
        self.transport = transport
        self.memory_base = config.memory_base
        self.memory_size = config.memory_size
        self._launched = False
        self._uncertain = False
        self._buffers = ()

    def validate(self, program):
        if program.format != self.format:
            raise ProgramFormatError("MMIO accepts only explicitly tagged fp16-hardware-v1 programs")
        if program.size_bytes > self.config.program_capacity:
            raise ValidationError("program exceeds configured instruction window")

    def idle(self):
        status = integer(self.transport.read32(self.config.status_offset), "status", 0, ADDRESS_LIMIT - 1)
        return not status & self.config.busy_mask

    def _require_idle(self):
        if not self.idle():
            raise HardwareSafetyError("configured hardware is busy; no memory may be accessed")

    def write(self, buffer, offset, data):
        self._require_idle()
        self.transport.write(buffer.address + offset, data)
        self.transport.sync_for_device(buffer.address + offset, len(data))

    def read(self, buffer, offset, size):
        self._require_idle()
        self.transport.sync_for_cpu(buffer.address + offset, size)
        data = self.transport.read(buffer.address + offset, size)
        if not isinstance(data, bytes) or len(data) != size:
            raise SubmissionError("MMIO transport returned a short or invalid memory read")
        return data

    def start(self, program, buffers, entry, max_steps):
        self._require_idle()
        config = self.config
        transport = self.transport
        self._buffers = buffers
        transport.write32(config.acknowledge_offset, config.acknowledge_value)
        transport.barrier()
        stale = integer(transport.read32(config.status_offset), "status", 0, ADDRESS_LIMIT - 1)
        if stale & (config.busy_mask | config.done_mask | config.error_mask):
            raise HardwareSafetyError("status acknowledgement did not clear busy/done/error")
        transport.write(config.program_base, program.to_bytes(config.byteorder))
        transport.sync_for_device(config.program_base, program.size_bytes)
        for buffer in buffers:
            transport.sync_for_device(buffer.address, buffer.size)
        transport.write32(config.entry_offset, entry * 4)
        transport.write32(config.word_count_offset, len(program.words))
        transport.barrier()
        self._launched = True
        self._uncertain = True
        transport.write32(config.control_offset, config.start_value)
        transport.barrier()
        accepted = integer(transport.read32(config.status_offset), "status", 0, ADDRESS_LIMIT - 1)
        if not accepted & (config.busy_mask | config.done_mask | config.error_mask):
            raise HardwareSafetyError("start was not acknowledged by busy/done/error; hardware state is uncertain")
        self._uncertain = False

    def poll(self):
        config = self.config
        status = integer(self.transport.read32(config.status_offset), "status", 0, ADDRESS_LIMIT - 1)
        if status & config.error_mask:
            raise SubmissionError(f"hardware status indicates error ({status:#x})")
        if status & config.done_mask and not status & config.busy_mask:
            return ExecutionResult(self.name)
        return None

    def quiesce(self):
        if self._uncertain or (self._launched and not self.idle()):
            if self.config.abort_offset is None:
                return False
            self.transport.write32(self.config.abort_offset, self.config.abort_value)
            self.transport.barrier()
        if not self.idle():
            return False
        self._launched = False
        self._uncertain = False
        return True

    def finish(self, buffers):
        self._require_idle()
        self.transport.barrier()
        for buffer in buffers:
            self.transport.sync_for_cpu(buffer.address, buffer.size)
        self._launched = False
        self._uncertain = False
        self._buffers = ()
