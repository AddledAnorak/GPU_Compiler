"""Execution of canonical C effects and opt-in sample-derived forms."""

from dataclasses import dataclass
import math
import operator

from . import fp16
from .isa import ASSUMPTIONS, CONSTANTS, PDF_WAIT_SLOTS, WORD_MASK, DecodeError, decode_word
from .memory import Memory, MemoryFault

_COMPARE = (operator.lt, operator.le, operator.eq, operator.ge, operator.gt, operator.ne)


class ExecutionFault(RuntimeError):
    def __init__(self, pc, word, detail):
        self.pc, self.word = pc, word
        prefix = f"pc={pc}" if word is None else f"pc={pc}, word=0x{word:08x}"
        super().__init__(f"{prefix}: {detail}")


class AssumptionRequired(ExecutionFault):
    def __init__(self, pc, word, key):
        self.key = key
        super().__init__(pc, word, f"enable {key!r} to proceed: {ASSUMPTIONS[key]}")


class StepLimit(ExecutionFault):
    pass


@dataclass(frozen=True, slots=True)
class _PendingResult:
    pc: int
    operation: str
    wait: str
    ready_step: int


class TimingViolation(ExecutionFault):
    def __init__(self, pc, word, operation, resource, producer, step):
        self.resource = resource
        self.producer_pc = producer.pc
        self.missing_slots = producer.ready_step - step
        super().__init__(pc, word,
                         f"timing violation: {operation} reads {resource} too early after "
                         f"{producer.operation} at pc={producer.pc}; "
                         f"{self.missing_slots} more intervening slot(s) needed "
                         f"({producer.wait} or independent instructions)")


@dataclass(frozen=True, slots=True)
class Step:
    pc: int
    word: int
    operation: str
    executed: bool
    next_pc: int
    register_changes: tuple[tuple[int, int], ...]
    vector_changes: tuple[tuple[int, tuple[int, ...]], ...]
    flags_after: int
    low_immediate_after: int
    bank_after: int
    stopped: bool
    irq: bool
    memory_reads: tuple[tuple[int, int], ...]
    memory_writes: tuple[tuple[int, int], ...]
    assumptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RunResult:
    reason: str
    pc: int
    steps: int
    assumptions_used: tuple[str, ...]


def _signed(value):
    return value - (1 << 32) if value & (1 << 31) else value


def _lane_mask(vector):
    return sum(1 << lane for lane, value in enumerate(vector) if value < 0x80000000)


class Emulator:
    def __init__(self, words, *, memory=None, allow=(), timing_check=False):
        self.program = tuple(words)
        if not self.program or any(type(word) is not int or not 0 <= word <= WORD_MASK
                                   for word in self.program):
            raise ValueError("program must contain unsigned 32-bit instruction words")
        self.memory = Memory() if memory is None else memory
        self.allowed = frozenset(allow)
        if self.allowed - ASSUMPTIONS.keys():
            raise ValueError(f"unknown assumptions: {sorted(self.allowed - ASSUMPTIONS.keys())}")
        self.timing_check = timing_check
        self.reset()

    def reset(self):
        self.registers = [0] * 64
        self.vectors = [(0, 0, 0, 0) for _ in range(8)]
        self.flags = self.pc = self.low_immediate = self.bank = self.steps = 0
        self.stopped = self.irq = False
        self.assumptions_used: set[str] = set()
        self._step_assumptions: set[str] = set()
        self._pending_results: dict[tuple[str, int], _PendingResult] = {}

    def _require(self, key):
        if key not in self.allowed:
            raise AssumptionRequired(self.pc, self.program[self.pc], key)
        self._step_assumptions.add(key)

    def _set_flag(self, index, value):
        self.flags = (self.flags & ~(1 << index)) | (bool(value) << index)

    def _check_result(self, kind, index, form):
        pending = self._pending_results.get((kind, index))
        if pending is not None and self.steps < pending.ready_step:
            raise TimingViolation(self.pc, self.program[self.pc], form.name, f"{kind}{index}", pending, self.steps)

    def _record_result(self, kind, index, form):
        resource = (kind, index)
        if form.result_wait is None:
            self._pending_results.pop(resource, None)
        else:
            ready_step = self.steps + 1 + PDF_WAIT_SLOTS[form.result_wait]
            self._pending_results[resource] = _PendingResult(self.pc, form.name, form.result_wait, ready_step)

    def _step(self, trace):
        pc = self.pc
        if self.stopped:
            raise ExecutionFault(pc, None, "core is stopped")
        if not 0 <= pc < len(self.program):
            raise ExecutionFault(pc, None, "instruction fetch outside program")
        word = self.program[pc]
        try:
            instruction = decode_word(word)
        except DecodeError as error:
            raise ExecutionFault(pc, word, str(error)) from error
        form = instruction.form
        if self.timing_check and instruction.condition is not None:
            self._check_result("cr", instruction.condition, form)
        executed = (instruction.condition is None
                    or bool(self.flags & (1 << instruction.condition)) != instruction.invert)
        if trace:
            prior_registers = tuple(self.registers)
            prior_vectors = tuple(self.vectors)
        self._step_assumptions.clear()
        next_pc, next_bank = pc + 1, 0
        reads = writes = ()
        if executed:
            if not form.modeled:
                raise ExecutionFault(pc, word, f"{form.name}: encoding known, effect not modeled")
            if form.assumption:
                self._require(form.assumption)
            opcode, selector = form.opcode, form.selector
            if opcode == 0:
                if selector:
                    next_bank = ((word >> 18) & 4) | ((word >> 8) & 2) | ((word >> 4) & 1)
            elif opcode == 21:
                next_pc = (pc + instruction.offset + form.pc_bias) & WORD_MASK
            else:
                try:
                    target, reads, writes = self._execute(instruction)
                except (MemoryFault, fp16.NumericFault) as error:
                    raise ExecutionFault(pc, word, str(error)) from error
                if target is not None:
                    next_pc = target
        self.pc, self.bank = next_pc, next_bank
        self.steps += 1
        if self._step_assumptions:
            self.assumptions_used.update(self._step_assumptions)
        if trace:
            return Step(
                pc, word, form.name, executed, next_pc,
                tuple((index, value) for index, value in enumerate(self.registers)
                      if value != prior_registers[index]),
                tuple((index, value) for index, value in enumerate(self.vectors)
                      if value != prior_vectors[index]),
                self.flags, self.low_immediate, self.bank, self.stopped, self.irq,
                reads, writes, tuple(sorted(self._step_assumptions)),
            )
        return None

    def _execute(self, instruction):
        form = instruction.form
        opcode, selector = form.opcode, form.selector
        registers, vectors = self.registers, self.vectors
        source_a, source_b, dst = instruction.source_a, instruction.source_b, instruction.destination
        reg_a = source_a | ((self.bank & 4) << 3)
        reg_b = source_b | ((self.bank & 2) << 4)
        reg_dst = dst | ((self.bank & 1) << 5)
        if self.timing_check:
            if "a" in form.operands and not instruction.constant_a:
                self._check_result("r", reg_a, form)
            if "b" in form.operands:
                self._check_result("r", reg_b, form)
        value_a = CONSTANTS[source_a & 7] if instruction.constant_a else registers[reg_a]
        value_b = registers[reg_b]
        target = None
        reads = writes = ()
        if opcode == 1:
            if selector:
                registers[reg_dst] = instruction.immediate << 16 | self.low_immediate
            else:
                self.low_immediate = instruction.immediate
        elif opcode == 2:
            if selector == 0:
                registers[reg_dst] = value_a
            elif selector == 1:
                registers[reg_dst] = value_a & 0x7FFF
            elif selector == 2:
                registers[reg_dst] = (value_a & 0xFFFF) ^ 0x8000
            else:
                registers[reg_dst] = (value_a & 0xFFFF) | 0x8000
        elif opcode == 3:
            power = source_b - 32 if source_b & 16 else source_b
            registers[reg_dst] = fp16.scale_power(value_a, power)
        elif 4 <= opcode <= 7:
            if opcode == 4 and selector == 0 and "sample-fadd-rne" in self.allowed:
                self._require("sample-fadd-rne")
                registers[reg_dst] = fp16.sample_binary(7, value_a, value_b)
            elif opcode == 7 and selector >= 2:
                registers[reg_dst] = fp16.root(selector, value_a, self._require)
            else:
                registers[reg_dst] = fp16.binary(opcode, selector, value_a, value_b, self._require)
        elif opcode == 8:
            registers[reg_dst] = fp16.round_to_integer(selector, value_a, self._require)
        elif opcode in (9, 10):
            width = 8 if opcode == 9 else 16
            mask = (1 << width) - 1
            shift = selector * width
            field = fp16.packed_integer(value_a, mask, self._require)
            registers[reg_dst] = (value_b & ~(mask << shift)) | field << shift
        elif opcode == 11:
            registers[reg_dst] = fp16.integer32(value_a, self._require)
        elif 12 <= opcode <= 14:
            if opcode != 14:
                width = 8 if opcode == 12 else 16
                value_b = (value_b >> (selector * width)) & ((1 << width) - 1)
            registers[reg_dst] = fp16.float_to_half(value_b, self._require)
        elif opcode == 15:
            left = fp16.half_to_float(value_a, self._require)
            right = fp16.half_to_float(value_b, self._require)
            ordered = not (math.isnan(left) or math.isnan(right))
            if selector < len(_COMPARE):
                value = ordered and _COMPARE[selector](left, right)
            else:
                value = ordered if selector == 6 else not ordered
            self._set_flag(dst, value)
        elif opcode == 16:
            if selector == 0:
                registers[reg_dst] = (value_a + value_b) & WORD_MASK
            elif selector == 1:
                registers[reg_dst] = value_a & value_b
            elif selector == 2:
                registers[reg_dst] = (value_a + 4 * value_b) & WORD_MASK
            elif selector == 3:
                vectors[dst] = tuple((value_a + lane * value_b) & WORD_MASK for lane in range(4))
            elif selector == 4:
                vectors[dst] = (value_a,) * 4
            elif selector == 5:
                vectors[dst] = tuple((left + right) & WORD_MASK
                                     for left, right in zip(vectors[source_a], vectors[source_b]))
            elif selector == 6:
                vectors[dst] = fp16.sample_setup(value_a, value_b)
            elif selector == 11:
                fp16_scale = "sample-rgb565-fp16-scale" in self.allowed
                if fp16_scale:
                    self._require("sample-rgb565-fp16-scale")
                vectors[dst] = tuple(fp16.sample_ceramic(value, fp16_scale) for value in vectors[source_a])
            else:
                vectors[dst] = tuple(fp16.sample_binary(selector, left, right)
                                     for left, right in zip(vectors[source_a], vectors[source_b]))
        elif opcode == 19 or opcode == 23:
            value = self.memory.read(value_b)
            reads = ((value_b, value),)
            if opcode == 19:
                registers[reg_dst] = value
            else:
                target = value
        elif opcode == 20:
            self.memory.write(value_b, value_a)
            writes = ((value_b, value_a),)
        elif opcode == 22:
            target = value_b
        elif opcode == 24:
            self.stopped = self.irq = True
        elif opcode == 25:
            values = (value_a,) * 4 if selector == 3 else vectors[source_a]
            writes = tuple((address, value) for lane, (address, value) in enumerate(zip(vectors[dst], values))
                           if value_b & (1 << lane))
            self.memory.write_lanes(writes)
        elif opcode == 26:
            self._set_flag(dst, _COMPARE[selector](_signed(value_a), _signed(value_b)))
        elif opcode == 27:
            vectors_a = vectors[source_a]
            registers[reg_dst] = _lane_mask(vectors_a) if selector == 1 else vectors_a[0]
        elif opcode == 28:
            registers[reg_dst] = _lane_mask(vectors[source_a]) & value_b
        else:
            raise ExecutionFault(self.pc, self.program[self.pc], f"{instruction.form.name}: no execution handler")
        if self.timing_check:
            if "d" in form.operands:
                self._record_result("r", reg_dst, form)
            elif "cr" in form.operands:
                self._record_result("cr", dst, form)
        return target, reads, writes

    def step(self) -> Step:
        """Execute one instruction and return its observable changes."""
        return self._step(trace=True)

    def run(self, *, max_steps: int, stop_pc: int | None = None, trace=None) -> RunResult:
        """Stop at a guest halt, an explicit host inspection PC, or a fault."""
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if stop_pc is not None and (type(stop_pc) is not int or not 0 <= stop_pc < len(self.program)):
            raise ValueError("inspection PC must be inside the program")
        initial = self.steps
        while not self.stopped and self.pc != stop_pc:
            if self.steps - initial >= max_steps:
                word = self.program[self.pc] if 0 <= self.pc < len(self.program) else None
                raise StepLimit(self.pc, word, f"instruction limit ({max_steps}) reached")
            instruction_trace = self._step(trace is not None)
            if trace is not None:
                trace(instruction_trace)
        return RunResult("stop_core" if self.stopped else "inspection_pc", self.pc,
                         self.steps - initial, tuple(sorted(self.assumptions_used)))
